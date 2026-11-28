"""Tuned single-stage baselines (answer to referee A10: "the baselines look under-tuned").

Same data, scaler and metrics as baselines.py. Every model family gets a small grid; each
configuration is fitted on the full train split and scored on the VALIDATION split only
(macro-F1). The best configuration per family is then scored once on the test split and
exported to ONNX for timing (one thread, one flow per call, same 20,000 flows as baselines.py).

Search spaces (fixed before any run):
  LightGBM  num_leaves {31, 127, 255} x class weights {sqrt, none}; learning rate 0.1,
            early stopping on validation (patience 50, at most 2,000 rounds)
  RF        100 trees, max_depth {16, 24} x min_samples_leaf {2, 10} x class weights {sqrt, none}
  MLP       hidden {(128,64), (256,128), (256,128,64)} x L2 alpha {1e-4, 1e-3};
            Adam, batch 4096, at most 30 epochs, early stopping on 10% of train, no class weights
            (as in baselines.py)

Usage: .venv/bin/python src/baselines_tuned.py search   # grid on validation -> results/baselines_tuned/search.csv
       .venv/bin/python src/baselines_tuned.py test     # best per family on test + ONNX export
       .venv/bin/python src/baselines_tuned.py time     # ONNX timing of the tuned models
"""
import argparse
import itertools
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

import baselines as B
from pipeline import class_weights

OUT = "results/baselines_tuned"
MODELS = f"{OUT}/models"
SEED = 42

GRIDS = {
    "lgbm": [{"num_leaves": nl, "weights": w} for nl, w in itertools.product((31, 127, 255), ("sqrt", "none"))],
    "rf": [{"max_depth": d, "min_samples_leaf": l, "weights": w}
           for d, l, w in itertools.product((16, 24), (2, 10), ("sqrt", "none"))],
    "mlp": [{"hidden": h, "alpha": a} for h, a in itertools.product(((128, 64), (256, 128), (256, 128, 64)),
                                                                    (1e-4, 1e-3))],
}


def key(fam, cfg):
    return fam + ":" + json.dumps(cfg, sort_keys=True)


def fit(fam, cfg, Xtr, ytr, Xva, yva, cw):
    sw = cw[ytr] if cfg.get("weights", "none") == "sqrt" else None
    if fam == "lgbm":
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=2000, learning_rate=0.1,
                               num_leaves=cfg["num_leaves"], subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=3.0, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)],
              eval_sample_weight=[cw[yva]] if sw is not None else None,
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return m
    if fam == "rf":
        m = RandomForestClassifier(n_estimators=100, max_depth=cfg["max_depth"],
                                   min_samples_leaf=cfg["min_samples_leaf"], n_jobs=-1, random_state=SEED)
        m.fit(Xtr, ytr, sample_weight=sw)
        return m
    m = MLPClassifier(hidden_layer_sizes=cfg["hidden"], alpha=cfg["alpha"], batch_size=4096, max_iter=30,
                      early_stopping=True, n_iter_no_change=3, random_state=SEED)
    m.fit(Xtr, ytr)
    return m


def predict(m, X):
    return np.asarray(m.predict(X)).astype(int)


def search():
    os.makedirs(MODELS, exist_ok=True)
    path = f"{OUT}/search.csv"
    done = set(pd.read_csv(path)["key"]) if os.path.exists(path) else set()
    Xtr, ytr, _ = B.scaled("train")
    Xva, yva, _ = B.scaled("validation")
    cw = class_weights(ytr, "sqrt")
    for fam, grid in GRIDS.items():
        for cfg in grid:
            k = key(fam, cfg)
            if k in done:
                continue
            t = time.time()
            m = fit(fam, cfg, Xtr, ytr, Xva, yva, cw)
            pv = predict(m, Xva)
            row = {"key": k, "family": fam, "config": json.dumps(cfg), "val_macro_f1": f1_score(yva, pv, average="macro"),
                   "val_accuracy": accuracy_score(yva, pv), "fit_s": time.time() - t,
                   "rounds": getattr(m, "best_iteration_", None)}
            pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            B.log(f"{fam} {cfg}: val macro-F1 {row['val_macro_f1']:.4f} [{row['fit_s']:.0f}s]")
            del m


def best_configs():
    s = pd.read_csv(f"{OUT}/search.csv")
    return {fam: json.loads(g.sort_values("val_macro_f1").iloc[-1].config) for fam, g in s.groupby("family")}


def test():
    Xtr, ytr, _ = B.scaled("train")
    Xva, yva, _ = B.scaled("validation")
    Xte, yte, gte = B.scaled("test")
    cw = class_weights(ytr, "sqrt")
    fams = [c for c in B.CLASSES if c != "Benign"]
    rows = []
    for fam, cfg in best_configs().items():
        if "hidden" in cfg:
            cfg["hidden"] = tuple(cfg["hidden"])
        m = fit(fam, cfg, Xtr, ytr, Xva, yva, cw)
        pred = predict(m, Xte)
        miss = {f: float((pred[gte == f] == B.BENIGN).mean()) for f in fams}
        ben = yte == B.BENIGN
        if fam == "lgbm":
            onx = convert_lightgbm(m.booster_, initial_types=[("x", FloatTensorType([None, 46]))], target_opset=15,
                                   zipmap=False)
        else:
            onx = to_onnx(m, Xtr[:1], target_opset=15, options={id(m): {"zipmap": False}})
        with open(f"{MODELS}/{fam}_tuned.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        rows.append({"method": f"{fam}_tuned", "config": json.dumps(cfg), "accuracy": accuracy_score(yte, pred),
                     "macro_f1": f1_score(yte, pred, average="macro"), "worst_family_miss": max(miss.values()),
                     "worst_family": max(miss, key=miss.get), "benign_flagged": float((pred[ben] != B.BENIGN).mean()),
                     "model_mb": os.path.getsize(f"{MODELS}/{fam}_tuned.onnx") / 1e6, **{f"miss_{f}": v for f, v in miss.items()}})
        B.log(f"{fam} tuned {cfg}: test macro-F1 {rows[-1]['macro_f1']:.4f}, worst miss {rows[-1]['worst_family_miss']:.4f}")
        del m
    pd.DataFrame(rows).to_csv(f"{OUT}/test.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def time_tuned(traffic="ns3/traffic/test_benign90_20000.csv", warmup=200, repeats=5):
    s = B.scaler_from_header(f"{B.RUN}/ids_params.h")
    d = pd.read_csv(traffic)
    X = ((d.iloc[:, :46].to_numpy(np.float32) - np.float32(s["means"])) / np.float32(s["scales"])).astype(np.float32)
    rows = []
    for fam in GRIDS:
        path = f"{MODELS}/{fam}_tuned.onnx"
        if not os.path.exists(path):
            continue
        sess = B.onnx_session(open(path, "rb").read())
        name = sess.get_inputs()[0].name
        per_rep, pooled = [], []
        for _ in range(repeats):
            ts = []
            for i in range(len(X)):
                t0 = time.perf_counter_ns()
                sess.run(None, {name: X[i:i + 1]})
                ts.append(time.perf_counter_ns() - t0)
            t = np.array(ts[warmup:]) / 1e3
            per_rep.append(t.mean())
            pooled.append(t)
        t = np.concatenate(pooled)
        rows.append({"method": f"{fam}_tuned", "mean_us": t.mean(), "mean_sd_over_repeats": float(np.std(per_rep, ddof=1)),
                     "p50_us": np.median(t), "p99_us": np.quantile(t, 0.99), "repeats": repeats})
        B.log(f"{fam}_tuned: mean {t.mean():.1f} us")
    pd.DataFrame(rows).to_csv(f"{OUT}/timing.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["search", "test", "time"])
    os.makedirs(MODELS, exist_ok=True)
    {"search": search, "test": test, "time": time_tuned}[ap.parse_args().phase]()
