"""Fair comparison of the proposed two-stage IDS with representative baselines (CICIoT2023).

Same splits, same scaler (runs/seed42_sqrt/ids_params.h), same metrics, and the same
inference engine for timing (ONNX Runtime, one thread, one flow per call), so library
overheads do not distort the comparison.

Methods
  xgb_single        single-stage XGBoost (the proposed Stage 2 alone)
  lgbm_single       single-stage LightGBM, same class weights and early stopping
  rf_single         single-stage Random Forest
  mlp_single        single-stage MLP (128-64)
  ae_filter         autoencoder trained on benign traffic only, as first stage (Wei et al.
                    style), threshold chosen with the same per-family rule, + proposed Stage 2
  unweighted_gk     unweighted Gatekeeper at tau=0.3 + proposed Stage 2
  proposed          family-balanced Gatekeeper at tau* + proposed Stage 2

All models are trained on the full train split (revision: no per-class cap), select their
hyperparameters/early stopping on the validation split only, and are timed in the same engine.
"proposed_certified" is the same system at the risk-controlled threshold (risk_control.py).

Usage:
  .venv/bin/python src/baselines.py train   # fits the new baselines, exports ONNX models
  .venv/bin/python src/baselines.py eval    # full test split metrics
  .venv/bin/python src/baselines.py time    # per-flow latency; run on an idle machine
"""
import argparse
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

from datasets import get_dataset
import pipeline
from pipeline import class_weights
from train_gatekeeper import scaler_from_header

RUN = "runs/seed42_sqrt"
GK_UNWEIGHTED = "results/gatekeeper_weighting_final/gatekeeper_unweighted.json"
OUT = "results/baselines"
MODELS = f"{OUT}/models"
CAP = None  # None = full train split for every model
SEED = 42
DS = get_dataset("ciciot2023")
CLASSES = DS.classes
BENIGN = CLASSES.index("Benign")
pipeline.CLASSES = CLASSES  # class_weights() reads the class list from the pipeline module


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scaled(split):
    X, g, raw, _ = DS.load(split)
    s = scaler_from_header(f"{RUN}/ids_params.h")
    X = (X - np.float32(s["means"])) / np.float32(s["scales"])
    y = np.array([CLASSES.index(c) for c in g])
    return X.astype(np.float32), y, g


def capped_index(y, cap, rng):
    if cap is None:
        return np.arange(len(y))
    idx = [rng.permutation(np.flatnonzero(y == c))[:cap] for c in range(len(CLASSES))]
    return np.sort(np.concatenate(idx))


def onnx_session(model_bytes):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])


def save_onnx(name, onx):
    path = f"{MODELS}/{name}.onnx"
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return path


def xgb_to_onnx(path, n_features=46):
    b = xgb.Booster()
    b.load_model(path)
    return convert_xgboost(b, initial_types=[("x", FloatTensorType([None, n_features]))], target_opset=15)


# ---------------------------------------------------------------------------- train
def train():
    os.makedirs(MODELS, exist_ok=True)
    rng = np.random.default_rng(SEED)
    Xtr, ytr, gtr = scaled("train")
    Xva, yva, gva = scaled("validation")
    cw = class_weights(ytr, "sqrt")
    log(f"train {len(ytr):,} rows, validation {len(yva):,}")

    # XGBoost models of the proposed system and the unweighted Gatekeeper -> ONNX.
    save_onnx("xgb_stage1_proposed", xgb_to_onnx(f"{RUN}/layer1_gatekeeper.ubj"))
    save_onnx("xgb_stage2", xgb_to_onnx(f"{RUN}/layer2_multiclass.ubj"))
    save_onnx("xgb_stage1_unweighted", xgb_to_onnx(GK_UNWEIGHTED))

    log("LightGBM (full train split, sqrt class weights, early stopping on validation)")
    lg = lgb.LGBMClassifier(objective="multiclass", n_estimators=1000, learning_rate=0.05,
                            num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            reg_lambda=3.0, random_state=SEED, n_jobs=-1, verbose=-1)
    lg.fit(Xtr, ytr, sample_weight=cw[ytr], eval_set=[(Xva, yva)], eval_sample_weight=[cw[yva]],
           callbacks=[lgb.early_stopping(50, verbose=False)])
    log(f"  best iteration {lg.best_iteration_}")
    lg.booster_.save_model(f"{MODELS}/lgbm.txt", num_iteration=lg.best_iteration_)
    lgb_small = lgb.Booster(model_file=f"{MODELS}/lgbm.txt")
    save_onnx("lgbm_single", convert_lightgbm(lgb_small, initial_types=[("x", FloatTensorType([None, 46]))],
                                              target_opset=15, zipmap=False))

    idx = capped_index(ytr, CAP, rng)
    Xc, yc = Xtr[idx], ytr[idx]
    log(f"training rows for RF/MLP: {len(yc):,}")

    log("Random Forest (100 trees, depth 16)")
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=2,
                                n_jobs=-1, random_state=SEED)
    rf.fit(Xc, yc, sample_weight=cw[yc])
    joblib.dump(rf, f"{MODELS}/rf.joblib")
    save_onnx("rf_single", to_onnx(rf, Xc[:1], target_opset=15, options={id(rf): {"zipmap": False}}))

    log("MLP (128-64)")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), batch_size=4096, max_iter=20,
                        early_stopping=True, n_iter_no_change=3, random_state=SEED)
    mlp.fit(Xc, yc)
    joblib.dump(mlp, f"{MODELS}/mlp.joblib")
    save_onnx("mlp_single", to_onnx(mlp, Xc[:1], target_opset=15, options={id(mlp): {"zipmap": False}}))

    log("Autoencoder filter on benign traffic only (46-24-8-24-46)")
    ben = Xtr[gtr == "Benign"]
    ae = MLPRegressor(hidden_layer_sizes=(24, 8, 24), batch_size=1024, max_iter=60,
                      early_stopping=True, random_state=SEED)
    ae.fit(ben, ben)
    joblib.dump(ae, f"{MODELS}/ae.joblib")
    save_onnx("ae_filter", to_onnx(ae, Xc[:1], target_opset=15))

    # Autoencoder threshold: same per-family rule as the proposed Gatekeeper (validation).
    err = ae_error(ae.predict(Xva), Xva)
    tau = pick_threshold(err, gva, 0.02)
    json.dump({"ae_tau": tau}, open(f"{MODELS}/ae_tau.json", "w"))
    log(f"autoencoder threshold {tau:.5f} (validation, worst-family miss <= 2%)")


def ae_error(recon, X):
    # skl2onnx returns the multi-output regressor flattened; restore (rows, features).
    recon = np.asarray(recon).reshape(X.shape)
    return ((recon - X) ** 2).mean(1)


def pick_threshold(score, g, target):
    """Largest threshold whose worst attack-family miss rate on validation is <= target.
    A flow is flagged (sent to Stage 2) when score >= threshold."""
    fams = [f for f in np.unique(g) if f != "Benign"]
    cand = np.unique(np.quantile(score, np.linspace(0, 1, 2001)))
    best = cand[0]
    for t in cand:
        if max((score[g == f] < t).mean() for f in fams) <= target:
            best = t
    return float(best)


# ----------------------------------------------------------------------------- eval
def certified_tau():
    rc = pd.read_csv("results/risk_control/summary.csv")
    q = rc[(rc.dataset == "ciciot2023") & (rc.calibration == "validation") & (rc.rule == "ucb_typ")]
    return float(q.tau.iloc[0])


def final_labels(method, X, sessions, ae_tau):
    def s2(Xs):
        return sessions["xgb_stage2"].run(None, {"x": Xs})[1].argmax(1)

    if method in ("xgb_single", "lgbm_single", "rf_single", "mlp_single"):
        sess = sessions["xgb_stage2" if method == "xgb_single" else method]
        out = sess.run(None, {sess.get_inputs()[0].name: X})
        probs = out[1] if len(out) > 1 else out[0]
        return np.asarray(probs).argmax(1), np.ones(len(X), bool)
    if method == "ae_filter":
        recon = sessions["ae_filter"].run(None, {sessions["ae_filter"].get_inputs()[0].name: X})[0]
        flag = ae_error(recon, X) >= ae_tau
    else:
        gk = "xgb_stage1_unweighted" if method == "unweighted_gk" else "xgb_stage1_proposed"
        tau = {"proposed": json.load(open(f"{RUN}/metrics.json"))["stage1"]["tau"],
               "proposed_certified": certified_tau(), "unweighted_gk": 0.3}[method]
        p = np.asarray(sessions[gk].run(None, {"x": X})[1])[:, 1]
        flag = p >= tau
    pred = np.full(len(X), BENIGN)
    pred[flag] = s2(X[flag])
    return pred, flag


METHODS = ["xgb_single", "lgbm_single", "rf_single", "mlp_single", "ae_filter", "unweighted_gk", "proposed",
           "proposed_certified"]


def load_sessions():
    return {n[:-5]: onnx_session(open(f"{MODELS}/{n}", "rb").read())
            for n in os.listdir(MODELS) if n.endswith(".onnx")}


def evaluate():
    sessions = load_sessions()
    ae_tau = json.load(open(f"{MODELS}/ae_tau.json"))["ae_tau"]
    Xte, yte, gte = scaled("test")
    rows, fam_rows = [], []
    fams = [c for c in CLASSES if c != "Benign"]
    for m in METHODS:
        t = time.time()
        pred, flag = final_labels(m, Xte, sessions, ae_tau)
        miss = {f: float((pred[gte == f] == BENIGN).mean()) for f in fams}
        ben = yte == BENIGN
        rows.append({"method": m, "accuracy": accuracy_score(yte, pred),
                     "macro_f1": f1_score(yte, pred, average="macro"),
                     "worst_family_miss": max(miss.values()),
                     "worst_family": max(miss, key=miss.get),
                     "overall_miss": float((pred[~ben] == BENIGN).mean()),
                     "benign_bypass": float((~flag[ben]).mean()),
                     "model_mb": sum(os.path.getsize(f"{MODELS}/{f}.onnx") for f in model_files(m)) / 1e6})
        fam_rows.append({"method": m, **miss})
        np.savez_compressed(f"{OUT}/pred_{m}.npz", pred=pred.astype(np.int8), flag=flag)
        log(f"{m}: macro-F1 {rows[-1]['macro_f1']:.4f}, worst miss {rows[-1]['worst_family_miss']:.4f} "
            f"({rows[-1]['worst_family']}) [{time.time() - t:.0f}s]")
    pd.DataFrame(rows).to_csv(f"{OUT}/accuracy.csv", index=False)
    pd.DataFrame(fam_rows).to_csv(f"{OUT}/miss_by_family.csv", index=False)
    np.save(f"{OUT}/y_test.npy", yte.astype(np.int8))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def model_files(m):
    return {"xgb_single": ["xgb_stage2"], "lgbm_single": ["lgbm_single"], "rf_single": ["rf_single"],
            "mlp_single": ["mlp_single"], "ae_filter": ["ae_filter", "xgb_stage2"],
            "unweighted_gk": ["xgb_stage1_unweighted", "xgb_stage2"],
            "proposed": ["xgb_stage1_proposed", "xgb_stage2"],
            "proposed_certified": ["xgb_stage1_proposed", "xgb_stage2"]}[m]


# ----------------------------------------------------------------------------- time
def time_methods(traffic="ns3/traffic/test_benign90_20000.csv", warmup=200, repeats=5):
    """Per-flow latency, one flow per ONNX call, one thread, same flows for every method."""
    sessions = load_sessions()
    ae_tau = json.load(open(f"{MODELS}/ae_tau.json"))["ae_tau"]
    tau_p = json.load(open(f"{RUN}/metrics.json"))["stage1"]["tau"]
    tau_c = certified_tau()
    s = scaler_from_header(f"{RUN}/ids_params.h")
    d = pd.read_csv(traffic)
    X = ((d.iloc[:, :46].to_numpy(np.float32) - np.float32(s["means"])) / np.float32(s["scales"])).astype(np.float32)
    names = {k: v.get_inputs()[0].name for k, v in sessions.items()}

    def run(name, row):
        return sessions[name].run(None, {names[name]: row})

    def one(m, row):
        if m in ("xgb_single", "lgbm_single", "rf_single", "mlp_single"):
            run("xgb_stage2" if m == "xgb_single" else m, row)
            return
        if m == "ae_filter":
            flag = ae_error(run("ae_filter", row)[0], row)[0] >= ae_tau
        else:
            gk = "xgb_stage1_unweighted" if m == "unweighted_gk" else "xgb_stage1_proposed"
            tau = {"proposed": tau_p, "proposed_certified": tau_c, "unweighted_gk": 0.3}[m]
            flag = np.asarray(run(gk, row)[1])[0, 1] >= tau
        if flag:
            run("xgb_stage2", row)

    rows = []
    for m in METHODS:
        per_rep, pooled = [], []
        for r in range(repeats):  # methods are interleaved by repeat order below
            times = []
            for i in range(len(X)):
                row = X[i:i + 1]
                t0 = time.perf_counter_ns()
                one(m, row)
                times.append(time.perf_counter_ns() - t0)
            t = np.array(times[warmup:]) / 1e3
            per_rep.append(t.mean())
            pooled.append(t)
        t = np.concatenate(pooled)
        rows.append({"method": m, "mean_us": t.mean(), "mean_sd_over_repeats": float(np.std(per_rep, ddof=1)),
                     "p50_us": np.median(t), "p95_us": np.quantile(t, 0.95), "p99_us": np.quantile(t, 0.99),
                     "flows_per_s": 1e6 / t.mean(), "repeats": repeats})
        log(f"{m}: mean {t.mean():.1f} +- {np.std(per_rep, ddof=1):.1f} us, p99 {np.quantile(t, 0.99):.1f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/timing.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "eval", "time"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"train": train, "eval": evaluate, "time": time_methods}[a.phase]()
