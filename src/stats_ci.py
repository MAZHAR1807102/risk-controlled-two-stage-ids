"""Uncertainty analysis for the revision (reviewer point 2.8).

1. CICIoT2023 two-stage metrics: 1,000 bootstrap replicates; each replicate draws one of the
   five training seeds at random and resamples the test flows with replacement, so the
   interval covers both test-sampling and training variation (percentile intervals).
2. CICIoMT2024: the test rows are in time order inside each capture, so a moving-block
   bootstrap (blocks of 500 consecutive rows) is compared with the i.i.d. bootstrap.
3. Gatekeeper skip rate per family and per attack type with exact Clopper-Pearson 95%
   intervals, at the efficiency threshold and at the certified threshold.
4. Paired bootstrap of the difference in macro-F1 and in worst-family miss between the
   proposed system and every baseline (same resampled flows for both systems).

Usage: .venv/bin/python src/stats_ci.py      (needs baselines.py eval to have saved predictions)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import beta

from datasets import get_dataset

B = 1000
RNG = np.random.default_rng(2026)
OUT = "results/stats"


def macro_f1(y, pred, k):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return f1.mean()


def worst_miss(y, pred, benign, k):
    m = [((pred[y == c] == benign).mean()) for c in range(k) if c != benign and (y == c).any()]
    return max(m)


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def two_stage(run):
    m = json.load(open(f"{run}/metrics.json"))
    z = np.load(f"{run}/test_scores.npz")
    C = m.get("classes", ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"])
    b = C.index("Benign")
    flag = z["p_att"] >= m["stage1"]["tau"]
    pred = np.where(flag, z["p_cls"].astype(np.float32).argmax(1), b)
    return z["y"].astype(np.int64), pred, b, len(C), z["p_att"]


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rows = []

    # 1. CICIoT2023, seeds + test sampling
    runs = [f"runs/seed{s}" for s in range(1, 6)]
    data = [two_stage(r) for r in runs]
    y0, _, b, k, _ = data[0]
    n = len(y0)
    f1s, accs, wms = [], [], []
    for _ in range(B):
        y, pred, _, _, _ = data[RNG.integers(len(data))]
        idx = RNG.integers(0, n, n)
        f1s.append(macro_f1(y[idx], pred[idx], k))
        accs.append((y[idx] == pred[idx]).mean())
        wms.append(worst_miss(y[idx], pred[idx], b, k))
    for name, v in (("macro_f1", f1s), ("accuracy", accs), ("worst_family_end_to_end_miss", wms)):
        rows.append({"dataset": "ciciot2023", "metric": name, "method": "seed+iid bootstrap",
                     "mean": float(np.mean(v)), "ci_low": float(np.quantile(v, 0.025)),
                     "ci_high": float(np.quantile(v, 0.975))})

    # 2. CICIoMT2024, block vs iid bootstrap
    y, pred, b, k, _ = two_stage("runs/iomt_seed42")
    n = len(y)
    L = 500
    starts = np.arange(0, n - L + 1)
    for scheme in ("iid", "block500"):
        v = []
        for _ in range(B):
            if scheme == "iid":
                idx = RNG.integers(0, n, n)
            else:
                s = RNG.choice(starts, size=int(np.ceil(n / L)))
                idx = (s[:, None] + np.arange(L)[None, :]).ravel()[:n]
            v.append(macro_f1(y[idx], pred[idx], k))
        rows.append({"dataset": "ciciomt2024", "metric": "macro_f1", "method": f"{scheme} bootstrap",
                     "mean": float(np.mean(v)), "ci_low": float(np.quantile(v, 0.025)),
                     "ci_high": float(np.quantile(v, 0.975))})
    pd.DataFrame(rows).to_csv(f"{OUT}/metric_ci.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 3. Gatekeeper skip per family and per type with Clopper-Pearson intervals
    rc = pd.read_csv("results/risk_control/summary.csv")
    gk_rows = []
    for ds_name, run in (("ciciot2023", "runs/seed42_sqrt"), ("ciciomt2024", "runs/iomt_seed42")):
        ds = get_dataset(ds_name)
        _, fam, typ, _ = ds.load("test")
        p = np.load(f"{run}/test_scores.npz")["p_att"]
        taus = {"efficiency": float(json.load(open(f"{run}/metrics.json"))["stage1"]["tau"]),
                "certified": float(rc[(rc.dataset == ds_name) & (rc.calibration == "validation")
                                      & (rc.rule == "ucb_typ")].tau.iloc[0])}
        for point, tau in taus.items():
            for level, labels in (("family", fam), ("type", typ)):
                for u in np.unique(labels[fam != "Benign"]):
                    m = labels == u
                    kk, nn = int((p[m] < tau).sum()), int(m.sum())
                    lo, hi = cp(kk, nn)
                    gk_rows.append({"dataset": ds_name, "threshold": point, "tau": tau, "level": level, "unit": u,
                                    "skipped": kk, "n": nn, "rate": kk / nn, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(gk_rows).to_csv(f"{OUT}/gatekeeper_skip_ci.csv", index=False)

    # 4. Paired differences vs baselines (CICIoT2023, seed-42 models)
    y = np.load("results/baselines/y_test.npy").astype(np.int64)
    b, k = 0, 8
    preds = {m: np.load(f"results/baselines/pred_{m}.npz")["pred"].astype(np.int64)
             for m in ["xgb_single", "lgbm_single", "rf_single", "mlp_single", "ae_filter", "unweighted_gk",
                       "proposed", "proposed_certified"]}
    n = len(y)
    diffs = {m: {"f1": [], "wm": []} for m in preds if m != "proposed"}
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        f_ref = macro_f1(y[idx], preds["proposed"][idx], k)
        w_ref = worst_miss(y[idx], preds["proposed"][idx], b, k)
        for m in diffs:
            diffs[m]["f1"].append(f_ref - macro_f1(y[idx], preds[m][idx], k))
            diffs[m]["wm"].append(w_ref - worst_miss(y[idx], preds[m][idx], b, k))
    drows = []
    for m, d in diffs.items():
        for key, name in (("f1", "macro_f1_diff (proposed - baseline)"), ("wm", "worst_miss_diff (proposed - baseline)")):
            v = np.array(d[key])
            drows.append({"baseline": m, "quantity": name, "mean": v.mean(), "ci_low": np.quantile(v, 0.025),
                          "ci_high": np.quantile(v, 0.975)})
    pd.DataFrame(drows).to_csv(f"{OUT}/paired_diff.csv", index=False)
    print(pd.DataFrame(drows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
