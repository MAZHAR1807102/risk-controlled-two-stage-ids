"""Aggregate pipeline runs: mean +/- std across seeds, plus bootstrap 95% CIs.

Seed variation only reflects Stage 2's row/column subsampling (Stage 1 and the data
splits are deterministic), so the bootstrap over test rows is reported alongside it
to capture sampling uncertainty in the small classes (e.g. 319 BruteForce flows).

Usage: .venv/bin/python src/aggregate.py --runs runs/seed1 runs/seed2 ... --out results/seeds
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from eval_offline import CLASSES as _CICIOT2023_CLASSES

# Overwritten in main() from the runs' metrics.json (older runs have no "classes" field).
CLASSES = list(_CICIOT2023_CLASSES)
BENIGN = CLASSES.index("Benign")
ATTACKS = [i for i in range(len(CLASSES)) if i != BENIGN]


def metrics_from_preds(y, pred, flag):
    """Per-class P/R/F1 from integer labels, plus Gatekeeper safety numbers."""
    k = len(CLASSES)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    missed = np.bincount(y, weights=~flag, minlength=k)
    group_miss = missed / np.maximum(np.bincount(y, minlength=k), 1)
    ben = y == BENIGN
    out = {"accuracy": tp.sum() / len(y), "macro_f1": f1.mean(),
           "macro_recall": rec.mean(),
           "benign_bypass": float((~flag[ben]).mean()),
           "overall_fnr": float((~flag[~ben]).mean()),
           "worst_group_miss": float(group_miss[ATTACKS].max())}
    out.update({f"f1_{c}": f1[i] for i, c in enumerate(CLASSES)})
    out.update({f"recall_{c}": rec[i] for i, c in enumerate(CLASSES)})
    out.update({f"miss_{CLASSES[i]}": group_miss[i] for i in ATTACKS})
    return out


def load_run(run):
    m = json.load(open(f"{run}/metrics.json"))
    z = np.load(f"{run}/test_scores.npz")
    y = z["y"].astype(np.int64)
    flag = z["p_att"] >= m["stage1"]["tau"]
    pred = np.where(flag, z["p_cls"].astype(np.float32).argmax(1), BENIGN)
    return m, y, pred, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/seeds")
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global CLASSES, BENIGN, ATTACKS
    CLASSES = json.load(open(f"{args.runs[0]}/metrics.json")).get("classes", CLASSES)
    BENIGN = CLASSES.index("Benign")
    ATTACKS = [i for i in range(len(CLASSES)) if i != BENIGN]

    rows, extra = [], []
    for run in args.runs:
        m, y, pred, flag = load_run(run)
        rows.append({"run": run, **metrics_from_preds(y, pred, flag)})
        extra.append({"run": run, "seed": m["seed"], "tau": m["stage1"]["tau"],
                      "stage2_rounds": m["stage2"]["best_iteration"] + 1,
                      "stage2_weighting": m["stage2"].get("weighting")})
    per_seed = pd.DataFrame(rows).set_index("run")
    per_seed.join(pd.DataFrame(extra).set_index("run")).to_csv(f"{args.out}/per_seed.csv")

    # Bootstrap over test rows, for every run, pooled.
    rng = np.random.default_rng(0)
    boot = []
    for run in args.runs:
        _, y, pred, flag = load_run(run)
        n = len(y)
        for _ in range(args.boot // len(args.runs) or 1):
            idx = rng.integers(0, n, n)
            boot.append(metrics_from_preds(y[idx], pred[idx], flag[idx]))
    boot = pd.DataFrame(boot)

    summary = pd.DataFrame({
        "mean": per_seed.mean(), "std_over_seeds": per_seed.std(ddof=1),
        "min": per_seed.min(), "max": per_seed.max(),
        "boot_ci95_low": boot.quantile(0.025), "boot_ci95_high": boot.quantile(0.975)})
    summary.to_csv(f"{args.out}/summary.csv")

    pd.set_option("display.width", 200)
    print(f"{len(args.runs)} runs, {len(boot)} bootstrap resamples\n")
    print(pd.DataFrame(extra).to_string(index=False))
    print()
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
