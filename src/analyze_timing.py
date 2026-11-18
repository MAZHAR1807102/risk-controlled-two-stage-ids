"""Summarise the ns-3 v2 timing matrix (ns3/timing/*.csv, written by ns3/run_timing.sh).

For each traffic mix / model / thread setting it pairs the two-stage run with the
single-stage run that was actually executed on the same flows, and reports measured
CPU saving, per-packet latency percentiles and Gatekeeper misses per attack group.

Usage: .venv/bin/python src/analyze_timing.py [--dir ns3/timing] [--warmup 200]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from eval_offline import CLASSES

BENIGN = CLASSES.index("Benign")


def summarize(path, warmup):
    mix, model, mode, threads, api = os.path.basename(path)[:-4].split("__")
    d = pd.read_csv(path).iloc[warmup:]
    total_us = (d.scale_ns + d.stage1_ns + d.stage2_ns) / 1e3
    att = d.label != BENIGN
    row = {"mix": mix, "model": model, "mode": mode, "threads": threads[1:], "api": api,
           "packets": len(d),
           "total_cpu_ms": total_us.sum() / 1e3,
           "mean_us": total_us.mean(),
           "p50_us": total_us.quantile(0.50), "p95_us": total_us.quantile(0.95),
           "p99_us": total_us.quantile(0.99),
           "scale_mean_us": d.scale_ns.mean() / 1e3,
           "stage1_mean_us": d.stage1_ns[d.stage1_ns > 0].mean() / 1e3 if mode == "two-stage" else np.nan,
           "stage2_mean_us": d.stage2_ns[d.inspected == 1].mean() / 1e3,
           "inspected_frac": d.inspected.mean(),
           "benign_bypass": 1 - d.inspected[~att].mean(),
           "attacks_missed": int((~d.inspected.astype(bool) & att).sum()),
           "attacks": int(att.sum()),
           "accuracy": (d.predicted == d.label).mean()}
    # Per-group miss (only meaningful for two-stage).
    for i, c in enumerate(CLASSES):
        if i != BENIGN:
            g = d.label == i
            row[f"miss_{c}"] = 1 - d.inspected[g].mean() if g.any() else np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ns3/timing")
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()

    df = pd.DataFrame([summarize(p, args.warmup) for p in sorted(glob.glob(f"{args.dir}/*.csv"))
                       if not p.endswith("summary.csv")])
    key = ["mix", "model", "threads", "api"]
    single = df[df["mode"] == "single-stage"].set_index(key)["total_cpu_ms"].rename("single_stage_cpu_ms")
    df = df.join(single, on=key)
    df["measured_saving"] = 1 - df.total_cpu_ms / df.single_stage_cpu_ms
    df.to_csv(f"{args.dir}/summary.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["mix", "model", "mode", "threads", "api", "total_cpu_ms", "measured_saving",
            "p50_us", "p95_us", "p99_us", "stage1_mean_us", "stage2_mean_us",
            "benign_bypass", "attacks_missed", "attacks", "accuracy"]
    print(df[cols].sort_values(["mix", "threads", "model", "mode"])
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    two = df[df["mode"] == "two-stage"]
    print("\nGatekeeper miss rate by attack group (two-stage runs):")
    print(two[["mix", "model", "threads"] + [c for c in df if c.startswith("miss_")]]
          .drop_duplicates(["mix", "model"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
