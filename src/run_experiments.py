"""Experiments 1-2 for the journal version, run through the ns-3 v2 gateway.

  breakeven : CPU cost per packet vs benign share, for single-stage, two-stage with
              uniform / per-stage thread settings, and the adaptive Gatekeeper.
  timeline  : normal -> DDoS flood -> normal traffic, comparing static and adaptive
              routing packet by packet.

All runs use the new models (runs/seed42_sqrt) and flows from the TEST split.
Run on an otherwise idle machine; timings are wall-clock per packet.

Usage: .venv/bin/python src/run_experiments.py breakeven|timeline|all
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from eval_offline import CLASSES, macro_group
from make_ns3_traffic import TEST, build

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS3 = os.path.expanduser("~/NS3/ns-allinone-3.42/ns-3.42")
RUN = f"{HERE}/runs/seed42_sqrt"
OUT = f"{HERE}/ns3/experiments"
BENIGN = CLASSES.index("Benign")

SHARES = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
BREAKEVEN_ROWS = 10000

# name -> (mode, nthread1, nthread2). Thread "0" = XGBoost default (all cores).
CONFIGS = {
    "single_t1":        ("single-stage", "1", "1"),
    "single_tall":      ("single-stage", "0", "0"),
    "two_t1":           ("two-stage", "1", "1"),
    "two_tall":         ("two-stage", "0", "0"),
    "two_hybrid":       ("two-stage", "1", "0"),
    "adaptive_tall":    ("adaptive", "0", "0"),
    "adaptive_hybrid":  ("adaptive", "1", "0"),
}
# Flood scenario: (rows, benign share) per phase, in order.
TIMELINE = [(6000, 0.90), (8000, 0.02), (6000, 0.90)]
TIMELINE_CONFIGS = ["single_tall", "two_tall", "adaptive_tall", "two_hybrid", "adaptive_hybrid"]


def write_traffic(sub, path):
    feats = list(sub.columns[:46])
    out = sub[feats].copy()
    out["class_id"] = sub["group"].map(CLASSES.index).astype(int)
    out["raw_label"] = sub["label"].values
    out.to_csv(path, index=False)


def load_test():
    df = pd.read_csv(TEST)
    df["group"] = df["label"].map({l: macro_group(l) for l in df["label"].unique()})
    return df


def simulate(traffic, config, out_csv, rows):
    mode, t1, t2 = CONFIGS[config]
    args = (f"IoT_TwoStage_IDS_v2 --data={traffic} --params={RUN}/ids_params.h "
            f"--l1={RUN}/layer1_gatekeeper.json --l2={RUN}/layer2_multiclass.json "
            f"--mode={mode} --nthread1={t1} --nthread2={t2} --api=inplace "
            f"--rows={rows} --out={out_csv}")
    r = subprocess.run(["./ns3", "run", "--no-build", args], cwd=NS3,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ns-3 failed for {config} on {traffic}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def packet_stats(csv, warmup=200):
    d = pd.read_csv(csv).iloc[warmup:]
    total = (d.scale_ns + d.stage1_ns + d.stage2_ns) / 1e3
    att = d.label != BENIGN
    return {"mean_us": total.mean(), "p50_us": total.quantile(0.5),
            "p99_us": total.quantile(0.99), "total_cpu_ms": total.sum() / 1e3,
            "inspected_frac": d.inspected.mean(), "direct_frac": d.direct.mean(),
            "attacks_missed": int((att & (d.inspected == 0)).sum()), "attacks": int(att.sum()),
            "accuracy": (d.predicted == d.label).mean()}


def breakeven(df, rng):
    os.makedirs(f"{OUT}/breakeven", exist_ok=True)
    rows = []
    for share in SHARES:
        traffic = f"{OUT}/breakeven/traffic_b{int(share * 100):02d}.csv"
        write_traffic(build(df, BREAKEVEN_ROWS, share, rng), traffic)
        for config in CONFIGS:
            csv = f"{OUT}/breakeven/b{int(share * 100):02d}__{config}.csv"
            simulate(traffic, config, csv, BREAKEVEN_ROWS)
            st = packet_stats(csv)
            rows.append({"benign_share": share, "config": config, **st})
            print(f"benign={share:.2f} {config:16s} mean={st['mean_us']:8.1f}us "
                  f"missed={st['attacks_missed']}/{st['attacks']} direct={st['direct_frac']:.2f}",
                  flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/breakeven_summary.csv", index=False)
    print(res.pivot(index="benign_share", columns="config", values="mean_us")
          .to_string(float_format=lambda v: f"{v:.1f}"))


def breakeven_repeat(reps):
    """Re-time the existing sweep traffic `reps` more times to average out OS noise."""
    rows = []
    for k in range(1, reps + 1):
        for share in SHARES:
            traffic = f"{OUT}/breakeven/traffic_b{int(share * 100):02d}.csv"
            for config in CONFIGS:
                csv = f"{OUT}/breakeven/b{int(share * 100):02d}__{config}__r{k}.csv"
                simulate(traffic, config, csv, BREAKEVEN_ROWS)
                rows.append({"rep": k, "benign_share": share, "config": config, **packet_stats(csv)})
            print(f"rep {k} benign={share:.2f} done", flush=True)
    first = pd.read_csv(f"{OUT}/breakeven_summary.csv").assign(rep=0)
    allr = pd.concat([first, pd.DataFrame(rows)], ignore_index=True)
    allr.to_csv(f"{OUT}/breakeven_all_reps.csv", index=False)
    agg = allr.groupby(["benign_share", "config"])["mean_us"].agg(["mean", "min", "max", "std"])
    agg.to_csv(f"{OUT}/breakeven_reps_summary.csv")
    print(agg["mean"].unstack().to_string(float_format=lambda v: f"{v:.1f}"))


def timeline(df, rng):
    os.makedirs(f"{OUT}/timeline", exist_ok=True)
    phases = [build(df, n, share, rng).assign(phase=i) for i, (n, share) in enumerate(TIMELINE)]
    seq = pd.concat(phases, ignore_index=True)
    traffic = f"{OUT}/timeline/traffic.csv"
    write_traffic(seq, traffic)
    seq[["phase"]].to_csv(f"{OUT}/timeline/phases.csv", index=False)
    rows = []
    for config in TIMELINE_CONFIGS:
        csv = f"{OUT}/timeline/{config}.csv"
        simulate(traffic, config, csv, len(seq))
        d = pd.read_csv(csv).assign(phase=seq["phase"].values)
        d["total_us"] = (d.scale_ns + d.stage1_ns + d.stage2_ns) / 1e3
        for ph, g in d.groupby("phase"):
            att = g.label != BENIGN
            rows.append({"config": config, "phase": ph, "mean_us": g.total_us.mean(),
                         "direct_frac": g.direct.mean(),
                         "attacks_missed": int((att & (g.inspected == 0)).sum()),
                         "attacks": int(att.sum())})
        rows.append({"config": config, "phase": "all", "mean_us": d.total_us.mean(),
                     "direct_frac": d.direct.mean(),
                     "attacks_missed": int(((d.label != BENIGN) & (d.inspected == 0)).sum()),
                     "attacks": int((d.label != BENIGN).sum())})
        print(f"timeline {config} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/timeline_summary.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["breakeven", "timeline", "all", "repeat"])
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    if args.what == "repeat":
        breakeven_repeat(args.reps)
        return
    df = load_test()
    rng = np.random.default_rng(7)
    if args.what in ("breakeven", "all"):
        breakeven(df, rng)
    if args.what in ("timeline", "all"):
        timeline(df, rng)


if __name__ == "__main__":
    main()
