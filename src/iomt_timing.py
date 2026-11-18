"""Timing of the two-stage IDS on CICIoMT2024 with the standalone C++ benchmark.

Builds labelled traffic mixes from the CICIoMT2024 test split (natural, 75% and 90%
benign, every attack family present), runs pi_bench/ids_bench in two-stage and
single-stage mode on one thread and on all cores, and writes a summary table.
Run on an otherwise idle machine.

Usage: .venv/bin/python src/iomt_timing.py --run runs/iomt_seed42
"""
import argparse
import os
import subprocess

import numpy as np
import pandas as pd

from datasets import get_dataset
from make_ns3_traffic import build

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_SRC = os.path.join(HERE, "pi_bench", "ids_bench.cc")
BENCH_BIN = os.path.join(HERE, "pi_bench", "ids_bench_host")
ROWS = 20000
WARMUP = 200


def build_bench():
    if os.path.exists(BENCH_BIN) and os.path.getmtime(BENCH_BIN) > os.path.getmtime(BENCH_SRC):
        return
    subprocess.run(["clang++", "-O2", "-std=c++17", BENCH_SRC, "-I", os.path.join(HERE, "pi_bench", "include"),
                    "-L/opt/homebrew/lib", "-lxgboost", "-Wl,-rpath,/opt/homebrew/lib", "-o", BENCH_BIN],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/iomt_seed42")
    ap.add_argument("--out", default="results/iomt_timing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    build_bench()

    ds = get_dataset("ciciomt2024")
    X, g, raw, feats = ds.load("test")
    df = pd.DataFrame(X, columns=feats).assign(group=g, label=raw)
    rng = np.random.default_rng(0)

    rows = []
    for mix, share in (("benign90", 0.90), ("benign75", 0.75), ("natural", None)):
        sub = df.sample(ROWS, random_state=1) if share is None else build(df, ROWS, share, rng)
        traffic = f"{args.out}/traffic_{mix}.csv"
        out = sub[feats].copy()
        out["class_id"] = sub["group"].map(ds.classes.index).astype(int)
        out["raw_label"] = sub["label"].values
        out.to_csv(traffic, index=False)
        for threads in ("1", "0"):
            for mode in ("two-stage", "single-stage"):
                csv = f"{args.out}/{mix}__{mode}__t{threads}.csv"
                subprocess.run([BENCH_BIN, f"--data={traffic}", f"--params={args.run}/ids_params.h",
                                f"--l1={args.run}/layer1_gatekeeper.ubj", f"--l2={args.run}/layer2_multiclass.ubj",
                                f"--mode={mode}", f"--nthread={threads}", f"--out={csv}",
                                f"--warmup={WARMUP}"], check=True, capture_output=True)
                d = pd.read_csv(csv).iloc[WARMUP:]
                t = (d.scale_ns + d.stage1_ns + d.stage2_ns) / 1e3
                att = d.label != ds.classes.index("Benign")
                rows.append({"mix": mix, "threads": "1" if threads == "1" else "all", "mode": mode,
                             "cpu_s": t.sum() / 1e6, "p50_us": t.quantile(0.5), "p99_us": t.quantile(0.99),
                             "missed": int((att & (d.inspected == 0)).sum()), "attacks": int(att.sum()),
                             "benign_share": float((~att).mean())})
                print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    single = res[res["mode"] == "single-stage"].set_index(["mix", "threads"])["cpu_s"]
    res["saving"] = [1 - r.cpu_s / single[(r.mix, r.threads)] for r in res.itertuples()]
    res.to_csv(f"{args.out}/summary.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
