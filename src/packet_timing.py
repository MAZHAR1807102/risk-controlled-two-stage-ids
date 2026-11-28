"""Inference-time saving of the two-stage IDS on packet records, with the standalone C++ benchmark
(pi_bench/ids_bench_host, XGBoost C API; host machine, not an edge board).

For every run: traffic mixes from its test split (90% / 75% benign, and the natural mix), the two-stage IDS at
the certified skip threshold and at the whole-system (added-miss) threshold, against the single-stage IDS
(Brain on every record), on one thread and on all cores. Run on an otherwise idle machine.

Usage: .venv/bin/python src/packet_timing.py <dataset>:<run>:<tag> ...  -> results/packet/timing.csv
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from datasets import get_dataset
from iomt_timing import BENCH_BIN, build_bench
from make_ns3_traffic import build

OUT = "results/packet"
ROWS, WARMUP = int(os.environ.get("PT_ROWS", 20000)), 200
TMP = "/tmp/packet_timing"


def bench(traffic, run, mode, threads, tau, out):
    subprocess.run([BENCH_BIN, f"--data={traffic}", f"--params={run}/ids_params.h",
                    f"--l1={run}/layer1_gatekeeper.ubj", f"--l2={run}/layer2_multiclass.ubj", f"--mode={mode}",
                    f"--nthread={threads}", f"--tau={tau}", f"--out={out}", f"--warmup={WARMUP}"],
                   check=True, capture_output=True)
    d = pd.read_csv(out).iloc[WARMUP:]
    return d, (d.scale_ns + d.stage1_ns + d.stage2_ns).sum() / 1e9


def main(specs):
    os.makedirs(TMP, exist_ok=True)
    build_bench()
    rows = []
    for spec in specs:
        ds_name, run, tag = spec.split(":")
        ds = get_dataset(ds_name)
        cert = pd.read_csv(f"{OUT}/cert_{tag}.csv").set_index("rule")
        X, fam, typ, feats = ds.load("test")
        df = pd.DataFrame(X, columns=feats).assign(group=fam, label=typ)
        rng = np.random.default_rng(0)
        for mix, share in (("benign90", 0.90), ("benign75", 0.75), ("natural", None)):
            sub = df.sample(ROWS, random_state=1) if share is None else build(df, ROWS, share, rng)
            traffic = f"{TMP}/{tag}_{mix}.csv"
            out = sub[feats].copy()
            out["class_id"] = sub["group"].map(ds.classes.index).astype(int)
            out.to_csv(traffic, index=False)
            for threads in ("1", "0"):
                _, single = bench(traffic, run, "single-stage", threads, 0.5, f"{TMP}/s.csv")
                for rule in ("skip", "added"):
                    if not bool(cert.loc[rule, "feasible"]):
                        continue
                    tau = float(cert.loc[rule, "tau"])
                    d, two = bench(traffic, run, "two-stage", threads, tau, f"{TMP}/t.csv")
                    att = d.label != ds.classes.index("Benign")
                    rows.append({"tag": tag, "dataset": ds_name, "rule": rule, "tau": tau, "mix": mix,
                                 "threads": "1" if threads == "1" else "all", "two_s": two, "single_s": single,
                                 "saving": 1 - two / single,
                                 "benign_bypass": float((d.inspected[~att] == 0).mean()),
                                 "benign_share": float((~att).mean())})
                    print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/timing{os.environ.get('PKT_SFX', '')}.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1:])
