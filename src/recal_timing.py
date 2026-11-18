"""Inference- and CPU-time saving at the CICIoMT2024 on-site re-calibrated certified threshold (Protocol B,
first 20% of each test capture), measured with the standalone C++ benchmark, three runs each (review 3, 2.5).
Usage: .venv/bin/python src/recal_timing.py  -> results/risk_control/recal_timing.csv"""
import os
import re
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(HERE, "pi_bench", "ids_bench_host")
rc = pd.read_csv(os.path.join(HERE, "results/risk_control/summary.csv"))
tau = float(rc.query("dataset=='ciciomt2024' and calibration=='first 20% of test files' and rule=='ucb_typ'").tau.iloc[0])
run = os.path.join(HERE, "runs/iomt_seed42")
rows = []
for mix in ("benign90", "benign75"):
    for t in ("1", "0"):
        for rep in range(3):
            res = {}
            for mode in ("two-stage", "single-stage"):
                out = f"/tmp/recal_{mix}_{mode}_{t}.csv"
                p = subprocess.run([BIN, f"--data={HERE}/results/iomt_timing/traffic_{mix}.csv", f"--params={run}/ids_params.h",
                                    f"--l1={run}/layer1_gatekeeper.ubj", f"--l2={run}/layer2_multiclass.ubj",
                                    f"--mode={mode}", f"--nthread={t}", f"--tau={tau}", f"--out={out}"],
                                   check=True, capture_output=True, text=True)
                cpu = float(re.search(r"process CPU time, user\+sys \(all flows\): ([0-9.]+)", p.stdout).group(1))
                wall = float(re.search(r"loop wall time \(all flows\): ([0-9.]+)", p.stdout).group(1))
                d = pd.read_csv(out).iloc[200:]
                res[mode] = ((d.scale_ns + d.stage1_ns + d.stage2_ns).sum() / 1e9, cpu, wall, d)
            two = res["two-stage"][3]
            att = two.label != 0
            rows.append({"tau": tau, "mix": mix, "threads": "1" if t == "1" else "all", "rep": rep,
                         "inference_saving": 1 - res["two-stage"][0] / res["single-stage"][0],
                         "cpu_saving": 1 - res["two-stage"][1] / res["single-stage"][1],
                         "benign_bypass": float((two.inspected[~att] == 0).mean())})
            print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "results/risk_control/recal_timing.csv"), index=False)
print(df.groupby(["mix", "threads"])[["inference_saving", "cpu_saving", "benign_bypass"]].mean().round(3))
