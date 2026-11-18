"""CPU saving at the certified thresholds (risk_control.py, rule ucb_typ), measured with the
standalone C++ benchmark on the same traffic mixes as before. Run on an idle machine.
Usage: .venv/bin/python src/certified_timing.py"""
import os, subprocess
import pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(HERE, "pi_bench", "ids_bench_host")
rc = pd.read_csv("results/risk_control/summary.csv")
cases = {
  "ciciot2023": ("runs/seed42_sqrt", "ns3/traffic/test_{mix}_20000.csv",
                 float(rc.query("dataset=='ciciot2023' and calibration=='validation' and rule=='ucb_typ'").tau.iloc[0])),
  "ciciomt2024": ("runs/iomt_seed42", "results/iomt_timing/traffic_{mix}.csv",
                  float(rc.query("dataset=='ciciomt2024' and calibration=='validation' and rule=='ucb_typ'").tau.iloc[0])),
}
rows = []
for ds, (run, traffic, tau) in cases.items():
    for mix in ("benign90", "benign75"):
        for t in ("1", "0"):
            res = {}
            for mode in ("two-stage", "single-stage"):
                out = f"/tmp/cert_{ds}_{mix}_{mode}_{t}.csv"
                subprocess.run([BIN, f"--data={traffic.format(mix=mix)}", f"--params={run}/ids_params.h",
                                f"--l1={run}/layer1_gatekeeper.ubj", f"--l2={run}/layer2_multiclass.ubj",
                                f"--mode={mode}", f"--nthread={t}", f"--tau={tau}", f"--out={out}"],
                               check=True, capture_output=True)
                d = pd.read_csv(out).iloc[200:]
                res[mode] = ((d.scale_ns + d.stage1_ns + d.stage2_ns).sum() / 1e9, d)
            two = res["two-stage"][1]
            att = two.label != 0
            rows.append({"dataset": ds, "tau": tau, "mix": mix, "threads": "1" if t == "1" else "all",
                         "two_cpu_s": res["two-stage"][0], "single_cpu_s": res["single-stage"][0],
                         "saving": 1 - res["two-stage"][0] / res["single-stage"][0],
                         "benign_bypass": float((two.inspected[~att] == 0).mean()),
                         "missed": int((att & (two.inspected == 0)).sum()), "attacks": int(att.sum())})
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/risk_control/certified_timing.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
