"""Gatekeeper + tuned Random Forest as the Brain (the Gatekeeper does not depend on the Brain).

CICIoT2023, deployed Gatekeeper (runs/seed42_sqrt) and the tuned Random Forest of baselines_tuned.py
(depth 24, min leaf 2, sqrt weights). The skip thresholds (observed 0.05, certified from
risk_control.py) depend only on the Gatekeeper, so they apply unchanged. The whole-system threshold
is not used: the Random Forest was selected on the validation split, which is also the calibration split.
ONNX Runtime, one thread, one record per call, the same 20,000 flows as baselines.py (90% benign)
and the natural mix; five interleaved repeats.

Usage: .venv/bin/python src/rf_brain.py -> results/baselines_tuned/rf_brain.csv
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import baselines as B
from risk_control import certified_tau

OUT = "results/baselines_tuned/rf_brain.csv"
REPEATS, WARMUP = 5, 200


def main():
    s1 = B.onnx_session(open(f"{B.MODELS}/xgb_stage1_proposed.onnx", "rb").read())
    rf = B.onnx_session(open("results/baselines_tuned/models/rf_tuned.onnx", "rb").read())
    n1, nr = s1.get_inputs()[0].name, rf.get_inputs()[0].name
    taus = {"observed": 0.05, "certified": certified_tau()}

    X, y, g = B.scaled("test")
    p = np.asarray(s1.run(None, {n1: X})[1])[:, 1]
    rf_lab = np.asarray(rf.run(None, {nr: X})[1]).argmax(1)
    fams = [c for c in B.CLASSES if c != "Benign"]
    ben = y == B.BENIGN
    rows = []
    for name, tau in [("single", None)] + list(taus.items()):
        pred = rf_lab.copy()
        if tau is not None:
            pred[p < tau] = B.BENIGN
        miss = {f: float((pred[g == f] == B.BENIGN).mean()) for f in fams}
        rows.append({"design": name, "tau": tau, "accuracy": accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro"), "worst_family_miss": max(miss.values()),
                     "worst_family": max(miss, key=miss.get), "benign_flagged": float((pred[ben] != B.BENIGN).mean()),
                     "benign_bypass": float((p[ben] < tau).mean()) if tau is not None else 0.0})

    s = B.scaler_from_header(f"{B.RUN}/ids_params.h")
    for mix in ("benign90", "natural"):
        d = pd.read_csv(f"ns3/traffic/test_{mix}_20000.csv")
        Xm = ((d.iloc[:, :46].to_numpy(np.float32) - np.float32(s["means"])) / np.float32(s["scales"])).astype(np.float32)
        res = {r["design"]: [] for r in rows}
        for rep in range(REPEATS):
            order = list(res) if rep % 2 == 0 else list(res)[::-1]
            for name in order:
                tau = taus.get(name)
                ts = []
                for i in range(len(Xm)):
                    row = Xm[i:i + 1]
                    t0 = time.perf_counter_ns()
                    if tau is None or np.asarray(s1.run(None, {n1: row})[1])[0, 1] >= tau:
                        rf.run(None, {nr: row})
                    ts.append(time.perf_counter_ns() - t0)
                res[name].append(np.mean(ts[WARMUP:]) / 1e3)
        for r in rows:
            r[f"mean_us_{mix}"] = float(np.mean(res[r["design"]]))
        single = np.mean(res["single"])
        for r in rows:
            r[f"saving_{mix}"] = 1 - r[f"mean_us_{mix}"] / single
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
