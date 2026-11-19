"""Gatekeeper without the IAT feature (robustness check for the flow-shaping test).

In CICIoT2023, IAT behaves like a capture timestamp and carries most of the Gatekeeper's evasion risk
(src/evasion.py). Here a Gatekeeper with the same settings (100 trees, depth 6, family-balanced weights,
seed 42) is trained on train without IAT; the certified threshold is picked on validation (ucb_typ) and
the test split is scored once. Then the timing-shaping attack of src/evasion.py is repeated.
Trees split on thresholds, so the features are used unscaled.

Usage: .venv/bin/python src/gatekeeper_no_iat.py  -> results/evasion/no_iat.csv
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from datasets import get_dataset
from pipeline import STAGE1_PARAMS, STAGE1_ROUNDS
from risk_control import pick
from train_gatekeeper import weights

OUT = "results/evasion"
LOW = ["BruteForce", "Recon", "Spoofing", "Web"]
TIMING = ["flow_duration", "Rate", "Srate", "Drate"]


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = get_dataset("ciciot2023")
    Xtr, gtr, _, feats = ds.load("train")
    keep = [i for i, f in enumerate(feats) if f != "IAT"]
    d = xgb.QuantileDMatrix(Xtr[:, keep], (gtr != "Benign").astype(np.float32), weight=weights(gtr, "group_balanced"))
    gk = xgb.train({**STAGE1_PARAMS, "seed": 42}, d, STAGE1_ROUNDS)
    del d, Xtr
    Xv, fv, tv, _ = ds.load("validation")
    pv = gk.predict(xgb.DMatrix(Xv[:, keep]))
    tau, ok = pick(pv, fv, tv, "ucb_typ")
    Xt, ft, tt, _ = ds.load("test")
    pt = gk.predict(xgb.DMatrix(Xt[:, keep]))
    rows = [{"check": "certified threshold", "tau": tau, "feasible": ok,
             "benign_bypass": float((pt[ft == "Benign"] < tau).mean())}]
    for g in LOW + ["DDoS", "DoS", "Mirai"]:
        rows.append({"check": "test skip", "family": g, "skip": float((pt[ft == g] < tau).mean())})
    # timing shaping without IAT (IAT is no longer used)
    rng = np.random.default_rng(3)
    col = {f: i for i, f in enumerate(feats)}
    ben = np.flatnonzero(ft == "Benign")
    att = np.flatnonzero(np.isin(ft, LOW))
    proto = Xt[:, col["Protocol Type"]]
    donors = np.empty(len(att), dtype=int)
    for v in np.unique(proto[att]):
        same, sel = ben[proto[ben] == v], proto[att] == v
        donors[sel] = rng.choice(same if len(same) else ben, size=sel.sum())
    for name, group in (("timing", TIMING),
                        ("timing+sizes", TIMING + ["Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "Number",
                                                   "Magnitue", "Radius", "Covariance", "Variance", "Weight"])):
        Xa = Xt[att].copy()
        for f in group:
            Xa[:, col[f]] = Xt[donors, col[f]]
        pa = gk.predict(xgb.DMatrix(Xa[:, keep]))
        for g in LOW:
            m = ft[att] == g
            rows.append({"check": f"shaped {name}", "family": g, "skip": float((pa[m] < tau).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/no_iat.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
