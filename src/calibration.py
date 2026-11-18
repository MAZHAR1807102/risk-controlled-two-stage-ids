"""How well does a miss-rate target chosen on validation hold on the test split?

For each dataset and validation target eps, tau is picked on validation with the same
per-family rule as the pipeline (Eq. tau*), then the worst-family miss rate and benign
bypass actually reached on the test split are reported. Test data is never used to pick tau.

Usage: .venv/bin/python src/calibration.py
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from datasets import get_dataset
from train_gatekeeper import pick_tau, scaler_from_header, summarize

RUNS = {"ciciot2023": "runs/seed42_sqrt", "ciciomt2024": "runs/iomt_seed42"}
TARGETS = [0.0025, 0.005, 0.01, 0.02, 0.05]

rows = []
for name, run in RUNS.items():
    ds = get_dataset(name)
    s = scaler_from_header(f"{run}/ids_params.h")
    gk = xgb.Booster()
    gk.load_model(f"{run}/layer1_gatekeeper.ubj")
    p = {}
    g = {}
    for split in ("validation", "test"):
        X, g[split], _, _ = ds.load(split)
        X = (X - np.float32(s["means"])) / np.float32(s["scales"])
        p[split] = gk.predict(xgb.DMatrix(X))
    for eps in TARGETS:
        tau = pick_tau(p["validation"], g["validation"], eps)
        te = summarize(p["test"], g["test"], tau, ds.stealthy)
        va = summarize(p["validation"], g["validation"], tau, ds.stealthy)
        rows.append({"dataset": name, "val_target": eps, "tau": tau,
                     "val_worst_miss": va["worst_group_miss"], "test_worst_miss": te["worst_group_miss"],
                     "test_benign_bypass": te["benign_bypass"], "test_overall_miss": te["overall_fnr"]})
        print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.to_csv("results/calibration.csv", index=False)
print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
