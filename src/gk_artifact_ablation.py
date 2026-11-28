"""CICIoT2023 Gatekeeper (100 trees, depth 6, family-balanced, seed 42) trained without the CIC artifact columns:
none / IAT only / IAT + Number + Weight. Certified threshold (ucb_typ) on validation, test scored once.
Usage: .venv/bin/python src/gk_artifact_ablation.py  -> results/packet/c23_gatekeeper_artifacts.csv"""
import numpy as np, pandas as pd, xgboost as xgb
from datasets import get_dataset
from pipeline import STAGE1_PARAMS, STAGE1_ROUNDS
from risk_control import pick
from train_gatekeeper import weights
ds = get_dataset("ciciot2023"); Xtr, gtr, _, feats = ds.load("train"); Xv, fv, tv, _ = ds.load("validation"); Xt, ft, tt, _ = ds.load("test")
rows = []
for drop in [(), ("IAT",), ("IAT", "Number", "Weight")]:
    keep = [i for i, f in enumerate(feats) if f not in drop]
    gk = xgb.train({**STAGE1_PARAMS, "seed": 42}, xgb.QuantileDMatrix(Xtr[:, keep], (gtr != "Benign").astype(np.float32),
                   weight=weights(gtr, "group_balanced")), STAGE1_ROUNDS)
    pv, pt = gk.predict(xgb.DMatrix(Xv[:, keep])), gk.predict(xgb.DMatrix(Xt[:, keep]))
    tau, ok = pick(pv, fv, tv, "ucb_typ")
    fam = {g: float((pt[ft == g] < tau).mean()) for g in np.unique(ft[ft != "Benign"])}
    w = max(fam, key=fam.get)
    rows.append({"dropped": "+".join(drop) or "none", "tau": tau, "feasible": ok,
                 "benign_bypass": float((pt[ft == "Benign"] < tau).mean()), "worst_family": w, "worst_family_skip": fam[w]})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/packet/c23_gatekeeper_artifacts.csv", index=False)
