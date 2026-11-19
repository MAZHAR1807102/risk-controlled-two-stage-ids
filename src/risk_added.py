"""Certified threshold on the misses the Gatekeeper ADDS to the single-stage IDS.

A flow of attack family g is an *added miss* at threshold tau if the Gatekeeper skips it
(p_a < tau) AND the Brain would have flagged it as an attack (Stage 2 class != Benign).
Attacks the Brain calls benign anyway are missed by the single-stage IDS too, so they do
not count against the Gatekeeper. For every family and type:

    two-stage miss = single-stage miss + added miss            (exact identity)

so certifying the added-miss rate <= eps says: the two-stage IDS misses at most eps more of
any family than running the Brain on every flow. The added-miss indicator still grows with
tau, so the fixed-sequence argument of Proposition 1 and risk_control.pick apply unchanged:
we only set the score of attacks the Brain calls benign to 2 (never below any tau).

The calibration flows must not have been used to fit or select either model. The Gatekeeper is
fitted on train only (fixed number of trees). The deployed Brain (runs/seed42_sqrt, runs/iomt_seed42)
was early-stopped on the validation split, so for Protocol A its added-miss counts are NOT a valid
certificate. With --sel, the Brain is the one early-stopped on 15% of the TRAIN split instead
(pipeline.py --select-frac 0.15: runs/seed42_sel, runs/iomt_seed42_sel), so validation stays pure
calibration data. Protocol B (calibration on the first 20% of the test captures) is valid for both.

Usage: .venv/bin/python src/risk_added.py [--sel]
"""
import sys
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from datasets import get_dataset
from risk_control import ALPHA, EPS, EPS_TYPE, RUNS, cp_upper, iomt_test_by_file, pick
from train_gatekeeper import scaler_from_header

OUT = "results/risk_control"


def scores(run, X, classes):
    """Gatekeeper attack score and 'Brain flags an attack' for every row."""
    s = scaler_from_header(f"{run}/ids_params.h")
    d = xgb.DMatrix((X - np.float32(s["means"])) / np.float32(s["scales"]))
    gk, br = xgb.Booster(), xgb.Booster()
    gk.load_model(f"{run}/layer1_gatekeeper.ubj")
    br.load_model(f"{run}/layer2_multiclass.ubj")
    p = gk.predict(d)
    flagged = br.predict(d).argmax(1) != classes.index("Benign")
    return p, flagged


def added_score(p, fam, flagged):
    """Scores for pick(): attacks the Brain calls benign can never count as added misses."""
    q = p.copy()
    q[(fam != "Benign") & ~flagged] = 2.0
    return q


def evaluate(p, fam, typ, flagged, tau):
    att = fam != "Benign"
    skip = p < tau
    rows = {"tau": tau, "benign_bypass": float(skip[~att].mean()), "overall_skip": float(skip[att].mean())}
    worst = {}
    for key, lab in (("family", fam), ("type", typ)):
        best = None
        for u in np.unique(lab[att]):
            m = lab == u
            n = int(m.sum())
            k_skip = int((skip & m).sum())
            k_add = int((skip & flagged & m).sum())
            miss_single = float((~flagged & m).mean() / m.mean())          # Brain alone calls it benign
            miss_two = float(((skip | ~flagged) & m).sum() / n)             # whole two-stage system
            assert abs(miss_two - (miss_single + k_add / n)) < 1e-12        # the identity above
            r = (k_add / n, u, k_add, n, k_skip / n, miss_single, miss_two)
            best = max(best, r) if best else r
        worst[key] = best
    for key in ("family", "type"):
        a, u, k, n, s, ms, mt = worst[key]
        rows.update({f"worst_{key}": u, f"worst_{key}_added": a, f"worst_{key}_added_n": f"{k}/{n}",
                     f"worst_{key}_added_ci95_high": float(cp_upper(k, n, ALPHA)),
                     f"worst_{key}_skip": s, f"worst_{key}_miss_single": ms, f"worst_{key}_miss_two": mt})
    # largest plain skip rate over families (what the old certificate controlled)
    rows["max_family_skip"] = max(float((skip & (fam == g)).sum() / (fam == g).sum()) for g in np.unique(fam[att]))
    return rows


SEL = "--sel" in sys.argv
if SEL:
    RUNS = {"ciciot2023": "runs/seed42_sel", "ciciomt2024": "runs/iomt_seed42_sel"}
SFX = "_sel" if SEL else ""


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    cache = {}
    for name, run in RUNS.items():
        ds = get_dataset(name)
        for split in ("validation", "test"):
            X, fam, typ, _ = ds.load(split)
            p, fl = scores(run, X, ds.classes)
            cache[(name, split)] = (p, fam, typ, fl)
            print(f"scored {name} {split}: {len(p)} rows", flush=True)
        pv, fv, tv, flv = cache[(name, "validation")]
        pt, ft, tt, flt = cache[(name, "test")]
        for target, q in (("skip", pv), ("added", added_score(pv, fv, flv))):
            tau, ok = pick(q, fv, tv, "ucb_typ")
            r = evaluate(pt, ft, tt, flt, tau)
            r.update({"dataset": name, "calibration": "validation", "certified_quantity": target, "feasible": ok})
            rows.append(r)
            print(r, flush=True)

    # CICIoMT2024, Protocol B: re-calibrate on the first 20% of every test capture
    X, fam, typ, fid, pos = iomt_test_by_file()
    p, fl = scores(RUNS["ciciomt2024"], X, get_dataset("ciciomt2024").classes)
    cal = pos < 0.20
    for target, q in (("skip", p[cal]), ("added", added_score(p[cal], fam[cal], fl[cal]))):
        tau, ok = pick(q, fam[cal], typ[cal], "ucb_typ")
        r = evaluate(p[~cal], fam[~cal], typ[~cal], fl[~cal], tau)
        r.update({"dataset": "ciciomt2024", "calibration": "first 20% of test files",
                  "certified_quantity": target, "feasible": ok})
        rows.append(r)
        print(r, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/added_miss{SFX}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["dataset", "calibration", "certified_quantity", "feasible", "tau", "benign_bypass",
               "max_family_skip", "worst_family", "worst_family_added", "worst_type", "worst_type_added",
               "worst_type_added_n"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"targets: family {EPS}, type {EPS_TYPE}, alpha {ALPHA}")


if __name__ == "__main__":
    main()
