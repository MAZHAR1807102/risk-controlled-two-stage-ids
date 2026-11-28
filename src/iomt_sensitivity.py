"""Protocol B (local re-calibration) on CICIoMT2024: how much does it depend on the window?

For every capture, a contiguous window of its rows (time order) is the calibration data and all other
rows of the same capture are the evaluation data. Rules: the certified skip rule (ucb_typ) and the
observed point rule.
  position - 20% window starting at 0, 10, ..., 80% of every capture
  size     - first 5, 10, 15, 20, 30, 40, 50% of every capture
  block    - design effect of the skips at the Protocol B threshold (first 20%), over the whole capture
             in time order, for block lengths 10-2000 rows, and the D-corrected rule with each
`swap` runs the same on the CIC train captures with a model trained on the test captures
(pipeline.py --dataset ciciomt2024s): an independent set of captures for every attack type.

Usage: .venv/bin/python src/iomt_sensitivity.py [orig|swap]  -> results/iomt_window/{position,size,block}_{orig|swap}.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from datasets import CICIoMT2024
from risk_control import ALPHA, EPS, EPS_TYPE, gatekeeper_scores, n_min, pick
from risk_coverage import design_effect, violations
from risk_dependence import pick_D

OUT = "results/iomt_window"
SETUPS = {"orig": ("runs/iomt_seed42", "test"), "swap": ("runs/iomt_swap_seed42_sel", "train")}


def by_capture(folder):
    """Rows of every capture in time order; numbered parts of one attack (ICMP1, ICMP2...) stay separate captures."""
    ds = CICIoMT2024()
    Xs, fams, typs, fid, pos = [], [], [], [], []
    for i, path in enumerate(sorted(glob.glob(os.path.join(ds.DIR, folder, "*.csv")))):
        df = pd.read_csv(path)
        a = ds.attack_name(path)
        Xs.append(df.to_numpy(np.float32))
        fams.append(np.full(len(df), ds.family(a), dtype=object))
        typs.append(np.full(len(df), a, dtype=object))
        fid.append(np.full(len(df), i))
        pos.append(np.arange(len(df)) / len(df))
    return (np.concatenate(Xs), np.concatenate(fams).astype(str), np.concatenate(typs).astype(str),
            np.concatenate(fid), np.concatenate(pos))


def run_rule(p, fam, typ, cal, rule):
    tau, ok = pick(p[cal], fam[cal], typ[cal], rule)
    ev = ~cal
    att = ev & (fam != "Benign")
    fs = {g: (p[ev & (fam == g)] < tau).mean() for g in np.unique(fam[att])}
    types, counts = np.unique(typ[cal & (fam != "Benign")], return_counts=True)
    cert = types[counts >= n_min(EPS_TYPE, ALPHA)]
    ts = {t: (p[ev & (typ == t)] < tau).mean() for t in cert if (ev & (typ == t)).any()}
    wf, wt = max(fs, key=fs.get), max(ts, key=ts.get) if ts else None
    fb, tb = violations(p[ev], fam[ev], typ[ev], tau, typ[cal], fam[cal]) if ok else (False, False)
    return {"rule": rule, "feasible": ok, "tau": tau, "bypass": float((p[ev & (fam == "Benign")] < tau).mean()) if ok else 0.0,
            "worst_family": wf, "worst_family_skip": fs[wf], "worst_type": wt, "worst_type_skip": ts.get(wt, np.nan),
            "violates_family": fb, "violates_type": tb,
            "cal_min_family_n": int(min((cal & (fam == g)).sum() for g in np.unique(fam[att]))),
            "uncertified_types": int(len(np.unique(typ[cal & (fam != "Benign")])) - len(cert))}


def main(setup="orig"):
    os.makedirs(OUT, exist_ok=True)
    run, folder = SETUPS[setup]
    X, fam, typ, fid, pos = by_capture(folder)
    p = gatekeeper_scores(run, X)
    print(f"{setup}: {len(p):,} rows from {fid.max() + 1} captures; families:",
          dict(zip(*np.unique(fam, return_counts=True))), flush=True)

    rows = []
    for start in np.round(np.arange(0, 0.81, 0.1), 2):
        cal = (pos >= start) & (pos < start + 0.2)
        for rule in ("ucb_typ", "point"):
            rows.append({"window_start": start, "window_size": 0.2, **run_rule(p, fam, typ, cal, rule)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/position_{setup}.csv", index=False)

    rows = []
    for size in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        cal = pos < size
        rows.append({"window_start": 0.0, "window_size": size, **run_rule(p, fam, typ, cal, "ucb_typ")})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/size_{setup}.csv", index=False)

    cal = pos < 0.2
    tau_b, _ = pick(p[cal], fam[cal], typ[cal], "ucb_typ")
    rows = []
    for L in (10, 50, 100, 200, 500, 1000, 2000):
        de = design_effect(p, fam, typ, tau_b, setup, L=L)
        D = {d["unit"]: d["deff"] for d in de}
        tau_d, ok_d = pick_D(p[cal], fam[cal], typ[cal], D)
        for d in de:
            rows.append({"L": L, "level": d["level"], "unit": d["unit"], "n": d["n"], "skipped": d["missed"],
                         "deff": d["deff"], "tau_B": tau_b, "D_rule_feasible": ok_d, "D_rule_tau": tau_d})
        print(f"L={L}: D " + ", ".join(f"{d['unit']} {d['deff']:.1f}" for d in de) +
              f" -> D-corrected rule feasible={ok_d} tau={tau_d:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/block_{setup}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
