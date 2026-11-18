"""Risk-controlled Gatekeeper thresholds (reviewer point 2.2).

Three ways to pick tau on held-out calibration data, all with the same target eps:
  point   largest tau whose *observed* worst-family miss rate <= eps (the observed rule)
  ucb_fam largest tau whose one-sided 95% Clopper-Pearson upper bound on every family's
          miss rate <= eps, Bonferroni-corrected over families
  ucb_typ the same bound per attack type for types with enough calibration flows to be
          certifiable at all (n >= n_min), and per family for the remaining small types

The bound is a finite-sample statement under the usual assumption that calibration and
deployment flows come from the same distribution (exchangeability). It is not a guarantee
under distribution shift, which is why the second experiment re-calibrates on-site.

Experiment 2 (CICIoMT2024, re-calibration): the test captures are split per file in time
order; the first f of each file is used to pick tau, the rest to evaluate it.

Usage: .venv/bin/python src/risk_control.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import beta

from datasets import CICIoMT2024, get_dataset
from train_gatekeeper import scaler_from_header

EPS = 0.02
ALPHA = 0.05
OUT = "results/risk_control"
RUNS = {"ciciot2023": "runs/seed42_sqrt", "ciciomt2024": "runs/iomt_seed42"}
GRID = np.unique(np.concatenate([np.linspace(0.0005, 0.9995, 2000), np.geomspace(1e-5, 0.999, 400)]))


def cp_upper(k, n, alpha):
    """One-sided (1-alpha) Clopper-Pearson upper bound for a binomial proportion."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    ub = beta.ppf(1 - alpha, k + 1, np.maximum(n - k, 1e-12))
    return np.where(k >= n, 1.0, ub)


def n_min(eps, alpha):
    """Smallest n for which zero observed misses can certify a miss rate <= eps."""
    return int(np.ceil(np.log(alpha) / np.log(1 - eps)))


def miss_counts(p, groups, taus):
    """For each group: (n, number of flows with score < tau for every tau in taus)."""
    out = {}
    for gname in np.unique(groups):
        s = np.sort(p[groups == gname])
        out[gname] = (len(s), np.searchsorted(s, taus, side="left"))
    return out


EPS_TYPE = 0.05  # per-type target used together with the family target in "ucb_typ"


def pick(p, fam, typ, rule, eps=EPS, alpha=ALPHA):
    """Largest tau meeting the rule on calibration data; returns (tau, feasible)."""
    att = fam != "Benign"
    taus = GRID
    if rule == "point":
        cnt = miss_counts(p[att], fam[att], taus)
        ok = np.max([k / n for n, k in cnt.values()], axis=0) <= eps
    else:
        fam_cnt = miss_counts(p[att], fam[att], taus)
        typ_cnt = {}
        if rule == "ucb_typ":
            # Error budget: alpha/2 shared by the families, alpha/2 by the attack types.
            types, counts = np.unique(typ[att], return_counts=True)
            big = types[counts >= n_min(EPS_TYPE, alpha / 2 / len(types))]  # certifiable types
            sel = np.isin(typ[att], big)
            typ_cnt = miss_counts(p[att][sel], typ[att][sel], taus)
        a_fam = alpha / (2 if typ_cnt else 1) / len(fam_cnt)
        ok = np.max([cp_upper(k, n, a_fam) for n, k in fam_cnt.values()], axis=0) <= eps
        if typ_cnt:
            a_typ = alpha / 2 / len(typ_cnt)
            ok &= np.max([cp_upper(k, n, a_typ) for n, k in typ_cnt.values()], axis=0) <= EPS_TYPE
    if not ok.any():
        return float(taus[0]), False
    return float(taus[np.flatnonzero(ok).max()]), True


def evaluate(p, fam, typ, tau, label):
    att = fam != "Benign"
    rows = {}
    fam_miss = {g: (int((p[fam == g] < tau).sum()), int((fam == g).sum())) for g in np.unique(fam[att])}
    typ_miss = {t: (int((p[typ == t] < tau).sum()), int((typ == t).sum())) for t in np.unique(typ[att])}
    wf = max(fam_miss, key=lambda g: fam_miss[g][0] / fam_miss[g][1])
    wt = max(typ_miss, key=lambda t: typ_miss[t][0] / typ_miss[t][1])
    k, n = fam_miss[wf]
    rows.update({"setting": label, "tau": tau,
                 "benign_bypass": float((p[~att] < tau).mean()),
                 "worst_family": wf, "worst_family_miss": k / n,
                 "worst_family_miss_ci95_high": float(cp_upper(k, n, ALPHA)),
                 "worst_type": wt, "worst_type_miss": typ_miss[wt][0] / typ_miss[wt][1],
                 "worst_type_missed_n": f"{typ_miss[wt][0]}/{typ_miss[wt][1]}",
                 "overall_miss": float((p[att] < tau).mean())})
    return rows, fam_miss, typ_miss


def gatekeeper_scores(run, X):
    s = scaler_from_header(f"{run}/ids_params.h")
    gk = xgb.Booster()
    gk.load_model(f"{run}/layer1_gatekeeper.ubj")
    return gk.predict(xgb.DMatrix((X - np.float32(s["means"])) / np.float32(s["scales"])))


def iomt_test_by_file():
    """CICIoMT2024 test rows with their file id and position, in file (time) order."""
    ds = CICIoMT2024()
    Xs, fams, typs, fid, pos = [], [], [], [], []
    for i, path in enumerate(sorted(glob.glob(os.path.join(ds.DIR, "test", "*.csv")))):
        df = pd.read_csv(path)
        a = ds.attack_name(path)
        Xs.append(df.to_numpy(np.float32))
        fams.append(np.full(len(df), ds.family(a), dtype=object))
        typs.append(np.full(len(df), a, dtype=object))
        fid.append(np.full(len(df), i))
        pos.append(np.arange(len(df)) / len(df))
    return (np.concatenate(Xs), np.concatenate(fams).astype(str), np.concatenate(typs).astype(str),
            np.concatenate(fid), np.concatenate(pos))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, detail = [], {}
    for name, run in RUNS.items():
        ds = get_dataset(name)
        data = {}
        for split in ("validation", "test"):
            X, fam, typ, _ = ds.load(split)
            data[split] = (gatekeeper_scores(run, X), fam, typ)
        pv, fv, tv = data["validation"]
        pt, ft, tt = data["test"]
        for rule in ("point", "ucb_fam", "ucb_typ"):
            tau, feasible = pick(pv, fv, tv, rule)
            r, fm, tm = evaluate(pt, ft, tt, tau, f"{name}:{rule}")
            r.update({"dataset": name, "rule": rule, "calibration": "validation", "feasible": feasible})
            rows.append(r)
            detail[f"{name}:{rule}"] = {"family": fm, "type": tm}
            print(r, flush=True)

    # ---- CICIoMT2024 on-site re-calibration ----
    X, fam, typ, fid, pos = iomt_test_by_file()
    p = gatekeeper_scores(RUNS["ciciomt2024"], X)
    for frac in (0.05, 0.10, 0.20):
        cal = pos < frac
        for rule in ("point", "ucb_fam", "ucb_typ"):
            tau, feasible = pick(p[cal], fam[cal], typ[cal], rule)
            r, fm, tm = evaluate(p[~cal], fam[~cal], typ[~cal], tau, f"iomt_recal{int(frac*100)}:{rule}")
            r.update({"dataset": "ciciomt2024", "rule": rule, "calibration": f"first {int(frac*100)}% of test files",
                      "feasible": feasible})
            rows.append(r)
            detail[f"iomt_recal{int(frac*100)}:{rule}"] = {"family": fm, "type": tm}
            print(r, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/summary.csv", index=False)
    json.dump(detail, open(f"{OUT}/detail.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(res[["dataset", "calibration", "rule", "feasible", "tau", "benign_bypass", "worst_family",
               "worst_family_miss", "worst_type", "worst_type_miss", "overall_miss"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"n_min for eps={EPS}, alpha={ALPHA}: {n_min(EPS, ALPHA)} flows")


if __name__ == "__main__":
    main()
