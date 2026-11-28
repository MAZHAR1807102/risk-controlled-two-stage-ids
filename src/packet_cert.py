"""Threshold rules on the packet-record datasets (and on CIC's CSV for comparison).

For one trained run (pipeline.py --select-frac 0.15, so the Brain never saw validation), tau is picked on
validation and the test split is scored once:
  point   observed rule (largest tau with every validation family skip rate <= 2%)
  skip    certified rule on the Gatekeeper's skips (ucb_typ)
  added   certified rule on the misses the Gatekeeper adds to the Brain's (whole system)
  skip_D  skip rule with counts divided by the design effect measured on validation in time order
          (an empirical correction, not a proved certificate)
Capture-separated datasets also get
  local   re-calibration on the first 20% (in time) of every test capture, evaluated on the other 80%
  novel   the certified skip / added thresholds evaluated only on test rows whose feature vector does not
          occur in train (exact duplicates removed)

Usage: .venv/bin/python src/packet_cert.py <dataset> <run> <tag>  -> results/packet/cert_<tag>.csv,
       families_<tag>.csv, deff_<tag>.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_added import added_score, evaluate as evaluate_added, scores
from risk_control import ALPHA, cp_upper, pick
from risk_coverage import design_effect
from risk_dependence import pick_D
from temporal_cert import skip_rates

OUT = "results/packet"


def row(name, ok, pt, ft, tt, flt, tau, extra=None):
    r = {"rule": name, "feasible": ok}
    r.update(evaluate_added(pt, ft, tt, flt, tau))
    r.update(skip_rates(pt, ft, tt, tau))
    r.update(extra or {})
    return r


def main(ds_name, run, tag):
    os.makedirs(OUT, exist_ok=True)
    ds = get_dataset(ds_name)
    data = {}
    for split in ("validation", "test"):
        X, fam, typ, _ = ds.load(split)
        p, fl = scores(run, X, ds.classes)
        data[split] = (X, p, fam, typ, fl)
        print(f"scored {split}: {len(p):,}", flush=True)
    Xv, pv, fv, tv, flv = data["validation"]
    Xt, pt, ft, tt, flt = data["test"]
    rows, taus = [], {}
    for name, q, rule in (("point", pv, "point"), ("skip", pv, "ucb_typ"), ("added", added_score(pv, fv, flv), "ucb_typ")):
        tau, ok = pick(q, fv, tv, rule)
        taus[name] = tau
        rows.append(row(name, ok, pt, ft, tt, flt, tau, {f"val_{k}": v for k, v in skip_rates(pv, fv, tv, tau).items()}))
        print(rows[-1]["rule"], ok, round(tau, 4), round(rows[-1]["benign_bypass"], 3), flush=True)

    de = pd.DataFrame(design_effect(pv, fv, tv, taus["skip"], f"{tag} validation"))
    de.to_csv(f"{OUT}/deff_{tag}.csv", index=False)
    if len(de):
        tau, ok = pick_D(pv, fv, tv, dict(zip(de.unit, de.deff)))
        rows.append(row("skip_D", ok, pt, ft, tt, flt, tau))

    if hasattr(ds, "meta") and ds.SPLIT == "C":
        m = ds.meta("test")
        cap = m["capture"]
        pos = pd.Series(np.arange(len(cap))).groupby(cap).rank(pct=True).to_numpy()   # rows are in time order
        cal = pos <= 0.20
        for name, q in (("local_skip", pt[cal]), ("local_added", added_score(pt[cal], ft[cal], flt[cal]))):
            tau, ok = pick(q, ft[cal], tt[cal], "ucb_typ")
            rows.append(row(name, ok, pt[~cal], ft[~cal], tt[~cal], flt[~cal], tau,
                            {"cal_rows": int(cal.sum())}))
        Xtr, _, _, _ = ds.load("train")
        seen = set(map(bytes, np.ascontiguousarray(Xtr).view(np.uint8).reshape(len(Xtr), -1)))
        del Xtr
        novel = np.array([bytes(r) not in seen for r in np.ascontiguousarray(Xt).view(np.uint8).reshape(len(Xt), -1)])
        for name in ("skip", "added"):
            rows.append(row(f"novel_{name}", True, pt[novel], ft[novel], tt[novel], flt[novel], taus[name],
                            {"novel_share": float(novel.mean())}))
        dup = pd.DataFrame({"family": ft, "novel": novel}).groupby("family").novel.mean()
        dup.to_csv(f"{OUT}/novel_share_{tag}.csv")
        print("share of test rows not in train:", dup.round(3).to_dict(), flush=True)

    res = pd.DataFrame(rows)
    res.insert(0, "dataset", ds_name)
    res.to_csv(f"{OUT}/cert_{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["rule", "feasible", "tau", "benign_bypass", "skip_worst_family", "skip_worst_family_rate",
               "skip_worst_type", "skip_worst_type_n", "worst_family_added", "worst_family_added_n"]].to_string(index=False))

    fam_rows = []
    for name in ("skip", "added"):
        for g in np.unique(ft[ft != "Benign"]):
            mk = ft == g
            k = int((pt[mk] < taus[name]).sum()) if name == "skip" else int(((pt[mk] < taus[name]) & flt[mk]).sum())
            n = int(mk.sum())
            fam_rows.append({"rule": name, "family": g, "test_n": n, "count": k, "rate": k / n,
                             "ci95_high": float(cp_upper(k, n, ALPHA)),
                             "brain_miss": float((~flt[mk]).mean()),
                             "two_stage_miss": float(((pt[mk] < taus[name]) | ~flt[mk]).mean())})
    pd.DataFrame(fam_rows).to_csv(f"{OUT}/families_{tag}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
