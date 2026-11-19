"""Threshold rules on the time-ordered CICIoT2023 split (answer to "independence not established").

Validation and test lie later in time than train inside every capture file, and test lies later
than validation (src/build_ciciot_temporal.py). The run must use a Brain that never saw validation
(pipeline.py --select-frac 0.15). For each rule, tau is picked on validation and the test split is
scored once:
  point  - observed rule: largest tau with every validation family skip rate <= 2%
  skip   - certified rule on the Gatekeeper's skips (Eq. tauhat, ucb_typ)
  added  - certified rule on the misses the Gatekeeper adds to the Brain's (whole system)
  skip_D - the skip rule with every count divided by the design effect D measured on validation in
           time order (blocks of 500 rows) at the certified threshold (risk_dependence.pick_D)
Control: dataset ciciot2023r = the same rows split at random (row order then carries no time, so no D).

Usage: .venv/bin/python src/temporal_cert.py [ciciot2023t|ciciot2023r] [run]
       -> results/temporal/{certificates,families_certified,design_effect}{_random}.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_added import added_score, evaluate as evaluate_added, scores
from risk_control import ALPHA, EPS, EPS_TYPE, cp_upper, n_min, pick
from risk_coverage import design_effect
from risk_dependence import pick_D

OUT = "results/temporal"


def skip_rates(p, fam, typ, tau):
    """Worst family and worst type by plain skip rate (all types, and certifiable types only)."""
    att = fam != "Benign"
    out = {}
    for key, lab in (("family", fam), ("type", typ)):
        r = {u: (int((p[lab == u] < tau).sum()), int((lab == u).sum())) for u in np.unique(lab[att])}
        w = max(r, key=lambda u: r[u][0] / r[u][1])
        out.update({f"skip_worst_{key}": w, f"skip_worst_{key}_rate": r[w][0] / r[w][1],
                    f"skip_worst_{key}_n": f"{r[w][0]}/{r[w][1]}"})
    return out


def main(ds_name="ciciot2023t", run=None):
    os.makedirs(OUT, exist_ok=True)
    temporal = ds_name == "ciciot2023t"
    run = run or ("runs/t23_seed42_sel" if temporal else "runs/r23_seed42_sel")
    sfx = "" if temporal else "_random"
    ds = get_dataset(ds_name)
    data = {}
    for split in ("validation", "test"):
        X, fam, typ, _ = ds.load(split)
        p, fl = scores(run, X, ds.classes)
        data[split] = (p, fam, typ, fl)
        print(f"scored {split}: {len(p):,} rows", flush=True)
    pv, fv, tv, flv = data["validation"]
    pt, ft, tt, flt = data["test"]

    att = fv != "Benign"
    types, counts = np.unique(tv[att], return_counts=True)
    print("validation counts per family:", dict(zip(*np.unique(fv[att], return_counts=True))))
    print(f"n_min family {n_min(EPS, ALPHA / 2 / 7)}, type {n_min(EPS_TYPE, ALPHA / 2 / len(types))}; "
          f"types below: {list(types[counts < n_min(EPS_TYPE, ALPHA / 2 / len(types))])}")

    rows = []
    for name, q, rule in (("point", pv, "point"), ("skip", pv, "ucb_typ"),
                          ("added", added_score(pv, fv, flv), "ucb_typ")):
        tau, ok = pick(q, fv, tv, rule)
        r = {"rule": name, "feasible": ok}
        r.update(evaluate_added(pt, ft, tt, flt, tau))
        r.update(skip_rates(pt, ft, tt, tau))
        r.update({f"val_{k}": v for k, v in skip_rates(pv, fv, tv, tau).items()})
        rows.append(r)
    tau_c = float(rows[1]["tau"])
    de = pd.DataFrame(design_effect(pv, fv, tv, tau_c, f"{ds_name} validation")) if temporal else pd.DataFrame()
    if len(de):
        de.to_csv(f"{OUT}/design_effect.csv", index=False)
        print(de.to_string(index=False))
        tau, ok = pick_D(pv, fv, tv, dict(zip(de.unit, de.deff)))
        r = {"rule": "skip_D", "feasible": ok}
        r.update(evaluate_added(pt, ft, tt, flt, tau))
        r.update(skip_rates(pt, ft, tt, tau))
        r.update({f"val_{k}": v for k, v in skip_rates(pv, fv, tv, tau).items()})
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/certificates{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["rule", "feasible", "tau", "benign_bypass", "val_skip_worst_family_rate", "skip_worst_family",
               "skip_worst_family_rate", "skip_worst_family_n", "skip_worst_type", "skip_worst_type_n",
               "worst_family_added", "worst_family_added_n", "worst_type_added_n"]].to_string(index=False))

    # Clopper-Pearson upper bound on each family's test skip rate at the certified threshold
    fam_rows = []
    for g in np.unique(ft[ft != "Benign"]):
        m = ft == g
        k, n = int((pt[m] < tau_c).sum()), int(m.sum())
        fam_rows.append({"family": g, "test_n": n, "skipped": k, "rate": k / n, "ci95_high": cp_upper(k, n, ALPHA)})
    pd.DataFrame(fam_rows).to_csv(f"{OUT}/families_certified{sfx}.csv", index=False)
    print(pd.DataFrame(fam_rows).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
