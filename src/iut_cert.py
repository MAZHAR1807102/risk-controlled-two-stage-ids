"""Certified threshold without the Bonferroni split (Learn Then Test, exact binomial, fixed sequence).

Referee A4/A14: our rule splits alpha over families and types (Bonferroni). In the fixed-sequence
rule the failure event is decided at the first bad threshold by one unit, so every unit can be tested
at the full alpha (intersection-union test) and the guarantee of Proposition 1 still holds under
(A1)-(A2). This script recomputes the skip and whole-system thresholds of the seven data settings of
the paper with both budgets ("ucb_typ" = paper, "ucb_typ_iut" = no split) and scores them on test.

Usage: .venv/bin/python src/iut_cert.py [tag ...]  -> results/packet/iut_cert.csv (tags given: rerun and keep the rest)
"""
import os

import pandas as pd

from datasets import get_dataset
from packet_cert import row
from risk_added import added_score, scores
from risk_control import pick

SETTINGS = [("c23", "ciciot2023", "runs/seed42_sel"), ("c23x", "ciciot2023x", "runs/c23_x_sel"),
            ("mt", "ciciomt2024", "runs/iomt_seed42_sel"), ("mtx", "ciciomt2024x", "runs/iomt_x_sel"),
            ("pR", "ciciomt2024pr", "runs/pkt_R_sel"), ("pCN", "ciciomt2024pn", "runs/pkt_CN_sel"),
            ("pMN", "ciciomt2024pmn", "runs/pkt_MN_sel")]
OUT = "results/packet/iut_cert.csv"


def main(only=None):
    rows = pd.read_csv(OUT).to_dict("records") if only and os.path.exists(OUT) else []
    for tag, ds_name, run in SETTINGS:
        if only and tag not in only:
            continue
        ds = get_dataset(ds_name)
        Xv, fv, tv, _ = ds.load("validation")
        pv, flv = scores(run, Xv, ds.classes)
        del Xv
        Xt, ft, tt, _ = ds.load("test")
        pt, flt = scores(run, Xt, ds.classes)
        del Xt
        for target, q in (("skip", pv), ("added", added_score(pv, fv, flv))):
            for rule in ("ucb_typ", "ucb_typ_iut"):
                tau, ok = pick(q, fv, tv, rule)
                r = row(f"{target}_{rule}", ok, pt, ft, tt, flt, tau)
                r.update({"setting": tag, "dataset": ds_name, "target": target, "budget": rule})
                rows.append(r)
                print(tag, target, rule, ok, round(tau, 5), round(r["benign_bypass"], 4),
                      r.get("skip_worst_family_rate"), r.get("worst_family_added"), flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    import sys
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    main(sys.argv[1:] or None)
