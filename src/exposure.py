"""What the guarantee covers, per attack family (CICIoT2023 test split, scored once).

At the certified skip threshold (main Brain, runs/seed42_sqrt) and at the whole-system threshold
(Brain that never saw validation, runs/seed42_sel), for every family:
  skip      - share skipped by the Gatekeeper, counted in flows, in packets and in bytes
  brain_fn  - share the Brain alone calls benign (single-stage miss)
  total_fn  - share the two-stage IDS calls benign (skipped, or inspected and called benign)
  added     - total_fn - brain_fn (misses the Gatekeeper adds)
Packets per flow record = 'Number', bytes = 'Tot sum' of the CIC record (window statistics).
Also: per attack type, calibration count, n_min and whether the type is certified on its own.

Usage: .venv/bin/python src/exposure.py  -> results/exposure/{families,types}.csv
"""
import os

import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_added import scores
from risk_control import ALPHA, EPS_TYPE, n_min

OUT = "results/exposure"
from risk_control import certified_tau
SETTINGS = [("certified skip", "runs/seed42_sqrt", certified_tau()),
            ("whole system", "runs/seed42_sel", certified_tau(quantity="added"))]


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = get_dataset("ciciot2023")
    raw = pd.read_csv("data/CICIoT2023/test/test.csv", usecols=["Number", "Tot sum"])
    pk, by = raw["Number"].to_numpy(float), raw["Tot sum"].to_numpy(float)
    X, fam, typ, _ = ds.load("test")
    rows = []
    for name, run, tau in SETTINGS:
        p, flagged = scores(run, X, ds.classes)
        skip = p < tau
        missed = skip | ~flagged          # two-stage calls it benign
        for g in [g for g in np.unique(fam) if g != "Benign"]:
            m = fam == g
            w = lambda v, x: float((v[m] * x[m]).sum() / x[m].sum())
            rows.append({"setting": name, "tau": tau, "family": g, "flows": int(m.sum()),
                         "skip_flows": float(skip[m].mean()), "skip_packets": w(skip, pk), "skip_bytes": w(skip, by),
                         "brain_fn": float((~flagged[m]).mean()), "total_fn": float(missed[m].mean()),
                         "added": float(missed[m].mean() - (~flagged[m]).mean()),
                         "total_fn_packets": w(missed, pk), "total_fn_bytes": w(missed, by)})
    fams = pd.DataFrame(rows)
    fams.to_csv(f"{OUT}/families.csv", index=False)
    pd.set_option("display.width", 250)
    print(fams.round(4).to_string(index=False))

    # attack types: calibration counts and individual certification (Bonferroni over certifiable types)
    _, fv, tv, _ = ds.load("validation")
    att = fv != "Benign"
    types, counts = np.unique(tv[att], return_counts=True)
    nmin = n_min(EPS_TYPE, ALPHA)
    p, _ = scores("runs/seed42_sqrt", X, ds.classes)
    trows = []
    for t, c in zip(types, counts):
        m = typ == t
        trows.append({"type": t, "family": fam[m][0], "calibration_n": int(c), "n_min": int(nmin),
                      "certified_alone": bool(c >= nmin), "test_n": int(m.sum()),
                      "test_skip": float((p[m] < SETTINGS[0][2]).mean())})
    types_df = pd.DataFrame(trows).sort_values(["family", "type"])
    types_df.to_csv(f"{OUT}/types.csv", index=False)
    print(types_df[types_df.family.isin(["BruteForce", "Recon", "Spoofing", "Web"])].to_string(index=False))


if __name__ == "__main__":
    main()
