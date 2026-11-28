"""Trade-off curve of the packet-level main model (pMN) on the separate test captures.

For a grid of Gatekeeper thresholds: benign bypass, worst-family skip rate and worst-family
added miss (skipped AND the Brain would flag it). Used by make_figures.fig1 panel (b).

Usage: .venv/bin/python src/pkt_tradeoff.py   -> results/packet/tradeoff_pMN.csv
"""
import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_added import scores

DS, RUN, OUT = "ciciomt2024pmn", "runs/pkt_MN_sel", "results/packet/tradeoff_pMN.csv"


def main():
    ds = get_dataset(DS)
    X, fam, _, _ = ds.load("test")
    p, flagged = scores(RUN, X, ds.classes)
    ben = fam == "Benign"
    groups = [g for g in np.unique(fam) if g != "Benign"]
    masks = {g: fam == g for g in groups}
    rows = []
    for t in np.unique(np.concatenate([np.geomspace(1e-6, 0.9999, 600), [0.5]])):
        skip = p < t
        rows.append({"tau": t, "benign_bypass": skip[ben].mean(),
                     "worst_family_skip": max(skip[m].mean() for m in masks.values()),
                     "worst_family_added": max((skip[m] & flagged[m]).mean() for m in masks.values())})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
