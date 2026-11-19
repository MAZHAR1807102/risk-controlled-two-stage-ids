"""Flow-shaping (mimicry) attack on the Gatekeeper and the whole IDS (CICIoT2023 test split).

The attacker keeps what the attack needs (protocol, flags, ports) but shapes the timing and the sizes
of its packets to look like benign traffic. For every low-volume attack flow, the features of the chosen
group are copied from a random benign test flow with the same protocol type:
  timing  - flow_duration, Rate, Srate, Drate, IAT
  sizes   - Tot sum, Min, Max, AVG, Std, Tot size, Number, Magnitue, Radius, Covariance, Variance, Weight
  both    - timing and sizes
  IAT only, timing without IAT - to see which timing feature carries the effect
Reported per family: Gatekeeper skip rate at the certified threshold (0.0166) and the share the whole
two-stage IDS calls benign. The calibration data has no shaped flows, so the certificate does not cover
this attacker; the numbers show how far it breaks.

Usage: .venv/bin/python src/evasion.py  -> results/evasion/mimicry.csv
"""
import os

import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_added import scores

OUT = "results/evasion"
TAU = 0.0166
GROUPS = {"timing": ["flow_duration", "Rate", "Srate", "Drate", "IAT"],
          "sizes": ["Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "Number", "Magnitue", "Radius",
                    "Covariance", "Variance", "Weight"]}
GROUPS["both"] = GROUPS["timing"] + GROUPS["sizes"]
GROUPS["IAT only"] = ["IAT"]
GROUPS["timing without IAT"] = ["flow_duration", "Rate", "Srate", "Drate"]
LOW = ["BruteForce", "Recon", "Spoofing", "Web"]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(3)
    ds = get_dataset("ciciot2023")
    X, fam, _, feats = ds.load("test")
    col = {f: i for i, f in enumerate(feats)}
    proto = X[:, col["Protocol Type"]]
    ben = np.flatnonzero(fam == "Benign")
    att = np.flatnonzero(np.isin(fam, LOW))
    # a benign donor with the same protocol type for every attack flow (any benign flow if none)
    donors = np.empty(len(att), dtype=int)
    for v in np.unique(proto[att]):
        same = ben[proto[ben] == v]
        sel = proto[att] == v
        donors[sel] = rng.choice(same if len(same) else ben, size=sel.sum())
    rows = []
    for name, group in [("none", [])] + list(GROUPS.items()):
        Xa = X[att].copy()
        for f in group:
            Xa[:, col[f]] = X[donors, col[f]]
        p, flagged = scores("runs/seed42_sqrt", Xa, ds.classes)
        for g in LOW:
            m = fam[att] == g
            rows.append({"shaped": name, "family": g, "flows": int(m.sum()),
                         "gatekeeper_skip": float((p[m] < TAU).mean()),
                         "ids_miss": float(((p[m] < TAU) | ~flagged[m]).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/mimicry.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.pivot(index="family", columns="shaped", values=["gatekeeper_skip", "ids_miss"]).round(3).to_string())


if __name__ == "__main__":
    main()
