"""Exact-duplicate analysis of the CICIoT2023 splits (reviewer: dependence across splits).

For every row we hash its 46 raw features. We report, per family and split, how many
rows have an exact copy (same features) inside the same split, and how many validation /
test rows have an exact copy in another split.

Usage: .venv/bin/python src/dup_analysis.py
"""
import os

import numpy as np
import pandas as pd

from datasets import get_dataset

OUT = "results/dependence"


def row_hash(X):
    X = np.ascontiguousarray(X)
    v = X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
    return pd.Series(v).map(hash).to_numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = get_dataset("ciciot2023")
    H, F, T = {}, {}, {}
    for s in ("train", "validation", "test"):
        X, fam, typ, _ = ds.load(s)
        H[s], F[s], T[s] = row_hash(X), fam, typ
        print("hashed", s, len(X), flush=True)
    sets = {s: set(H[s]) for s in H}
    rows = []
    for s in H:
        h = pd.Series(H[s])
        within = h.duplicated(keep=False).to_numpy()
        others = [o for o in H if o != s]
        cross = {o: np.isin(H[s], list(sets[o])) for o in others}
        for g in ["ALL"] + sorted(np.unique(F[s])):
            m = np.ones(len(h), bool) if g == "ALL" else F[s] == g
            r = {"split": s, "family": g, "n": int(m.sum()),
                 "unique_vectors": int(h[m].nunique()),
                 "dup_within_split": float(within[m].mean())}
            for o in others:
                r[f"in_{o}"] = float(cross[o][m].mean())
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/exact_duplicates.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
