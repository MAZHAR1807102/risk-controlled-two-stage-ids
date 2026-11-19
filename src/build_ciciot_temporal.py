"""Time-ordered CICIoT2023 splits from the per-attack CSV files of the CIC release.

The published train/validation/test.csv come from one shuffled pool, so neighbouring flows of the
same capture can land in different splits. The per-attack files (CSV/CSV/<attack>/<attack>N.pcap.csv,
39 features, re-extracted by CIC in 2024) keep extraction order: adjacent rows are 6-10x more often
identical than shuffled rows. Here every file is cut by position: first 70% train, next 15%
validation, last 15% test. So validation and test always lie later in time than train, inside
every capture file.

Large flood types are thinned to at most CAP rows per type by keeping every k-th row (order kept).
Benign and all rare types are kept in full. +-inf (a few rows of Rate) is replaced by the largest
finite value of that column in the same file. A few files end in a half row (cut on the server);
such rows are dropped.

Usage: .venv/bin/python src/build_ciciot_temporal.py   -> data/CICIoT2023/temporal/{train,validation,test}.npz
"""
import glob
import math
import os

import numpy as np
import pandas as pd

from eval_offline import macro_group

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "CICIoT2023")
SRC, OUT = os.path.join(ROOT, "per_attack_csv"), os.path.join(ROOT, "temporal")
CAP = 300_000
CUTS = (0.70, 0.85)
# folder name -> label of the original release (used by macro_group and the paper's tables)
RENAME = {"Benign_Final": "BenignTraffic", "DDoS-PSHACK_FLOOD": "DDoS-PSHACK_Flood",
          "DDoS-RSTFINFLOOD": "DDoS-RSTFINFlood"}


def file_order(path):
    stem = os.path.basename(path).split(".pcap")[0].rstrip("-")
    digits = stem[len(stem.rstrip("0123456789")):]
    return int(digits) if digits else -1


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = {s: {"X": [], "label": [], "file": [], "pos": []} for s in ("train", "validation", "test")}
    feats, rows = None, []
    for folder in sorted(os.listdir(SRC)):
        files = sorted(glob.glob(os.path.join(SRC, folder, "*.csv")), key=file_order)
        if not files:
            continue
        label = RENAME.get(folder, folder)
        fam = macro_group(label)
        n_total = sum(sum(1 for _ in open(f)) - 1 for f in files)
        k = 1 if fam == "Benign" else max(1, math.ceil(n_total / CAP))
        for f in files:
            df = pd.read_csv(f, on_bad_lines="skip")
            if feats is None:
                feats = list(df.columns)
            elif list(df.columns) != feats:
                raise ValueError(f"column mismatch in {f}")
            n_raw = len(df)
            df = df.dropna()  # half row at the end of a cut file
            X = df.to_numpy(np.float64)
            for j in range(X.shape[1]):
                bad = ~np.isfinite(X[:, j])
                if bad.any():
                    fin = X[~bad, j]
                    X[bad, j] = np.where(X[bad, j] > 0, fin.max(), fin.min())
            X = X[::k].astype(np.float32)
            n = len(X)
            a, b = int(round(n * CUTS[0])), int(round(n * CUTS[1]))
            name = os.path.relpath(f, SRC)
            for s, sl in (("train", slice(0, a)), ("validation", slice(a, b)), ("test", slice(b, n))):
                m = len(X[sl])
                parts[s]["X"].append(X[sl])
                parts[s]["label"].append(np.full(m, label, dtype=object))
                parts[s]["file"].append(np.full(m, name, dtype=object))
                parts[s]["pos"].append(np.arange(sl.start, sl.start + m))
            rows.append({"label": label, "family": fam, "file": name, "rows_raw": n_raw,
                         "dropped": n_raw - len(df), "stride": k, "kept": n, "train": a, "validation": b - a,
                         "test": n - b})
        print(f"{label:26s} {fam:10s} stride {k:3d}", flush=True)
    for s, p in parts.items():
        np.savez(os.path.join(OUT, f"{s}.npz"), X=np.concatenate(p["X"]), label=np.concatenate(p["label"]).astype(str),
                 src_file=np.concatenate(p["file"]).astype(str), pos=np.concatenate(p["pos"]), features=np.array(feats))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, "files.csv"), index=False)
    by = summary.groupby("family")[["rows_raw", "kept", "train", "validation", "test"]].sum()
    by.to_csv(os.path.join(OUT, "families.csv"))
    print(by.to_string())
    print("total", by.sum().to_dict())


if __name__ == "__main__":
    main()
