"""Datasets from the clean packet records (src/flow_records.py) of CICIoMT2024.

Records of attack captures that do not use the attack's protocol (background traffic of the other devices)
are dropped. Every attack type is thinned to at most CAP records per side (train captures / test captures)
by taking every k-th record in time order; benign records are kept in full.

Splits (data/CICIoMT2024/packet/<split>/{train,validation,test}.npz):
  C  capture-separated: train captures -> first 85% of each capture in time = train, last 15% = validation
     (calibration); test captures (other recordings) = test. Within a split, rows are in capture, time order.
  R  control: the same records pooled and split 70/15/15 at random (seed 0); records of one flow can land
     in different splits, as with a shuffled CSV.
Each npz holds X (float32), fam, typ, capture, t0, flow, host_a, host_b and the feature names.

Usage: .venv/bin/python src/build_packet_dataset.py [records_dir out_dir]
       (default: records -> packet; with context features: records_ctx -> packet_ctx)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from flow_records import FEATURES, attack_mask

LABEL = os.environ.get("PKT_LABEL", "proto")   # 'mac' = attacker-MAC ground truth

SRC = "data/CICIoMT2024/records"
OUT = "data/CICIoMT2024/packet"
CAP = 250_000
VAL_FRAC = 0.15
META = ["capture", "t0", "flow", "host_a", "host_b"]


def load_side(side):
    parts = []
    for f in sorted(glob.glob(f"{SRC}/{side}/*.parquet")):
        d = pd.read_parquet(f)
        d = d[attack_mask(d, LABEL)].copy()
        d["type"] = d["type"].str.replace(r"\d+$", "", regex=True)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    keep = []
    for t, g in d.groupby("type", sort=True):
        g = g.sort_values(["capture", "t0"])
        if g.family.iloc[0] != "Benign" and len(g) > CAP:
            g = g.iloc[np.linspace(0, len(g) - 1, CAP).round().astype(int)]
        keep.append(g)
    return pd.concat(keep, ignore_index=True).sort_values(["capture", "t0"], kind="stable").reset_index(drop=True)


def save(d, path):
    np.savez(path, X=d[FEATURES].to_numpy(np.float32), fam=d.family.to_numpy(str), typ=d.type.to_numpy(str),
             capture=d.capture.to_numpy(str), t0=d.t0.to_numpy(float), flow=d.flow.to_numpy(np.int64),
             host_a=d.host_a.to_numpy(str), host_b=d.host_b.to_numpy(str), features=np.array(FEATURES))


def main():
    global SRC, OUT, FEATURES
    if len(sys.argv) > 2:
        SRC, OUT = sys.argv[1], sys.argv[2]
    tr, te = load_side("train"), load_side("test")
    FEATURES = [f for f in FEATURES if f in tr.columns]
    if os.environ.get("PKT_SWAP_ARP"):
        a = "ARP_Spoofing"
        tr_a, te_a = tr[tr.type == a], te[te.type == a]
        tr = pd.concat([tr[tr.type != a], te_a], ignore_index=True)
        te = pd.concat([te[te.type != a], tr_a], ignore_index=True)
        print(f"swapped ARP: train now {len(te_a)} ARP rows, test now {len(tr_a)}")
    # C: time cut inside every train capture
    pos = tr.groupby("capture").cumcount() / tr.groupby("capture").capture.transform("size")
    os.makedirs(f"{OUT}/C", exist_ok=True)
    save(tr[pos < 1 - VAL_FRAC], f"{OUT}/C/train.npz")
    save(tr[pos >= 1 - VAL_FRAC], f"{OUT}/C/validation.npz")
    save(te, f"{OUT}/C/test.npz")
    # R: pooled, random 70/15/15
    allr = pd.concat([tr, te], ignore_index=True)
    perm = np.random.default_rng(0).permutation(len(allr))
    a, b = int(0.70 * len(allr)), int(0.85 * len(allr))
    os.makedirs(f"{OUT}/R", exist_ok=True)
    for name, idx in (("train", perm[:a]), ("validation", perm[a:b]), ("test", perm[b:])):
        save(allr.iloc[np.sort(idx)], f"{OUT}/R/{name}.npz")
    rows = []
    for sp in ("C", "R"):
        for name in ("train", "validation", "test"):
            z = np.load(f"{OUT}/{sp}/{name}.npz")
            c = pd.Series(z["fam"]).value_counts()
            rows.append({"split": sp, "part": name, "rows": len(z["fam"]), **c.to_dict()})
    t = pd.DataFrame(rows).fillna(0)
    os.makedirs("results/packet", exist_ok=True)
    t.to_csv(f"results/packet/dataset_counts{'_ctx' if 'ctx' in OUT else ''}.csv", index=False)
    print(t.to_string(index=False))
    tv = pd.Series(np.load(f"{OUT}/C/validation.npz")["typ"]).value_counts()
    print("C validation per type:\n", tv.to_string())


if __name__ == "__main__":
    main()
