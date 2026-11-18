"""Build labelled traffic files for the ns-3 gateway simulation, from the TEST split.

Each file has 46 raw features + integer class label (+ the
raw CICIoT2023 label), and every attack group is represented in proportion to the
test split, with the rare groups guaranteed a floor so they actually get exercised.

Usage: .venv/bin/python src/make_ns3_traffic.py --rows 20000 --out ns3/traffic
"""
import argparse
import os

import numpy as np
import pandas as pd

from eval_offline import CLASSES, macro_group

TEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "CICIoT2023", "test", "test.csv")
MIN_PER_GROUP = 100  # floor so Web/BruteForce/Recon/Spoofing appear in every file


def build(df, rows, benign_frac, rng):
    n_ben = int(round(rows * benign_frac))
    n_att = rows - n_ben
    att = df[df.group != "Benign"]
    # Allocate attacks by test-split share, then raise rare groups to the floor.
    share = att.group.value_counts(normalize=True)
    floor = min(MIN_PER_GROUP, n_att // 20)  # keep the floor small when attacks are few
    alloc = (share * n_att).round().astype(int).clip(lower=floor)
    alloc = alloc.clip(upper=att.group.value_counts())
    alloc[share.idxmax()] -= alloc.sum() - n_att  # keep the total exact via the largest group
    if n_att == 0:
        alloc = alloc * 0
    parts = [df[df.group == "Benign"].sample(n_ben, random_state=rng.integers(1 << 31))]
    parts += [att[att.group == g].sample(k, random_state=rng.integers(1 << 31))
              for g, k in alloc.items()]
    out = pd.concat(parts).sample(frac=1.0, random_state=rng.integers(1 << 31))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--out", default="ns3/traffic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(TEST)
    df["group"] = df["label"].map({l: macro_group(l) for l in df["label"].unique()})
    feats = list(df.columns[:46])

    mixes = {"natural": None, "benign75": 0.75, "benign90": 0.90}
    for name, bf in mixes.items():
        if bf is None:  # plain random sample: the real test-split mix (~2.4% benign)
            sub = df.sample(args.rows, random_state=rng.integers(1 << 31))
        else:
            sub = build(df, args.rows, bf, rng)
        out = sub[feats].copy()
        out["class_id"] = sub["group"].map(CLASSES.index).astype(int)
        out["raw_label"] = sub["label"].values
        path = f"{args.out}/test_{name}_{args.rows}.csv"
        out.to_csv(path, index=False)
        counts = out["class_id"].map(lambda i: CLASSES[i]).value_counts()
        print(f"{path}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
