"""Step 1: cost-sensitive Gatekeeper retraining.

A Gatekeeper trained with XGBoost defaults (unweighted, 97.6% attack prior)
misses 9-14% of low-volume attacks (Recon, BruteForce, Spoofing). This script
retrains it at the SAME capacity (100 trees, depth 6 -> same latency) with
different sample weightings, then picks tau on the validation split so that the
WORST attack group's miss rate stays under a target, and reports on test.

Protocol: train -> fit, validation -> tau selection, test -> reporting only.
Usage: .venv/bin/python src/train_gatekeeper.py --params runs/seed42_sqrt/ids_params.h
"""
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from eval_offline import macro_group

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "CICIoT2023", "{0}", "{0}.csv")
OUT = "results/gatekeeper_weighting_final"
ATTACK_GROUPS = ["BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
STEALTHY = ["BruteForce", "Recon", "Spoofing", "Web"]
TARGETS = [0.01, 0.02, 0.05]  # max per-group miss rate allowed on validation

# Same capacity as the deployed Gatekeeper so latency is directly comparable.
PARAMS = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.3,
          "tree_method": "hist", "eval_metric": "logloss", "nthread": -1}
N_TREES = 100


SCALER = {}  # set in main(): {"means": [...], "scales": [...]}


def scaler_from_header(path):
    """Read SCALER_MEANS / SCALER_SCALES from a pipeline-generated ids_params.h."""
    text = open(path).read()
    arr = lambda name: [float(v) for v in
                        text.split(name)[1].split("{")[1].split("}")[0].split(",")]
    return {"means": arr("SCALER_MEANS"), "scales": arr("SCALER_SCALES")}


def load(split):
    t = time.time()
    df = pd.read_csv(DATA.format(split))
    s = SCALER
    X = (df.iloc[:, :46].to_numpy(np.float32) - np.float32(s["means"])) / np.float32(s["scales"])
    labels = df["label"].to_numpy()
    lut = {l: macro_group(l) for l in np.unique(labels)}
    g = np.array([lut[l] for l in labels])
    print(f"  {split}: {len(df):,} rows in {time.time() - t:.0f}s")
    return X, g, labels


def weights(g, scheme):
    """Per-sample weights. Each scheme keeps the mean weight at 1."""
    if scheme == "unweighted":
        w = np.ones(len(g))
    elif scheme == "binary_balanced":       # benign total == attack total
        ben = g == "Benign"
        w = np.where(ben, 0.5 / ben.mean(), 0.5 / (~ben).mean())
    elif scheme in ("group_balanced", "group_sqrt"):
        # Benign gets half the total weight; the 7 attack groups share the other
        # half equally (group_balanced) or in proportion to sqrt(count) (group_sqrt).
        groups, counts = np.unique(g, return_counts=True)
        cnt = dict(zip(groups, counts))
        att = [k for k in groups if k != "Benign"]
        share = {k: (1.0 if scheme == "group_balanced" else np.sqrt(cnt[k])) for k in att}
        z = sum(share.values())
        per = {"Benign": 0.5 / cnt["Benign"]}
        per.update({k: 0.5 * share[k] / z / cnt[k] for k in att})
        w = np.array([per[k] for k in g]) * len(g)
    else:
        raise ValueError(scheme)
    return w.astype(np.float32)


def group_miss(p, g, tau):
    """Miss rate per attack family; families are taken from the data (any dataset)."""
    families = [k for k in np.unique(g) if k != "Benign"]
    return {k: float((p[g == k] < tau).mean()) for k in families}


def pick_tau(p, g, target):
    """Largest tau whose worst attack-group miss rate is <= target (max bypass)."""
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), [0.3]]))
    ok = [t for t in grid if max(group_miss(p, g, t).values()) <= target]
    return float(max(ok)) if ok else float(grid[0])


def summarize(p, g, tau, stealthy=STEALTHY):
    ben = g == "Benign"
    miss = group_miss(p, g, tau)
    flagged_benign = float((p[ben] >= tau).mean())
    att = ~ben
    return {"tau": tau,
            "benign_bypass": 1 - flagged_benign,
            "overall_fnr": float((p[att] < tau).mean()),
            "worst_group_miss": max(miss.values()),
            "stealthy_miss": float((p[np.isin(g, stealthy)] < tau).mean()),
            **{f"miss_{k}": v for k, v in miss.items()}}


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="runs/seed42_sqrt/ids_params.h",
                    help="ids_params.h to take the scaler from")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    SCALER.update(scaler_from_header(args.params))
    os.makedirs(OUT, exist_ok=True)
    print("Loading splits ...")
    Xtr, gtr, _ = load("train")
    Xva, gva, _ = load("validation")
    Xte, gte, lte = load("test")
    ytr = (gtr != "Benign").astype(np.float32)
    dva, dte = xgb.DMatrix(Xva, missing=-1.0), xgb.DMatrix(Xte, missing=-1.0)

    models = {}
    for scheme in ["unweighted", "binary_balanced", "group_sqrt", "group_balanced"]:
        t = time.time()
        dtr = xgb.QuantileDMatrix(Xtr, ytr, weight=weights(gtr, scheme), missing=-1.0)
        models[scheme] = xgb.train(PARAMS, dtr, N_TREES)
        models[scheme].save_model(f"{OUT}/gatekeeper_{scheme}.json")
        del dtr
        print(f"  trained {scheme} in {time.time() - t:.0f}s")

    rows, per_attack = [], []
    for name, m in models.items():
        pva, pte = m.predict(dva), m.predict(dte)
        rows.append({"model": name, "selection": "tau=0.3 (paper)", **summarize(pte, gte, 0.3)})
        for target in TARGETS:
            tau = pick_tau(pva, gva, target)
            rows.append({"model": name, "selection": f"val worst-group miss<={target:.0%}",
                         **summarize(pte, gte, tau)})
        # Per-attack-type miss at the 2% operating point, for the paper's table.
        tau2 = pick_tau(pva, gva, 0.02)
        df = pd.DataFrame({"label": lte, "group": gte, "missed": pte < tau2})
        df = df[df.group != "Benign"].groupby(["group", "label"])["missed"].mean()
        per_attack.append(df.rename(name))
        np.save(f"{OUT}/test_scores_{name}.npy", pte.astype(np.float32))

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/comparison.csv", index=False)
    pd.concat(per_attack, axis=1).to_csv(f"{OUT}/miss_by_attack_at_2pct.csv")
    np.save(f"{OUT}/test_groups.npy", gte)

    pd.set_option("display.width", 250)
    cols = ["model", "selection", "tau", "benign_bypass", "overall_fnr",
            "worst_group_miss", "stealthy_miss"] + [f"miss_{k}" for k in STEALTHY]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
