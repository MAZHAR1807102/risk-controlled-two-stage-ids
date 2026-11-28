"""Adaptive router: estimator bias and unseen load patterns (offline copy of the ns-3 router).

The router (ns3/IoT_TwoStage_IDS_v2.cc, UpdateAdaptive) keeps EWMAs (alpha 0.01) of the Gatekeeper's
bypass share, of the share of flows that end benign (bypassed or Stage 2 says benign), and of the stage
costs. In direct mode it cannot see the bypass share and uses benign_share * B0 (B0 = 0.77). It switches
when the saved Stage 2 time leaves the band (1 +- h) C1 (h = 0.2), at most every 200 flows.
This script runs the same rule on flows of the CICIoT2023 test split (deployed models, tau = 0.05),
with fixed per-flow costs from the ns-3 timing runs (one thread: C1 24 us, C2 2,000 us; all cores:
C1 150 us, C2 750 us). The defaults were fixed in the first router version and are not tuned here.
  bias      - for benign shares and attack mixtures: true bypass share B vs the direct-mode estimate,
              and whether the estimate picks the cheaper design
  patterns  - six load patterns not used before; cost vs an oracle that picks the cheaper static design
              per 500-flow block, static designs, switches, and the error of the router's saving estimate

Usage: .venv/bin/python src/router_offline.py [onnx] -> results/router/{bias,patterns}[_onnx].csv
"""
import os
import sys

import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_added import scores

OUT = "results/router"
TAU, B0, ALPHA, H, DWELL = 0.05, 0.77, 0.01, 0.2, 200
COSTS = {"1 thread": (24.0, 2000.0), "all cores": (150.0, 750.0)}
POOLS = {"floods": ["DDoS", "DoS", "Mirai"], "low-volume": ["BruteForce", "Recon", "Spoofing", "Web"],
         "web+bruteforce": ["Web", "BruteForce"]}


def load():
    ds = get_dataset("ciciot2023")
    X, fam, _, _ = ds.load("test")
    p, flagged = scores("runs/seed42_sqrt", X, ds.classes)
    return p < TAU, ~flagged, fam  # skipped by the Gatekeeper, Stage 2 says benign, family


def bias(skip, brain_benign, fam):
    rows = []
    ben = fam == "Benign"
    for pool, fams in POOLS.items():
        att = np.isin(fam, fams)
        s_b, s_a = skip[ben].mean(), skip[att].mean()          # bypass of benign / of these attacks
        bb_b, bb_a = brain_benign[ben].mean(), brain_benign[att].mean()
        for pi in (0.02, 0.10, 0.25, 0.50, 0.75, 0.90):
            B = pi * s_b + (1 - pi) * s_a                         # true bypass share
            est = (pi * bb_b + (1 - pi) * bb_a) * B0              # direct-mode estimate
            r = {"attacks": pool, "benign_share": pi, "true_bypass": B, "estimate": est, "bias": est - B,
                 "brain_benign_on_attacks": bb_a}
            for name, (c1, c2) in COSTS.items():
                r[f"right_choice_{name}"] = bool((B * c2 > c1) == (est * c2 > c1))
            rows.append(r)
    return pd.DataFrame(rows)


def make_pattern(name, idx, rng, n=20000):
    """List of (pool name or 'benign') per flow."""
    def seg(m, pi, pool):
        return [("benign" if rng.random() < pi else pool) for _ in range(m)]
    if name == "gradual escalation":
        return sum((seg(n // 20, 0.9 - (0.9 - 0.02) * i / 19, "floods") for i in range(20)), [])
    if name == "alternating 1,000-flow bursts":
        return sum((seg(1000, 0.9 if i % 2 == 0 else 0.02, "floods") for i in range(n // 1000)), [])
    if name == "alternating 100-flow bursts":
        return sum((seg(100, 0.9 if i % 2 == 0 else 0.02, "floods") for i in range(n // 100)), [])
    if name == "distributed low-rate attacks":
        return seg(n, 0.75, "low-volume")
    if name == "Brain-blind attack mix":
        return seg(n // 2, 0.9, "floods") + seg(n // 2, 0.3, "web+bruteforce")
    if name == "changed benign traffic":
        return seg(n, 0.9, "floods")  # benign drawn from flows the Gatekeeper keeps more often (see below)
    raise ValueError(name)


def run_router(skip, bb, c1, c2):
    direct, since = False, 0
    e_byp = e_ben = None
    cost, switches, est_err = 0.0, 0, []
    ew = lambda s, x: x if s is None else ALPHA * x + (1 - ALPHA) * s
    for i in range(len(skip)):
        if direct:
            cost += c2
            inspected, pred_benign = True, bb[i]
        else:
            cost += c1 + (0 if skip[i] else c2)
            inspected, pred_benign = not skip[i], bb[i]
            e_byp = ew(e_byp, 0.0 if inspected else 1.0)
        e_ben = ew(e_ben, 1.0 if (not inspected or pred_benign) else 0.0)
        since += 1
        if since < DWELL or e_byp is None:
            continue
        byp = e_ben * B0 if direct else e_byp
        if direct:
            lo = max(0, i - 99)
            est_err.append(byp - skip[lo:i + 1].mean())  # vs the bypass the Gatekeeper would really give
        saved = byp * c2
        if (not direct and saved < c1 * (1 - H)) or (direct and saved > c1 * (1 + H)):
            direct, since, switches = not direct, 0, switches + 1
    return cost, switches, (float(np.mean(np.abs(est_err))) if est_err else 0.0)


def patterns(skip, bb, fam, rng):
    idx = {"benign": np.flatnonzero(fam == "Benign")}
    for pool, fams in POOLS.items():
        idx[pool] = np.flatnonzero(np.isin(fam, fams))
    # benign flows that the Gatekeeper bypasses less often (bypass ~0.4 instead of 0.77)
    ben = idx["benign"]
    kept, passed = ben[~skip[ben]], ben[skip[ben]]
    changed = np.concatenate([kept, rng.choice(passed, size=int(len(kept) * 0.4 / 0.6), replace=False)])
    rows = []
    for name in ["gradual escalation", "alternating 1,000-flow bursts", "alternating 100-flow bursts",
                 "distributed low-rate attacks", "Brain-blind attack mix", "changed benign traffic"]:
        kinds = make_pattern(name, idx, rng)
        pick = np.array([rng.choice(changed if (k == "benign" and name == "changed benign traffic") else idx[k])
                         for k in kinds])
        s, b = skip[pick], bb[pick]
        for cname, (c1, c2) in COSTS.items():
            two = c1 + np.where(s, 0, c2)
            single = np.full(len(s), c2)
            blocks = np.arange(len(s)) // 500
            oracle = sum(min(two[blocks == k].sum(), single[blocks == k].sum()) for k in np.unique(blocks))
            cost, sw, err = run_router(s, b, c1, c2)
            rows.append({"pattern": name, "costs": cname, "benign_bypass_actual": float(skip[pick][[k == "benign" for k in kinds]].mean()),
                         "router_vs_oracle": cost / oracle - 1, "two_stage_vs_oracle": two.sum() / oracle - 1,
                         "single_stage_vs_oracle": single.sum() / oracle - 1, "switches": sw,
                         "direct_mode_bypass_estimate_mae": err})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    # "onnx": per-record costs of the fast inference path, measured by src/onnx_cost.py
    global COSTS
    suffix = ""
    if len(sys.argv) > 1 and sys.argv[1] == "onnx":
        st = pd.read_csv("results/onnx_cost/stages.csv").iloc[0]
        COSTS = {"ONNX 1 thread": (float(st.T1_us), float(st.T2_us))}
        suffix = "_onnx"
    os.makedirs(OUT, exist_ok=True)
    skip, bb, fam = load()
    b = bias(skip, bb, fam)
    b.to_csv(f"{OUT}/bias{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(b.round(3).to_string(index=False))
    p = patterns(skip, bb, fam, np.random.default_rng(5))
    p.to_csv(f"{OUT}/patterns{suffix}.csv", index=False)
    print(p.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
