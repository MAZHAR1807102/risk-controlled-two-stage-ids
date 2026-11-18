"""Sensitivity of the load-adaptive router (reviewer point 2.6), ns-3 v2, all cores.

Scenarios (20,000 CICIoT2023 test flows each, built like make_ns3_traffic.build):
  step     6k at 90% benign -> 8k flood at 2% benign -> 6k at 90%
  ramp     benign share falls linearly from 90% to 2% and back, in 1,000-flow steps
  bursty   alternating 1,500-flow bursts at 90% and 2% benign
  lowvol   like step, but the flood contains only low-volume attacks (Recon, Spoofing,
           BruteForce, Web)
For every scenario, the static two-stage and single-stage IDS are run once, and the adaptive
router is run for every (alpha, hysteresis, dwell) combination.

Metrics per adaptive run: total CPU, regret against an oracle that picks the cheaper static
design in every 500-flow window, number of switches, flows needed to react to each regime
change, missed attacks, and the error of the router's benign-share estimate.

Usage: .venv/bin/python src/adaptive_sensitivity.py
"""
import itertools
import os
import subprocess

import numpy as np
import pandas as pd

from eval_offline import CLASSES
from make_ns3_traffic import TEST, build

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS3 = os.path.expanduser("~/NS3/ns-allinone-3.42/ns-3.42")
RUN = f"{HERE}/runs/seed42_sqrt"
OUT = f"{HERE}/ns3/adaptive_sensitivity"
LOWVOL = {"Recon", "Spoofing", "BruteForce", "Web"}
ALPHAS = [0.005, 0.01, 0.05]
HYST = [0.1, 0.2, 0.4]
DWELL = [50, 200, 1000]
WINDOW = 500


def make_traffic(df, rng):
    def seg(n, share, pool=None):
        src = df if pool is None else df[(df.group == "Benign") | df.group.isin(pool)]
        return build(src, n, share, rng)

    scen = {
        "step": [seg(6000, 0.90), seg(8000, 0.02), seg(6000, 0.90)],
        "ramp": [seg(1000, s) for s in list(np.linspace(0.90, 0.02, 10)) + list(np.linspace(0.02, 0.90, 10))],
        "bursty": [seg(1500, s) for s in [0.90, 0.02] * 6] + [seg(2000, 0.90)],
        "lowvol": [seg(6000, 0.90), seg(8000, 0.02, LOWVOL), seg(6000, 0.90)],
    }
    changes = {"step": [6000, 14000], "lowvol": [6000, 14000], "bursty": list(range(1500, 18000, 1500)), "ramp": []}
    paths = {}
    for name, parts in scen.items():
        seq = pd.concat(parts, ignore_index=True)
        out = seq[list(df.columns[:46])].copy()
        out["class_id"] = seq["group"].map(CLASSES.index).astype(int)
        out["raw_label"] = seq["label"].values
        paths[name] = f"{OUT}/traffic_{name}.csv"
        out.to_csv(paths[name], index=False)
    return paths, changes


def sim(traffic, mode, out, extra=""):
    args = (f"IoT_TwoStage_IDS_v2 --data={traffic} --params={RUN}/ids_params.h "
            f"--l1={RUN}/layer1_gatekeeper.ubj --l2={RUN}/layer2_multiclass.ubj --mode={mode} "
            f"--nthread1=0 --nthread2=0 --api=inplace --rows=20000 --warmup=0 --out={out} {extra}")
    r = subprocess.run(["./ns3", "run", "--no-build", args], cwd=NS3, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ns-3 failed: {r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    d = pd.read_csv(out)
    d["cost_us"] = (d.scale_ns + d.stage1_ns + d.stage2_ns) / 1e3
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(TEST)
    from eval_offline import macro_group
    df["group"] = df["label"].map({l: macro_group(l) for l in df["label"].unique()})
    paths, changes = make_traffic(df, np.random.default_rng(11))
    rows = []
    for scen, traffic in paths.items():
        two = sim(traffic, "two-stage", f"{OUT}/{scen}__static_two.csv")
        one = sim(traffic, "single-stage", f"{OUT}/{scen}__static_single.csv")
        win = np.arange(len(two)) // WINDOW
        oracle = np.minimum(two.groupby(win).cost_us.sum(), one.groupby(win).cost_us.sum()).sum()
        true_benign = (two.label == 0).astype(float).rolling(100, min_periods=1).mean()
        for name, d in (("static_two", two), ("static_single", one)):
            rows.append({"scenario": scen, "config": name, "cpu_ms": d.cost_us.sum() / 1e3,
                         "regret": d.cost_us.sum() / oracle - 1,
                         "missed": int(((d.label != 0) & (d.inspected == 0)).sum())})
        for a, h, dw in itertools.product(ALPHAS, HYST, DWELL):
            cfg = f"a{a}_h{h}_d{dw}"
            d = sim(traffic, "adaptive", f"{OUT}/{scen}__{cfg}.csv",
                    f"--alpha={a} --hysteresis={h} --dwell={dw}")
            direct = d.direct.to_numpy()
            sw = np.flatnonzero(np.diff(direct) != 0) + 1
            react = []
            for c in changes[scen]:
                after = sw[sw >= c]
                react.append(int(after[0] - c) if len(after) else np.nan)
            rows.append({"scenario": scen, "config": cfg, "alpha": a, "hysteresis": h, "dwell": dw,
                         "cpu_ms": d.cost_us.sum() / 1e3, "regret": d.cost_us.sum() / oracle - 1,
                         "switches": len(sw), "react_mean_flows": float(np.nanmean(react)) if react else np.nan,
                         "missed": int(((d.label != 0) & (d.inspected == 0)).sum()),
                         "est_benign_mae": float((d.est_benign - true_benign).abs().mean())})
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.groupby("scenario").apply(lambda g: g.sort_values("regret").head(4)).to_string())


if __name__ == "__main__":
    main()
