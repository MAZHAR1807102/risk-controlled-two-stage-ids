"""Noise-free regret for the adaptive-router sensitivity study.

Wall-clock CPU of separate ns-3 runs on all cores drifts by up to ~35% between runs
(even in phases where routing is identical), so total-CPU regret is not comparable
across runs. Here each adaptive run is re-costed counterfactually: every flow costs what
the *static* run measured for that same flow (single-stage cost if the router sent it
direct, two-stage cost otherwise). The oracle picks the cheaper design per 500-flow window
from the same per-flow costs.

Usage: .venv/bin/python src/adaptive_reanalyze.py   -> ns3/adaptive_sensitivity/summary_costed.csv
"""
import numpy as np
import pandas as pd

OUT = "ns3/adaptive_sensitivity"
WINDOW = 500


def cost(d):
    return ((d.scale_ns + d.stage1_ns + d.stage2_ns) / 1e3).to_numpy()


def main():
    s = pd.read_csv(f"{OUT}/summary.csv")
    rows = []
    for scen in s.scenario.unique():
        c2 = cost(pd.read_csv(f"{OUT}/{scen}__static_two.csv"))
        c1 = cost(pd.read_csv(f"{OUT}/{scen}__static_single.csv"))
        win = np.arange(len(c2)) // WINDOW
        oracle = np.minimum(np.bincount(win, c2), np.bincount(win, c1)).sum()
        for _, r in s[s.scenario == scen].iterrows():
            if r.config == "static_two":
                tot = c2.sum()
            elif r.config == "static_single":
                tot = c1.sum()
            else:
                direct = pd.read_csv(f"{OUT}/{scen}__{r.config}.csv").direct.to_numpy().astype(bool)
                tot = np.where(direct, c1, c2).sum()
            rr = r.to_dict()
            rr.update({"cpu_costed_ms": tot / 1e3, "regret_costed": tot / oracle - 1})
            rows.append(rr)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/summary_costed.csv", index=False)
    pd.set_option("display.width", 220)
    print(res[res.config.str.startswith("static")][["scenario", "config", "regret_costed", "missed"]].to_string(index=False))
    a = res[~res.config.str.startswith("static")]
    print(a.groupby("scenario")[["regret_costed", "switches", "react_mean_flows", "missed"]]
          .agg(["min", "median", "max"]).round(3).to_string())
    print(a[a.config == "a0.01_h0.2_d200"][["scenario", "regret_costed", "switches", "react_mean_flows", "missed"]]
          .to_string(index=False))
    print(a.groupby(["alpha", "hysteresis", "dwell"]).regret_costed.mean().sort_values().head(8).round(4))


if __name__ == "__main__":
    main()
