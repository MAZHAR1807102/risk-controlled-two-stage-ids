"""Paper figures, generated only from saved results (no hardcoded numbers).

  fig1_tradeoff      Gatekeeper safety/efficiency curve: worst attack-group miss vs
                     benign bypass, unweighted vs family-balanced, with deployed operating points
  fig2_breakeven     CPU per packet vs benign share (ns-3), all-cores vs hybrid threading
  fig3_latency       per-packet latency ECDF, single- vs two-stage (75% benign)
  fig4_confusion     two-stage confusion matrix (row-normalised, test split)
  fig5_timeline      normal -> flood -> normal: static vs adaptive routing

Usage: .venv/bin/python src/make_figures.py   (writes figures/*.png and *.pdf)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from eval_offline import CLASSES

OUT = "figures"
BENIGN = CLASSES.index("Benign")

# Validated categorical slots (light/print): 1 blue, 2 orange, 3 aqua.
NEW, ORIG, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.png/.pdf")


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points",
                color=INK, fontsize=7, va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8))


# ---- Figure 1 ---------------------------------------------------------------------------
def tradeoff_curve(p, y, taus):
    att_groups = [i for i in range(len(CLASSES)) if i != BENIGN]
    ben = y == BENIGN
    xs, ys = [], []
    for t in taus:
        flag = p >= t
        xs.append(100 * (~flag[ben]).mean())
        ys.append(100 * max((~flag[y == g]).mean() for g in att_groups))
    return np.array(xs), np.array(ys)


def fig1():
    # All three Gatekeepers come from the same weighting study (same data, scaler, capacity).
    D = "results/gatekeeper_weighting_final"
    groups = np.load(f"{D}/test_groups.npy", allow_pickle=True)
    y = np.array([CLASSES.index(g) for g in groups])
    tau_new = float(pd.read_json("runs/seed42_sqrt/metrics.json", typ="series")["stage1"]["tau"])
    taus = np.unique(np.concatenate([np.geomspace(1e-4, 0.999, 400), [0.3, tau_new]]))
    series = [("unweighted", ORIG, "--", "Unweighted"),
              ("binary_balanced", THIRD, "-.", "Benign/attack balanced"),
              ("group_balanced", NEW, "-", "Family-balanced (proposed)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    scores = {}
    for name, color, ls, label in series:
        scores[name] = np.load(f"{D}/test_scores_{name}.npy")
        xs, ys = tradeoff_curve(scores[name], y, taus)
        ax.plot(xs, ys, color=color, ls=ls, label=label)
    for name, tau, color, text, dy in (("unweighted", 0.3, ORIG, "\u03c4=0.3", 8),
                                       ("group_balanced", tau_new, NEW, f"\u03c4*={tau_new:.2f}", -8)):
        xs, ys = tradeoff_curve(scores[name], y, [tau])
        ax.plot(xs, ys, "o", ms=6, color=color, mec="white", mew=1.2, zorder=5)
        label_end(ax, xs[0], ys[0], f"{text}: {xs[0]:.1f}% bypass, {ys[0]:.1f}% miss", color, dy)
    ax.axhline(2, color=INK2, lw=0.8, ls=":")
    ax.text(1, 2.3, "2% target", color=INK2, fontsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Benign traffic bypassing Stage 2 (%)")
    ax.set_ylabel("Worst attack-family miss rate (%)")
    ax.legend(loc="upper left")
    ax.set_title("Gatekeeper trade-off on the CICIoT2023 test split", loc="left")
    save(fig, "fig1_tradeoff")


# ---- Figure 2 ---------------------------------------------------------------------------
def fig2():
    reps = "ns3/experiments/breakeven_reps_summary.csv"
    if os.path.exists(reps):
        d = pd.read_csv(reps)
    else:  # single repetition: no band
        d = pd.read_csv("ns3/experiments/breakeven_summary.csv").rename(columns={"mean_us": "mean"})
        d["min"] = d["max"] = d["mean"]
    d["share"] = 100 * d.benign_share
    panels = [("(a) Both stages on 1 thread",
               [("single_t1", ORIG, "--", "Single-stage"),
                ("two_t1", NEW, "-", "Two-stage")]),
              ("(b) Both stages on all cores",
               [("single_tall", ORIG, "--", "Single-stage"),
                ("two_tall", NEW, "-", "Two-stage"),
                ("adaptive_tall", THIRD, "-.", "Adaptive")]),
              ("(c) Stage 1: 1 thread, Stage 2: all cores",
               [("single_tall", ORIG, "--", "Single-stage"),
                ("two_hybrid", NEW, "-", "Two-stage"),
                ("adaptive_hybrid", THIRD, "-.", "Adaptive")])]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    for ax, (title, series) in zip(axes, panels):
        for cfg, color, ls, name in series:
            s = d[d.config == cfg].sort_values("share")
            ax.fill_between(s.share, s["min"], s["max"], color=color, alpha=0.15, lw=0)
            ax.plot(s.share, s["mean"], color=color, ls=ls, marker="o", ms=3.5,
                    mec="white", mew=0.8, label=name)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Benign share of traffic (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Mean inference time per packet (\u00b5s)")
    save(fig, "fig2_breakeven")


# ---- Figure 3 ---------------------------------------------------------------------------
def fig3():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for cfg, color, ls, name in (("single_tall", ORIG, "--", "Single-stage"),
                                 ("two_hybrid", NEW, "-", "Two-stage (hybrid threads)")):
        d = pd.read_csv(f"ns3/experiments/breakeven/b75__{cfg}.csv").iloc[200:]
        t = np.sort((d.scale_ns + d.stage1_ns + d.stage2_ns).to_numpy() / 1e3)
        ax.step(t, np.arange(1, len(t) + 1) / len(t), where="post", color=color, ls=ls, label=name)
        for q in (0.5, 0.99):
            v = np.quantile(t, q)
            ax.plot(v, q, "o", ms=4, color=color, mec="white", mew=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Per-packet IDS latency (µs, log scale)")
    ax.set_ylabel("Fraction of packets")
    ax.set_ylim(0, 1.01)
    ax.legend(loc="upper left")
    ax.set_title("Latency distribution, 75% benign traffic", loc="left")
    save(fig, "fig3_latency")


# ---- Figure 4 ---------------------------------------------------------------------------
def fig4():
    cm = pd.read_csv("runs/seed42_sqrt/two_stage_confusion.csv", index_col=0)
    norm = cm.div(cm.sum(1), axis=0)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.imshow(norm.values, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = norm.values[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), [f"{c} (n={cm.loc[c].sum():,})" for c in CLASSES])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Two-stage IDS, test split (row-normalised)", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig4_confusion")


# ---- Figure 5 ---------------------------------------------------------------------------
def fig5():
    phases = pd.read_csv("ns3/experiments/timeline/phases.csv")["phase"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    bounds = np.flatnonzero(np.diff(phases)) + 1
    ax.axvspan(bounds[0], bounds[1], color="#f0efec", lw=0)
    ax.text((bounds[0] + bounds[1]) / 2, 0.97, "DDoS flood (2% benign)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=INK2, fontsize=7)
    for cfg, color, ls, name in (("single_tall", ORIG, "--", "Single-stage"),
                                 ("two_tall", NEW, "-", "Two-stage (static)"),
                                 ("adaptive_tall", THIRD, "-.", "Adaptive")):
        d = pd.read_csv(f"ns3/experiments/timeline/{cfg}.csv")
        t = (d.scale_ns + d.stage1_ns + d.stage2_ns) / 1e3
        ax.plot(d.idx, t.rolling(300, min_periods=50).mean(), color=color, ls=ls, label=name)
    ax.set_xlim(0, len(phases))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Packet index")
    ax.set_ylabel("Inference time per packet (µs, rolling mean)")
    # The caption carries the title; the legend sits above the plot area.
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.01), borderaxespad=0)
    save(fig, "fig5_timeline")


# ---- Figure 6 ---------------------------------------------------------------------------
def fig6():
    d = pd.read_csv("ns3/overload/summary_mean.csv")
    series = [("single", ORIG, "--", "o", "Single-stage"),
              ("two", NEW, "-", "s", "Two-stage"),
              ("single_prio", "#eda100", ":", "^", "Single-stage + priority"),
              ("two_prio", THIRD, "-.", "D", "Two-stage + priority"),
              ("two_prio_block50", "#e87ba4", (0, (5, 1)), "v", "Two-stage + priority + blocking (50 bots)")]
    panels = [("robot_loss", 100, "(a) Robot loss (%)", False),
              ("robot_p99_ms", 1, "(b) Robot p99 delay (ms)", True),
              ("device_loss", 100, "(c) Other devices' loss (%)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    for ax, (col, mult, title, logy) in zip(axes, panels):
        for setup, color, ls, mk, name in series:
            t = d[d.setup == setup].sort_values("attack_rate")
            ax.plot(t.attack_rate.replace(0, 100), t[col] * mult, color=color, ls=ls, marker=mk, ms=3.5,
                    mec="white", mew=0.6, label=name)
        ax.set_xscale("log")
        ax.set_xticks([100, 250, 500, 1000, 2000, 4000, 8000],
                      ["0", "250", "500", "1k", "2k", "4k", "8k"])
        ax.minorticks_off()
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Flood rate (packets/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig6_overload")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig4()
    if os.path.exists("ns3/experiments/breakeven_summary.csv"):
        fig2()
        fig3()
    if os.path.exists("ns3/experiments/timeline_summary.csv"):
        fig5()
    if os.path.exists("ns3/overload/summary_mean.csv"):
        fig6()


if __name__ == "__main__":
    main()
