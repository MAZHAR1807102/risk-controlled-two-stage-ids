"""Paper figures, generated only from saved results (no hardcoded numbers).

  fig1_tradeoff      Gatekeeper safety/efficiency curve: worst attack-group miss vs
                     benign bypass, original vs retrained, with deployed operating points
  fig2_breakeven     CPU per packet vs benign share (ns-3), all-cores vs hybrid threading
  fig3_latency       per-packet latency ECDF, single- vs two-stage (75% benign)
  fig4_confusion     two-stage confusion matrix (row-normalised, test split)
  fig5_timeline      normal -> flood -> normal: static vs adaptive routing
  fig6_overload      old per-packet gateway queue (kept for history)
  fig6_flowmeter     flow-record gateway: time to block, leaked flood packets, robot delay

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
    # (a) CSV features: all three Gatekeepers come from the same weighting study (same data, scaler, capacity).
    D = "results/gatekeeper_weighting_final"
    groups = np.load(f"{D}/test_groups.npy", allow_pickle=True)
    y = np.array([CLASSES.index(g) for g in groups])
    tau_new = float(pd.read_json("runs/seed42_sqrt/metrics.json", typ="series")["stage1"]["tau"])
    taus = np.unique(np.concatenate([np.geomspace(1e-4, 0.999, 400), [0.3, tau_new]]))
    series = [("unweighted", ORIG, "--", "Unweighted"),
              ("binary_balanced", THIRD, "-.", "Benign/attack balanced"),
              ("group_balanced", NEW, "-", "Family-balanced (proposed)")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.7))
    scores = {}
    for name, color, ls, label in series:
        scores[name] = np.load(f"{D}/test_scores_{name}.npy")
        xs, ys = tradeoff_curve(scores[name], y, taus)
        ax.plot(xs, ys, color=color, ls=ls, label=label)
    for name, tau, color, text, dy in (("unweighted", 0.3, ORIG, "\u03c4=0.3", 8),
                                       ("group_balanced", tau_new, NEW, f"\u03c4*={tau_new:.2f}", -8)):
        xs, ys = tradeoff_curve(scores[name], y, [tau])
        ax.plot(xs, ys, "o", ms=6, color=color, mec="white", mew=1.2, zorder=5)
        label_end(ax, xs[0], ys[0], f"{text}: {xs[0]:.0f}% bypass, {ys[0]:.1f}% miss", color, dy)
    cert = pd.read_csv("results/risk_control/summary.csv")
    tau_hat = float(cert[(cert.dataset == "ciciot2023") & (cert.calibration == "validation")
                         & (cert.rule == "ucb_typ")].tau.iloc[0])
    xs, ys = tradeoff_curve(scores["group_balanced"], y, [tau_hat])
    ax.plot(xs, ys, "D", ms=6, color=NEW, mec="white", mew=1.2, zorder=5)
    ax.annotate(f"certified \u03c4\u0302={tau_hat:.3f}:\n{xs[0]:.0f}% bypass, {ys[0]:.2f}% miss", (xs[0], ys[0]),
                xytext=(-8, 30), textcoords="offset points", ha="right", fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", color=NEW, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=NEW, lw=0.8))
    ax.axhline(2, color=INK2, lw=0.8, ls=":")
    ax.text(1, 2.3, "2% target", color=INK2, fontsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Benign traffic bypassing the Brain (%)")
    ax.set_ylabel("Worst attack-family miss rate (%)")
    ax.legend(loc="upper left")
    ax.set_title("(a) CICIoT2023, CIC CSV features", loc="left")

    # (b) packet records (pMN) on separate CICIoMT2024 test captures.
    t = pd.read_csv("results/packet/tradeoff_pMN.csv")
    bx.plot(100 * t.benign_bypass, 100 * t.worst_family_skip, color=NEW, label="Skipped by the Gatekeeper")
    bx.plot(100 * t.benign_bypass, 100 * t.worst_family_added, color=THIRD, ls="-.",
            label="Added to the Brain's misses")
    c = pd.read_csv("results/packet/cert_pMN.csv").set_index("rule")
    for rule, col, color, mk, text, off in (
            ("skip", "max_family_skip", NEW, "D", "CP-selected skip:\n", (-10, 26)),
            ("point", "max_family_skip", ORIG, "o", "observed threshold", (-8, 0)),
            ("added", "worst_family_added", THIRD, "D", "whole-system", (-14, 32))):
        r = c.loc[rule]
        x0, y0 = 100 * r.benign_bypass, 100 * r[col]
        bx.plot(x0, y0, mk, ms=6, color=color, mec="white", mew=1.2, zorder=5)
        bx.annotate(f"{text}{"" if text.endswith(chr(10)) else ": "}{x0:.0f}% bypass, {y0:.1f}%", (x0, y0), xytext=off, textcoords="offset points",
                    ha="right", va="center", fontsize=7, color=INK,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8))
    bx.axhline(2, color=INK2, lw=0.8, ls=":")
    bx.text(99.5, 1.0, "2% target", color=INK2, fontsize=6.5, ha="right")
    bx.set_xlim(70, 100)
    bx.set_ylim(0, 20)
    bx.set_xlabel("Benign traffic bypassing the Brain (%)")
    bx.legend(loc="upper left")
    bx.set_title("(b) CICIoMT2024, packet-derived flow records, separate captures", loc="left")
    fig.tight_layout(w_pad=1.5)
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
    panels = [("(a) CSV, 1 thread",
               [("single_t1", ORIG, "--", "Single-stage"),
                ("two_t1", NEW, "-", "Two-stage")]),
              ("(b) CSV, all cores",
               [("single_tall", ORIG, "--", "Single-stage"),
                ("two_tall", NEW, "-", "Two-stage"),
                ("adaptive_tall", THIRD, "-.", "Adaptive")]),
              ("(c) CSV, hybrid",
               [("single_tall", ORIG, "--", "Single-stage"),
                ("two_hybrid", NEW, "-", "Two-stage"),
                ("adaptive_hybrid", THIRD, "-.", "Adaptive")])]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
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
        ax.legend(loc="lower left", fontsize=6.5)
    axes[0].set_ylabel("Mean inference time per record (\u00b5s)")
    # (d) packet records (pMN), standalone C++ gateway: saving over the Brain on every record.
    ax = axes[3]
    t = pd.read_csv("results/packet/timing_mac_nosplit.csv")
    t = t[t.tag == "pMN"]
    for rule, color, rname in (("added", THIRD, "Whole"), ("skip", NEW, "CP skip")):
        for thr, ls, tname in (("1", "-", "1 thr."), ("all", "--", "all")):
            u = t[(t.rule == rule) & (t.threads == thr)].sort_values("benign_share")
            ax.plot(100 * u.benign_share, 100 * u.saving, color=color, ls=ls, marker="o", ms=3.5,
                    mec="white", mew=0.8, label=f"{rname}, {tname}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("(d) Packet-derived records", loc="left")
    ax.set_xlabel("Benign share of traffic (%)")
    ax.set_ylabel("Saving over single stage (%)", labelpad=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(-80, 92)
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)
    fig.tight_layout(w_pad=0.8)
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
    ax.set_xlabel("Record index (one request = one flow record)")
    ax.set_ylabel("Inference time per record (µs)")
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


# ---- Figure 6 (flow-record gateway) ---------------------------------------------------------
def fig6_flowmeter():
    d = pd.read_csv("ns3/flowmeter/summary.csv")
    d = d[(d.variant == "main") & (d.attack_rate > 0)]
    series = [("two-stage", 50, NEW, "-", "s", "Two-stage, 50 bots"),
              ("single-stage", 50, ORIG, "--", "o", "Single-stage, 50 bots"),
              ("two-stage", 1000, THIRD, "-.", "D", "Two-stage, 1,000 bots"),
              ("single-stage", 1000, "#e87ba4", ":", "v", "Single-stage, 1,000 bots")]
    panels = [("d_total_ms_p50", 1e-3, "(a) Time to block, median (s)"),
              ("attack_leak_share", 100, "(b) Flood packets let through (%)"),
              ("robot_max_ms", 1e3, "(c) Largest robot delay ($\\mu$s)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    for ax, (col, mult, title) in zip(axes, panels):
        for mode, bots, color, ls, mk, name in series:
            t = d[(d["mode"] == mode) & (d.n_bots == bots)].sort_values("attack_rate")
            y = t[col] * mult
            if col == "d_total_ms_p50":
                y = y.where(t.bots_blocked_share > 0)
            ax.plot(t.attack_rate, y, color=color, ls=ls, marker=mk, ms=3.5, mec="white", mew=0.6, label=name)
        ax.set_xscale("log")
        ax.set_xticks([1000, 4000, 16000, 64000, 128000], ["1k", "4k", "16k", "64k", "128k"])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Flood rate (packets/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig6_flowmeter")


# ---- Threshold rules: how often each breaks the family target -------------------------------
def fig7_rules():
    d = pd.read_csv("results/method_compare/coverage_v2.csv").set_index("method")
    rows = [("crc_marginal", "Conformal (CRC), all attacks"),
            ("sgr_marginal", "Selective class., all attacks"),
            ("point", "Observed threshold"),
            ("crc_family", "Mondrian CRC, per family"),
            ("ltt_hoeffding", "LTT, Hoeffding"),
            ("cp_family", "LTT exact, Bonferroni (earlier)"),
            ("ltt_binom", "LTT exact, no split (ours)")]
    io = pd.read_csv("results/risk_control/coverage.csv")
    io = io[io.dataset == "ciciomt2024"].set_index("rule")
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.5, 3.15), gridspec_kw={"height_ratios": [7, 2.3]})
    for i, (m, name) in enumerate(rows[::-1]):
        r = d.loc[m]
        color = NEW if m == "ltt_binom" else "#9b9a95"
        if r.feasible == 0:
            ax.text(1, i, "never feasible (~6,000 flows/family)", va="center", fontsize=6.5, color=INK2)
            continue
        v = 100 * r.family_violation
        ax.barh(i, max(v, 0.6), height=0.6, color=color)
        ax.text(max(v, 0.6) + 1.5, i, f"{v:.1f}%  (bypass {100 * r.bypass_mean:.0f}%)", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows)), [n for _, n in rows[::-1]])
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) CICIoT2023, 200 random splits (family target)", loc="left")
    for i, (rule, name) in enumerate((("ucb_typ", "CP-selected (ours)"), ("point", "Observed threshold"))):
        r = io.loc[rule]
        v = 100 * r.violation_any
        bx.barh(i, v, height=0.6, color=NEW if rule == "ucb_typ" else "#9b9a95")
        bx.text(v + 1.5, i, f"{v:.0f}%  (bypass {100 * r.bypass_mean:.0f}%)", va="center", fontsize=6.5)
    bx.set_yticks([0, 1], ["CP-selected (ours)", "Observed threshold"])
    bx.set_xlim(0, 135)
    bx.set_xticks([0, 25, 50, 75, 100])
    bx.grid(axis="y", visible=False)
    bx.set_title("(b) CICIoMT2024, 100 random 20% windows (any target)", loc="left")
    bx.set_xlabel("Runs that broke a miss target (%)")
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig7_rules")


# ---- Where the bypass comes from: benign bypass per data condition ---------------------------
def fig8_conditions():
    settings = [("c23", "CICIoT2023, CIC CSV"),
                ("c23x", "CICIoT2023, CSV without artifacts"),
                ("mt", "CICIoMT2024, CIC CSV"),
                ("mtx", "CICIoMT2024, CSV without artifacts"),
                ("pR", "Packet-derived records, random split"),
                ("pCN", "Packet-derived records, separate captures"),
                ("pMN", "Same, labels from packet headers")]
    rules = [("point", ORIG, "Observed threshold", "max_family_skip"),
             ("skip", NEW, "CP-selected skip", "max_family_skip"),
             ("added", THIRD, "Whole-system", "worst_family_added")]
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    h = 0.26
    n = len(settings)
    for j, (rule, color, name, col) in enumerate(rules):
        for i, (tag, _) in enumerate(settings):
            r = pd.read_csv(f"results/packet/cert_{tag}.csv").set_index("rule").loc[rule]
            yy = (n - 1 - i) + (1 - j) * h
            broke = r[col] > 0.02
            ax.barh(yy, 100 * r.benign_bypass, height=h * 0.92, color="white" if broke else color,
                    edgecolor=color, hatch="////" if broke else None, lw=0.8,
                    label=name if i == 0 else None)
            if broke:
                ax.text(100 * r.benign_bypass + 1.5, yy, "\u2717", va="center", fontsize=7, color=INK)
    ax.barh(-5, 1, color="white", edgecolor=INK2, hatch="////", lw=0.8, label="Broke 2% family target (\u2717)")
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n), [s for _, s in settings][::-1])
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Benign traffic that skips the Brain (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2, fontsize=6.5, handlelength=1.2)
    fig.tight_layout()
    save(fig, "fig8_conditions")


# ---- Robustness per attack: own test, padding attacker, other dataset ------------------------
def fig9_robustness():
    pad = pd.read_csv("results/packet/adaptive_attack.csv").set_index("capture")
    fam = pd.read_csv("results/packet/families_pMN.csv")
    fam = fam[fam.rule == "added"].set_index("family")
    kit = {k: pd.read_csv(f"results/packet/kitsune_{k}_pkt_MN_sel_added.csv").iloc[0].two_stage_calls_benign
           for k in ("OS_Scan", "SSDP_Flood", "ARP_MitM")}
    rows = [("OS scan", pad.loc["Recon-OS_Scan"].miss_no_attack, pad.loc["Recon-OS_Scan"].miss_adaptive_pad,
             kit["OS_Scan"]),
            ("Port scan", pad.loc["Recon-Port_Scan"].miss_no_attack, pad.loc["Recon-Port_Scan"].miss_adaptive_pad,
             None),
            ("Vulnerability scan", pad.loc["Recon-VulScan"].miss_no_attack,
             pad.loc["Recon-VulScan"].miss_adaptive_pad, None),
            ("Ping sweep", pad.loc["Recon-Ping_Sweep"].miss_no_attack, pad.loc["Recon-Ping_Sweep"].miss_adaptive_pad,
             None),
            ("MQTT malformed data", pad.loc["MQTT-Malformed_Data"].miss_no_attack,
             pad.loc["MQTT-Malformed_Data"].miss_adaptive_pad, None),
            ("UDP flood", fam.loc["DDoS"].two_stage_miss, None, kit["SSDP_Flood"]),
            ("ARP spoofing", pad.loc["ARP_Spoofing"].miss_no_attack, pad.loc["ARP_Spoofing"].miss_adaptive_pad,
             kit["ARP_MitM"])]
    marks = [(NEW, "o", "Own test captures"), (ORIG, "s", "Padding attacker"),
             (THIRD, "D", "Other dataset (Kitsune)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    n = len(rows)
    for i, (name, *vals) in enumerate(rows):
        yy = n - 1 - i
        got = [100 * v for v in vals if v is not None]
        ax.plot([min(got), max(got)], [yy, yy], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        for k, (v, (color, mk, label)) in enumerate(zip(vals, marks)):
            if v is None:
                continue
            ax.plot(100 * v, yy + (k - 1) * 0.22, mk, ms=5.5, color=color, mec="white", mew=0.8, zorder=3,
                    label=label if name in ("OS scan", "UDP flood") and (k != 1 or name == "OS scan") else None)
    ax.set_yticks(range(n), [r[0] for r in rows][::-1])
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Attack records the two-stage IDS calls benign (%)")
    h, l = ax.get_legend_handles_labels()
    keep = dict(zip(l, h))
    ax.legend(keep.values(), keep.keys(), loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=3, fontsize=6.5,
              handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "fig9_robustness")


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
    if os.path.exists("ns3/flowmeter/summary.csv"):
        fig6_flowmeter()
    fig7_rules()
    fig8_conditions()
    fig9_robustness()


if __name__ == "__main__":
    main()
