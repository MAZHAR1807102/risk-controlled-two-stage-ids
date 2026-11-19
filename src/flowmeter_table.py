"""LaTeX rows of Supplementary Table S24 (flow-record gateway) from ns3/flowmeter/summary*.csv.

Usage: .venv/bin/python src/flowmeter_table.py  -> ns3/flowmeter/table_s24.tex
"""
import pandas as pd

OUT = "ns3/flowmeter/table_s24.tex"


def fmt_s(ms):
    return "--" if pd.isna(ms) else f"{ms / 1e3:.2f}"


def row(r, first):
    blocked = "--" if pd.isna(r.bots_blocked_share) else f"{100 * r.bots_blocked_share:.0f}\\%"
    num = lambda x, f: format(x, f).replace(",", "{,}")
    return (f"{first} & {num(r.robot_max_ms * 1e3, ',.1f')} & {num(r.device_p99_ms * 1e3, ',.1f')} & {blocked} & "
            f"{fmt_s(r.d_total_ms_p50)} & {fmt_s(r.d_formation_ms_mean)} & {fmt_s(r.d_queue_ms_mean)} & "
            f"{100 * r.attack_leak_share:.0f}\\% & {num(r.rec_dropped_attack, ',.0f')} \\\\")


def main():
    d = pd.read_csv("ns3/flowmeter/summary.csv")
    k1 = pd.read_csv("ns3/flowmeter/summary_raw_k1.csv")
    k1["bots_blocked_share"] = k1.bots_blocked / k1.bots.where(k1.bots > 0)
    k1 = k1.groupby(["mode", "n_bots", "attack_rate"]).mean(numeric_only=True).reset_index()
    lines = [r"\multicolumn{9}{@{}l}{(a) Two-stage IDS (certified threshold), mean of three seeds} \\",
             r"Bots / rate & Robot & Dev.\ p99 & Blocked & $D$ med. & Form. & Queue & Through & Rec.\ lost \\ \midrule"]
    main_ = d[(d.variant == "main") & (d["mode"] == "two-stage") & (d.attack_rate > 0)]
    for bots in (50, 1000):
        for _, r in main_[main_.n_bots == bots].sort_values("attack_rate").iterrows():
            lines.append(row(r, f"{bots} / {int(r.attack_rate / 1000)}k"))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{@{}l}{(b) Changes to the setup (two-stage; 50 bots at 16k, 1{,}000 bots at 64k; one seed, main setup three)} \\ \midrule")
    names = {"main": "Main setup", "timeout1s": "Records also closed after 1 s", "fwd1us": "Network stack 1 $\\mu$s",
             "fwd20us": "Network stack 20 $\\mu$s", "slow4x": "All times $\\times 4$ (slower board)"}
    for v, name in names.items():
        for bots, rate in ((50, 16000), (1000, 64000)):
            t = d[(d.variant == v) & (d["mode"] == "two-stage") & (d.n_bots == bots) & (d.attack_rate == rate)]
            if len(t):
                lines.append(row(t.iloc[0], f"{name} / {bots}"))
    for bots, rate in ((50, 16000), (1000, 64000)):
        t = k1[(k1["mode"] == "two-stage") & (k1.n_bots == bots) & (k1.attack_rate == rate)]
        lines.append(row(t.iloc[0], f"Block on first verdict / {bots}"))
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
