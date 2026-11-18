"""Gateway overload experiment in ns-3 (IoT_IDS_Overload): IDS as a queue under flooding.

For each flood rate and gateway setup, runs the ns-3 program for three seeds and collects
robot loss/delay, benign-device loss/delay, blocking effect and IDS busy time.
Run on an otherwise idle machine (service times are measured wall-clock inference times).

Usage: .venv/bin/python src/run_overload.py            # full matrix
       .venv/bin/python src/run_overload.py --quick    # one seed, fewer rates
"""
import argparse
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS3 = os.path.expanduser("~/NS3/ns-allinone-3.42/ns-3.42")
RUN = f"{HERE}/runs/seed42_sqrt"
OUT = f"{HERE}/ns3/overload"

RATES = [0, 250, 500, 1000, 2000, 4000, 8000]
SETUPS = {
    "single":               "--mode=single-stage",
    "two":                  "--mode=two-stage",
    "single_prio":          "--mode=single-stage --priority=1",
    "two_prio":             "--mode=two-stage --priority=1",
    "two_prio_block50":     "--mode=two-stage --priority=1 --block=1 --nBots=50",
    "two_prio_block1000":   "--mode=two-stage --priority=1 --block=1 --nBots=1000",
}
# Queuing-design study: fair queuing for devices, spoofed robot priority, and its defences.
SETUPS_QUEUING = {
    "two_prio_drr":         "--mode=two-stage --priority=1 --drr=1",
    "two_prio_spoof":       "--mode=two-stage --priority=1 --drr=1 --spoofRobot=0.5",
    "two_prio_spoof_cap":   "--mode=two-stage --priority=1 --drr=1 --spoofRobot=0.5 --prioCap=200",
    "two_prio_spoof_auth":  "--mode=two-stage --priority=1 --drr=1 --spoofRobot=0.5 --authPriority=1",
}
TRACE_RATE = 1000
TRACE_SETUPS = ["single", "two", "two_prio", "two_prio_block50"]


def run(setup, rate, seed, summary, trace=None):
    flags = SETUPS.get(setup) or SETUPS_QUEUING[setup]
    args = (f"IoT_IDS_Overload --params={RUN}/ids_params.h --l1={RUN}/layer1_gatekeeper.ubj "
            f"--l2={RUN}/layer2_multiclass.ubj --benign={HERE}/ns3/traffic/test_benign90_20000.csv "
            f"--flood={HERE}/ns3/traffic/test_natural_20000.csv --attackRate={rate} --seed={seed} "
            f"--summary={summary} {flags}")
    if trace:
        args += f" --trace={trace}"
    r = subprocess.run(["./ns3", "run", "--no-build", args], cwd=NS3, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ns-3 failed ({setup}, {rate}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    print(f"{setup:20s} rate={rate:5d} seed={seed}  {r.stdout.strip().splitlines()[-1]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--queuing", action="store_true", help="run the queuing-design study instead")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    setups = SETUPS_QUEUING if a.queuing else SETUPS
    tag = "_queuing" if a.queuing else ""
    seeds = [1] if a.quick else [1, 2, 3]
    rates = [0, 1000, 4000] if a.quick else RATES
    summary = f"{OUT}/summary_raw{tag}.csv"
    if os.path.exists(summary):
        os.remove(summary)
    rows = []
    for seed in seeds:
        for rate in rates:
            for setup in setups:
                trace = (f"{OUT}/trace_{setup}" if seed == 1 and rate == TRACE_RATE and setup in TRACE_SETUPS
                         else None)
                run(setup, rate, seed, summary, trace)
                rows.append({"setup": setup, "seed": seed})
    raw = pd.read_csv(summary)
    raw.insert(0, "setup", [r["setup"] for r in rows])
    raw.insert(1, "seed", [r["seed"] for r in rows])
    for c in ("robot", "device", "attacker"):
        raw[f"{c}_loss"] = (raw[f"{c}_drop_queue"] + raw[f"{c}_drop_blocked"]) / raw[f"{c}_arrived"].clip(lower=1)
    raw["attack_blocked_frac"] = raw["attacker_drop_blocked"] / raw["attacker_arrived"].clip(lower=1)
    raw.to_csv(f"{OUT}/summary{tag}.csv", index=False)
    cols = ["robot_loss", "robot_p99_ms", "device_loss", "device_p99_ms", "attack_blocked_frac",
            "devices_blocked", "busy_frac"]
    agg = raw.groupby(["setup", "attack_rate"])[cols].mean().reset_index()
    agg.to_csv(f"{OUT}/summary_mean{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
