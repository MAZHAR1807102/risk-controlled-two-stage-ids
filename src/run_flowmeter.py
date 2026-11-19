"""Flow-record gateway under flooding (ns-3 IoT_IDS_FlowMeter): packets forwarded at once, the IDS
checks flow records (10 packets, 100 for floods) and installs block rules.

Answers "the packet-level flood simulation is inconsistent with the flow-level IDS": inspection
requests are created only when the flow meter closes a record, robot packets never wait for an
inference, and the security delay of every attack flow is split into formation, queue, inference
and mitigation. Two-stage runs use the certified threshold (0.0166).
Run on an otherwise idle machine (service times are measured wall-clock times).

Main matrix: 3 seeds; sensitivity variants (timeout, forwarding cost, 4x slower board): 1 seed.
Usage: .venv/bin/python src/run_flowmeter.py [--quick]          # main matrix (new summary_raw.csv)
       .venv/bin/python src/run_flowmeter.py --variants --append # variants, appended
       .venv/bin/python src/run_flowmeter.py --aggregate         # only rebuild summary.csv
"""
import argparse
import itertools
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS3 = os.path.expanduser("~/NS3/ns-allinone-3.42/ns-3.42")
RUN = f"{HERE}/runs/seed42_sqrt"
OUT = f"{HERE}/ns3/flowmeter"
TAU_CERT = 0.0166
RATES = [0, 1000, 4000, 16000, 64000, 128000]
MODES = {"two": f"--mode=two-stage --tau={TAU_CERT}", "single": "--mode=single-stage"}
# name -> extra flags; every variant runs for both modes, both botnet sizes and all rates
VARIANTS = {
    "main": "",
    "timeout1s": "--activeTimeout=1",
    "fwd1us": "--fwdNs=1000 --dropNs=200",
    "fwd20us": "--fwdNs=20000 --dropNs=4000",
    "slow4x": "--timeScale=4",
}


def run(label, flags, seed, summary):
    args = (f"IoT_IDS_FlowMeter --params={RUN}/ids_params.h --l1={RUN}/layer1_gatekeeper.ubj "
            f"--l2={RUN}/layer2_multiclass.ubj --benign={HERE}/ns3/traffic/test_benign90_20000.csv "
            f"--flood={HERE}/ns3/traffic/test_natural_20000.csv --seed={seed} --summary={summary} "
            f"--label={label} {flags}")
    r = subprocess.run(["./ns3", "run", "--no-build", args], cwd=NS3, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ns-3 failed ({label}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    print(r.stdout.strip().splitlines()[-1], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variants", action="store_true", help="run the sensitivity variants (1 seed)")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = f"{OUT}/summary_raw.csv"
    if not a.aggregate:
        if os.path.exists(summary) and not a.append:
            os.remove(summary)
        if a.variants:
            variants, seeds = {k: v for k, v in VARIANTS.items() if k != "main"}, [1]
        else:
            variants, seeds = {"main": ""}, ([1] if a.quick else [1, 2, 3])
        for (v, vf), (m, mf), bots, rate, seed in itertools.product(variants.items(), MODES.items(), (50, 1000),
                                                                   RATES, seeds):
            run(f"{v}_{m}", f"{mf} {vf} --nBots={bots} --attackRate={rate}", seed, summary)
    d = pd.read_csv(summary)
    d["variant"] = d.label.str.rsplit("_", n=1).str[0]
    d["robot_loss"] = 1 - d.robot_forwarded / d.robot_arrived
    d["device_loss"] = 1 - d.device_forwarded / d.device_arrived
    d["bots_blocked_share"] = d.bots_blocked / d.bots.where(d.bots > 0)
    keys = ["variant", "mode", "n_bots", "attack_rate"]
    cols = ["robot_loss", "robot_max_ms", "device_loss", "device_p99_ms", "dp_busy", "ids_busy", "dp_ns_max",
            "rec_wait_ms_p99", "rec_dropped_attack", "devices_blocked", "robot_false_alarms", "bots_blocked_share",
            "attack_leak_share", "d_total_ms_p50", "d_total_ms_p95", "d_total_ms_max", "d_formation_ms_mean",
            "d_queue_ms_mean", "d_inference_ms_mean", "d_mitigation_ms_mean"]
    agg = d.groupby(keys)[cols].agg(lambda x: x.max() if x.name.endswith("max_ms") or x.name == "dp_ns_max"
                                    else x.mean()).reset_index()
    agg.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg[agg.variant == "main"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
