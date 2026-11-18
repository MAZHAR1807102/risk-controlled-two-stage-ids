"""Wall-clock time vs. process CPU time of the two-stage IDS (reviewer point 2.3).

The per-flow timers in ns-3 and in pi_bench/ids_bench.cc measure elapsed (wall-clock) time.
With a multi-threaded Stage 2, the threads together can use much more CPU time than the
elapsed time. Here the standalone C++ benchmark (same decisions as ns-3) also reads the
process CPU time (user + system, all threads, getrusage) around the whole inference loop.

For both datasets, three traffic mixes, three thread settings (1 thread, all cores, hybrid =
Stage 1 on one thread and Stage 2 on all cores), the efficiency and certified thresholds and
the single-stage IDS; 3 repeats each.

Usage: .venv/bin/python src/cpu_timing.py   -> results/cpu_timing/{raw,summary}.csv
"""
import os
import re
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(HERE, "pi_bench", "ids_bench_host")
OUT = os.path.join(HERE, "results", "cpu_timing")
ROWS = 10000
REPEATS = 3
RC = pd.read_csv(os.path.join(HERE, "results", "risk_control", "summary.csv"))


def certified(ds):
    r = RC[(RC.dataset == ds) & (RC.calibration == "validation") & (RC.rule == "ucb_typ")]
    return float(r.tau.iloc[0])


DATASETS = {
    "ciciot2023": ("runs/seed42_sqrt", "pi_bench/traffic/main/test_{mix}_20000.csv"),
    "ciciomt2024": ("runs/iomt_seed42", "results/iomt_timing/traffic_{mix}.csv"),
}
THREADS = {"1 thread": ("1", "1"), "all cores": ("0", "0"), "hybrid": ("1", "0")}


def run(run_dir, traffic, mode, t1, t2, tau=None):
    cmd = [BENCH, f"--data={traffic}", f"--params={run_dir}/ids_params.h",
           f"--l1={run_dir}/layer1_gatekeeper.ubj", f"--l2={run_dir}/layer2_multiclass.ubj",
           f"--mode={mode}", f"--nthread1={t1}", f"--nthread2={t2}", f"--rows={ROWS}"]
    if tau is not None:
        cmd.append(f"--tau={tau}")
    out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=True).stdout
    get = lambda key: float(re.search(re.escape(key) + r"[^:]*:\s*([0-9.]+)", out).group(1))
    return {"wall_ms": get("loop wall time"), "cpu_ms": get("loop process CPU time"),
            "summed_ms": get("summed per-flow inference time")}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ds, (run_dir, pattern) in DATASETS.items():
        tau_c = certified(ds)
        for mix in ("benign90", "benign75", "natural"):
            traffic = pattern.format(mix=mix)
            for tname, (t1, t2) in THREADS.items():
                for rep in range(REPEATS):
                    for design, mode, tau in (("single-stage", "single-stage", None),
                                              ("two-stage, efficiency", "two-stage", None),
                                              ("two-stage, certified", "two-stage", tau_c)):
                        r = run(run_dir, traffic, mode, t1, t2, tau)
                        r.update({"dataset": ds, "mix": mix, "threads": tname, "design": design, "repeat": rep})
                        rows.append(r)
                        print(r, flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(f"{OUT}/raw.csv", index=False)
    m = raw.groupby(["dataset", "mix", "threads", "design"])[["wall_ms", "cpu_ms", "summed_ms"]].mean().reset_index()
    single = m[m.design == "single-stage"].set_index(["dataset", "mix", "threads"])
    for col in ("wall_ms", "cpu_ms"):
        m[f"saving_{col[:-3]}"] = [1 - r[col] / single.loc[(r.dataset, r.mix, r.threads), col] for _, r in m.iterrows()]
    m["cpu_per_wall"] = m.cpu_ms / m.wall_ms
    m.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(m.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
