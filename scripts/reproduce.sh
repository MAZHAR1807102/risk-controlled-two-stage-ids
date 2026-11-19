#!/bin/zsh
# Full reproduction of every result in the paper, in order. Run from anywhere; it cd's to the repo root.
# Expect many hours in total. Run timing steps on an otherwise idle machine.
# Needs: data/ (see data/README.md), .venv (requirements.txt), the XGBoost C library, and ns-3.42 with the
# programs from ns3/ built in scratch/nested-subdir (set NS3 below).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NS3=${NS3:-~/NS3/ns-allinone-3.42/ns-3.42}
mkdir -p logs/runs logs/results logs/ns3

# 1. Models: deployed run, five seeds, Stage 2 ablation, second dataset
$PY -u src/pipeline.py --run runs/seed42_sqrt --seed 42            > logs/runs/seed42_sqrt.log 2>&1
for s in 1 2 3 4 5; do $PY -u src/pipeline.py --run runs/seed$s --seed $s > logs/runs/seed$s.log 2>&1; done
$PY -u src/pipeline.py --run runs/seed42_balanced --seed 42 --stage2-weighting balanced > logs/runs/seed42_balanced.log 2>&1
$PY -u src/pipeline.py --dataset ciciomt2024 --run runs/iomt_seed42 --seed 42 > logs/runs/iomt_seed42.log 2>&1
# Brain early-stopped on 15% of train (never sees validation), for the whole-system certificate
$PY -u src/pipeline.py --run runs/seed42_sel --seed 42 --select-frac 0.15 > logs/runs/seed42_sel.log 2>&1
$PY -u src/pipeline.py --dataset ciciomt2024 --run runs/iomt_seed42_sel --seed 42 --select-frac 0.15 > logs/runs/iomt_seed42_sel.log 2>&1
$PY src/aggregate.py --runs runs/seed1 runs/seed2 runs/seed3 runs/seed4 runs/seed5 --out results/seeds --boot 200

# 2. Gatekeeper studies and threshold rules
$PY -u src/train_gatekeeper.py --params runs/seed42_sqrt/ids_params.h --out results/gatekeeper_weighting_final
$PY -u src/calibration.py
$PY -u src/risk_control.py
$PY -u src/risk_coverage.py
$PY -u src/risk_added.py --sel     # whole-system (added-miss) certificate, Brain that never saw validation
$PY -u src/dup_analysis.py         # exact copies of flows across the splits
$PY -u src/dup_robustness.py 3     # certificate without near copies (3 and 2 significant digits)
$PY -u src/dup_robustness.py 2
$PY -u src/dup_robustness.py 3 --sel
# Time-ordered CICIoT2023 (per-attack CIC CSV files in data/CICIoT2023/per_attack_csv) and its random-split control
$PY -u src/build_ciciot_temporal.py
$PY -u src/pipeline.py --dataset ciciot2023t --run runs/t23_seed42_sel --seed 42 --select-frac 0.15 > logs/runs/t23_seed42_sel.log 2>&1
$PY -u src/pipeline.py --dataset ciciot2023r --run runs/r23_seed42_sel --seed 42 --select-frac 0.15 > logs/runs/r23_seed42_sel.log 2>&1
$PY -u src/temporal_cert.py ciciot2023t
$PY -u src/temporal_cert.py ciciot2023r
# CICIoMT2024 Protocol B in detail: window time/addresses (full PCAPs), window position/size/block length, swapped captures
$PY -u src/iomt_window_stats.py test
$PY -u src/pipeline.py --dataset ciciomt2024s --run runs/iomt_swap_seed42_sel --seed 42 --select-frac 0.15 > logs/runs/iomt_swap_seed42_sel.log 2>&1
$PY -u src/iomt_sensitivity.py orig
$PY -u src/iomt_sensitivity.py swap

# 3. ns-3 (two-stage gateway)
$PY src/make_ns3_traffic.py --rows 20000 --out ns3/traffic
zsh ns3/run_timing.sh && $PY src/analyze_timing.py > ns3/timing/analysis.txt
$PY -u src/run_experiments.py all
$PY -u src/run_experiments.py repeat --reps 2
$PY -u src/adaptive_sensitivity.py && $PY src/adaptive_reanalyze.py

# 4. ns-3 (gateway under flooding)
$PY -u src/run_overload.py
$PY -u src/run_overload.py --queuing
# Flow-record gateway (packets forwarded at once, IDS checks flow records, block rules); used in the paper.
# Build first: copy ns3/IoT_IDS_FlowMeter.cc to ns-3 scratch/nested-subdir, add it to CMakeLists, ./ns3 build IoT_IDS_FlowMeter
$PY -u src/run_flowmeter.py && $PY -u src/run_flowmeter.py --variants --append
$PY -u src/run_flowmeter.py --queues --append   # queue sizes and robot traffic
$PY src/flowmeter_table.py
# Exposure in packets/bytes, other threshold rules, router estimate, flow shaping, no-IAT Gatekeeper
$PY -u src/exposure.py
$PY -u src/method_compare.py
$PY -u src/router_offline.py
$PY -u src/evasion.py
$PY -u src/gatekeeper_no_iat.py
$PY -u src/baselines.py time   # also writes p99.9 and max (timing_p999.csv)
$PY -u src/run_overload.py --enforced   # authentication cost + per-check time limit

# 5. Standalone C++ benchmark: second dataset, certified threshold, process CPU time
$PY -u src/iomt_timing.py --run runs/iomt_seed42
$PY -u src/certified_timing.py
$PY -u src/cpu_timing.py
$PY -u src/added_timing.py --sel

# 6. Baselines and statistics
$PY -u src/baselines.py train && $PY -u src/baselines.py eval && $PY -u src/baselines.py time
$PY -u src/stats_ci.py

# 7. Figures
$PY src/make_figures.py
