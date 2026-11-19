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
$PY src/aggregate.py --runs runs/seed1 runs/seed2 runs/seed3 runs/seed4 runs/seed5 --out results/seeds --boot 200

# 2. Gatekeeper studies and threshold rules
$PY -u src/train_gatekeeper.py --params runs/seed42_sqrt/ids_params.h --out results/gatekeeper_weighting_final
$PY -u src/calibration.py
$PY -u src/risk_control.py
$PY -u src/risk_coverage.py
$PY -u src/risk_added.py           # whole-system (added-miss) certificate
$PY -u src/dup_analysis.py         # exact copies of flows across the splits
$PY -u src/dup_robustness.py 3     # certificate without near copies (3 and 2 significant digits)
$PY -u src/dup_robustness.py 2

# 3. ns-3 (two-stage gateway)
$PY src/make_ns3_traffic.py --rows 20000 --out ns3/traffic
zsh ns3/run_timing.sh && $PY src/analyze_timing.py > ns3/timing/analysis.txt
$PY -u src/run_experiments.py all
$PY -u src/run_experiments.py repeat --reps 2
$PY -u src/adaptive_sensitivity.py && $PY src/adaptive_reanalyze.py

# 4. ns-3 (gateway under flooding)
$PY -u src/run_overload.py
$PY -u src/run_overload.py --queuing
$PY -u src/run_overload.py --enforced   # authentication cost + per-check time limit

# 5. Standalone C++ benchmark: second dataset, certified threshold, process CPU time
$PY -u src/iomt_timing.py --run runs/iomt_seed42
$PY -u src/certified_timing.py
$PY -u src/cpu_timing.py
$PY -u src/added_timing.py

# 6. Baselines and statistics
$PY -u src/baselines.py train && $PY -u src/baselines.py eval && $PY -u src/baselines.py time
$PY -u src/stats_ci.py

# 7. Figures
$PY src/make_figures.py
