#!/bin/bash
# Full measurement campaign on the Pi. Run inside tmux/screen or with nohup; it takes hours.
#   nohup ./run_all.sh > run_all.log 2>&1 &
# Output layout matches the ns-3 experiments, so analyze_timing.py and make_figures.py
# on the laptop can read it directly.
set -uo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p $OUT/timing $OUT/experiments/breakeven $OUT/experiments/timeline
MAIN_ROWS=${MAIN_ROWS:-10000}
SWEEP_ROWS=${SWEEP_ROWS:-5000}

device_state() { # temperature, clock and throttling flags, when vcgencmd exists
  if command -v vcgencmd >/dev/null; then
    echo "$(date +%T) $(vcgencmd measure_temp) $(vcgencmd measure_clock arm) $(vcgencmd get_throttled)"
  else
    echo "$(date +%T) temp=$(awk '{print $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null)C"
  fi
}

{
  echo "== device"; cat /proc/device-tree/model 2>/dev/null; echo; uname -a; lscpu | head -20
  free -m; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
  .venv/bin/python -c 'import xgboost; print("xgboost", xgboost.__version__)'
} > $OUT/device_info.txt

run() { # traffic model mode nthread1 nthread2 rows outfile
  echo "[$(device_state)] $(basename "$7")" | tee -a $OUT/device_state.log
  ./ids_bench --data="$1" --params=models/$2/ids_params.h --l1=models/$2/layer1_gatekeeper.ubj \
    --l2=models/$2/layer2_multiclass.ubj --mode=$3 --nthread1=$4 --nthread2=$5 --rows=$6 \
    --out="$7" > "${7%.csv}.log" 2>&1 || echo "FAILED $7"
  grep -E "summed per-flow|process CPU|missed" "${7%.csv}.log"
}

echo "=== A. main matrix: two-stage vs single-stage ($MAIN_ROWS flows per run)"
for mix in benign90 benign75 natural; do
  for t in 1 0; do
    for mode in two-stage single-stage; do
      run traffic/main/test_${mix}_20000.csv new $mode $t $t $MAIN_ROWS \
          $OUT/timing/${mix}__new__${mode}__t${t}__inplace.csv
    done
  done
done

echo "=== B. break-even sweep ($SWEEP_ROWS flows per run, new models)"
# name:mode:nthread1:nthread2, as in run_experiments.py
CONFIGS="single_t1:single-stage:1:1 single_tall:single-stage:0:0 two_t1:two-stage:1:1 two_tall:two-stage:0:0 two_hybrid:two-stage:1:0 adaptive_tall:adaptive:0:0 adaptive_hybrid:adaptive:1:0"
for f in traffic/breakeven/traffic_b*.csv; do
  share=$(basename "$f" .csv | sed 's/traffic_//')
  for c in $CONFIGS; do
    IFS=: read -r name mode t1 t2 <<< "$c"
    run "$f" new $mode $t1 $t2 $SWEEP_ROWS $OUT/experiments/breakeven/${share}__${name}.csv
  done
done

echo "=== C. flood timeline (new models)"
cp traffic/timeline/phases.csv $OUT/experiments/timeline/
for c in single_tall:single-stage:0:0 two_tall:two-stage:0:0 adaptive_tall:adaptive:0:0 two_hybrid:two-stage:1:0 adaptive_hybrid:adaptive:1:0; do
  IFS=: read -r name mode t1 t2 <<< "$c"
  run traffic/timeline/traffic.csv new $mode $t1 $t2 20000 $OUT/experiments/timeline/${name}.csv
done

echo "=== DONE. Copy the 'results' folder back to the laptop."
tar czf results_pi.tar.gz $OUT && echo "Packed: results_pi.tar.gz"
