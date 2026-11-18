#!/bin/zsh
# Timing matrix for the ns-3 gateway simulation. Run on an otherwise idle machine.
# Output: ns3/timing/<mix>__new__<mode>__t<nthread>__inplace.csv (per-packet) + .log
P=${0:A:h:h}
NS3=~/NS3/ns-allinone-3.42/ns-3.42
OUT=$P/ns3/timing
mkdir -p $OUT
ROWS=${ROWS:-20000}
M=$P/runs/seed42_sqrt

run() { # mix mode nthread
  local name="$1__new__$2__t$3__inplace"
  echo "=== $name $(date +%H:%M:%S)"
  (cd $NS3 && ./ns3 run --no-build "IoT_TwoStage_IDS_v2 --data=$P/ns3/traffic/test_$1_20000.csv \
     --params=$M/ids_params.h --l1=$M/layer1_gatekeeper.ubj --l2=$M/layer2_multiclass.ubj \
     --mode=$2 --nthread=$3 --api=inplace --rows=$ROWS --out=$OUT/$name.csv") > $OUT/$name.log 2>&1 || echo "=== FAILED $name"
  grep -E "inspected by|total CPU|per-packet total" $OUT/$name.log
}

for mix in benign90 benign75 natural; do
  for nt in 1 0; do
    run $mix two-stage    $nt
    run $mix single-stage $nt
  done
done
echo "=== ALL DONE $(date +%H:%M:%S)"
