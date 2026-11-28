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

# 7. Packet-derived flow records (CICIoMT2024 PCAPs in data/CICIoMT2024/PCAP_full/{train,test}) - main result of the paper
# Records of up to 10 packets per flow, with attacker-MAC ground truth and context columns -> data/CICIoMT2024/records_ctx
$PY -u src/flow_records.py test && $PY -u src/flow_records.py train
PKT_LABEL=mac $PY -u src/build_packet_dataset.py data/CICIoMT2024/records_ctx data/CICIoMT2024/packet_mac
$PY -u src/pipeline.py --dataset ciciomt2024pmn  --run runs/pkt_MN_sel  --select-frac 0.15 > logs/runs/pkt_MN_sel.log 2>&1   # main model pMN
$PY -u src/pipeline.py --dataset ciciomt2024pmno --run runs/pkt_MNO_sel --select-frac 0.15 > logs/runs/pkt_MNO_sel.log 2>&1  # without TTL/header length
# Seven data settings (Fig. 3): CSV with/without artifacts, packet records (random split pR, separate captures pCN, MAC labels pMN).
# pR/pCN use protocol-based labels and were built (src/build_packet_dataset.py, default arguments -> data/CICIoMT2024/packet)
# from records made before the context columns were added; their datasets drop all timing features.
$PY -u src/pipeline.py --dataset ciciomt2024pr --run runs/pkt_R_sel  --select-frac 0.15 > logs/runs/pkt_R_sel.log 2>&1
$PY -u src/pipeline.py --dataset ciciomt2024pn --run runs/pkt_CN_sel --select-frac 0.15 > logs/runs/pkt_CN_sel.log 2>&1
$PY -u src/pipeline.py --dataset ciciomt2024x  --run runs/iomt_x_sel --select-frac 0.15 > logs/runs/iomt_x_sel.log 2>&1
$PY -u src/pipeline.py --dataset ciciot2023x   --run runs/c23_x_sel  --select-frac 0.15 > logs/runs/c23_x_sel.log 2>&1
for a in "ciciot2023 runs/seed42_sel c23" "ciciot2023x runs/c23_x_sel c23x" "ciciomt2024 runs/iomt_seed42_sel mt" \
         "ciciomt2024x runs/iomt_x_sel mtx" "ciciomt2024pr runs/pkt_R_sel pR" "ciciomt2024pn runs/pkt_CN_sel pCN" \
         "ciciomt2024pmn runs/pkt_MN_sel pMN" "ciciomt2024pmno runs/pkt_MNO_sel pMNO"; do $PY -u src/packet_cert.py ${=a}; done
$PY -u src/iut_cert.py                # certified rule with and without the Bonferroni split (Table S39)
$PY -u src/pkt_tradeoff.py            # Fig. 2(b)
$PY -u src/adaptive_attack.py         # padding attacker (Fig. 4)
$PY -u src/benign_block.py            # benign hosts blocked by the three-verdict rule
for k in "OS_Scan OS_Scan_pcap.pcapng Recon" "SSDP_Flood SSDP_Flood_pcap.pcap DDoS" "ARP_MitM ARP_MitM_pcap.pcapng Spoofing"; do
  set -- ${=k}; $PY -u src/kitsune_eval.py data/kitsune/$2 data/kitsune/$1_labels.csv runs/pkt_MN_sel --attack $3 \
    --tau $(awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next} $h["rule"]=="added"{print $h["tau"]}' results/packet/cert_pMN.csv) --tag _added
done
PKT_SFX=_mac_nosplit $PY -u src/packet_timing.py ciciomt2024pmn:runs/pkt_MN_sel:pMN   # timing (idle machine)
# CIC extractor reconstruction and IAT = capture timestamp
$PY -u src/cic_validate.py test && $PY -u src/iat_dates.py
# C++ feature extractor check (exports packets and builds ns3/pcap_feature_check.cc with clang++ and the XGBoost C library)
$PY -u src/cpp_feature_check.py

# 8. Reviewer-requested runs: fast inference path, forged-robot and line-rate floods, tuned baselines
$PY -u src/onnx_cost.py && $PY -u src/router_offline.py onnx
$PY -u src/run_flowmeter.py --set onnx && $PY -u src/run_flowmeter.py --set forged && $PY -u src/run_flowmeter.py --set linerate
$PY -u src/baselines_tuned.py search && $PY -u src/baselines_tuned.py test && $PY -u src/baselines_tuned.py time
$PY -u src/rf_brain.py                # our Gatekeeper with the tuned Random Forest as Stage 2

# 9. Figures
$PY src/make_figures.py
