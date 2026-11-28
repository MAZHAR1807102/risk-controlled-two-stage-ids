# Two-Stage Intrusion Detection With Per-Family Miss Control

Code, trained models and result data for the paper

> M. Islam, U. H. Tisha, "Two-Stage Intrusion Detection With Per-Family Miss Control: Certified Under Independence,
> Tested on Time-Ordered Traffic" (manuscript under review).

The system is a two-stage intrusion detection system (IDS) for IoT edge gateways.

- **Stage 1, the Gatekeeper:** a small binary XGBoost model (100 trees, depth 6). Flow records it scores
  below a threshold τ skip deep inspection.
- **Stage 2, the Brain:** a multiclass XGBoost model that labels every remaining record. Any model can be
  used here; we also test a Random Forest.

The Gatekeeper gives every attack family equal weight. Its threshold is chosen on calibration data with
one-sided Clopper–Pearson bounds, every unit (family, and attack type with enough data) tested at the full
α = 5% in a fixed order (Learn Then Test, no Bonferroni split). If the calibration flows are independent and
match future traffic, then with 95% confidence no attack family **bypasses inspection** at a rate above 2%
and no well-sampled type above 5%. The guarantee covers the bypass decision, not detection: the whole system
still misses what the Brain misses. A variant bounds the misses the Gatekeeper adds to those of the Brain.

## Key results

**Main setting: our own flow records built from raw CICIoMT2024 packets**, labels from the attacker MAC
addresses, separate time-ordered test captures (model `pMN`). Independence fails here, so these are empirical:

| | CP-selected threshold | Whole-system threshold |
|---|---|---|
| Benign records skipping Stage 2 | 90.0% | 97.7% |
| Worst attack family skipped / added | 1.3% (1 of 78 spoofing records) | 2.6% added (above the 2% target) |
| Inference-time saving, 90% benign (1 thread / all cores) | 76% / 44% | 81% / 51% |

Over seven data settings (CIC CSV with and without artifacts, packet records with random and separate-capture
splits), the Clopper–Pearson-selected rule met its 2% family target on test every time; the observed threshold
failed in four. Macro-F1 of pMN is only 0.69 (DDoS and DoS confused per flow; ARP spoofing hard).

**Reference setting: CICIoT2023 CSV** (independent random splits, where the certificate applies):

| | Observed threshold (τ = 0.05) | Certified threshold (τ = 0.031) | Whole system (τ = 0.297) |
|---|---|---|---|
| Benign flows skipping Stage 2 | 76.7% | 69.6% | 94.8% |
| Worst attack family skipped / added | 1.94% skipped | 1.00% skipped | 1.62% added |
| Inference-time saving, 90% benign (1 thread / all cores) | 68% / 52% | 62% / 44% | 84% / 65% |

**Caveats.** In the merged CICIoT2023 CSV, IAT holds the capture time stamp and Number/Weight encode CIC's
packet window per attack type, so much of the CSV bypass rests on dataset artifacts. A padding attacker evades
some scans and the spoofing records. All timings are host measurements (Apple M5), not gateway-board
measurements. In the simulated flow-record gateway, the priority device lost no packet up to line rate, but above
the data plane's capacity (about 330,000 packets/s on our host model) the IDS stops until the flood ends.

## Repository layout

```
├── src/          Python: training pipeline, threshold rules, experiments, statistics, figures
├── ns3/          ns-3 programs (two-stage gateway, gateway under flooding) and result summaries
├── pi_bench/     Standalone C++ gateway benchmark (same decisions as ns-3; also reads CPU time)
├── scripts/      reproduce.sh: every experiment, in order
├── runs/         Trained models and their metrics (see below)
├── results/      Result tables (CSV/JSON) behind the tables and figures of the paper
├── figures/      Figures (PDF + PNG)
└── data/         Datasets (not included; see data/README.md)
```

### Trained models

| Folder | Content |
|---|---|
| `runs/seed42_sqrt/` | Deployed CICIoT2023 models: `layer1_gatekeeper.ubj`, `layer2_multiclass.ubj`, `ids_params.h` (scaler means, standard deviations and τ for C++), metrics and reports |
| `runs/iomt_seed42/` | The same for CICIoMT2024 |
| `runs/seed1` … `runs/seed5` | Metrics of the five-seed study |
| `runs/seed42_balanced/` | Metrics of the Stage 2 weighting ablation |
| `runs/pkt_MN_sel/` | Main packet-record model pMN (CICIoMT2024 flow records, attacker-MAC labels, no timing features), Brain early-stopped on 15% of the train split |
| `runs/seed42_sel/`, `runs/iomt_seed42_sel/` | Brains early-stopped on 15% of the train split (whole-system certificate) |

The models are in XGBoost's binary format (`.ubj`) and give the same predictions as the JSON
format. Load them with `xgboost.Booster(model_file=...)` in Python, or `XGBoosterLoadModel` in C/C++.
Inputs are the raw flow features, scaled with the values in `ids_params.h`.

### Main source files

| File | Purpose |
|---|---|
| `src/datasets.py` | Loaders for CICIoT2023 and CICIoMT2024 (family mapping, validation split) |
| `src/pipeline.py` | Trains both stages, picks τ, evaluates, writes the models and `ids_params.h` |
| `src/train_gatekeeper.py` | Gatekeeper weighting study (unweighted, binary-balanced, family-balanced) |
| `src/risk_control.py` | Observed and certified threshold rules (rule `ucb_typ`: no Bonferroni split; `ucb_typ_split`: the earlier, more conservative budget); on-site re-calibration |
| `src/iut_cert.py` | Certified rule with and without the Bonferroni split over seven data settings |
| `src/flow_records.py`, `src/build_packet_dataset.py` | Flow records built from the CICIoMT2024 packet captures; attacker-MAC labels; dataset splits |
| `src/packet_cert.py`, `src/packet_timing.py`, `src/pkt_tradeoff.py` | Threshold rules, timing and trade-off curve per data setting |
| `src/adaptive_attack.py`, `src/kitsune_eval.py`, `src/benign_block.py` | Padding attacker, cross-dataset test on Kitsune, benign hosts blocked by the three-verdict rule |
| `src/cic_extractor.py`, `src/cic_validate.py`, `src/iat_dates.py` | Re-implementation of CIC's feature extraction, validated against the CSV; IAT = capture time stamp |
| `ns3/flow_features.h`, `ns3/pcap_feature_check.cc`, `src/cpp_feature_check.py` | C++ port of the flow-record features, checked against Python |
| `src/onnx_cost.py`, `src/baselines_tuned.py`, `src/rf_brain.py` | Cost with ONNX Runtime; tuned baselines; the Gatekeeper with a Random Forest Brain |
| `src/risk_coverage.py`, `src/risk_dependence.py` | Resampling diagnostic and design effect of the certified rule |
| `src/risk_added.py --sel`, `src/added_timing.py --sel` | Whole-system certificate on the misses the Gatekeeper adds, and its saving. Uses the Brain in `runs/*_sel`, early-stopped on 15% of the train split (`pipeline.py --select-frac 0.15`), so the calibration flows are never used to fit or select a model |
| `src/build_ciciot_temporal.py`, `src/temporal_cert.py` | Time-ordered CICIoT2023 test: every per-attack CIC file cut 70/15/15 in time order, threshold rules (observed, certified, design-effect corrected, whole system) and a random-split control. The two Brain models (~90 MB) are not stored; `pipeline.py --dataset ciciot2023t` rebuilds them |
| `src/iomt_window_stats.py`, `src/iomt_sensitivity.py` | CICIoMT2024 Protocol B in detail: what the calibration window holds (rows, seconds, addresses; needs the CIC PCAPs), window position and size, block length, and a replication with the CIC train and test captures swapped (`pipeline.py --dataset ciciomt2024s`) |
| `ns3/IoT_IDS_FlowMeter.cc`, `src/run_flowmeter.py`, `src/flowmeter_table.py` | Flow-record gateway under flooding: packets are forwarded at once, a flow meter closes records of 10 (floods: 100) packets, the IDS checks records off the forwarding path and installs block rules. Robot delay, and time to block each attack flow split into formation, queue, inference and mitigation |
| `src/exposure.py` | Skip and miss rates per family counted in flows, packets and bytes; calibration counts per attack type |
| `src/method_compare.py` | Our rule against conformal risk control, selective classification, Mondrian CRC, PAC Mondrian calibration and Learn Then Test (Hoeffding; exact binomial with and without Bonferroni) |
| `src/router_offline.py` | Offline copy of the adaptive router: bias of its bypass estimate and six further load patterns |
| `src/evasion.py`, `src/gatekeeper_no_iat.py` | Flow-shaping attacker (benign timing or sizes copied), and a Gatekeeper trained without the IAT feature |
| `src/dup_analysis.py`, `src/dup_robustness.py` | Exact and near copies of flows across the splits; certified threshold re-picked without them |
| `src/baselines.py` | Six baselines; ONNX Runtime timing |
| `src/stats_ci.py` | Bootstrap (seed + test, moving-block) and paired-difference intervals |
| `src/run_experiments.py`, `src/adaptive_sensitivity.py`, `src/run_overload.py` | ns-3 experiments |
| `src/cpu_timing.py`, `src/certified_timing.py`, `src/iomt_timing.py`, `src/recal_timing.py` | C++ benchmark campaigns |
| `src/make_figures.py` | All figures from saved results |
| `ns3/IoT_TwoStage_IDS_v2.cc` | ns-3 gateway: two-stage, single-stage and adaptive modes, per-packet timing |
| `ns3/IoT_IDS_Overload.cc` | ns-3 gateway as a queue under flooding: priority, fair queuing, blocking |

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The C++ parts need the XGBoost C library (`brew install xgboost` on macOS; the header is in
`pi_bench/include`) and ns-3.42 for the simulations. Copy the programs in `ns3/` to
`ns-3.42/scratch/nested-subdir/` and build them with `./ns3 build IoT_TwoStage_IDS_v2` and
`./ns3 build IoT_IDS_Overload`.

## Data

The CIC datasets may not be redistributed. Download CICIoT2023 and CICIoMT2024 from the Canadian
Institute for Cybersecurity and place them as described in `data/README.md`.

## Reproducing the results

```bash
.venv/bin/python src/pipeline.py --run runs/seed42_sqrt --seed 42   # deployed models, about 7 min
zsh scripts/reproduce.sh                                             # everything, many hours
```

Run the timing steps on an otherwise idle machine. All timings in the paper come from one
Apple M5 host; absolute times will differ on other machines, but the relative savings should hold.

## Citation

If you use this code, please cite the paper above. The full reference will be added here when the
paper is published.

## License

The code is released under the MIT License (see `LICENSE`). `pi_bench/include/xgboost/c_api.h`
is part of XGBoost and is under the Apache License 2.0. The datasets keep their own licenses.
