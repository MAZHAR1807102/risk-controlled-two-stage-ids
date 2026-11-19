# Risk-Controlled Two-Stage Intrusion Detection for IoT Edge Gateways

Code, trained models and result data for the paper

> M. Islam, "Risk-Controlled Two-Stage Intrusion Detection for IoT Edge Gateways" (manuscript under review).

The system is a two-stage intrusion detection system (IDS) for IoT edge gateways. A robot that
collects biohazard waste serves as an illustrative safety-critical device; it is not tested on real
robot traffic.

- **Stage 1, the Gatekeeper:** a small binary XGBoost model (100 trees, depth 6). Flows it scores
  below a threshold τ skip deep inspection.
- **Stage 2, the Brain:** a multiclass XGBoost model (about 2,100 trees, 8 classes) that labels
  every remaining flow.

The Gatekeeper gives every attack family equal weight. Its threshold is chosen on the validation
split with one-sided Clopper–Pearson bounds, so that, with 95% confidence, no attack family is
skipped more than 2% and no well-sampled attack type more than 5%. Both stages run in C++ inside
ns-3, next to an executed single-stage baseline, and the gateway is studied under flooding.

## Key results (CICIoT2023 test split)

| | Efficiency threshold (τ = 0.05) | Certified threshold (τ = 0.017) |
|---|---|---|
| Benign flows skipping Stage 2 | 76.7% | 61.2% |
| Worst attack family skipped | 1.94% | 0.27% |
| Worst attack type skipped | 4.9% | 0.66% |
| Inference-time saving vs. single-stage, 90% benign (1 thread / all cores) | 68% / 52% (ns-3) | 54% / 38% (C++ gateway) |
| Accuracy / macro-F1 | 99.61% / 0.868 (five seeds) | 99.61% / 0.866 (seed 42) |

The same certified rule can also limit the misses the Gatekeeper **adds** to the whole system (attacks it skips that
Stage 2 would have caught). At τ = 0.022, 64.6% of benign flows skip Stage 2, the Gatekeeper added at most 0.01% to the
misses of any attack family on test, and the saving rises to 57% / 40% (standalone C++ gateway, 90% benign).

In a simulated flow-record gateway (packets forwarded at once, the IDS checks flow records of 10 or 100 packets
off the forwarding path and installs block rules), the priority device lost no packet and waited at most 12 µs at
up to 128,000 flood packets/s. Floods from 50 bots were blocked after 0.4 to 3.7 s; a flood spread over 1,000
slow bots was not stopped in time.

**Caveats.** The guarantee limits what the Gatekeeper skips, not what the whole IDS misses. Much of the bypass on
CICIoT2023 comes from the IAT feature, which in the merged CSV behaves like a capture-session time stamp: an
attacker who copies benign timing makes the Gatekeeper skip up to half of a family, and a Gatekeeper trained
without IAT keeps the certificate but lets only 3% of benign flows skip Stage 2. All timings are host
measurements (Apple M5), not gateway-board measurements.

The second dataset, CICIoMT2024 (Wi-Fi and MQTT), reaches 99.81% accuracy and a macro-F1 of
0.936. There, the threshold needs on-site re-calibration, because the test captures differ from
the training captures.

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

The models are in XGBoost's binary format (`.ubj`) and give the same predictions as the JSON
format. Load them with `xgboost.Booster(model_file=...)` in Python, or `XGBoosterLoadModel` in C/C++.
Inputs are the raw flow features, scaled with the values in `ids_params.h`.

### Main source files

| File | Purpose |
|---|---|
| `src/datasets.py` | Loaders for CICIoT2023 and CICIoMT2024 (family mapping, validation split) |
| `src/pipeline.py` | Trains both stages, picks τ, evaluates, writes the models and `ids_params.h` |
| `src/train_gatekeeper.py` | Gatekeeper weighting study (unweighted, binary-balanced, family-balanced) |
| `src/risk_control.py` | Observed and certified threshold rules; on-site re-calibration |
| `src/risk_coverage.py`, `src/risk_dependence.py` | Resampling diagnostic and design effect of the certified rule |
| `src/risk_added.py --sel`, `src/added_timing.py --sel` | Whole-system certificate on the misses the Gatekeeper adds, and its saving. Uses the Brain in `runs/*_sel`, early-stopped on 15% of the train split (`pipeline.py --select-frac 0.15`), so the calibration flows are never used to fit or select a model |
| `src/build_ciciot_temporal.py`, `src/temporal_cert.py` | Time-ordered CICIoT2023 test: every per-attack CIC file cut 70/15/15 in time order, threshold rules (observed, certified, design-effect corrected, whole system) and a random-split control. The two Brain models (~90 MB) are not stored; `pipeline.py --dataset ciciot2023t` rebuilds them |
| `src/iomt_window_stats.py`, `src/iomt_sensitivity.py` | CICIoMT2024 Protocol B in detail: what the calibration window holds (rows, seconds, addresses; needs the CIC PCAPs), window position and size, block length, and a replication with the CIC train and test captures swapped (`pipeline.py --dataset ciciomt2024s`) |
| `ns3/IoT_IDS_FlowMeter.cc`, `src/run_flowmeter.py`, `src/flowmeter_table.py` | Flow-record gateway under flooding: packets are forwarded at once, a flow meter closes records of 10 (floods: 100) packets, the IDS checks records off the forwarding path and installs block rules. Robot delay, and time to block each attack flow split into formation, queue, inference and mitigation |
| `src/exposure.py` | Skip and miss rates per family counted in flows, packets and bytes; calibration counts per attack type |
| `src/method_compare.py` | Our family-wise Clopper-Pearson rule against conformal risk control, selective classification, group-conditional CRC and Learn-then-Test |
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
