# Two-stage IDS benchmark on a Raspberry Pi

Runs the same detection pipeline as the ns-3 experiments (inline scaler, Gatekeeper,
Brain, adaptive routing) on real gateway hardware. Its decisions are identical to the
ns-3 version; only the timings change.

## Needs
- Raspberry Pi 4 or 5 (2 GB RAM or more), **64-bit** Raspberry Pi OS (Lite is fine)
- Power supply rated for the board, and a heatsink or fan if you have one
- About 1 GB free on the SD card, and internet for the setup

## Steps
1. Copy this folder to the Pi, e.g. from the laptop:
   `scp -r pi_bench pi@<pi-address>:~/`
2. On the Pi:
   ```
   cd ~/pi_bench
   ./setup_pi.sh                            # ~10 min; ends with a 500-flow smoke test
   nohup ./run_all.sh > run_all.log 2>&1 &  # several hours; safe to disconnect
   tail -f run_all.log                      # watch progress
   ```
3. When it prints `DONE`, copy `results_pi.tar.gz` back to the laptop:
   `scp pi@<pi-address>:~/pi_bench/results_pi.tar.gz .`

## Measurement hygiene
- Keep the Pi otherwise idle during `run_all.sh` (no desktop, no browser).
- Note in the paper whether it had a heatsink/fan. `results/device_state.log` records
  temperature, clock and throttling flags before every run.
- To shorten the campaign: `MAIN_ROWS=5000 SWEEP_ROWS=2000 ./run_all.sh`.

## What it measures
| Part | Content |
|---|---|
| A. main matrix | 3 traffic mixes × 1 thread/all cores × two-stage/single-stage |
| B. break-even | 8 benign shares × 7 configurations (single, two-stage, hybrid, adaptive) |
| C. flood timeline | normal → flood → normal, static vs adaptive routing |
