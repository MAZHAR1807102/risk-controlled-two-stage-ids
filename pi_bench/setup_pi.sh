#!/bin/bash
# One-time setup on the Raspberry Pi (64-bit Raspberry Pi OS / Ubuntu, aarch64).
# Installs a compiler, gets the XGBoost shared library from pip, builds ids_bench.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "This needs a 64-bit OS (uname -m = aarch64). Re-flash the Pi with 64-bit Raspberry Pi OS." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y build-essential python3-venv libgomp1

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install xgboost

XGB_LIB_DIR=$(.venv/bin/python -c "import xgboost, os; print(os.path.join(os.path.dirname(xgboost.__file__), 'lib'))")
echo "XGBoost $(.venv/bin/python -c 'import xgboost; print(xgboost.__version__)') library: $XGB_LIB_DIR"

g++ -O2 -std=c++17 ids_bench.cc -Iinclude -L"$XGB_LIB_DIR" -lxgboost -Wl,-rpath,"$XGB_LIB_DIR" -o ids_bench

# Smoke test: 500 flows through the new models.
./ids_bench --data=traffic/main/test_benign90_20000.csv --params=models/new/ids_params.h \
  --l1=models/new/layer1_gatekeeper.ubj --l2=models/new/layer2_multiclass.ubj \
  --mode=two-stage --nthread=1 --rows=500 --warmup=50
echo "Setup OK. Next: ./run_all.sh"
