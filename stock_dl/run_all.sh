#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/quick.yaml}"
export PYTHONPATH="$(pwd)"

python3 scripts/01_build_panel.py --config "$CONFIG"
python3 scripts/03_train.py --config "$CONFIG"
python3 scripts/04_eval_ic.py --config "$CONFIG"
python3 scripts/05_backtest.py --config "$CONFIG"
python3 scripts/08_baselines.py --config "$CONFIG"
python3 scripts/09_diagnostics.py --config "$CONFIG"
python3 scripts/07_visualize.py --config "$CONFIG"
python3 scripts/06_infer_orders.py --config "$CONFIG"
echo "Done. See outputs*/"
