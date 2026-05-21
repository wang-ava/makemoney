#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/quick.yaml}"
export PYTHONPATH="$(pwd)"
export WANDB_RUN_ID="${WANDB_RUN_ID:-stockdl-$(date +%Y%m%d-%H%M%S)}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

python3 scripts/01_build_panel.py --config "$CONFIG"
python3 scripts/03_train.py --config "$CONFIG"
python3 scripts/10_train_lgbm.py --config "$CONFIG"
python3 scripts/11_blend_scores.py --config "$CONFIG"
python3 scripts/04_eval_ic.py --config "$CONFIG"
python3 scripts/05_backtest.py --config "$CONFIG"
python3 scripts/08_baselines.py --config "$CONFIG"
python3 scripts/09_diagnostics.py --config "$CONFIG"
python3 scripts/07_visualize.py --config "$CONFIG"
python3 scripts/06_infer_orders.py --config "$CONFIG"
echo "Done. See outputs*/"
