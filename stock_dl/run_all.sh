#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/quick.yaml}"
START_AT="${START_AT:-panel}"
STOP_AFTER="${STOP_AFTER:-infer}"
HOLDINGS="${HOLDINGS:-}"
export PYTHONPATH="$(pwd)"
export WANDB_RUN_ID="${WANDB_RUN_ID:-stockdl-$(date +%Y%m%d-%H%M%S)}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

stage_index() {
  case "$1" in
    panel) echo 0 ;;
    train) echo 1 ;;
    lgbm) echo 2 ;;
    blend) echo 3 ;;
    eval) echo 4 ;;
    backtest) echo 5 ;;
    baselines) echo 6 ;;
    diagnostics) echo 7 ;;
    visualize) echo 8 ;;
    infer) echo 9 ;;
    *)
      echo "Unknown stage: $1" >&2
      echo "Valid stages: panel train lgbm blend eval backtest baselines diagnostics visualize infer" >&2
      exit 2
      ;;
  esac
}

START_IDX="$(stage_index "$START_AT")"
STOP_IDX="$(stage_index "$STOP_AFTER")"
if [ "$START_IDX" -gt "$STOP_IDX" ]; then
  echo "START_AT must be before or equal to STOP_AFTER." >&2
  exit 2
fi

should_run() {
  local idx
  idx="$(stage_index "$1")"
  [ "$idx" -ge "$START_IDX" ] && [ "$idx" -le "$STOP_IDX" ]
}

run_stage() {
  local stage="$1"
  shift
  if should_run "$stage"; then
    echo "== stage: $stage =="
    "$@"
  else
    echo "skip stage: $stage"
  fi
}

echo "Pipeline range: $START_AT -> $STOP_AFTER"
run_stage panel python3 scripts/01_build_panel.py --config "$CONFIG"
run_stage train python3 scripts/03_train.py --config "$CONFIG"
run_stage lgbm python3 scripts/10_train_lgbm.py --config "$CONFIG"
run_stage blend python3 scripts/11_blend_scores.py --config "$CONFIG"
run_stage eval python3 scripts/04_eval_ic.py --config "$CONFIG"
run_stage backtest python3 scripts/05_backtest.py --config "$CONFIG"
run_stage baselines python3 scripts/08_baselines.py --config "$CONFIG"
run_stage diagnostics python3 scripts/09_diagnostics.py --config "$CONFIG"
run_stage visualize python3 scripts/07_visualize.py --config "$CONFIG"
run_stage infer python3 scripts/06_infer_orders.py --config "$CONFIG" --holdings "$HOLDINGS"
echo "Done. See outputs*/"
