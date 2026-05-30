#!/bin/bash
#SBATCH --job-name=stock-dl-full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

# 用法（推荐，从任何目录都可以）：
#   STOCK_DL=~/finalex/finalex/stock_dl
#   mkdir -p "$STOCK_DL/logs"
#   REPO="$STOCK_DL" sbatch "$STOCK_DL/jobs/train.sbatch"
#
# 用法（已经 cd 到 stock_dl 时）：
#   mkdir -p logs
#   REPO="$PWD" sbatch jobs/train.sbatch
#
# 方案选择（通过 CONFIG 环境变量）：
#   CONFIG=configs/a_share_pure_dl.yaml     # 方案A: 纯深度学习
#   CONFIG=configs/a_share_with_lgbm.yaml    # 方案B: 深度学习+LightGBM（默认）
#
# 可选环境变量：
#   HOLDINGS="000001.SZ,600000.SH" sbatch jobs/train.sbatch
#   SKIP_PANEL=0 sbatch jobs/train.sbatch      # 强制重建 panel
#   SKIP_PANEL=1 sbatch jobs/train.sbatch      # 强制复用已有 panel
#   START_AT=lgbm sbatch jobs/train.sbatch     # 从 LGBM 继续，跳过深度训练
#   START_AT=visualize STOP_AFTER=visualize sbatch jobs/train.sbatch  # 只重画图
#   TRAIN_MAX_MINUTES=360 sbatch jobs/train.sbatch
#   SKIP_LGBM=1 sbatch jobs/train.sbatch       # 跳过LGBM（用于方案A）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-$SCRIPT_REPO}"
CONFIG="${CONFIG:-configs/a_share_with_lgbm.yaml}"  # 默认使用方案B配置
CONDA_ENV="${CONDA_ENV:-dl-homework}"
HOLDINGS="${HOLDINGS:-}"
SKIP_PANEL="${SKIP_PANEL:-auto}"
SKIP_LGBM="${SKIP_LGBM:-0}"
START_AT="${START_AT:-panel}"
STOP_AFTER="${STOP_AFTER:-infer}"
export CONFIG

cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-stockdl-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

set +u
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "== Job info =="
echo "host: $(hostname)"
echo "repo: $REPO"
echo "config: $CONFIG"
echo "conda env: $CONDA_ENV"
echo "started at: $(date)"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
nvidia-smi || true

echo "== Pipeline =="
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

echo "pipeline range: $START_AT -> $STOP_AFTER"
PANEL_PATH="$(python - <<'PY'
import os
from pathlib import Path
from src.config import load_config
cfg = load_config(os.environ["CONFIG"])
print(Path(cfg["output_dir"]) / "panel.parquet")
PY
)"
if [ "$SKIP_PANEL" = "auto" ]; then
  if [ -f "$PANEL_PATH" ]; then
    SKIP_PANEL_EFFECTIVE=1
  else
    SKIP_PANEL_EFFECTIVE=0
  fi
else
  SKIP_PANEL_EFFECTIVE="$SKIP_PANEL"
fi

if [ "$SKIP_PANEL_EFFECTIVE" != "1" ]; then
  run_stage panel python scripts/01_build_panel.py --config "$CONFIG"
else
  echo "skip panel build: $PANEL_PATH"
fi

if should_run train; then
  if [ -z "${TRAIN_MAX_MINUTES:-}" ]; then
    if [ "$SKIP_PANEL_EFFECTIVE" = "1" ]; then
      export TRAIN_MAX_MINUTES="${TRAIN_MAX_MINUTES_SKIP_PANEL:-330}"
    else
      export TRAIN_MAX_MINUTES="${TRAIN_MAX_MINUTES_BUILD_PANEL:-270}"
    fi
  else
    export TRAIN_MAX_MINUTES
  fi
  echo "train max minutes: $TRAIN_MAX_MINUTES"
fi

run_stage train python scripts/03_train.py --config "$CONFIG"
# 只有在不跳过LGBM时才运行lgbm和blend阶段
if [ "$SKIP_LGBM" = "1" ]; then
  echo "skip lgbm (SKIP_LGBM=1)"
else
  run_stage lgbm python scripts/10_train_lgbm.py --config "$CONFIG"
  run_stage blend python scripts/11_blend_scores.py --config "$CONFIG"
fi
run_stage eval python scripts/04_eval_ic.py --config "$CONFIG"
run_stage backtest python scripts/05_backtest.py --config "$CONFIG"
run_stage baselines python scripts/08_baselines.py --config "$CONFIG"
run_stage diagnostics python scripts/09_diagnostics.py --config "$CONFIG"
run_stage visualize python scripts/07_visualize.py --config "$CONFIG"
run_stage infer python scripts/06_infer_orders.py --config "$CONFIG" --holdings "$HOLDINGS"

echo "== Done =="
echo "finished at: $(date)"
echo "outputs are under: $(python - <<'PY'
import os
from src.config import load_config
cfg = load_config(os.environ["CONFIG"])
print(cfg["output_dir"])
PY
)"
