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

# 用法：
#   cd ~/finalex/finalex/stock_dl
#   mkdir -p logs
#   sbatch jobs/train.sbatch
#
# 可选：
#   CONFIG=configs/default.yaml sbatch jobs/train.sbatch
#   HOLDINGS="000001.SZ,600000.SH" sbatch jobs/train.sbatch
#   SKIP_PANEL=0 sbatch jobs/train.sbatch      # 强制重建 panel
#   SKIP_PANEL=1 sbatch jobs/train.sbatch      # 强制复用已有 panel
#   TRAIN_MAX_MINUTES=360 sbatch jobs/train.sbatch

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/finalex/finalex/stock_dl}}"
CONFIG="${CONFIG:-configs/server_8h.yaml}"
CONDA_ENV="${CONDA_ENV:-dl-homework}"
HOLDINGS="${HOLDINGS:-}"
SKIP_PANEL="${SKIP_PANEL:-auto}"
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
  python scripts/01_build_panel.py --config "$CONFIG"
else
  echo "skip panel build: $PANEL_PATH"
fi

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

python scripts/03_train.py --config "$CONFIG"
python scripts/10_train_lgbm.py --config "$CONFIG"
python scripts/11_blend_scores.py --config "$CONFIG"
python scripts/04_eval_ic.py --config "$CONFIG"
python scripts/05_backtest.py --config "$CONFIG"
python scripts/08_baselines.py --config "$CONFIG"
python scripts/09_diagnostics.py --config "$CONFIG"
python scripts/07_visualize.py --config "$CONFIG"
python scripts/06_infer_orders.py --config "$CONFIG" --holdings "$HOLDINGS"

echo "== Done =="
echo "finished at: $(date)"
echo "outputs are under: $(python - <<'PY'
import os
from src.config import load_config
cfg = load_config(os.environ["CONFIG"])
print(cfg["output_dir"])
PY
)"
