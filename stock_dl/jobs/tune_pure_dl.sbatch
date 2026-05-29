#!/bin/bash
#SBATCH --job-name=stock-dl-tune-pure-dl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# 方案A: 纯深度学习调参作业
# GRU + Transformer，不使用LightGBM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-$SCRIPT_REPO}"
CONDA_ENV="${CONDA_ENV:-dl-homework}"

cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-stockdl-pure-dl-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_GROUP="${WANDB_GROUP:-finalex-scheme-a-tune}"

set +u
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "== Job info =="
echo "host: $(hostname)"
echo "repo: $REPO"
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

echo "== Running Pure DL Tuning (Scheme A) =="

# 调参配置
TUNE_DIR="outputs_tuning_pure_dl"
BASE_CONFIG="configs/a_share_pure_dl.yaml"
TRIALS="${TRIALS_PER_JOB:-1}"
MAX_MINUTES="${TRIAL_MAX_MINUTES:-60}"

echo "Tune directory: $TUNE_DIR"
echo "Base config: $BASE_CONFIG"
echo "Trials per job: $TRIALS"
echo "Max minutes per trial: $MAX_MINUTES"

# 运行调参（纯DL模式，跳过LGBM）
python scripts/12_tune_once.py \
    --base-config "$BASE_CONFIG" \
    --tune-dir "$TUNE_DIR" \
    --trials "$TRIALS" \
    --max-minutes "$MAX_MINUTES" \
    --pure-dl \
    --force-panel

echo "== Tuning complete =="
echo "finished at: $(date)"
echo "Results saved to: $TUNE_DIR"
