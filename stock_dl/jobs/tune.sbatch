#!/bin/bash
#SBATCH --job-name=stock-dl-tune
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
#   sbatch jobs/tune.sbatch
#
# 可选：
#   TRIALS_PER_JOB=2 sbatch jobs/tune.sbatch           # 一次提交连续跑 2 组，注意 8h 上限
#   sbatch --array=1-4 jobs/tune.sbatch               # 并行跑 4 组，脚本会自动抢不同 trial
#   FORCE_TUNE_PANEL=1 sbatch jobs/tune.sbatch        # 强制重建共享 panel
#   TRIAL_MAX_MINUTES=300 sbatch jobs/tune.sbatch     # 单个 trial 深度模型训练时长上限
#   TUNE_SKIP_LGBM=1 sbatch jobs/tune.sbatch          # 临时只调深度模型，不跑 LGBM 融合

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/finalex/finalex/stock_dl}}"
CONDA_ENV="${CONDA_ENV:-dl-homework}"
TUNE_BASE_CONFIG="${TUNE_BASE_CONFIG:-configs/server_8h.yaml}"
TUNE_DIR="${TUNE_DIR:-outputs_tuning}"
TRIALS_PER_JOB="${TRIALS_PER_JOB:-1}"
TRIAL_MAX_MINUTES="${TRIAL_MAX_MINUTES:-330}"
export TUNE_BASE_CONFIG TUNE_DIR TRIALS_PER_JOB TRIAL_MAX_MINUTES

cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-stockdl-tune-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

set +u
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "== Tune job info =="
echo "host: $(hostname)"
echo "repo: $REPO"
echo "base config: $TUNE_BASE_CONFIG"
echo "tune dir: $TUNE_DIR"
echo "trials per job: $TRIALS_PER_JOB"
echo "trial max minutes: $TRIAL_MAX_MINUTES"
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

python scripts/12_tune_once.py \
  --base-config "$TUNE_BASE_CONFIG" \
  --tune-dir "$TUNE_DIR" \
  --trials "$TRIALS_PER_JOB" \
  --max-minutes "$TRIAL_MAX_MINUTES"

echo "== Tune done =="
echo "finished at: $(date)"
echo "results:"
echo "  $TUNE_DIR/tuning_results.csv"
echo "  $TUNE_DIR/best_params.json"
echo "  $TUNE_DIR/best_params.yaml"
echo "  $TUNE_DIR/best_config.yaml"
