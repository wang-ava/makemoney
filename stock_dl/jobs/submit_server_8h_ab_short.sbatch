#!/usr/bin/env bash
#SBATCH --job-name=stockdl-ab-short
#SBATCH --output=logs/stockdl-ab-short-%j.out
#SBATCH --error=logs/stockdl-ab-short-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STOCK_DL_DIR:-$SCRIPT_DIR}"
mkdir -p logs

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-470}"
export A_TRAIN_MAX_MINUTES="${A_TRAIN_MAX_MINUTES:-150}"
export B_TRAIN_MAX_MINUTES="${B_TRAIN_MAX_MINUTES:-150}"
export B_LGBM_MAX_MINUTES="${B_LGBM_MAX_MINUTES:-65}"
export AUDIT_TOP_K="${AUDIT_TOP_K:-30}"
export DEPLOY_REQUIRE_GREEN="${DEPLOY_REQUIRE_GREEN:-0}"

bash scripts/run_server_8h_ab_short.sh
