#!/bin/bash
#SBATCH --job-name=stock-dl-infer
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_low
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -euo pipefail
REPO=~/finalex/stock_dl
cd "$REPO"
export PYTHONPATH="$(pwd)"

set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dl-homework
set -u

# 当前持仓通过环境变量传入，逗号分隔
HOLDINGS="${HOLDINGS:-}"
python scripts/06_infer_orders.py --config configs/default.yaml --holdings "$HOLDINGS"
