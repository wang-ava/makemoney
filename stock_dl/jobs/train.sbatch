#!/bin/bash
# USTC 107 本科生算力平台 — 参考:
# https://xinchengo.github.io/ustc107/guides/ai/deep-learning-homework/
#SBATCH --job-name=stock-dl-train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

# 改成你的路径
REPO=~/finalex/stock_dl
DATA=~/finalex/documents-export-2026-5-19
CONFIG=configs/default.yaml

cd "$REPO"
export PYTHONPATH="$(pwd)"
export DATA_DIR="$DATA"

# 激活 conda（按你集群上的环境名修改）
set +u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dl-homework
set -u

# 若 config 里 data_dir 是相对路径，可先 sed 或复制一份集群专用 config
python scripts/01_build_panel.py --config "$CONFIG"
python scripts/03_train.py --config "$CONFIG"
python scripts/04_eval_ic.py --config "$CONFIG"
python scripts/05_backtest.py --config "$CONFIG"
