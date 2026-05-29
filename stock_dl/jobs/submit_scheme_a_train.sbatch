#!/bin/bash
# ============================================================================
# 服务器提交脚本 - 深度学习大作业
# ============================================================================
#
# 使用方法：
#   方案A训练: sbatch submit_scheme_a_train.sbatch
#   方案B训练: sbatch submit_scheme_b_train.sbatch
#   方案A每日推理: sbatch submit_scheme_a_infer.sbatch
#   方案B每日推理: sbatch submit_scheme_b_infer.sbatch
#
# ============================================================================

# 设置环境
export PATH="$HOME/miniforge3/bin:$PATH"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dl-homework

cd /home/scc/pb23061103/finalex/finalex/stock_dl
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export WANDB_RESUME=allow
export WANDB_GROUP="finalex-$(date +%Y%m%d)"

# ============================================================================
# 方案A: 纯深度学习训练
# ============================================================================
#SBATCH --job-name=stock-dl-scheme-a-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

echo "========================================"
echo "方案A: 纯深度学习训练"
echo "时间: $(date)"
echo "========================================"

# 配置
export CONFIG="configs/a_share_pure_dl.yaml"
export OUTPUT_DIR="outputs_scheme_a"

echo "配置: $CONFIG"
echo "输出: $OUTPUT_DIR"

# 构建面板
python scripts/01_build_panel.py --config "$CONFIG"

# 训练深度学习模型
python scripts/03_train.py --config "$CONFIG"

# 评估IC
python scripts/04_eval_ic.py --config "$CONFIG"

# 回测
python scripts/05_backtest.py --config "$CONFIG"

# 生成交易信号
python scripts/06_infer_orders.py --config "$CONFIG"

echo ""
echo "========================================"
echo "方案A训练完成!"
echo "========================================"
echo "输出目录: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null | tail -5
