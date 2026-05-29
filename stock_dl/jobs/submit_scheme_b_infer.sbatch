#!/bin/bash
# ============================================================================
# 服务器提交脚本 - 方案B每日推理
# ============================================================================

# 设置环境
export PATH="$HOME/miniforge3/bin:$PATH"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dl-homework

cd /home/scc/pb23061103/finalex/finalex/stock_dl
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# ============================================================================
# 方案B: 深度学习 + LightGBM 每日推理
# ============================================================================
#SBATCH --job-name=stock-dl-scheme-b-infer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -euo pipefail

echo "========================================"
echo "方案B: 深度学习 + LightGBM 每日推理"
echo "时间: $(date)"
echo "========================================"

# 配置
export CONFIG="configs/a_share_with_lgbm.yaml"
export OUTPUT_DIR="outputs_scheme_b"

# 检查模型是否存在
if [ ! -f "$OUTPUT_DIR/model.pt" ]; then
    echo "错误: 模型不存在 - $OUTPUT_DIR/model.pt"
    echo "请先运行训练: sbatch jobs/submit_scheme_b_train.sbatch"
    exit 1
fi

echo "配置: $CONFIG"
echo "输出: $OUTPUT_DIR"

# 构建最新数据面板
python scripts/01_build_panel.py --config "$CONFIG"

# 生成交易信号
python scripts/06_infer_orders.py --config "$CONFIG"

echo ""
echo "========================================"
echo "方案B每日推理完成!"
echo "========================================"
echo "交易信号:"
ls -la "$OUTPUT_DIR"/orders_*.csv 2>/dev/null | tail -1
