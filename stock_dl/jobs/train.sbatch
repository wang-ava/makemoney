#!/bin/bash
# ============================================================================
# 股票量化交易训练流水线 - 双方案一键运行
# ============================================================================
#
# 用法（单命令运行两个方案）:
#   cd /home/scc/pb23061103/finalex/finalex/stock_dl
#   bash jobs/train.sbatch
#
# 或分开运行:
#   SCHEME=a bash jobs/train.sbatch    # 只跑方案A
#   SCHEME=b bash jobs/train.sbatch    # 只跑方案B
#
# ============================================================================
#SBATCH --job-name=stock-dl-both
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-dl-homework}"
SCHEME="${SCHEME:-both}"  # both, a, b

cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export WANDB_RESUME=allow

set +u
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "=========================================="
echo "股票量化交易训练流水线"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "方案: $SCHEME"
echo "=========================================="

python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY

# ============================================================================
# 方案A: 纯深度学习 (GRU + Transformer)
# ============================================================================
run_scheme_a() {
    echo ""
    echo "=========================================="
    echo ">>> 方案A: 纯深度学习 (GRU + Transformer)"
    echo "=========================================="

    export CONFIG="configs/a_share_pure_dl.yaml"
    export OUTPUT_DIR="outputs_scheme_a"

    echo "构建数据面板..."
    python scripts/01_build_panel.py --config "$CONFIG"

    echo "训练深度学习模型..."
    python scripts/03_train.py --config "$CONFIG"

    echo "评估IC..."
    python scripts/04_eval_ic.py --config "$CONFIG"

    echo "回测..."
    python scripts/05_backtest.py --config "$CONFIG"

    echo "生成交易信号..."
    python scripts/06_infer_orders.py --config "$CONFIG"

    echo "方案A完成! 输出: $OUTPUT_DIR"
}

# ============================================================================
# 方案B: 深度学习 + LightGBM (GRU + Transformer + LightGBM)
# ============================================================================
run_scheme_b() {
    echo ""
    echo "=========================================="
    echo ">>> 方案B: 深度学习 + LightGBM"
    echo "=========================================="

    export CONFIG="configs/a_share_with_lgbm.yaml"
    export OUTPUT_DIR="outputs_scheme_b"

    echo "构建数据面板..."
    python scripts/01_build_panel.py --config "$CONFIG"

    echo "训练深度学习模型..."
    python scripts/03_train.py --config "$CONFIG"

    echo "训练LightGBM..."
    python scripts/10_train_lgbm.py --config "$CONFIG"

    echo "集成分数..."
    python scripts/11_blend_scores.py --config "$CONFIG"

    echo "评估IC..."
    python scripts/04_eval_ic.py --config "$CONFIG"

    echo "回测..."
    python scripts/05_backtest.py --config "$CONFIG"

    echo "生成交易信号..."
    python scripts/06_infer_orders.py --config "$CONFIG"

    echo "方案B完成! 输出: $OUTPUT_DIR"
}

# ============================================================================
# 执行
# ============================================================================
case "$SCHEME" in
    both|a|A)
        run_scheme_a
        ;;
esac

case "$SCHEME" in
    both|b|B)
        run_scheme_b
        ;;
esac

echo ""
echo "=========================================="
echo "全部完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "方案A输出: outputs_scheme_a/"
echo "方案B输出: outputs_scheme_b/"
echo ""
echo "查看交易信号:"
ls -la outputs_scheme_a/orders_*.csv 2>/dev/null | tail -1
ls -la outputs_scheme_b/orders_*.csv 2>/dev/null | tail -1
