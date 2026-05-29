#!/bin/bash
# 方案A: 纯深度学习 (GRU + Transformer)
# 不使用LightGBM

set -euo pipefail

cd "$(dirname "$0")"

echo "=============================================="
echo "方案A: 纯深度学习 (GRU + Transformer)"
echo "=============================================="

# 配置
CONFIG="configs/a_share_pure_dl.yaml"
SCHEME_NAME="scheme_a_pure_dl"

# 设置环境变量
export PYTHONPATH="$PWD"
export SKIP_LGBM=1
export WANDB_GROUP="finalex-${SCHEME_NAME}"

echo "Config: $CONFIG"
echo "Output: outputs_scheme_a"
echo "SKIP_LGBM: $SKIP_LGBM"

# 创建必要的目录
mkdir -p logs
mkdir -p outputs_scheme_a

# 检查GPU
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "=============================================="
echo "Step 1: 构建数据面板"
echo "=============================================="
python scripts/01_build_panel.py --config "$CONFIG"

echo ""
echo "=============================================="
echo "Step 2: 训练深度学习模型 (GRU-Transformer)"
echo "=============================================="
python scripts/03_train.py --config "$CONFIG"

echo ""
echo "=============================================="
echo "Step 3: 评估IC"
echo "=============================================="
python scripts/04_eval_ic.py --config "$CONFIG"

echo ""
echo "=============================================="
echo "Step 4: 回测"
echo "=============================================="
python scripts/05_backtest.py --config "$CONFIG"

echo ""
echo "=============================================="
echo "Step 5: 生成交易信号"
echo "=============================================="
python scripts/06_infer_orders.py --config "$CONFIG"

echo ""
echo "=============================================="
echo "方案A完成!"
echo "=============================================="
echo "结果保存在: outputs_scheme_a/"
ls -la outputs_scheme_a/*.csv 2>/dev/null || echo "No CSV files found"
