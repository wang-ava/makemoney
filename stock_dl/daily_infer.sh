#!/bin/bash
# 每日推理脚本 - 用于比赛提交
# 用法: ./daily_infer.sh [scheme_a|scheme_b]
# 示例: ./daily_infer.sh scheme_a

set -euo pipefail

SCHEME="${1:-scheme_b}"

echo "=============================================="
echo "每日推理脚本"
echo "当前方案: $SCHEME"
echo "运行时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

cd "$(dirname "$0")"

# 设置环境
export PYTHONPATH="$PWD"

case "$SCHEME" in
    scheme_a)
        echo ">>> 使用方案A: 纯深度学习 (GRU + Transformer)"
        CONFIG="configs/a_share_pure_dl.yaml"
        OUTPUT_DIR="outputs_scheme_a"
        ;;
    scheme_b)
        echo ">>> 使用方案B: 深度学习 + LightGBM (GRU + Transformer + LightGBM)"
        CONFIG="configs/a_share_with_lgbm.yaml"
        OUTPUT_DIR="outputs_scheme_b"
        ;;
    *)
        echo "错误: 未知的方案 '$SCHEME'"
        echo "用法: $0 [scheme_a|scheme_b]"
        exit 1
        ;;
esac

echo ""
echo "配置: $CONFIG"
echo "输出: $OUTPUT_DIR"
echo ""

# 检查模型文件是否存在
if [ ! -f "$OUTPUT_DIR/model.pt" ]; then
    echo "错误: 模型文件不存在 - $OUTPUT_DIR/model.pt"
    echo ""
    echo "请先完成模型训练:"
    echo "  方案A: ./run_scheme_a.sh"
    echo "  方案B: ./run_scheme_b.sh"
    exit 1
fi

echo ">>> Step 1: 构建最新数据面板"
python scripts/01_build_panel.py --config "$CONFIG"

echo ""
echo ">>> Step 2: 生成交易信号"
python scripts/06_infer_orders.py --config "$CONFIG"

echo ""
echo "=============================================="
echo "每日推理完成!"
echo "=============================================="
echo ""
echo "交易信号文件:"
ls -la "$OUTPUT_DIR"/orders_*.csv 2>/dev/null | tail -1
echo ""
echo "评分文件:"
ls -la "$OUTPUT_DIR"/scores_*.csv 2>/dev/null | tail -1
echo ""
echo "请根据 orders_YYYYMMDD.csv 在同花顺模拟盘执行交易"
