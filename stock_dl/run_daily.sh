#!/bin/bash
# 一键运行每日交易信号生成
# 用法: ./run_daily.sh 或者 ./run_daily.sh "000001.SZ,600016.SH,..."

set -e

# 配置
SCHEME="a"  # a = 纯DL, b = DL+LGBM
PORTFOLIO_VALUE=1000000  # 组合总价值
CONFIG_FILE="configs/local_scheme_${SCHEME}.yaml"
OUTPUT_DIR="outputs_scheme_${SCHEME}"

# 获取持仓参数
HOLDINGS="${1:-}"

echo "=========================================="
echo "股票量化交易 - 每日信号生成"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "方案: ${SCHEME}"
echo "组合价值: ${PORTFOLIO_VALUE} 元"
if [ -n "$HOLDINGS" ]; then
    echo "持仓: ${HOLDINGS}"
else
    echo "持仓: (空，建仓日)"
fi
echo "=========================================="

# 运行推理
echo ""
echo ">>> 运行推理..."
python scripts/06_infer_orders.py \
    --config ${CONFIG_FILE} \
    --holdings "${HOLDINGS}" \
    --portfolio-value ${PORTFOLIO_VALUE} 2>&1

# 找到最新的输出文件
LATEST_ORDER=$(ls -t ${OUTPUT_DIR}/order_details_*.csv 2>/dev/null | head -1)

if [ -n "$LATEST_ORDER" ]; then
    echo ""
    echo ">>> 生成交易指南..."
    python scripts/14_generate_trading_guide.py \
        --scheme ${SCHEME} \
        --portfolio-value ${PORTFOLIO_VALUE}

    echo ""
    echo "=========================================="
    echo "✅ 完成！"
    echo "=========================================="
    echo "交易指南: ${OUTPUT_DIR}/trading_guide.txt"
    echo ""
else
    echo ""
    echo "❌ 未找到输出文件"
    exit 1
fi
