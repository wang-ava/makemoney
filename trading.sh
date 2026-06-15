#!/bin/bash
#===============================================================================
# 同花顺手动下单 - 一键运行脚本
#
# 用法:
#   ./trading.sh                           # 交互式输入
#   ./trading.sh a "000001.SZ:1000,600000.SH:500" 1000000  # 直接运行，数量为股数
#   ./trading.sh a "000001.SZ:10,600000.SH:5" 1000000 hands  # 历史手数口径
#
# 持仓格式: 代码:股数,代码:股数,代码:股数
# 示例:   "000001.SZ:1000,600000.SH:500,600519.SH:50"
#===============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_DIR="$SCRIPT_DIR"
CONFIG_DIR="$PROJECT_DIR/stock_dl/configs"
SCRIPT_FILE="$PROJECT_DIR/stock_dl/scripts/14_generate_trading_guide.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  同花顺手动下单交易指南生成器"
echo "=============================================="

# 解析参数
SCHEME="${1:-}"
HOLDINGS="${2:-}"
PORTFOLIO="${3:-1000000}"
HOLDINGS_UNIT="${4:-shares}"

# 如果没有参数，显示交互式菜单
if [ -z "$SCHEME" ]; then
    echo ""
    echo "请选择方案:"
    echo "  1) 方案A - 纯深度学习 (GRU-Transformer)"
    echo "  2) 方案B - DL+LGBM融合"
    echo ""
    read -p "请输入选择 (1/2) [1]: " scheme_choice
    scheme_choice="${scheme_choice:-1}"

    if [ "$scheme_choice" = "1" ]; then
        SCHEME="a"
        SCHEME_NAME="方案A(纯DL)"
        CONFIG_FILE="server_8h_scheme_a_short.yaml"
        OUTPUT_DIR="outputs_server8h_scheme_a_short"
    else
        SCHEME="b"
        SCHEME_NAME="方案B(DL+LGBM)"
        CONFIG_FILE="server_8h_scheme_b_short.yaml"
        OUTPUT_DIR="outputs_server8h_scheme_b_short"
    fi

    echo ""
    echo "请输入持仓 (格式: 代码:股数,代码:股数；同花顺持仓页显示的数量可直接填)"
    echo "示例: 000001.SZ:1000,600000.SH:500,600519.SH:100"
    echo "无持仓请直接回车:"
    read -r HOLDINGS

    echo ""
    read -p "持仓数量单位 shares=股数 / hands=手数 [shares]: " HOLDINGS_UNIT
    HOLDINGS_UNIT="${HOLDINGS_UNIT:-shares}"

    echo ""
    read -p "请输入账户总资产 [1000000]: " PORTFOLIO
    PORTFOLIO="${PORTFOLIO:-1000000}"

else
    # 命令行参数模式
    if [ "$SCHEME" = "a" ]; then
        SCHEME_NAME="方案A(纯DL)"
        CONFIG_FILE="server_8h_scheme_a_short.yaml"
        OUTPUT_DIR="outputs_server8h_scheme_a_short"
    elif [ "$SCHEME" = "b" ]; then
        SCHEME_NAME="方案B(DL+LGBM)"
        CONFIG_FILE="server_8h_scheme_b_short.yaml"
        OUTPUT_DIR="outputs_server8h_scheme_b_short"
    else
        echo -e "${RED}错误: 方案必须是 'a' 或 'b'${NC}"
        exit 1
    fi
fi

CONFIG_PATH="$CONFIG_DIR/$CONFIG_FILE"

# 检查文件是否存在
if [ ! -f "$SCRIPT_FILE" ]; then
    echo -e "${RED}错误: 脚本不存在: $SCRIPT_FILE${NC}"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}错误: 配置不存在: $CONFIG_PATH${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}运行参数:${NC}"
echo "  方案: $SCHEME_NAME"
echo "  配置: configs/$CONFIG_FILE"
echo "  输出: $OUTPUT_DIR"
echo "  持仓: ${HOLDINGS:-无}"
echo "  持仓单位: $HOLDINGS_UNIT"
echo "  资产: ${PORTFOLIO} 元"
echo ""

# 构建命令
CMD="cd $PROJECT_DIR/stock_dl && python scripts/14_generate_trading_guide.py \
    --config configs/$CONFIG_FILE \
    --scheme-name \"$SCHEME_NAME\" \
    --portfolio-value $PORTFOLIO \
    --holdings-unit $HOLDINGS_UNIT \
    --allocation score_weighted"

if [ -n "$HOLDINGS" ]; then
    CMD="$CMD --holdings \"$HOLDINGS\""
fi

# 执行
echo -e "${YELLOW}开始运行...${NC}"
echo ""
eval $CMD

echo ""
echo -e "${GREEN}=============================================="
echo "  运行完成!"
echo -e "==============================================${NC}"
echo ""
echo "下一步操作:"
echo "1. 查看生成的交易指南:"
echo "   cat stock_dl/$OUTPUT_DIR/trading_guides/$(date +%Y%m%d)/trading_guide.txt"
echo "2. 在同花顺模拟账户中按指南下单"
echo "3. 收盘后记录实际成交"
echo ""
