#!/usr/bin/env python3
"""
一键运行每日交易信号生成
用法:
    python run_daily.py                    # 建仓日（无持仓）
    python run_daily.py "000001.SZ,600016.SH"  # 已有持仓
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_command(cmd: list[str], desc: str) -> bool:
    """运行命令并打印输出"""
    print(f"\n>>> {desc}...")
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="一键运行每日交易信号")
    parser.add_argument(
        "--holdings",
        default="",
        help="当前持仓股票代码，逗号分隔，如 000001.SZ,600016.SH；建仓日留空"
    )
    parser.add_argument(
        "--scheme",
        choices=["a", "b"],
        default="a",
        help="方案: a=纯深度学习, b=DL+LightGBM"
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=1000000,
        help="组合总价值（默认100万）"
    )
    args = parser.parse_args()

    scheme = args.scheme
    portfolio_value = args.portfolio_value
    holdings = args.holdings.strip().strip('"').strip("'")

    print("=" * 60)
    print("📊 股票量化交易 - 每日信号生成")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"方案: {'A (纯深度学习)' if scheme == 'a' else 'B (DL+LightGBM)'}")
    print(f"组合价值: {portfolio_value:,.0f} 元")
    if holdings:
        print(f"持仓数量: {len(holdings.split(','))} 只")
        print(f"持仓: {holdings[:50]}{'...' if len(holdings) > 50 else ''}")
    else:
        print("持仓: (空，建仓日)")
    print("=" * 60)

    config_file = f"configs/local_scheme_{scheme}.yaml"
    output_dir = f"outputs_scheme_{scheme}"

    # 1. 运行推理
    cmd_infer = [
        sys.executable,
        "scripts/06_infer_orders.py",
        "--config", config_file,
        "--holdings", holdings,
        "--portfolio-value", str(portfolio_value)
    ]

    if not run_command(cmd_infer, "运行推理"):
        print("❌ 推理失败")
        sys.exit(1)

    # 2. 生成交易指南
    cmd_guide = [
        sys.executable,
        "scripts/14_generate_trading_guide.py",
        "--scheme", scheme,
        "--portfolio-value", str(portfolio_value)
    ]

    if not run_command(cmd_guide, "生成交易指南"):
        print("❌ 生成指南失败")
        sys.exit(1)

    # 3. 显示下一步指引
    output_path = ROOT / output_dir / "trading_guide.txt"
    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("=" * 60)
    print(f"\n📄 交易指南: {output_path}")
    print("\n📋 下一步:")
    print("  1. 打开 trading_guide.txt 查看详细交易清单")
    print("  2. 登录同花顺，按清单买入/卖出")
    print("  3. 下次运行时，将持仓代码传入 --holdings 参数")

    # 提取买入股票代码供下次参考
    print("\n💡 下次运行命令（复制保存）:")
    order_file = sorted((ROOT / output_dir).glob("order_details_*.csv"))
    if order_file:
        holdings_str = ",".join([
            line.strip().split(",")[1]
            for line in open(order_file[-1], encoding="utf-8")
            if line.startswith("buy,") or line.startswith("hold,")
        ])
        print(f"\n  python run_daily.py --holdings \"{holdings_str}\" --scheme {scheme} --portfolio-value {portfolio_value}")


if __name__ == "__main__":
    main()
