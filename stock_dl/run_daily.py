#!/usr/bin/env python3
"""
一键运行每日交易信号生成
用法:
    python run_daily.py                    # 建仓日（无持仓）
    python run_daily.py --holdings "000001.SZ:1000,600016.SH:500"  # 已有持仓，数量为股数
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


def _clean_holdings(raw: str) -> str:
    return raw.strip().strip('"').strip("'")


def _holdings_codes_only(holdings: str) -> str:
    codes = []
    for item in holdings.split(","):
        item = item.strip()
        if not item:
            continue
        codes.append(item.split(":", 1)[0].strip())
    return ",".join(codes)


def run_scheme(
    scheme: str,
    holdings: str,
    holdings_unit: str,
    portfolio_value: float,
) -> None:
    config_file = f"configs/local_scheme_{scheme}.yaml"
    output_dir = f"outputs_scheme_{scheme}"
    scheme_name = "方案A(纯DL)" if scheme == "a" else "方案B(DL+LGBM)"

    print("=" * 60)
    print("📊 股票量化交易 - 每日信号生成")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"方案: {scheme_name}")
    print(f"组合价值: {portfolio_value:,.0f} 元")
    if holdings:
        print(f"持仓数量: {len(holdings.split(','))} 只")
        print(f"持仓: {holdings[:50]}{'...' if len(holdings) > 50 else ''}")
    else:
        print("持仓: (空，建仓日)")
    print("=" * 60)

    # 1. 运行推理。06 脚本只需要代码列表，真正的股数/手数交给 14 脚本生成最终订单。
    cmd_infer = [
        sys.executable,
        "scripts/06_infer_orders.py",
        "--config", config_file,
        "--holdings", _holdings_codes_only(holdings),
        "--portfolio-value", str(portfolio_value),
    ]

    if not run_command(cmd_infer, f"运行推理 {scheme_name}"):
        print("❌ 推理失败")
        sys.exit(1)

    # 2. 生成交易指南。14 脚本已锁定历史最优交易策略，A/B 仅模型分数不同。
    cmd_guide = [
        sys.executable,
        "scripts/14_generate_trading_guide.py",
        "--config", config_file,
        "--scheme-name", scheme_name,
        "--portfolio-value", str(portfolio_value),
        "--holdings-unit", holdings_unit,
    ]
    if holdings:
        cmd_guide.extend(["--holdings", holdings])

    if not run_command(cmd_guide, f"生成交易指南 {scheme_name}"):
        print("❌ 生成指南失败")
        sys.exit(1)

    guide_root = ROOT / output_dir / "trading_guides"
    latest_dirs = sorted([p for p in guide_root.glob("*") if p.is_dir()])
    output_path = latest_dirs[-1] / "trading_guide.txt" if latest_dirs else guide_root

    print("\n" + "=" * 60)
    print(f"✅ {scheme_name} 完成!")
    print("=" * 60)
    print(f"\n📄 交易指南: {output_path}")
    print(f"📄 下单 CSV: {output_path.parent / 'trading_orders.csv'}")


def main():
    parser = argparse.ArgumentParser(description="一键运行每日交易信号")
    parser.add_argument(
        "--holdings",
        default="",
        help="当前持仓，格式 代码:股数,代码:股数；建仓日留空"
    )
    parser.add_argument(
        "--holdings-unit",
        choices=["shares", "hands"],
        default="shares",
        help="持仓数量单位：shares=股数（默认），hands=手数"
    )
    parser.add_argument(
        "--scheme",
        choices=["a", "b", "both"],
        default="a",
        help="方案: a=纯深度学习, b=DL+LightGBM, both=同时保留A/B每日结果"
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=1000000,
        help="组合总价值（默认100万）"
    )
    parser.add_argument("--holdings-a", default=None, help="方案A当前持仓；不填则使用 --holdings")
    parser.add_argument("--holdings-b", default=None, help="方案B当前持仓；不填则使用 --holdings")
    parser.add_argument("--portfolio-value-a", type=float, default=None, help="方案A组合总价值；不填则使用 --portfolio-value")
    parser.add_argument("--portfolio-value-b", type=float, default=None, help="方案B组合总价值；不填则使用 --portfolio-value")
    args = parser.parse_args()

    scheme = args.scheme
    holdings = _clean_holdings(args.holdings)
    holdings_unit = args.holdings_unit

    if scheme == "both":
        run_scheme(
            "a",
            _clean_holdings(args.holdings_a) if args.holdings_a is not None else holdings,
            holdings_unit,
            args.portfolio_value_a if args.portfolio_value_a is not None else args.portfolio_value,
        )
        run_scheme(
            "b",
            _clean_holdings(args.holdings_b) if args.holdings_b is not None else holdings,
            holdings_unit,
            args.portfolio_value_b if args.portfolio_value_b is not None else args.portfolio_value,
        )
    else:
        run_scheme(scheme, holdings, holdings_unit, args.portfolio_value)


if __name__ == "__main__":
    main()
