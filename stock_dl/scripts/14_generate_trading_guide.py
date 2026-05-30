#!/usr/bin/env python3
"""
生成优化的交易建议 - 更清晰的输出格式
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def generate_trading_guide(order_details_path: Path, output_path: Path, portfolio_value: float = 1000000):
    """生成清晰的交易指南"""

    df = pd.read_csv(order_details_path)

    # 从文件名获取日期
    import re
    match = re.search(r'(\d{8})', order_details_path.name)
    data_date = match.group(1) if match else "unknown"

    # 过滤买入的股票
    buy_df = df[df["side"] == "buy"].copy()

    # 计算总买入金额
    total_buy = buy_df["target_amount"].sum()
    target_position = buy_df["target_position_ratio"].iloc[0] if len(buy_df) > 0 else 0.947

    print(f"\n{'='*80}")
    print(f"📊 5.31 股票交易指南")
    print(f"{'='*80}")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据日期: {data_date}")
    print(f"{'='*80}")

    print(f"\n📈 策略参数:")
    print(f"   • 持仓数量: {len(buy_df)} 只")
    print(f"   • 目标仓位: {target_position:.2%}")
    print(f"   • 投入资金: {portfolio_value * target_position:,.0f} 元")
    print(f"   • 每只股票: 按分数加权分配")

    print(f"\n{'='*80}")
    print(f"🛒 买入清单 ({len(buy_df)} 只)")
    print(f"{'='*80}")
    print(f"{'序号':>4} {'股票代码':>12} {'参考价格':>10} {'目标金额':>12} {'买入股数':>10} {'权重':>8}")
    print("-" * 80)

    for i, (_, row) in enumerate(buy_df.iterrows(), 1):
        print(f"{i:>4} {row['ts_code']:>12} {row['reference_price']:>10.2f} {row['target_amount']:>12,.0f} {int(row['estimated_shares']):>10} {row['weight_ratio']:>8.2%}")

    print(f"\n{'='*80}")
    print(f"💰 资金汇总:")
    print(f"   • 总目标金额: {total_buy:,.0f} 元")
    print(f"   • 实际可用: {portfolio_value * target_position:,.0f} 元")
    print(f"   • 差额: {portfolio_value * target_position - total_buy:,.0f} 元 (备用金)")
    print(f"{'='*80}")

    # 生成同花顺格式
    print(f"\n{'='*80}")
    print(f"📋 同花顺买入格式")
    print(f"{'='*80}")
    print(f"证券代码        买入价格      买入数量")
    print("-" * 80)
    for _, row in buy_df.iterrows():
        print(f"{row['ts_code']:<14} {row['reference_price']:>10.2f} {int(row['estimated_shares']):>10}")

    # 保存到文件
    guide_text = f"""
================================================================================
                        5.31 股票交易指南
================================================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据日期: {data_date}

--------------------------------------------------------------------------------
策略参数
--------------------------------------------------------------------------------
持仓数量: {len(buy_df)} 只
目标仓位: {target_position:.2%}
投入资金: {portfolio_value * target_position:,.0f} 元
资金分配: 按分数加权 (高分股票买入更多)

--------------------------------------------------------------------------------
买入清单 ({len(buy_df)} 只)
--------------------------------------------------------------------------------
{'序号':>4} {'股票代码':>12} {'参考价格':>10} {'目标金额':>12} {'买入股数':>10} {'权重':>8}
{'='*80}
"""
    for i, (_, row) in enumerate(buy_df.iterrows(), 1):
        guide_text += f"{i:>4} {row['ts_code']:>12} {row['reference_price']:>10.2f} {row['target_amount']:>12,.0f} {int(row['estimated_shares']):>10} {row['weight_ratio']:>8.2%}\n"

    guide_text += f"""
--------------------------------------------------------------------------------
资金汇总
--------------------------------------------------------------------------------
总目标金额: {total_buy:,.0f} 元
实际可用: {portfolio_value * target_position:,.0f} 元
差额(备用金): {portfolio_value * target_position - total_buy:,.0f} 元

--------------------------------------------------------------------------------
同花顺买入格式
--------------------------------------------------------------------------------
证券代码        买入价格      买入数量
{'='*80}
"""
    for _, row in buy_df.iterrows():
        guide_text += f"{row['ts_code']:<14} {row['reference_price']:>10.2f} {int(row['estimated_shares']):>10}\n"

    guide_text += f"""
================================================================================
注意事项:
1. 参考价格为5.29收盘价，31日开盘价可能略有不同
2. 可使用"市价买入"自动成交
3. 资金不够时，优先购买排名靠前的股票
================================================================================
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(guide_text)

    print(f"\n✅ 交易指南已保存: {output_path}")

    return buy_df


def main():
    parser = argparse.ArgumentParser(description="生成优化的交易指南")
    parser.add_argument("--scheme", choices=["a", "b"], default="a", help="方案A或方案B")
    parser.add_argument("--portfolio-value", type=float, default=1000000, help="组合总价值")
    parser.add_argument("--output", default=None, help="输出文件")
    args = parser.parse_args()

    scheme = "scheme_a" if args.scheme == "a" else "scheme_b"
    cfg_path = ROOT / "configs" / f"local_{scheme}.yaml"
    cfg = load_config(str(cfg_path))
    out = Path(cfg["output_dir"])

    # 找到最新的order_details文件
    order_files = sorted(out.glob("order_details_*.csv"))
    if not order_files:
        print(f"Error: No order_details files found in {out}")
        sys.exit(1)

    latest_order = order_files[-1]
    output_path = Path(args.output) if args.output else out / "trading_guide.txt"

    print(f"Reading from: {latest_order}")
    generate_trading_guide(latest_order, output_path, args.portfolio_value)


if __name__ == "__main__":
    main()
