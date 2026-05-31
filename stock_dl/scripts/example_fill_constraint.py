#!/usr/bin/env python3
"""
成交量约束回测示例

演示如何使用:
1. 历史成交率估算
2. 带成交约束的回测
3. 智能下单执行器

用法:
    python scripts/example_fill_constraint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.fill_rate import (
    estimate_daily_volume_stats,
    estimate_fill_rate,
    estimate_fill_rate_by_amount,
    get_market_condition,
    MarketCondition,
    FillRateConfig,
    SmartExecutor,
)
from src.backtest.fill_constraint_backtest import run_backtest_with_fill_constraint
from src.config import load_config
from src.data.dataset import load_panel


def example_fill_rate_estimation():
    """示例1: 成交率估算"""
    print("\n" + "="*60)
    print("示例1: 成交率估算")
    print("="*60)

    # 示例: 估算不同参与率的成交率
    scenarios = [
        # (目标股数, 日均成交量, 股票类型)
        (10000, 1_000_000, "大盘蓝筹"),
        (100000, 1_000_000, "中盘股"),
        (500000, 1_000_000, "小盘股"),
        (10000, 100_000, "微盘股"),
    ]

    print(f"\n{'股票类型':<10} {'目标股数':>12} {'日均成交量':>15} {'参与率':>10} {'成交率':>10}")
    print("-" * 60)

    for target_shares, daily_vol, stock_type in scenarios:
        participation = target_shares / daily_vol

        for condition in [MarketCondition.HOT, MarketCondition.NORMAL]:
            fill_rate = estimate_fill_rate(
                target_shares=target_shares,
                daily_volume=daily_vol,
                price=10.0,  # 假设价格10元
                market_condition=condition,
            )

            if condition == MarketCondition.NORMAL:
                print(f"{stock_type:<10} {target_shares:>12,} {daily_vol:>15,} {participation:>10.1%} {fill_rate:>10.1%}")


def example_fill_rate_from_panel():
    """示例2: 从历史数据计算成交率"""
    print("\n" + "="*60)
    print("示例2: 从历史数据计算日均成交量")
    print("="*60)

    # 这个需要实际数据才能运行
    # 以下是代码示例

    """
    # 1. 加载panel数据
    panel = load_panel(Path("outputs/panel.parquet"))

    # 2. 计算每只股票的日均成交量
    volume_stats = estimate_daily_volume_stats(panel, window=20)

    # 3. 查看结果
    print(volume_stats.head(10))

    # 4. 根据目标金额估算成交率
    for _, row in volume_stats.head(5).iterrows():
        code = row['ts_code']
        avg_amount = row['avg_amount']

        fill_rate = estimate_fill_rate_by_amount(
            target_amount=1_000_000,  # 目标买入100万
            avg_daily_amount=avg_amount,
            market_condition=MarketCondition.NORMAL,
        )
        print(f"{code}: 日均成交额 {avg_amount/1e6:.1f}M, 成交率: {fill_rate:.1%}")
    """


def example_backtest_with_constraint():
    """示例3: 带成交约束的回测"""
    print("\n" + "="*60)
    print("示例3: 带成交约束的回测")
    print("="*60)

    """
    # 这个需要实际数据和模型才能运行
    # 以下是代码示例

    # 1. 加载数据和配置
    cfg = load_config("configs/quick.yaml")
    pred = pd.read_csv(cfg["output_dir"] / "val_predictions.csv")
    panel = load_panel(cfg["output_dir"] / "panel.parquet")

    # 2. 运行带成交约束的回测
    result = run_backtest_with_fill_constraint(
        scores=pred,
        prices=panel[["trade_date", "ts_code", "open", "close", "vol", "amount", "pct_chg"]],
        panel=panel,
        n_hold=50,
        k_trade=1,
        initial_cash=1_000_000,
        use_fill_constraint=True,
        volume_window=20,
    )

    # 3. 查看结果
    print("\n回测指标:")
    for key, value in result["metrics"].items():
        if key != "fill_rate_stats":
            print(f"  {key}: {value}")

    print("\n成交约束统计:")
    for key, value in result["metrics"]["fill_rate_stats"].items():
        print(f"  {key}: {value}")
    """


def example_smart_executor():
    """示例4: 智能下单执行器"""
    print("\n" + "="*60)
    print("示例4: 智能下单执行器")
    print("="*60)

    # 创建执行器
    executor = SmartExecutor(
        config={
            "max_retry": 3,
            "price_improvement": 0.002,  # 0.2%
            "wait_seconds": 30,
            "simulation_mode": True,  # 模拟模式
        }
    )

    # 模拟执行几个订单
    from src.backtest.fill_rate import estimate_fill_rate_by_amount, MarketCondition
    from dataclasses import dataclass

    @dataclass
    class MockOrder:
        ts_code: str
        side: str
        price: float
        shares: int

    orders = [
        MockOrder(ts_code="000001.SZ", side="buy", price=10.5, shares=10000),
        MockOrder(ts_code="000002.SZ", side="buy", price=8.2, shares=50000),
        MockOrder(ts_code="600000.SH", side="sell", price=12.0, shares=20000),
    ]

    print("\n模拟执行订单:")
    for order in orders:
        # 使用成交率估算模拟
        avg_daily_amount = 1_000_000_000  # 假设日均成交10亿
        target_amount = order.shares * order.price
        fill_rate = estimate_fill_rate_by_amount(
            target_amount=target_amount,
            avg_daily_amount=avg_daily_amount,
            market_condition=MarketCondition.NORMAL,
        )
        filled_shares = int(order.shares * fill_rate)
        avg_price = order.price * (1.001 if order.side == "buy" else 0.999)

        print(f"  {order.side.upper()} {order.ts_code}: "
              f"目标{order.shares:,}股, "
              f"成交{filled_shares:,}股 ({fill_rate:.1%}), "
              f"均价{avg_price:.2f}")

    # 打印摘要
    print(f"\n执行摘要:")
    print(f"  总订单数: {len(orders)}")
    print(f"  整体成交率: 约75-90% (取决于股票流动性)")


def main():
    print("="*60)
    print("成交量约束交易示例")
    print("="*60)

    # 示例1: 成交率估算
    example_fill_rate_estimation()

    # 示例2: 从历史数据计算 (需要数据)
    # example_fill_rate_from_panel()

    # 示例3: 带约束回测 (需要数据)
    # example_backtest_with_constraint()

    # 示例4: 智能执行器
    example_smart_executor()

    print("\n" + "="*60)
    print("运行完成!")
    print("="*60)


if __name__ == "__main__":
    main()
