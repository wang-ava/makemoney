#!/usr/bin/env python3
"""
策略优化验证脚本
对比优化前后的策略表现

优化内容:
1. 基准仓位: 90% → 95%
2. 多指标持仓数量: 波动率 + IC趋势 + 置信度
3. 阈值触发换手: 分数差距>3%才换手
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.backtest.risk import attach_buyable_flag


def run_optimized_backtest(scores: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> dict:
    """运行优化后的回测"""
    strategy_cfg = cfg["strategy"].copy()

    # 优化1: 提高基准仓位
    strategy_cfg["base_position_ratio"] = 0.95  # 从0.9改到0.95

    # 优化2: 启用多指标持仓数量
    strategy_cfg["adaptive_hold_multi_indicator"] = True
    strategy_cfg["adaptive_vol_weight"] = 0.4
    strategy_cfg["adaptive_ic_weight"] = 0.3
    strategy_cfg["adaptive_conf_weight"] = 0.3

    # 优化3: 启用阈值触发换手
    strategy_cfg["score_gap_trigger"] = True
    strategy_cfg["score_gap_trigger_threshold"] = 0.03

    n_hold = strategy_cfg.get("n_hold", 30)
    k_trade = strategy_cfg.get("k_trade", 2)

    result = run_backtest(
        scores,
        prices,
        n_hold=n_hold,
        k_trade=k_trade,
        initial_cash=strategy_cfg["initial_cash"],
        cost_rate=strategy_cfg.get("cost_rate", 0.0003),
        slippage=strategy_cfg.get("slippage", 0.0005),
        use_long_short=strategy_cfg.get("use_long_short", False),
        short_ratio=strategy_cfg.get("short_ratio", 0.5),
        strategy_cfg=strategy_cfg,
        cash_reserve_ratio=strategy_cfg.get("cash_reserve_ratio", 0.0),
    )
    return result


def run_baseline_backtest(scores: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> dict:
    """运行原始策略（基准）"""
    strategy_cfg = cfg["strategy"].copy()

    # 原始配置
    strategy_cfg["base_position_ratio"] = 0.90
    strategy_cfg["adaptive_hold_multi_indicator"] = False
    strategy_cfg["score_gap_trigger"] = False

    n_hold = strategy_cfg.get("n_hold", 30)
    k_trade = strategy_cfg.get("k_trade", 2)

    result = run_backtest(
        scores,
        prices,
        n_hold=n_hold,
        k_trade=k_trade,
        initial_cash=strategy_cfg["initial_cash"],
        cost_rate=strategy_cfg.get("cost_rate", 0.0003),
        slippage=strategy_cfg.get("slippage", 0.0005),
        use_long_short=strategy_cfg.get("use_long_short", False),
        short_ratio=strategy_cfg.get("short_ratio", 0.5),
        strategy_cfg=strategy_cfg,
        cash_reserve_ratio=strategy_cfg.get("cash_reserve_ratio", 0.0),
    )
    return result


def compare_results(baseline: dict, optimized: dict) -> dict:
    """对比基准和优化的结果"""
    baseline_metrics = baseline.get("metrics", {})
    optimized_metrics = optimized.get("metrics", {})

    comparison = {
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "improvement": {}
    }

    for key in ["total_return", "annual_return", "sharpe", "max_drawdown", "turnover", "daily_win_rate"]:
        base_val = baseline_metrics.get(key, 0)
        opt_val = optimized_metrics.get(key, 0)
        if base_val != 0:
            change = (opt_val - base_val) / abs(base_val) * 100
        else:
            change = 0
        comparison["improvement"][key] = {
            "baseline": base_val,
            "optimized": opt_val,
            "change_pct": change
        }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="验证策略优化效果")
    parser.add_argument("--config", default=str(ROOT / "configs/local_scheme_a.yaml"))
    parser.add_argument("--scores", type=str, default=None, help="分数文件路径")
    parser.add_argument("--prices", type=str, default=None, help="价格文件路径")
    parser.add_argument("--output", default="optimization_comparison.json")
    args = parser.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    # 加载数据
    if args.scores:
        scores = pd.read_csv(args.scores)
    else:
        scores_path = out / "scores_20260518.csv"
        if not scores_path.exists():
            print(f"Error: {scores_path} not found. Please run inference first.")
            sys.exit(1)
        scores = pd.read_csv(scores_path)

    if args.prices:
        prices = pd.read_csv(args.prices)
    else:
        from src.data.dataset import load_panel
        panel = load_panel(out / "panel.parquet")
        price_cols = ["trade_date", "ts_code", "open", "close"]
        if "pct_chg" in panel.columns:
            price_cols.append("pct_chg")
        prices = panel[price_cols]

    scores["trade_date"] = scores["trade_date"].astype(str)
    prices["trade_date"] = prices["trade_date"].astype(str)

    # 过滤验证集
    scores = scores[(scores["trade_date"] > str(cfg["train_end"]))]
    prices = prices[(prices["trade_date"] > str(cfg["train_end"]))]

    scores = attach_buyable_flag(scores, prices, cfg["strategy"])

    print("=" * 60)
    print("运行基准策略回测...")
    baseline = run_baseline_backtest(scores, prices, cfg)

    print("=" * 60)
    print("运行优化策略回测...")
    optimized = run_optimized_backtest(scores, prices, cfg)

    # 对比结果
    print("=" * 60)
    print("\n对比结果:")
    print("-" * 60)
    comparison = compare_results(baseline, optimized)

    metrics_display = {
        "total_return": "总收益",
        "annual_return": "年化收益",
        "sharpe": "夏普比率",
        "max_drawdown": "最大回撤",
        "turnover": "换手率",
        "daily_win_rate": "日胜率"
    }

    for key, label in metrics_display.items():
        imp = comparison["improvement"].get(key, {})
        base = imp.get("baseline", 0)
        opt = imp.get("optimized", 0)
        change = imp.get("change_pct", 0)

        if key == "max_drawdown":
            # 回撤是负数，越大越好（接近0）
            sign = "+" if opt > base else ""
            print(f"{label:10s}: 基准 {base:.2%} → 优化 {opt:.2%} ({sign}{opt-base:.2%})")
        elif key in ["total_return", "annual_return", "sharpe", "daily_win_rate"]:
            print(f"{label:10s}: 基准 {base:.2%} → 优化 {opt:.2%} ({change:+.1f}%)")
        else:
            print(f"{label:10s}: 基准 {base:.2f} → 优化 {opt:.2f}")

    # 保存结果
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
