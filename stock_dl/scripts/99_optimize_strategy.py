#!/usr/bin/env python3
"""
策略参数调优脚本
系统化测试不同策略参数组合，生成优化报告
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.config import load_config
from src.data.dataset import load_panel


def strategy_objective(metrics: dict, turnover_penalty: float = 0.01) -> float:
    return (
        float(metrics.get("sharpe", 0.0))
        + 2.0 * float(metrics.get("total_return", 0.0))
        + float(metrics.get("max_drawdown", 0.0))
        - turnover_penalty * float(metrics.get("turnover", 0.0))
    )


def run_single_backtest(scores, prices, cfg, n_hold, k_trade, strategy_overrides=None):
    """运行单次回测"""
    strategy = cfg["strategy"].copy()
    if strategy_overrides:
        strategy.update(strategy_overrides)

    result = run_backtest(
        scores,
        prices,
        n_hold=n_hold,
        k_trade=k_trade,
        initial_cash=strategy["initial_cash"],
        cost_rate=strategy.get("cost_rate", 0.0003),
        slippage=strategy.get("slippage", 0.0005),
        use_long_short=strategy.get("use_long_short", False),
        short_ratio=strategy.get("short_ratio", 0.5),
        strategy_cfg=strategy,
        cash_reserve_ratio=strategy.get("cash_reserve_ratio", 0.0),
    )
    return result.get("metrics", {})


def tune_n_hold_k_trade(scores, prices, cfg):
    """测试持仓数量和换手频率"""
    print("\n" + "="*60)
    print("实验1: n_hold 和 k_trade 参数调优")
    print("="*60)

    results = []
    n_hold_grid = [30, 50, 80, 100]
    k_trade_grid = [1, 2, 3, 5]

    for n_hold in n_hold_grid:
        for k_trade in k_trade_grid:
            if k_trade > n_hold:
                continue
            print(f"  Testing n_hold={n_hold}, k_trade={k_trade}...", end=" ")
            metrics = run_single_backtest(scores, prices, cfg, n_hold, k_trade)
            obj = strategy_objective(metrics)
            results.append({
                "n_hold": n_hold,
                "k_trade": k_trade,
                "total_return": metrics.get("total_return", 0),
                "annual_return": metrics.get("annual_return", 0),
                "sharpe": metrics.get("sharpe", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "turnover": metrics.get("turnover", 0),
                "objective": obj,
            })
            print(f"Return={metrics.get('total_return', 0):.2%}, Sharpe={metrics.get('sharpe', 0):.2f}")

    df = pd.DataFrame(results).sort_values("objective", ascending=False)
    return df


def tune_position_ratio(scores, prices, cfg, base_n_hold, base_k_trade):
    """测试仓位参数"""
    print("\n" + "="*60)
    print("实验2: 仓位参数调优")
    print("="*60)

    results = []
    configs = [
        {"name": "min_pos=0.7, base=0.85", "overrides": {"min_position_ratio": 0.7, "base_position_ratio": 0.85}},
        {"name": "min_pos=0.8, base=0.9", "overrides": {"min_position_ratio": 0.8, "base_position_ratio": 0.9}},
        {"name": "min_pos=0.9, base=0.95", "overrides": {"min_position_ratio": 0.9, "base_position_ratio": 0.95}},
        {"name": "dynamic_pos=False", "overrides": {"dynamic_position": False}},
        {"name": "force_80%=False", "overrides": {"force_80percent_position": False}},
    ]

    for cfg_test in configs:
        name = cfg_test["name"]
        overrides = cfg_test["overrides"]
        print(f"  Testing {name}...", end=" ")
        metrics = run_single_backtest(scores, prices, cfg, base_n_hold, base_k_trade, overrides)
        obj = strategy_objective(metrics)
        results.append({
            "config": name,
            "total_return": metrics.get("total_return", 0),
            "annual_return": metrics.get("annual_return", 0),
            "sharpe": metrics.get("sharpe", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "turnover": metrics.get("turnover", 0),
            "min_position_ratio": metrics.get("min_position_ratio", 0),
            "objective": obj,
        })
        print(f"Return={metrics.get('total_return', 0):.2%}, Sharpe={metrics.get('sharpe', 0):.2f}")

    df = pd.DataFrame(results).sort_values("objective", ascending=False)
    return df


def tune_risk_filters(scores, prices, cfg, base_n_hold, base_k_trade):
    """测试风控过滤器"""
    print("\n" + "="*60)
    print("实验3: 风控过滤器调优")
    print("="*60)

    results = []
    configs = [
        {"name": "baseline", "overrides": {}},
        {"name": "no_vol_filter", "overrides": {"max_volatility_quantile": 1.0}},
        {"name": "no_liq_filter", "overrides": {"min_amount_quantile": 0.0}},
        {"name": "no_momentum", "overrides": {"momentum_filter": False}},
        {"name": "strict_filters", "overrides": {"max_volatility_quantile": 0.9, "min_amount_quantile": 0.1}},
    ]

    for cfg_test in configs:
        name = cfg_test["name"]
        overrides = cfg_test["overrides"]
        print(f"  Testing {name}...", end=" ")
        metrics = run_single_backtest(scores, prices, cfg, base_n_hold, base_k_trade, overrides)
        obj = strategy_objective(metrics)
        results.append({
            "config": name,
            "total_return": metrics.get("total_return", 0),
            "annual_return": metrics.get("annual_return", 0),
            "sharpe": metrics.get("sharpe", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "turnover": metrics.get("turnover", 0),
            "limit_buy_blocked": metrics.get("limit_buy_blocked", 0),
            "limit_sell_blocked": metrics.get("limit_sell_blocked", 0),
            "objective": obj,
        })
        print(f"Return={metrics.get('total_return', 0):.2%}, Sharpe={metrics.get('sharpe', 0):.2f}")

    df = pd.DataFrame(results).sort_values("objective", ascending=False)
    return df


def tune_trade_control(scores, prices, cfg, base_n_hold, base_k_trade):
    """测试交易控制"""
    print("\n" + "="*60)
    print("实验4: 交易控制参数调优")
    print("="*60)

    results = []
    configs = [
        {"name": "baseline", "overrides": {}},
        {"name": "no_T1", "overrides": {"enforce_t1": False}},
        {"name": "score_gap=0.05", "overrides": {"score_gap_trigger": True, "score_gap_trigger_threshold": 0.05}},
        {"name": "score_gap=0.10", "overrides": {"score_gap_trigger": True, "score_gap_trigger_threshold": 0.10}},
        {"name": "turnover_pen=0.005", "overrides": {"turnover_penalty": 0.005}},
        {"name": "turnover_pen=0.02", "overrides": {"turnover_penalty": 0.02}},
    ]

    for cfg_test in configs:
        name = cfg_test["name"]
        overrides = cfg_test["overrides"]
        print(f"  Testing {name}...", end=" ")
        metrics = run_single_backtest(scores, prices, cfg, base_n_hold, base_k_trade, overrides)
        obj = strategy_objective(metrics, turnover_penalty=overrides.get("turnover_penalty", 0.01))
        results.append({
            "config": name,
            "total_return": metrics.get("total_return", 0),
            "annual_return": metrics.get("annual_return", 0),
            "sharpe": metrics.get("sharpe", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "turnover": metrics.get("turnover", 0),
            "objective": obj,
        })
        print(f"Return={metrics.get('total_return', 0):.2%}, Sharpe={metrics.get('sharpe', 0):.2f}, Turnover={metrics.get('turnover', 0):.1f}")

    df = pd.DataFrame(results).sort_values("objective", ascending=False)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/quick.yaml"))
    parser.add_argument("--output", default="strategy_optimization_report.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    # 加载数据
    pred = pd.read_csv(out / "val_predictions.csv")
    pred["trade_date"] = pred["trade_date"].astype(str)
    panel = load_panel(out / "panel.parquet")
    panel["trade_date"] = panel["trade_date"].astype(str)
    price_cols = ["trade_date", "ts_code", "open", "close"]
    if "pct_chg" in panel.columns:
        price_cols.append("pct_chg")
    prices = panel[price_cols]

    # 验证集区间
    pred = pred[(pred["trade_date"] > str(cfg["train_end"])) & (pred["trade_date"] <= str(cfg["val_end"]))]
    scores = pred[["trade_date", "ts_code", "score"]]

    # 运行实验
    exp1 = tune_n_hold_k_trade(scores, prices, cfg)
    best_params = exp1.iloc[0]
    base_n_hold = int(best_params["n_hold"])
    base_k_trade = int(best_params["k_trade"])
    print(f"\n  Best n_hold={base_n_hold}, k_trade={base_k_trade}")

    exp2 = tune_position_ratio(scores, prices, cfg, base_n_hold, base_k_trade)
    exp3 = tune_risk_filters(scores, prices, cfg, base_n_hold, base_k_trade)
    exp4 = tune_trade_control(scores, prices, cfg, base_n_hold, base_k_trade)

    # 生成报告
    report = f"""# 策略参数优化报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 实验1: 持仓数量和换手频率 (n_hold, k_trade)

| n_hold | k_trade | 总收益 | 年化收益 | 夏普比率 | 最大回撤 | 换手率 | 综合评分 |
|--------|---------|--------|----------|----------|----------|--------|----------|
"""

    for _, row in exp1.iterrows():
        report += f"| {int(row['n_hold'])} | {int(row['k_trade'])} | {row['total_return']:.2%} | {row['annual_return']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['turnover']:.1f} | {row['objective']:.4f} |\n"

    best1 = exp1.iloc[0]
    report += f"""
**最优配置**: n_hold={int(best1['n_hold'])}, k_trade={int(best1['k_trade'])}

---

## 实验2: 仓位参数调优

| 配置 | 总收益 | 年化收益 | 夏普比率 | 最大回撤 | 换手率 | 最小仓位 | 综合评分 |
|------|--------|----------|----------|----------|--------|----------|----------|
"""

    for _, row in exp2.iterrows():
        report += f"| {row['config']} | {row['total_return']:.2%} | {row['annual_return']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['turnover']:.1f} | {row['min_position_ratio']:.2%} | {row['objective']:.4f} |\n"

    best2 = exp2.iloc[0]
    report += f"""
**最优配置**: {best2['config']}

---

## 实验3: 风控过滤器调优

| 配置 | 总收益 | 年化收益 | 夏普比率 | 最大回撤 | 涨跌停拦截 |
|------|--------|----------|----------|----------|------------|
"""

    for _, row in exp3.iterrows():
        report += f"| {row['config']} | {row['total_return']:.2%} | {row['annual_return']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | 买:{row['limit_buy_blocked']} 卖:{row['limit_sell_blocked']} |\n"

    best3 = exp3.iloc[0]
    report += f"""
**最优配置**: {best3['config']}

---

## 实验4: 交易控制参数调优

| 配置 | 总收益 | 年化收益 | 夏普比率 | 最大回撤 | 换手率 | 综合评分 |
|------|--------|----------|----------|----------|--------|----------|
"""

    for _, row in exp4.iterrows():
        report += f"| {row['config']} | {row['total_return']:.2%} | {row['annual_return']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['turnover']:.1f} | {row['objective']:.4f} |\n"

    best4 = exp4.iloc[0]
    report += f"""
**最优配置**: {best4['config']}

---

## 优化结论与建议

### 最优配置总结

| 类别 | 最优参数 |
|------|----------|
| 持仓数量 | n_hold={int(best1['n_hold'])} |
| 换手频率 | k_trade={int(best1['k_trade'])} |
| 仓位管理 | {best2['config']} |
| 风控过滤 | {best3['config']} |
| 交易控制 | {best4['config']} |

### 关键发现

1. **持仓数量**: {"更多持仓能分散风险" if int(best1['n_hold']) > 50 else "适度持仓集中度更有效"}
2. **换手频率**: {"低换手减少交易成本" if int(best1['k_trade']) <= 2 else "高换手能更快适应市场"}
3. **仓位管理**: {"动态仓位调整有效" if best2['config'].startswith('min_pos=0.8') else "静态仓位更稳定"}
4. **风控过滤**: {"启用过滤能降低风险" if best3['config'] != 'baseline' else "当前过滤阈值适中"}
5. **交易控制**: {"T+1限制有效" if 'no_T1' not in best4['config'] else "可考虑取消T+1"}

### 预期提升

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 总收益 | {best1['total_return']:.2%} | {best1['total_return']:.2%} | - |
| 夏普比率 | {best1['sharpe']:.3f} | {best1['sharpe']:.3f} | - |
| 最大回撤 | {best1['max_drawdown']:.2%} | {best1['max_drawdown']:.2%} | - |

---

*报告由系统自动生成*
"""

    # 保存报告
    report_path = Path(args.output)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存至: {report_path}")

    # 保存详细数据
    exp1.to_csv("optimization_exp1_n_hold_k_trade.csv", index=False)
    exp2.to_csv("optimization_exp2_position.csv", index=False)
    exp3.to_csv("optimization_exp3_risk_filters.csv", index=False)
    exp4.to_csv("optimization_exp4_trade_control.csv", index=False)
    print("详细数据已保存至 CSV 文件")


if __name__ == "__main__":
    main()
