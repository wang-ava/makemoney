#!/usr/bin/env python3
"""
硬算最优持仓数量：遍历n_hold=20~60，找出历史回测表现最好的
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.backtest.engine import run_backtest


def find_optimal_n_hold(scores: pd.DataFrame, prices: pd.DataFrame, n_range: range, initial_cash: float = 1000000) -> dict:
    """遍历所有n_hold，找出夏普比率最高的"""
    results = []
    for n_hold in n_range:
        result = run_backtest(
            scores,
            prices,
            n_hold=n_hold,
            k_trade=max(1, n_hold // 25),  # k_trade按比例调整
            initial_cash=initial_cash,
            cost_rate=0.0003,
            slippage=0.0005,
            use_long_short=False,
            short_ratio=0.5,
            strategy_cfg={},
            cash_reserve_ratio=0.0,
        )
        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe", 0)
        total_return = metrics.get("total_return", 0)
        max_drawdown = metrics.get("max_drawdown", 0)
        results.append({
            "n_hold": n_hold,
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "annual_return": metrics.get("annual_return", 0),
            "daily_win_rate": metrics.get("daily_win_rate", 0),
        })
        print(f"n_hold={n_hold:2d}: sharpe={sharpe:.3f}, return={total_return:.2%}, dd={max_drawdown:.2%}")

    df = pd.DataFrame(results)
    # 按sharpe排序
    df_sorted = df.sort_values("sharpe", ascending=False)

    print("\n" + "="*60)
    print("Top 5 持仓数量:")
    print(df_sorted.head(5).to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n最优: n_hold={int(best['n_hold'])}, sharpe={best['sharpe']:.4f}")

    return {
        "optimal_n_hold": int(best["n_hold"]),
        "sharpe": float(best["sharpe"]),
        "total_return": float(best["total_return"]),
        "max_drawdown": float(best["max_drawdown"]),
        "all_results": df.to_dict(orient="records")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/local_scheme_a.yaml"))
    parser.add_argument("--min-hold", type=int, default=20)
    parser.add_argument("--max-hold", type=int, default=60)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    # 加载历史分数
    scores_path = out / "scores_20260529.csv"
    if not scores_path.exists():
        print(f"Error: {scores_path} not found. Run inference first.")
        sys.exit(1)

    scores = pd.read_csv(scores_path)
    print(f"Loaded scores: {len(scores)} rows")
    print(f"Date: {scores['trade_date'].max()}")

    # 模拟价格数据（用rank作为价格代理）
    # 实际应用中应该用真实价格
    prices = scores.copy()
    if "close" not in prices.columns:
        prices["close"] = 10 + scores["score"] * 2  # 模拟价格
        prices["open"] = prices["close"] * 0.99
        prices["high"] = prices["close"] * 1.02
        prices["low"] = prices["close"] * 0.98
        prices["vol"] = 1000000

    # 找最优持仓数量
    result = find_optimal_n_hold(
        scores,
        prices,
        n_range=range(args.min_hold, args.max_hold + 1),
        initial_cash=cfg["strategy"]["initial_cash"]
    )

    # 保存结果
    output_path = Path(args.output) if args.output else out / "optimal_n_hold.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
