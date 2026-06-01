#!/usr/bin/env python3
"""
k_trade 精细化扫描：n_hold × k_trade 对比

设计原则：与 14_generate_trading_guide.py **使用相同的 strategy_cfg 行为**：
- dynamic_k=True（让 _choose_dynamic_k 生效，这是 k_trade 的核心机制）
- dynamic_position=True
- adaptive_hold=False（让 n_hold 等于传入值，便于横向对比 k_trade）
- score_gap_trigger=True

n_hold_grid = [20, 50]（方案A 实际配置边界：adaptive_min_hold=20, adaptive_max_hold=50）
k_trade_grid = [1, 2, 3, 4, 5]

输出：stock_dl/outputs_quick/ktrade_comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.config import load_config
from src.data.dataset import load_panel


def make_strategy(cfg, override_n_hold, override_k_trade):
    """生成与 14_generate_trading_guide.py 完全一致的 strategy_cfg

    关键：强制启用 dynamic_k / dynamic_position / score_gap_trigger
    (这与 14 脚本的 _choose_dynamic_k 行为一致，是 k_trade 的核心机制)
    """
    s = dict(cfg["strategy"])
    s["n_hold"] = override_n_hold
    s["k_trade"] = override_k_trade
    # 关闭 adaptive_hold 让 n_long = n_hold exactly（便于横向对比 k_trade）
    s["adaptive_hold"] = False
    s["adaptive_hold_multi_indicator"] = False
    # 强制启用（与方案A local_scheme_a.yaml 一致）
    s["dynamic_k"] = True
    s["dynamic_position"] = True
    s["score_gap_trigger"] = True
    s["score_gap_trigger_threshold"] = 0.03
    s["dynamic_k_step"] = 2
    s["score_gap_high"] = 0.10
    s["score_gap_low"] = 0.02
    return s


def run_single(scores, prices, cfg, n_hold, k_trade):
    strategy = make_strategy(cfg, n_hold, k_trade)
    result = run_backtest(
        scores, prices,
        n_hold=n_hold, k_trade=k_trade,
        initial_cash=strategy["initial_cash"],
        cost_rate=strategy.get("cost_rate", 0.0003),
        slippage=strategy.get("slippage", 0.0005),
        use_long_short=strategy.get("use_long_short", False),
        short_ratio=strategy.get("short_ratio", 0.5),
        strategy_cfg=strategy,
        cash_reserve_ratio=strategy.get("cash_reserve_ratio", 0.0),
    )
    return result.get("metrics", {})


def main():
    cfg = load_config(str(ROOT / "configs/quick.yaml"))
    out = Path(cfg["output_dir"])

    pred = pd.read_csv(out / "val_predictions.csv")
    pred["trade_date"] = pred["trade_date"].astype(str)
    panel = load_panel(out / "panel.parquet")
    panel["trade_date"] = panel["trade_date"].astype(str)
    price_cols = ["trade_date", "ts_code", "open", "close"]
    if "pct_chg" in panel.columns:
        price_cols.append("pct_chg")
    prices = panel[price_cols]
    scores = pred[(pred["trade_date"] > str(cfg["train_end"])) &
                  (pred["trade_date"] <= str(cfg["val_end"]))][
        ["trade_date", "ts_code", "score"]
    ]
    print(f"val 区间: {scores['trade_date'].min()} ~ {scores['trade_date'].max()}, "
          f"dates={scores['trade_date'].nunique()}, "
          f"stocks/date≈{scores.groupby('trade_date').size().mean():.0f}")

    # 加载 quick 的 strategy 字段
    print(f"原始 strategy 关键参数: n_hold={cfg['strategy']['n_hold']}, "
          f"k_trade={cfg['strategy']['k_trade']}, "
          f"adaptive_hold={cfg['strategy'].get('adaptive_hold')}, "
          f"dynamic_k={cfg['strategy'].get('dynamic_k')}, "
          f"dynamic_position={cfg['strategy'].get('dynamic_position')}, "
          f"score_gap_trigger={cfg['strategy'].get('score_gap_trigger')}")
    print("扫描时强制覆盖: dynamic_k=True, dynamic_position=True, "
          "score_gap_trigger=True, adaptive_hold=False")

    # 扫描 n_hold × k_trade
    n_hold_grid = [20, 50]   # 方案A 实际配置边界
    k_trade_grid = list(range(1, 21))  # k_trade = 1..20
    rows = []
    for nh in n_hold_grid:
        for kt in k_trade_grid:
            if kt > nh:
                continue
            print(f"  n_hold={nh}, k_trade={kt} ...", end=" ", flush=True)
            m = run_single(scores, prices, cfg, nh, kt)
            obj = (m.get("sharpe", 0)
                   + 2.0 * m.get("total_return", 0)
                   + m.get("max_drawdown", 0)
                   - 0.01 * m.get("turnover", 0))
            row = {
                "n_hold": nh, "k_trade": kt,
                "total_return": m.get("total_return", 0),
                "annual_return": m.get("annual_return", 0),
                "sharpe": m.get("sharpe", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "turnover": m.get("turnover", 0),
                "objective": obj,
            }
            rows.append(row)
            print(f"return={row['total_return']:.2%}, sharpe={row['sharpe']:.3f}, "
                  f"turnover={row['turnover']:.1f}")

    df = pd.DataFrame(rows).sort_values(["n_hold", "k_trade"])

    # 保存
    out_path = out / "ktrade_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n保存对比表: {out_path} ({len(df)} 行)")

    # 打印对比
    print("\n" + "=" * 90)
    print("方案A k_trade 对比（n_hold=20 / 50, k_trade=1~5）")
    print("注：adaptive_hold=False 锁定 n_long=n_hold；dynamic_k 等其它策略与 14 脚本一致")
    print("=" * 90)
    print(f"{'n_hold':>6} {'k_trade':>7} {'总收益':>8} {'年化':>8} {'Sharpe':>8} "
          f"{'最大回撤':>8} {'换手率':>8} {'评分':>8}")
    print("-" * 90)
    for _, r in df.iterrows():
        print(f"{int(r['n_hold']):>6} {int(r['k_trade']):>7} "
              f"{r['total_return']:>8.2%} {r['annual_return']:>8.2%} "
              f"{r['sharpe']:>8.3f} {r['max_drawdown']:>8.2%} "
              f"{r['turnover']:>8.1f} {r['objective']:>8.4f}")
    print("=" * 90)

    # 最优
    print("\n每个 n_hold 的最优 k_trade（按 objective 排序）：")
    for nh in n_hold_grid:
        sub = df[df["n_hold"] == nh].sort_values("objective", ascending=False)
        if sub.empty:
            continue
        best = sub.iloc[0]
        print(f"  n_hold={nh}: best k_trade={int(best['k_trade'])}, "
              f"sharpe={best['sharpe']:.3f}, annual={best['annual_return']:.2%}, "
              f"turnover={best['turnover']:.1f}")

    # k_trade=2 vs 其它（关键对比）
    print("\n关键对比（k_trade=2 vs 其它，n_hold=20）：")
    sub = df[df["n_hold"] == 20].set_index("k_trade")
    if 2 in sub.index:
        base = sub.loc[2]
        print(f"  基准 k_trade=2: sharpe={base['sharpe']:.3f}, "
              f"annual={base['annual_return']:.2%}, turnover={base['turnover']:.1f}")
        for kt in sorted(sub.index):
            if kt == 2:
                continue
            r = sub.loc[kt]
            delta_sharpe = r['sharpe'] - base['sharpe']
            delta_annual = r['annual_return'] - base['annual_return']
            marker = " 🏆" if (r['sharpe'] > base['sharpe'] and
                                r['annual_return'] > base['annual_return']) else ""
            print(f"    vs k_trade={kt:>2}: Δsharpe={delta_sharpe:+.3f}, "
                  f"Δannual={delta_annual:+.2%}, turnover={r['turnover']:.1f}{marker}")

    # 全局最优 k_trade
    print("\n" + "=" * 90)
    print("🏆 全局最优 k_trade（按 objective 排序，Top 10）")
    print("=" * 90)
    top10 = df.sort_values("objective", ascending=False).head(10)
    print(f"{'排名':>4} {'n_hold':>6} {'k_trade':>7} {'年化':>8} {'Sharpe':>8} "
          f"{'换手率':>8} {'评分':>8}")
    print("-" * 90)
    for rank, (_, r) in enumerate(top10.iterrows(), 1):
        print(f"{rank:>4} {int(r['n_hold']):>6} {int(r['k_trade']):>7} "
              f"{r['annual_return']:>8.2%} {r['sharpe']:>8.3f} "
              f"{r['turnover']:>8.1f} {r['objective']:>8.4f}")

    best_overall = df.sort_values("objective", ascending=False).iloc[0]
    best_sharpe = df.sort_values("sharpe", ascending=False).iloc[0]
    best_annual = df.sort_values("annual_return", ascending=False).iloc[0]
    print(f"\n按 objective 最优: n_hold={int(best_overall['n_hold'])}, "
          f"k_trade={int(best_overall['k_trade'])} "
          f"(年化={best_overall['annual_return']:.2%}, Sharpe={best_overall['sharpe']:.3f})")
    print(f"按 Sharpe 最优:    n_hold={int(best_sharpe['n_hold'])}, "
          f"k_trade={int(best_sharpe['k_trade'])} "
          f"(年化={best_sharpe['annual_return']:.2%}, Sharpe={best_sharpe['sharpe']:.3f})")
    print(f"按年化收益最优:    n_hold={int(best_annual['n_hold'])}, "
          f"k_trade={int(best_annual['k_trade'])} "
          f"(年化={best_annual['annual_return']:.2%}, Sharpe={best_annual['sharpe']:.3f})")
    print("=" * 90)


if __name__ == "__main__":
    main()
