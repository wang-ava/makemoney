#!/usr/bin/env python3
"""
硬算最优持仓数量
遍历 n_hold = 20~60，找出历史回测表现最好的
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.panel import build_panel
from src.data.features import add_features, feature_columns
from src.models.factory import build_model_from_checkpoint
from src.backtest.engine import run_backtest


def score_all_dates(
    panel: pd.DataFrame,
    feat_cols: list[str],
    seq_len: int,
    ckpt: dict,
    device,
    score_dates: list[str]
) -> pd.DataFrame:
    """对所有日期打分"""
    model, flatten = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    all_scores = []
    for date in score_dates:
        date_panel = panel[panel["trade_date"].astype(str) == date]
        if date_panel.empty:
            continue

        rows = []
        for code, g in date_panel.groupby("ts_code"):
            g = g.sort_values("trade_date")
            if len(g) < seq_len:
                continue
            window = g[feat_cols].astype(np.float32).values[-seq_len:]
            if np.isnan(window).any():
                continue
            mu = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True) + 1e-6
            window = (window - mu) / std
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            x_np = window.reshape(-1) if flatten else window
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            with torch.no_grad():
                score = float(model(x).cpu().item())
            rows.append({"trade_date": date, "ts_code": code, "score": score})

        if rows:
            all_scores.extend(rows)

    return pd.DataFrame(all_scores)


def run_strategy_scan(scores: pd.DataFrame, prices: pd.DataFrame, n_range: range, initial_cash: float) -> pd.DataFrame:
    """遍历所有n_hold，找出最优"""
    results = []
    for n_hold in n_range:
        k_trade = max(1, n_hold // 25)
        result = run_backtest(
            scores,
            prices,
            n_hold=n_hold,
            k_trade=k_trade,
            initial_cash=initial_cash,
            cost_rate=0.0003,
            slippage=0.0005,
            use_long_short=False,
            short_ratio=0.5,
            strategy_cfg={},
            cash_reserve_ratio=0.0,
        )
        metrics = result.get("metrics", {})
        results.append({
            "n_hold": n_hold,
            "k_trade": k_trade,
            "sharpe": metrics.get("sharpe", 0),
            "total_return": metrics.get("total_return", 0),
            "annual_return": metrics.get("annual_return", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "daily_win_rate": metrics.get("daily_win_rate", 0),
            "turnover": metrics.get("turnover", 0),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="硬算最优持仓数量")
    parser.add_argument("--config", default=str(ROOT / "configs/local_scheme_a.yaml"))
    parser.add_argument("--min-hold", type=int, default=20)
    parser.add_argument("--max-hold", type=int, default=60)
    parser.add_argument("--val-start", default="20250101", help="回测开始日期")
    parser.add_argument("--val-end", default="auto", help="回测结束日期")
    parser.add_argument("--output", default=None, help="输出目录")
    parser.add_argument("--skip-scoring", action="store_true", help="跳过打分，直接用已有分数")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output) if args.output else Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    ckpt_path = out / "model.pt"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"Model loaded: {ckpt_path}")

    # 确定回测日期
    end_date = cfg.get("end_date", "auto")
    if end_date == "auto":
        end_date = datetime.now().strftime("%Y%m%d")

    # 每个月取一个交易日做回测（节省时间）
    val_start = args.val_start
    val_end = end_date

    print(f"\n{'='*60}")
    print(f"回测区间: {val_start} ~ {val_end}")
    print(f"持仓数量范围: {args.min_hold} ~ {args.max_hold}")
    print(f"{'='*60}\n")

    # 构建面板数据
    print("Building panel data...")
    panel = build_panel(
        cfg["data_dir"],
        start_date="20180101",
        end_date=end_date,
        use_metric=cfg["features"]["use_metric"],
        use_moneyflow=cfg["features"]["use_moneyflow"],
        use_market=cfg["features"].get("use_market", True),
        use_news=cfg["features"].get("use_news", False),
        universe=cfg["universe"],
    )
    panel = add_features(
        panel,
        cross_section_rank=cfg["features"]["cross_section_rank"],
        label_horizon=cfg.get("label_horizon", 1),
        fill_missing=cfg["features"].get("fill_missing", True),
    )
    feat_cols = ckpt.get("feat_cols") or feature_columns(panel)
    for c in feat_cols:
        if c not in panel.columns:
            panel[c] = 0.0

    print(f"Panel shape: {panel.shape}")

    # 获取回测日期（每月一个）
    trade_dates = sorted(panel["trade_date"].unique())
    val_dates = [d for d in trade_dates if val_start <= str(d) <= val_end]
    # 每月取一个
    selected_dates = []
    last_month = None
    for d in val_dates:
        month = str(d)[:6]
        if month != last_month:
            selected_dates.append(str(d))
            last_month = month
    print(f"Selected {len(selected_dates)} dates for backtesting")

    # 打分
    scores_path = out / f"scores_scan_{val_start}_{val_end}.parquet"
    if args.skip_scoring and scores_path.exists():
        print(f"Loading existing scores from {scores_path}")
        scores = pd.read_parquet(scores_path)
    else:
        print("Scoring all dates (this may take a while)...")
        scores = score_all_dates(panel, feat_cols, ckpt["seq_len"], ckpt, device, selected_dates)
        scores.to_parquet(scores_path)
        print(f"Scores saved to {scores_path}")

    if scores.empty:
        print("Error: No scores generated")
        sys.exit(1)

    print(f"Scored {len(scores)} records, {scores['trade_date'].nunique()} dates")

    # 准备价格数据
    prices = panel[["trade_date", "ts_code", "close", "open", "high", "low", "vol"]].copy()

    # 遍历找最优
    print(f"\nRunning strategy scan for n_hold = {args.min_hold} ~ {args.max_hold}...")
    results = run_strategy_scan(
        scores,
        prices,
        n_range=range(args.min_hold, args.max_hold + 1),
        initial_cash=cfg["strategy"]["initial_cash"]
    )

    # 排序
    results_sorted = results.sort_values("sharpe", ascending=False)

    # 输出结果
    print(f"\n{'='*80}")
    print("回测结果 (按夏普比率排序)")
    print(f"{'='*80}")
    print(f"{'n_hold':>6} {'k_trade':>7} {'夏普比率':>10} {'总收益':>10} {'年化收益':>10} {'最大回撤':>10} {'胜率':>8}")
    print("-" * 80)
    for _, row in results_sorted.iterrows():
        print(f"{row['n_hold']:>6} {row['k_trade']:>7} {row['sharpe']:>10.4f} {row['total_return']:>10.2%} {row['annual_return']:>10.2%} {row['max_drawdown']:>10.2%} {row['daily_win_rate']:>8.2%}")

    # 最优结果
    best = results_sorted.iloc[0]
    print(f"\n{'='*80}")
    print(f"🏆 最优持仓数量: n_hold = {int(best['n_hold'])}")
    print(f"   夏普比率: {best['sharpe']:.4f}")
    print(f"   总收益: {best['total_return']:.2%}")
    print(f"   年化收益: {best['annual_return']:.2%}")
    print(f"   最大回撤: {best['max_drawdown']:.2%}")
    print(f"{'='*80}")

    # 保存结果
    result_json = {
        "optimal_n_hold": int(best["n_hold"]),
        "optimal_k_trade": int(best["k_trade"]),
        "metrics": {
            "sharpe": float(best["sharpe"]),
            "total_return": float(best["total_return"]),
            "annual_return": float(best["annual_return"]),
            "max_drawdown": float(best["max_drawdown"]),
            "daily_win_rate": float(best["daily_win_rate"]),
            "turnover": float(best["turnover"]),
        },
        "all_results": results.to_dict(orient="records"),
        "config": {
            "val_start": val_start,
            "val_end": val_end,
            "min_hold": args.min_hold,
            "max_hold": args.max_hold,
            "num_dates": len(selected_dates),
        }
    }

    result_path = out / "optimal_n_hold_scan.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {result_path}")


if __name__ == "__main__":
    main()
