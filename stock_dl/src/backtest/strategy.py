from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .engine import run_backtest


def strategy_objective(
    metrics: dict,
    turnover_penalty: float = 0.01,
    recent_return_key: str | None = None,
    recent_return_weight: float = 0.0,
) -> float:
    score = (
        float(metrics.get("sharpe", 0.0))
        + 2.0 * float(metrics.get("total_return", 0.0))
        + float(metrics.get("max_drawdown", 0.0))
        - turnover_penalty * float(metrics.get("turnover", 0.0))
    )
    if recent_return_key and recent_return_weight:
        score += float(recent_return_weight) * float(metrics.get(recent_return_key, 0.0))
    return score


def tune_strategy(scores: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    strategy = cfg["strategy"]
    n_grid = strategy.get("n_hold_grid", [strategy["n_hold"]])
    k_grid = strategy.get("k_trade_grid", [strategy["k_trade"]])
    rows = []
    for n_hold in n_grid:
        for k_trade in k_grid:
            if int(k_trade) > int(n_hold):
                continue
            result = run_backtest(
                scores,
                prices,
                n_hold=int(n_hold),
                k_trade=int(k_trade),
                initial_cash=strategy["initial_cash"],
                cost_rate=strategy.get("cost_rate", 0.0003),
                slippage=strategy.get("slippage", 0.0005),
                use_long_short=strategy.get("use_long_short", False),
                short_ratio=strategy.get("short_ratio", 0.5),
                strategy_cfg=strategy,
                cash_reserve_ratio=strategy.get("cash_reserve_ratio", 0.0),
            )
            metrics = result.get("metrics", {})
            recent_window = int(strategy.get("objective_recent_window", 0) or 0)
            recent_return_key = None
            if recent_window > 1:
                eq = result.get("equity_curve")
                if isinstance(eq, pd.DataFrame) and not eq.empty and "equity" in eq.columns:
                    sub = eq.tail(recent_window)
                    if len(sub) > 1 and float(sub["equity"].iloc[0]) > 0:
                        recent_return_key = f"recent_return_{recent_window}"
                        metrics[recent_return_key] = float(sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0)
            row = {"n_hold": int(n_hold), "k_trade": int(k_trade), **metrics}
            row["objective"] = strategy_objective(
                metrics,
                turnover_penalty=float(strategy.get("turnover_penalty", 0.01)),
                recent_return_key=recent_return_key,
                recent_return_weight=float(strategy.get("recent_return_weight", 0.0)),
            )
            rows.append(row)

    tuning = pd.DataFrame(rows).sort_values("objective", ascending=False)
    if tuning.empty:
        return {"n_hold": strategy["n_hold"], "k_trade": strategy["k_trade"]}, tuning
    best = tuning.iloc[0].to_dict()
    return {"n_hold": int(best["n_hold"]), "k_trade": int(best["k_trade"])}, tuning


def save_best_strategy(best: dict, out_dir: Path) -> None:
    (out_dir / "best_strategy.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_best_strategy(cfg: dict, out_dir: Path) -> tuple[int, int]:
    strategy = cfg["strategy"]
    fp = out_dir / "best_strategy.json"
    if strategy.get("auto_tune", False) and fp.exists():
        best = json.loads(fp.read_text(encoding="utf-8"))
        return int(best.get("n_hold", strategy["n_hold"])), int(best.get("k_trade", strategy["k_trade"]))
    return int(strategy["n_hold"]), int(strategy["k_trade"])
