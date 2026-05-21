#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.backtest.risk import attach_buyable_flag
from src.backtest.strategy import save_best_strategy, tune_strategy
from src.config import load_config
from src.data.dataset import load_panel
from src.utils.wandb_utils import finish_wandb, init_wandb, wandb_log, wandb_log_artifact, wandb_summary_update


def _curve_metrics(equity: pd.Series) -> dict:
    if len(equity) < 2:
        return {}
    ret = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)
    max_drawdown = (equity / equity.cummax() - 1).min()
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }


def add_benchmarks(eq: pd.DataFrame, data_dir: Path, initial_cash: float) -> tuple[pd.DataFrame, dict]:
    bench_map = {
        "bench_sh": "000001.SH",
        "bench_hs300": "000300.SH",
        "bench_cyb": "399006.SZ",
    }
    metrics = {}
    for col, code in bench_map.items():
        fp = data_dir / "market" / f"{code}.csv"
        if not fp.exists() or eq.empty:
            continue
        m = pd.read_csv(fp, usecols=["trade_date", "close"])
        m["trade_date"] = m["trade_date"].astype(str)
        m = m.sort_values("trade_date")
        merged = eq[["trade_date"]].merge(m, on="trade_date", how="left")
        merged["close"] = merged["close"].ffill().bfill()
        if merged["close"].notna().sum() < 2:
            continue
        eq[col] = merged["close"] / merged["close"].iloc[0] * initial_cash
        metrics[col] = _curve_metrics(eq[col])
    return eq, metrics


def _flatten_metrics(metrics: dict, prefix: str = "backtest") -> dict[str, float]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in _flatten_metrics(value, f"{prefix}/{key}").items():
                flat[sub_key] = sub_value
        elif isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            flat[f"{prefix}/{key}"] = float(value)
    return flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    pred = pd.read_csv(out / "val_predictions.csv")
    pred["trade_date"] = pred["trade_date"].astype(str)
    panel = load_panel(out / "panel.parquet")
    panel["trade_date"] = panel["trade_date"].astype(str)
    prices = panel[["trade_date", "ts_code", "open", "close"]]

    # 验证集区间
    pred = pred[(pred["trade_date"] > str(cfg["train_end"])) & (pred["trade_date"] <= str(cfg["val_end"]))]
    scores = pred.rename(columns={"score": "score"})[["trade_date", "ts_code", "score"]]
    scores = attach_buyable_flag(scores, panel, cfg["strategy"])

    n_hold = cfg["strategy"]["n_hold"]
    k_trade = cfg["strategy"]["k_trade"]
    if cfg["strategy"].get("auto_tune", False):
        best, tuning = tune_strategy(scores, prices, cfg)
        tuning.to_csv(out / "strategy_tuning.csv", index=False)
        save_best_strategy(best, out)
        n_hold, k_trade = best["n_hold"], best["k_trade"]
        print(f"Tuned strategy: n_hold={n_hold}, k_trade={k_trade}")

    result = run_backtest(
        scores,
        prices,
        n_hold=n_hold,
        k_trade=k_trade,
        initial_cash=cfg["strategy"]["initial_cash"],
        cost_rate=cfg["strategy"].get("cost_rate", 0.0003),
        slippage=cfg["strategy"].get("slippage", 0.0005),
        use_long_short=cfg["strategy"].get("use_long_short", False),
        short_ratio=cfg["strategy"].get("short_ratio", 0.5),
        strategy_cfg=cfg["strategy"],
    )
    eq = result["equity_curve"]
    if isinstance(eq, pd.DataFrame) and not eq.empty:
        eq, bench_metrics = add_benchmarks(eq, Path(cfg["data_dir"]), cfg["strategy"]["initial_cash"])
        eq.to_csv(out / "equity_curve.csv", index=False)
        result["metrics"]["benchmarks"] = bench_metrics
    (out / "backtest_metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    wandb_run = init_wandb(cfg, job_type="backtest", extra_config={"script": "05_backtest.py"})
    flat_metrics = _flatten_metrics(result["metrics"])
    wandb_log(wandb_run, flat_metrics)
    wandb_summary_update(wandb_run, flat_metrics)
    if cfg.get("wandb", {}).get("log_artifacts", True):
        wandb_log_artifact(wandb_run, out / "backtest_metrics.json", name="stock-dl-backtest-metrics", artifact_type="metrics")
        wandb_log_artifact(wandb_run, out / "equity_curve.csv", name="stock-dl-equity-curve", artifact_type="metrics")
        wandb_log_artifact(wandb_run, out / "strategy_tuning.csv", name="stock-dl-strategy-tuning", artifact_type="metrics")
    finish_wandb(wandb_run)
    print("Backtest metrics:", result["metrics"])


if __name__ == "__main__":
    main()
