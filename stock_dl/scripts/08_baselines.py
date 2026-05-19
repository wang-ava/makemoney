#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.backtest.risk import attach_buyable_flag
from src.backtest.strategy import load_best_strategy
from src.config import load_config
from src.data.dataset import load_panel
from src.metrics.ic import daily_ic, ic_summary


def _normalize_by_day(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("trade_date")[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    return ((df[col] - mean) / (std + 1e-12)).fillna(0.0)


def build_baseline_scores(panel: pd.DataFrame, out: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    val = panel[
        (panel["trade_date"].astype(str) > str(cfg["train_end"]))
        & (panel["trade_date"].astype(str) <= str(cfg["val_end"]))
        & panel["label"].notna()
    ].copy()
    val["trade_date"] = val["trade_date"].astype(str)

    baselines: dict[str, pd.DataFrame] = {}
    pred_fp = out / "val_predictions.csv"
    if pred_fp.exists():
        pred = pd.read_csv(pred_fp)
        pred["trade_date"] = pred["trade_date"].astype(str)
        baselines["deep_temporal_attention"] = pred[["trade_date", "ts_code", "score"]]

    recipes = {
        "momentum_20d": "ret_20d",
        "momentum_5d": "ret_5d",
        "short_reversal_1d": "ret_1d",
        "moneyflow_5d": "mf_ratio_5d",
        "small_cap": "log_circ_mv",
        "value_bp": "bp",
        "liquidity_turnover": "turnover_rate",
    }

    for name, col in recipes.items():
        if col not in val.columns:
            continue
        score = _normalize_by_day(val, col)
        if name in {"short_reversal_1d", "small_cap"}:
            score = -score
        baselines[name] = pd.DataFrame(
            {"trade_date": val["trade_date"], "ts_code": val["ts_code"], "score": score}
        )

    if {"ret_20d", "mf_ratio_5d", "bp", "log_circ_mv"}.issubset(val.columns):
        combo = (
            0.35 * _normalize_by_day(val, "ret_20d")
            + 0.25 * _normalize_by_day(val, "mf_ratio_5d")
            + 0.20 * _normalize_by_day(val, "bp")
            - 0.20 * _normalize_by_day(val, "log_circ_mv")
        )
        baselines["handcrafted_combo"] = pd.DataFrame(
            {"trade_date": val["trade_date"], "ts_code": val["ts_code"], "score": combo}
        )

    rng = np.random.default_rng(cfg["train"].get("seed", 42))
    baselines["random"] = pd.DataFrame(
        {"trade_date": val["trade_date"], "ts_code": val["ts_code"], "score": rng.normal(size=len(val))}
    )
    return baselines


def evaluate_one(name: str, scores: pd.DataFrame, labels: pd.DataFrame, prices: pd.DataFrame, panel: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    scores = scores.dropna(subset=["score"]).copy()
    scores = attach_buyable_flag(scores, panel, cfg["strategy"])
    out = Path(cfg["output_dir"])
    n_hold, k_trade = load_best_strategy(cfg, out)
    ic_df = daily_ic(
        scores.rename(columns={"score": "value"})[["trade_date", "ts_code", "value"]],
        labels.rename(columns={"label": "value"})[["trade_date", "ts_code", "value"]],
    )
    ic_stats = ic_summary(ic_df)
    bt = run_backtest(
        scores[["trade_date", "ts_code", "score"]],
        prices,
        n_hold=n_hold,
        k_trade=k_trade,
        initial_cash=cfg["strategy"]["initial_cash"],
        cost_rate=cfg["strategy"].get("cost_rate", 0.0003),
        slippage=cfg["strategy"].get("slippage", 0.0005),
    )
    metrics = {
        "model": name,
        "ic_mean": ic_stats.get("ic_mean", 0.0),
        "icir": ic_stats.get("icir", 0.0),
        **bt.get("metrics", {}),
    }
    eq = bt.get("equity_curve", pd.DataFrame())
    if isinstance(eq, pd.DataFrame) and not eq.empty:
        eq = eq[["trade_date", "equity"]].rename(columns={"equity": name})
    return metrics, eq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    panel = load_panel(out / "panel.parquet")
    panel["trade_date"] = panel["trade_date"].astype(str)
    labels = panel[
        (panel["trade_date"] > str(cfg["train_end"]))
        & (panel["trade_date"] <= str(cfg["val_end"]))
        & panel["label"].notna()
    ][["trade_date", "ts_code", "label"]]
    prices = panel[["trade_date", "ts_code", "open", "close"]]

    baselines = build_baseline_scores(panel, out, cfg)
    rows = []
    equity = None
    for name, scores in baselines.items():
        metrics, eq = evaluate_one(name, scores, labels, prices, panel, cfg)
        rows.append(metrics)
        if not eq.empty:
            equity = eq if equity is None else equity.merge(eq, on="trade_date", how="outer")

    comp = pd.DataFrame(rows).sort_values(["ic_mean", "total_return"], ascending=False)
    comp.to_csv(out / "baseline_comparison.csv", index=False)
    if equity is not None:
        equity = equity.sort_values("trade_date")
        equity.to_csv(out / "baseline_equity_curves.csv", index=False)
    (out / "baseline_comparison.json").write_text(
        json.dumps(comp.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
