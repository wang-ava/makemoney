#!/usr/bin/env python3
"""
根据最新交易日数据输出「次日调仓建议」CSV，供同花顺手动下单。
用法：盘后更新数据 -> 运行本脚本 -> 按 orders_YYYYMMDD.csv 在同花顺模拟盘操作
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.backtest.risk import attach_buyable_flag
from src.backtest.strategy import load_best_strategy
from src.data.features import add_features, feature_columns
from src.data.panel import build_panel
from src.models.factory import build_model_from_checkpoint


def score_latest(panel: pd.DataFrame, feat_cols: list[str], seq_len: int, ckpt: dict, device) -> pd.DataFrame:
    model, flatten = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    last_date = panel["trade_date"].max()
    rows = []
    for code, g in panel.groupby("ts_code"):
        g = g.sort_values("trade_date")
        if len(g) < seq_len or g["trade_date"].iloc[-1] != last_date:
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
        rows.append({"trade_date": last_date, "ts_code": code, "score": score})
    return pd.DataFrame(rows)


def score_latest_lgbm(panel: pd.DataFrame, out: Path) -> pd.DataFrame:
    model_path = out / "lgbm_model.txt"
    meta_path = out / "lgbm_meta.json"
    status_path = out / "lgbm_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status", "ok") != "ok":
            return pd.DataFrame()
    if not model_path.exists() or not meta_path.exists():
        return pd.DataFrame()
    try:
        import lightgbm as lgb
    except ImportError:
        return pd.DataFrame()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feat_cols = meta.get("feat_cols", [])
    if not feat_cols:
        return pd.DataFrame()
    last_date = panel["trade_date"].max()
    latest = panel[panel["trade_date"] == last_date].copy()
    if latest.empty:
        return pd.DataFrame()
    for col in feat_cols:
        if col not in latest.columns:
            latest[col] = 0.0
    latest[feat_cols] = latest[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    booster = lgb.Booster(model_file=str(model_path))
    latest["score_lgbm"] = booster.predict(latest[feat_cols], num_iteration=booster.best_iteration)
    return latest[["trade_date", "ts_code", "score_lgbm"]]


def blend_latest_scores(deep_scores: pd.DataFrame, lgbm_scores: pd.DataFrame, out: Path, cfg: dict) -> pd.DataFrame:
    if lgbm_scores.empty:
        return deep_scores
    merged = deep_scores.rename(columns={"score": "score_deep"}).merge(
        lgbm_scores,
        on=["trade_date", "ts_code"],
        how="inner",
    )
    if merged.empty:
        return deep_scores
    meta_path = out / "blend_meta.json"
    if meta_path.exists():
        alpha = float(json.loads(meta_path.read_text(encoding="utf-8")).get("best_alpha", 0.6))
    else:
        alpha = float(cfg.get("ensemble", {}).get("alpha", 0.6))
    merged["rank_deep"] = merged.groupby("trade_date")["score_deep"].rank(pct=True)
    merged["rank_lgbm"] = merged.groupby("trade_date")["score_lgbm"].rank(pct=True)
    merged["score"] = alpha * merged["rank_deep"] + (1.0 - alpha) * merged["rank_lgbm"]
    return merged[["trade_date", "ts_code", "score", "score_deep", "score_lgbm", "rank_deep", "rank_lgbm"]]


def _buy_scores_with_filters(scores: pd.DataFrame, strategy_cfg: dict) -> pd.Series:
    s = scores.set_index("ts_code")["score"]
    if "buyable" in scores.columns:
        buy_s = scores[scores["buyable"].astype(bool)].set_index("ts_code")["score"]
    else:
        buy_s = s
    if strategy_cfg.get("momentum_filter", False):
        col = strategy_cfg.get("momentum_rank_col", "rank_ret_5d")
        if col in scores.columns:
            threshold = float(strategy_cfg.get("min_momentum_rank", 0.2))
            keep = scores[scores[col].fillna(0.5) >= threshold].set_index("ts_code")
            filtered = buy_s[buy_s.index.isin(keep.index)]
            if not filtered.empty:
                buy_s = filtered
    return buy_s


def _dynamic_k(scores: pd.DataFrame, holdings: list[str], buy_s: pd.Series, k_trade: int, strategy_cfg: dict) -> int:
    if not strategy_cfg.get("dynamic_k", False) or not holdings:
        return int(k_trade)
    s = scores.set_index("ts_code")["score"]
    held = [c for c in holdings if c in s.index]
    candidates = buy_s.drop(index=[c for c in holdings if c in buy_s.index], errors="ignore")
    if not held or candidates.empty:
        return int(k_trade)
    gap = float(candidates.max() - s.loc[held].min())
    if gap > float(strategy_cfg.get("score_gap_high", 0.10)):
        return int(k_trade) + int(strategy_cfg.get("dynamic_k_step", 2))
    if gap < float(strategy_cfg.get("score_gap_low", 0.02)):
        return max(1, int(k_trade) - int(strategy_cfg.get("dynamic_k_step", 1)))
    return int(k_trade)


def make_orders(scores: pd.DataFrame, holdings: list[str], n_hold: int, k_trade: int, strategy_cfg: dict | None = None) -> dict:
    strategy_cfg = strategy_cfg or {}
    s = scores.set_index("ts_code")["score"]
    buy_s = _buy_scores_with_filters(scores, strategy_cfg)
    k_trade = min(max(1, _dynamic_k(scores, holdings, buy_s, k_trade, strategy_cfg)), max(n_hold, 1))
    if not holdings:
        buy = buy_s.nlargest(n_hold).index.tolist()
        return {"buy": buy, "sell": [], "hold": []}

    held = [c for c in holdings if c in s.index]
    sell = s.loc[held].nsmallest(min(k_trade, len(held))).index.tolist() if held else []
    post_sell_count = len(holdings) - len(sell)
    buy_n = max(k_trade, n_hold - post_sell_count)
    candidates = buy_s.nlargest(n_hold + buy_n).index.tolist()
    buy = [c for c in candidates if c not in holdings][:buy_n]
    hold = [c for c in holdings if c not in sell]
    return {"buy": buy, "sell": sell, "hold": hold}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument(
        "--holdings",
        default="",
        help="当前持仓逗号分隔，如 000001.SZ,600000.SH；首日留空",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    ckpt = torch.load(out / "model.pt", map_location="cpu", weights_only=True)

    from datetime import datetime, timedelta

    end = cfg["end_date"]
    end_dt = datetime.strptime(end, "%Y%m%d")
    start = (end_dt - timedelta(days=120)).strftime("%Y%m%d")
    panel = build_panel(
        cfg["data_dir"],
        start_date=start,
        end_date=end,
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
    missing_features = [c for c in feat_cols if c not in panel.columns]
    if missing_features:
        for c in missing_features:
            panel[c] = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_scores = score_latest(panel, feat_cols, ckpt["seq_len"], ckpt, device)
    if deep_scores.empty:
        raise ValueError("No latest scores generated. Check seq_len, end_date, and available feature history.")
    lgbm_scores = score_latest_lgbm(panel, out) if cfg.get("lgbm", {}).get("enabled", False) else pd.DataFrame()
    scores = blend_latest_scores(deep_scores, lgbm_scores, out, cfg)
    scores = attach_buyable_flag(scores, panel, cfg["strategy"])
    holdings = [x.strip() for x in args.holdings.split(",") if x.strip()]
    n_hold, k_trade = load_best_strategy(cfg, out)
    orders = make_orders(
        scores,
        holdings,
        n_hold,
        k_trade,
        cfg["strategy"],
    )

    last_date = scores["trade_date"].max()
    order_path = out / f"orders_{last_date}.csv"
    rows = []
    for side, codes in orders.items():
        for c in codes:
            sc = float(scores.loc[scores["ts_code"] == c, "score"].iloc[0]) if c in set(scores["ts_code"]) else None
            buyable = bool(scores.loc[scores["ts_code"] == c, "buyable"].iloc[0]) if "buyable" in scores.columns and c in set(scores["ts_code"]) else None
            rows.append({"side": side, "ts_code": c, "score": sc, "buyable": buyable})
    pd.DataFrame(rows).to_csv(order_path, index=False)
    scores.to_csv(out / f"scores_{last_date}.csv", index=False)

    print(f"Signal date (data as of): {last_date}")
    print(f"SELL: {orders['sell']}")
    print(f"BUY:  {orders['buy']}")
    print(f"HOLD: {orders['hold']}")
    print(f"Saved: {order_path}")


if __name__ == "__main__":
    main()
