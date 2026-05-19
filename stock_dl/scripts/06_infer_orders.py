#!/usr/bin/env python3
"""
根据最新交易日数据输出「次日调仓建议」CSV，供同花顺手动下单。
用法：盘后更新数据 -> 运行本脚本 -> 按 orders_YYYYMMDD.csv 在同花顺模拟盘操作
"""
from __future__ import annotations

import argparse
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


def make_orders(scores: pd.DataFrame, holdings: list[str], n_hold: int, k_trade: int) -> dict:
    s = scores.set_index("ts_code")["score"]
    if "buyable" in scores.columns:
        buy_s = scores[scores["buyable"].astype(bool)].set_index("ts_code")["score"]
    else:
        buy_s = s
    if not holdings:
        buy = buy_s.nlargest(n_hold).index.tolist()
        return {"buy": buy, "sell": [], "hold": []}

    held = [c for c in holdings if c in s.index]
    sell = s.loc[held].nsmallest(min(k_trade, len(held))).index.tolist() if held else []
    candidates = buy_s.nlargest(n_hold + k_trade).index.tolist()
    buy = [c for c in candidates if c not in holdings][:k_trade]
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
    ckpt = torch.load(out / "model.pt", map_location="cpu", weights_only=False)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = score_latest(panel, feat_cols, ckpt["seq_len"], ckpt, device)
    scores = attach_buyable_flag(scores, panel, cfg["strategy"])
    holdings = [x.strip() for x in args.holdings.split(",") if x.strip()]
    n_hold, k_trade = load_best_strategy(cfg, out)
    orders = make_orders(
        scores,
        holdings,
        n_hold,
        k_trade,
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
