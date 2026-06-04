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

from src.config import load_config
from src.data.dataset import load_panel


WINDOWS = (0, 120, 60, 20, 10, 5)
TOP_KS = (10, 30, 50, 70, 100, 300)


def _load_panel_for_audit(out: Path) -> pd.DataFrame:
    errors: list[str] = []
    for name in ("panel.parquet", "panel_new.parquet"):
        fp = out / name
        if not fp.exists():
            continue
        try:
            return load_panel(fp)
        except Exception as exc:
            errors.append(f"{fp}: {type(exc).__name__}: {exc}")
    detail = "\n".join(errors) if errors else "no panel candidates found"
    raise FileNotFoundError(f"Could not load a panel for audit:\n{detail}")


def _safe_float(value: float | int | np.floating | None) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def _add_tradable_labels(panel: pd.DataFrame, horizon: int, limit_up_pct: float = 9.5) -> pd.DataFrame:
    df = panel.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"])
    by_code = df.groupby("ts_code", sort=False)
    future_close = by_code["close"].shift(-max(int(horizon), 1))
    next_open = by_code["open"].shift(-1)
    next2_close = by_code["close"].shift(-2)
    df["audit_close_to_next_close"] = future_close / df["close"].replace(0, np.nan) - 1.0
    df["audit_next_open_to_next_close"] = future_close / next_open.replace(0, np.nan) - 1.0
    df["audit_next_open_to_next2_close"] = next2_close / next_open.replace(0, np.nan) - 1.0
    df["audit_close_to_next_open"] = next_open / df["close"].replace(0, np.nan) - 1.0

    invalid_entry = (~np.isfinite(next_open)) | (next_open <= 0)
    open_gap = next_open / df["close"].replace(0, np.nan) - 1.0
    invalid_entry |= open_gap >= float(limit_up_pct) / 100.0
    if "vol" in df.columns:
        invalid_entry |= by_code["vol"].shift(-1).fillna(0) <= 0
    if "amount" in df.columns:
        invalid_entry |= by_code["amount"].shift(-1).fillna(0) <= 0
    for col in ("audit_next_open_to_next_close", "audit_next_open_to_next2_close"):
        df.loc[invalid_entry, col] = np.nan

    return df[[
        "trade_date",
        "ts_code",
        "audit_close_to_next_close",
        "audit_next_open_to_next_close",
        "audit_next_open_to_next2_close",
        "audit_close_to_next_open",
    ]]


def _daily_ic(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for d, g in df.dropna(subset=["score", label_col]).groupby("trade_date"):
        if len(g) < 20 or g["score"].nunique() < 3 or g[label_col].nunique() < 3:
            continue
        rows.append({
            "trade_date": d,
            "ic": _safe_float(g["score"].corr(g[label_col], method="spearman")),
        })
    return pd.DataFrame(rows)


def _topk_windows(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    day_rows = []
    for d, g in df.dropna(subset=["score", label_col]).groupby("trade_date"):
        universe = float(g[label_col].mean())
        for k in TOP_KS:
            top = g.nlargest(min(k, len(g)), "score")
            if top.empty:
                continue
            day_rows.append({
                "trade_date": d,
                "label_col": label_col,
                "top_k": k,
                "universe_return": universe,
                "top_return": float(top[label_col].mean()),
                "top_excess": float(top[label_col].mean() - universe),
                "top_win_rate": float((top[label_col] > 0).mean()),
            })
    daily = pd.DataFrame(day_rows)
    if daily.empty:
        return daily

    for window in WINDOWS:
        sub = daily if window == 0 else daily.groupby("top_k", group_keys=False).tail(window)
        for k, g in sub.groupby("top_k"):
            rows.append({
                "label_col": label_col,
                "window_days": "all" if window == 0 else window,
                "top_k": int(k),
                "days": int(g["trade_date"].nunique()),
                "top_return_mean": float(g["top_return"].mean()),
                "top_excess_mean": float(g["top_excess"].mean()),
                "top_win_rate_mean": float(g["top_win_rate"].mean()),
                "bad_excess_days": int((g["top_excess"] <= 0).sum()),
            })
    return pd.DataFrame(rows)


def _deciles(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    work = df.dropna(subset=["score", label_col]).copy()
    if work.empty:
        return pd.DataFrame()
    work["score_decile"] = work.groupby("trade_date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")
    )
    return (
        work.groupby("score_decile")[label_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "return_mean", "count": "samples"})
    )


def _equity_recent(out: Path) -> pd.DataFrame:
    fp = out / "equity_curve.csv"
    if not fp.exists():
        return pd.DataFrame()
    eq = pd.read_csv(fp)
    if eq.empty or "equity" not in eq.columns:
        return pd.DataFrame()
    rows = []
    for window in (5, 10, 20, 60):
        sub = eq.tail(window)
        if len(sub) < 2:
            continue
        ret = sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0
        win_rate = sub["equity"].pct_change().dropna().gt(0).mean()
        rows.append({"window_days": window, "return": float(ret), "win_rate": float(win_rate)})
    return pd.DataFrame(rows)


def _status(summary: dict[str, float]) -> str:
    recent_ic = summary.get("tradable_ic_last20", 0.0)
    recent_excess = summary.get("tradable_topk_excess_last20", summary.get("tradable_top70_excess_last20", 0.0))
    t1_excess = summary.get("realizable_t1_topk_excess_last20", summary.get("realizable_t1_top70_excess_last20", 0.0))
    equity_20 = summary.get("equity_return_last20", 0.0)
    if recent_ic < 0 or recent_excess < 0 or t1_excess < 0 or equity_20 < -0.05:
        return "red"
    if recent_ic < 0.03 or recent_excess < 0.001 or t1_excess < 0.001 or equity_20 < -0.02:
        return "yellow"
    return "green"


def _write_report(out: Path, cfg: dict, summary: dict, topk: pd.DataFrame, deciles: pd.DataFrame, equity: pd.DataFrame) -> None:
    audit_top_k = int(summary.get("audit_top_k", 70))
    status_text = {
        "red": "红灯：近期信号或组合表现不支持继续激进交易",
        "yellow": "黄灯：信号偏弱，只能小仓位验证",
        "green": "绿灯：信号近期仍有效，但仍需执行风控",
    }[summary["status"]]
    lines = [
        "# 模型审计报告",
        "",
        f"- 配置: `{cfg.get('output_dir')}`",
        f"- 当前训练标签: `{cfg.get('label_mode', 'close_to_next_close')}` / horizon={cfg.get('label_horizon', 1)}",
        f"- 状态: **{status_text}**",
        "",
        "## 核心指标",
        "",
        f"- 原始验证标签 IC 均值: {summary['stored_ic_mean']:.4f}",
        f"- 可交易标签 IC 均值: {summary['tradable_ic_mean']:.4f}",
        f"- 可交易标签最近20日 IC: {summary['tradable_ic_last20']:.4f}",
        f"- Top{audit_top_k} 可交易超额收益均值: {summary['tradable_topk_excess_all']:.4%}",
        f"- Top{audit_top_k} 可交易最近20日超额收益: {summary['tradable_topk_excess_last20']:.4%}",
        f"- Top{audit_top_k} T+1 可兑现最近20日超额收益: {summary.get('realizable_t1_topk_excess_last20', 0.0):.4%}",
        f"- 回测最近20日收益: {summary.get('equity_return_last20', 0.0):.2%}",
        "",
        "## 判断",
        "",
    ]
    if summary["status"] == "red":
        lines.extend([
            "- 不建议直接按 raw TopK 全量交易。",
            "- 先用可交易标签重训，并把最近20日作为上线前压力测试。",
            f"- 若重训后可交易 IC 或 Top{audit_top_k} 超额仍弱，模型只能作为观察列表，不能作为下单引擎。",
        ])
    elif summary["status"] == "yellow":
        lines.extend([
            "- 模型可以保留为候选池，但仓位和换手必须收紧。",
            "- 每日只换少数票，并过滤主力净流出、低流动性、冲高回落标的。",
        ])
    else:
        lines.extend([
            "- 模型有一定近期排序能力。",
            "- 仍应使用可交易标签、低换手和风控过滤，避免把统计优势消耗在滑点里。",
        ])

    if not topk.empty:
        show = topk[
            (topk["label_col"].isin(["audit_next_open_to_next_close", "audit_next_open_to_next2_close"]))
            & (topk["top_k"].isin([audit_top_k, 50, 70]))
            & (topk["window_days"].isin(["all", 20, 10, 5]))
        ].copy()
        lines.extend(["", "## TopK 窗口摘要", "", "```text", show.to_string(index=False), "```"])
    if not deciles.empty:
        lines.extend(["", "## 十分位收益", "", "```text", deciles.to_string(index=False), "```"])
    if not equity.empty:
        lines.extend(["", "## 近期回测权益", "", "```text", equity.to_string(index=False), "```"])

    (out / "model_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether the current model is tradable.")
    parser.add_argument("--config", default=str(ROOT / "configs/local_scheme_a.yaml"))
    parser.add_argument("--top-k", type=int, default=70, help="Primary TopK used for status summary.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    pred_fp = out / "val_predictions.csv"
    if not pred_fp.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_fp}")

    pred = pd.read_csv(pred_fp)
    pred["trade_date"] = pred["trade_date"].astype(str)
    panel = _load_panel_for_audit(out)
    audit_labels = _add_tradable_labels(panel, int(cfg.get("label_horizon", 1)), float(cfg.get("label_limit_up_pct", 9.5)))
    df = pred.merge(audit_labels, on=["trade_date", "ts_code"], how="left")

    stored_ic = _daily_ic(df.rename(columns={"label": "stored_label"}), "stored_label")
    tradable_ic = _daily_ic(df, "audit_next_open_to_next_close")
    t1_ic = _daily_ic(df, "audit_next_open_to_next2_close")
    topk_stored = _topk_windows(df.rename(columns={"label": "stored_label"}), "stored_label")
    topk_tradable = _topk_windows(df, "audit_next_open_to_next_close")
    topk_t1 = _topk_windows(df, "audit_next_open_to_next2_close")
    topk = pd.concat([topk_stored, topk_tradable, topk_t1], ignore_index=True)
    deciles = _deciles(df, "audit_next_open_to_next_close")
    equity = _equity_recent(out)

    def _topk_value_for(label_col: str, top_k: int, window, metric: str) -> float:
        row = topk[
            (topk["label_col"] == label_col)
            & (topk["top_k"] == int(top_k))
            & (topk["window_days"].astype(str) == str(window))
        ]
        if row.empty:
            return 0.0
        return float(row.iloc[0][metric])

    primary_top_k = int(args.top_k)
    tradable_excess_all = _topk_value_for("audit_next_open_to_next_close", primary_top_k, "all", "top_excess_mean")
    tradable_excess_last20 = _topk_value_for("audit_next_open_to_next_close", primary_top_k, 20, "top_excess_mean")
    t1_excess_all = _topk_value_for("audit_next_open_to_next2_close", primary_top_k, "all", "top_excess_mean")
    t1_excess_last20 = _topk_value_for("audit_next_open_to_next2_close", primary_top_k, 20, "top_excess_mean")
    tradable_top70_excess_all = _topk_value_for("audit_next_open_to_next_close", 70, "all", "top_excess_mean")
    tradable_top70_excess_last20 = _topk_value_for("audit_next_open_to_next_close", 70, 20, "top_excess_mean")
    t1_top70_excess_all = _topk_value_for("audit_next_open_to_next2_close", 70, "all", "top_excess_mean")
    t1_top70_excess_last20 = _topk_value_for("audit_next_open_to_next2_close", 70, 20, "top_excess_mean")
    summary = {
        "output_dir": str(out),
        "label_mode": cfg.get("label_mode", "close_to_next_close"),
        "label_horizon": cfg.get("label_horizon", 1),
        "audit_top_k": primary_top_k,
        "stored_ic_mean": float(stored_ic["ic"].mean()) if not stored_ic.empty else 0.0,
        "tradable_ic_mean": float(tradable_ic["ic"].mean()) if not tradable_ic.empty else 0.0,
        "tradable_ic_last20": float(tradable_ic.tail(20)["ic"].mean()) if not tradable_ic.empty else 0.0,
        "tradable_topk_excess_all": tradable_excess_all,
        "tradable_topk_excess_last20": tradable_excess_last20,
        "tradable_top70_excess_all": tradable_top70_excess_all,
        "tradable_top70_excess_last20": tradable_top70_excess_last20,
        "realizable_t1_ic_mean": float(t1_ic["ic"].mean()) if not t1_ic.empty else 0.0,
        "realizable_t1_ic_last20": float(t1_ic.tail(20)["ic"].mean()) if not t1_ic.empty else 0.0,
        "realizable_t1_topk_excess_all": t1_excess_all,
        "realizable_t1_topk_excess_last20": t1_excess_last20,
        "realizable_t1_top70_excess_all": t1_top70_excess_all,
        "realizable_t1_top70_excess_last20": t1_top70_excess_last20,
    }
    if not equity.empty:
        for _, row in equity.iterrows():
            summary[f"equity_return_last{int(row['window_days'])}"] = float(row["return"])
    summary["status"] = _status(summary)

    topk.to_csv(out / "model_audit_topk_windows.csv", index=False)
    deciles.to_csv(out / "model_audit_deciles.csv", index=False)
    if not stored_ic.empty:
        stored_ic.to_csv(out / "model_audit_stored_daily_ic.csv", index=False)
    if not tradable_ic.empty:
        tradable_ic.to_csv(out / "model_audit_tradable_daily_ic.csv", index=False)
    if not t1_ic.empty:
        t1_ic.to_csv(out / "model_audit_realizable_t1_daily_ic.csv", index=False)
    (out / "model_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(out, cfg, summary, topk, deciles, equity)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {out / 'model_audit_report.md'}")


if __name__ == "__main__":
    main()
