from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def daily_ic(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """scores/labels: columns [trade_date, ts_code, value]."""
    merged = scores.merge(labels, on=["trade_date", "ts_code"], suffixes=("_pred", "_label"))
    rows = []
    for d, g in merged.groupby("trade_date"):
        if len(g) < 10:
            continue
        ic, _ = spearmanr(g["value_pred"], g["value_label"])
        if np.isfinite(ic):
            rows.append({"trade_date": d, "ic": ic})
    return pd.DataFrame(rows)


def daily_pearson_ic(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional Pearson IC."""
    merged = scores.merge(labels, on=["trade_date", "ts_code"], suffixes=("_pred", "_label"))
    rows = []
    for d, g in merged.groupby("trade_date"):
        if len(g) < 10:
            continue
        pred = g["value_pred"].astype(float)
        label = g["value_label"].astype(float)
        if pred.nunique(dropna=True) < 2 or label.nunique(dropna=True) < 2:
            continue
        ic, _ = pearsonr(pred, label)
        if np.isfinite(ic):
            rows.append({"trade_date": d, "pearson_ic": ic})
    return pd.DataFrame(rows)


def ic_summary(ic_df: pd.DataFrame) -> dict[str, float]:
    if ic_df.empty:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0}
    m = ic_df["ic"].mean()
    s = ic_df["ic"].std() + 1e-9
    return {"ic_mean": float(m), "ic_std": float(s), "icir": float(m / s)}


def direction_accuracy(scores: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    """Measure whether predicted and realized returns point in the same direction."""
    merged = scores.merge(labels, on=["trade_date", "ts_code"], suffixes=("_pred", "_label"))
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["value_pred", "value_label"])
    if merged.empty:
        return {
            "direction_accuracy": 0.0,
            "long_accuracy": 0.0,
            "short_accuracy": 0.0,
            "daily_direction_accuracy_mean": 0.0,
        }

    pred_up = merged["value_pred"] > 0
    label_up = merged["value_label"] > 0
    correct = pred_up == label_up
    long_mask = pred_up
    short_mask = ~pred_up
    daily_acc = correct.groupby(merged["trade_date"]).mean()
    return {
        "direction_accuracy": float(correct.mean()),
        "long_accuracy": float(correct[long_mask].mean()) if long_mask.any() else 0.0,
        "short_accuracy": float(correct[short_mask].mean()) if short_mask.any() else 0.0,
        "daily_direction_accuracy_mean": float(daily_acc.mean()) if not daily_acc.empty else 0.0,
    }
