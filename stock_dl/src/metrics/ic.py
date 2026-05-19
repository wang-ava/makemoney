from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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


def ic_summary(ic_df: pd.DataFrame) -> dict[str, float]:
    if ic_df.empty:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0}
    m = ic_df["ic"].mean()
    s = ic_df["ic"].std() + 1e-9
    return {"ic_mean": float(m), "ic_std": float(s), "icir": float(m / s)}
