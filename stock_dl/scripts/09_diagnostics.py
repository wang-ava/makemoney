#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import load_panel
from src.data.features import feature_columns


def feature_ic(panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in tqdm(feat_cols, desc="feature ic"):
        vals = []
        valid_days = 0
        for _, g in panel[["trade_date", col, "label"]].dropna().groupby("trade_date"):
            if len(g) < 20 or g[col].nunique() < 3:
                continue
            ic, _ = spearmanr(g[col], g["label"])
            if np.isfinite(ic):
                vals.append(ic)
                valid_days += 1
        if vals:
            arr = np.array(vals, dtype=float)
            rows.append(
                {
                    "feature": col,
                    "ic_mean": float(arr.mean()),
                    "ic_std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                    "icir": float(arr.mean() / (arr.std(ddof=1) + 1e-9) if len(arr) > 1 else 0.0),
                    "abs_ic_mean": float(abs(arr.mean())),
                    "valid_days": valid_days,
                }
            )
    return pd.DataFrame(rows).sort_values("abs_ic_mean", ascending=False)


def prediction_diagnostics(out: Path) -> None:
    fp = out / "val_predictions.csv"
    if not fp.exists():
        return
    pred = pd.read_csv(fp).dropna(subset=["score", "label"])
    if pred.empty:
        return
    pred["score_rank"] = pred.groupby("trade_date")["score"].rank(pct=True)
    pred["label_rank"] = pred.groupby("trade_date")["label"].rank(pct=True)
    pred["top_decile"] = pred["score_rank"] >= 0.9
    pred["bottom_decile"] = pred["score_rank"] <= 0.1
    top = pred[pred["top_decile"]]["label"].mean()
    bottom = pred[pred["bottom_decile"]]["label"].mean()
    diag = {
        "score_mean": float(pred["score"].mean()),
        "score_std": float(pred["score"].std()),
        "label_mean": float(pred["label"].mean()),
        "top_decile_next_return": float(top),
        "bottom_decile_next_return": float(bottom),
        "top_minus_bottom": float(top - bottom),
    }
    pd.Series(diag).to_frame("value").to_csv(out / "prediction_diagnostics.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    panel = load_panel(out / "panel.parquet")
    panel["trade_date"] = panel["trade_date"].astype(str)
    val = panel[
        (panel["trade_date"] > str(cfg["train_end"]))
        & (panel["trade_date"] <= str(cfg["val_end"]))
        & panel["label"].notna()
    ].copy()
    feat_cols = feature_columns(val)
    fi = feature_ic(val, feat_cols)
    fi.to_csv(out / "feature_ic.csv", index=False)
    prediction_diagnostics(out)
    print(fi.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
