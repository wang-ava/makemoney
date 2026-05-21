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
from src.metrics.ic import daily_ic, ic_summary
from src.utils.wandb_utils import finish_wandb, init_wandb, wandb_log, wandb_log_artifact, wandb_summary_update


def _rank_by_day(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("trade_date")[col].rank(pct=True).fillna(0.5)


def _score_ic(df: pd.DataFrame, col: str) -> dict:
    scores = df.rename(columns={col: "value"})[["trade_date", "ts_code", "value"]]
    labels = df.rename(columns={"label": "value"})[["trade_date", "ts_code", "value"]]
    return ic_summary(daily_ic(scores, labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    deep_fp = out / "val_predictions_deep.csv"
    if not deep_fp.exists():
        deep_fp = out / "val_predictions.csv"
    if not deep_fp.exists():
        raise FileNotFoundError("No deep validation predictions found.")

    deep = pd.read_csv(deep_fp)
    deep["trade_date"] = deep["trade_date"].astype(str)
    deep = deep.rename(columns={"score": "score_deep"})[["trade_date", "ts_code", "score_deep", "label"]]

    lgbm_fp = out / "lgbm_val_predictions.csv"
    lgbm_status_fp = out / "lgbm_status.json"
    lgbm_status = {}
    if lgbm_status_fp.exists():
        lgbm_status = json.loads(lgbm_status_fp.read_text(encoding="utf-8"))
    lgbm_ready = cfg.get("lgbm", {}).get("enabled", False) and lgbm_status.get("status", "ok") == "ok"
    if not lgbm_ready or not lgbm_fp.exists():
        final = deep.rename(columns={"score_deep": "score"})
        final.to_csv(out / "val_predictions.csv", index=False)
        final.to_csv(out / "val_predictions_blend.csv", index=False)
        reason = lgbm_status.get("status", "lgbm predictions not found")
        meta = {"best_alpha": 1.0, "reason": reason}
        (out / "blend_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"LightGBM blend skipped ({reason}); final score uses deep model only.")
        return

    lgbm = pd.read_csv(lgbm_fp)
    lgbm["trade_date"] = lgbm["trade_date"].astype(str)
    if "score_lgbm" not in lgbm.columns:
        lgbm = lgbm.rename(columns={"score": "score_lgbm"})
    lgbm = lgbm[["trade_date", "ts_code", "score_lgbm"]]

    merged = deep.merge(lgbm, on=["trade_date", "ts_code"], how="inner")
    if merged.empty:
        final = deep.rename(columns={"score_deep": "score"})
        final.to_csv(out / "val_predictions.csv", index=False)
        final.to_csv(out / "val_predictions_blend.csv", index=False)
        meta = {"best_alpha": 1.0, "reason": "deep/lgbm predictions have no overlap"}
        (out / "blend_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("No overlap between deep and LightGBM predictions; final score uses deep model only.")
        return

    merged["rank_deep"] = _rank_by_day(merged, "score_deep")
    merged["rank_lgbm"] = _rank_by_day(merged, "score_lgbm")

    alpha_grid = cfg.get("ensemble", {}).get("alpha_grid", [0.3, 0.4, 0.5, 0.6, 0.7])
    rows = []
    best_alpha = float(alpha_grid[0])
    best_ic = -np.inf
    for alpha in alpha_grid:
        alpha = float(alpha)
        col = f"blend_{alpha:.2f}"
        merged[col] = alpha * merged["rank_deep"] + (1.0 - alpha) * merged["rank_lgbm"]
        stats = _score_ic(merged, col)
        row = {"alpha": alpha, **stats}
        rows.append(row)
        ic = float(stats.get("ic_mean", 0.0))
        if ic > best_ic:
            best_ic = ic
            best_alpha = alpha

    merged["score"] = best_alpha * merged["rank_deep"] + (1.0 - best_alpha) * merged["rank_lgbm"]
    final_cols = ["trade_date", "ts_code", "score", "label", "score_deep", "score_lgbm", "rank_deep", "rank_lgbm"]
    merged[final_cols].to_csv(out / "val_predictions_blend.csv", index=False)
    merged[final_cols].to_csv(out / "val_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "blend_alpha_search.csv", index=False)
    meta = {"best_alpha": best_alpha, "ic_mean": best_ic, "alpha_search": rows}
    (out / "blend_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    wandb_run = init_wandb(cfg, job_type="blend", extra_config={"script": "11_blend_scores.py"})
    payload = {"blend/best_alpha": float(best_alpha), "blend/ic_mean": float(best_ic)}
    wandb_log(wandb_run, payload)
    wandb_summary_update(wandb_run, payload)
    if cfg.get("wandb", {}).get("log_artifacts", True):
        wandb_log_artifact(wandb_run, out / "val_predictions_blend.csv", name="stock-dl-blend-val-predictions", artifact_type="predictions")
        wandb_log_artifact(wandb_run, out / "blend_alpha_search.csv", name="stock-dl-blend-alpha-search", artifact_type="metrics")
    finish_wandb(wandb_run)
    print(f"Hybrid blend selected alpha={best_alpha:.2f}, val IC={best_ic:.6f}")


if __name__ == "__main__":
    main()
