#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import load_panel
from src.data.labels import attach_expected_labels
from src.metrics.ic import daily_ic, daily_pearson_ic, direction_accuracy, ic_summary
from src.utils.wandb_utils import finish_wandb, init_wandb, wandb_log, wandb_log_artifact, wandb_summary_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    pred = pd.read_csv(out / "val_predictions.csv")
    panel = load_panel(out / "panel.parquet")
    pred, label_check = attach_expected_labels(
        pred,
        panel,
        label_mode=cfg.get("label_mode", "close_to_next_close"),
        label_horizon=cfg.get("label_horizon", 1),
        tradable_label_filter=cfg.get("tradable_label_filter", True),
        label_limit_up_pct=cfg.get("label_limit_up_pct", 9.5),
    )
    scores = pred.rename(columns={"score": "value"})[["trade_date", "ts_code", "value"]]
    labels = pred.rename(columns={"label": "value"})[["trade_date", "ts_code", "value"]]

    ic_df = daily_ic(scores, labels)
    pearson_ic_df = daily_pearson_ic(scores, labels)
    summary = ic_summary(ic_df)
    summary.update(direction_accuracy(scores, labels))
    if label_check is not None:
        summary["label_bad_ratio"] = label_check.bad_ratio
        summary["label_mean_abs_diff"] = label_check.mean_abs_diff
        summary["label_max_abs_diff"] = label_check.max_abs_diff
    if not pearson_ic_df.empty:
        pearson_std = pearson_ic_df["pearson_ic"].std()
        summary["pearson_ic_mean"] = float(pearson_ic_df["pearson_ic"].mean())
        summary["pearson_ic_std"] = float((pearson_std if pd.notna(pearson_std) else 0.0) + 1e-9)
        summary["pearson_icir"] = float(summary["pearson_ic_mean"] / summary["pearson_ic_std"])
    ic_df.to_csv(out / "daily_ic.csv", index=False)
    pearson_ic_df.to_csv(out / "daily_pearson_ic.csv", index=False)
    (out / "ic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wandb_run = init_wandb(cfg, job_type="eval_ic", extra_config={"script": "04_eval_ic.py"})
    wandb_log(
        wandb_run,
        {
            "eval/ic_mean": summary.get("ic_mean", 0.0),
            "eval/ic_std": summary.get("ic_std", 0.0),
            "eval/icir": summary.get("icir", 0.0),
            "eval/pearson_ic_mean": summary.get("pearson_ic_mean", 0.0),
            "eval/direction_accuracy": summary.get("direction_accuracy", 0.0),
            "eval/long_accuracy": summary.get("long_accuracy", 0.0),
            "eval/short_accuracy": summary.get("short_accuracy", 0.0),
        },
    )
    wandb_summary_update(wandb_run, {f"eval/{k}": v for k, v in summary.items()})
    if cfg.get("wandb", {}).get("log_artifacts", True):
        wandb_log_artifact(wandb_run, out / "daily_ic.csv", name="stock-dl-daily-ic", artifact_type="metrics")
        wandb_log_artifact(wandb_run, out / "daily_pearson_ic.csv", name="stock-dl-daily-pearson-ic", artifact_type="metrics")
        wandb_log_artifact(wandb_run, out / "ic_summary.json", name="stock-dl-ic-summary", artifact_type="metrics")
    finish_wandb(wandb_run)
    print("IC summary:", summary)


if __name__ == "__main__":
    main()
