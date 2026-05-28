#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import load_panel
from src.data.features import feature_columns
from src.models.lgbm_ranker import (
    predict_ranker,
    prepare_ranker_frame,
    save_feature_importance,
    train_lambdarank,
)
from src.utils.wandb_utils import finish_wandb, init_wandb, wandb_log, wandb_log_artifact, wandb_summary_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    lgbm_cfg = cfg.get("lgbm", {})
    out = Path(cfg["output_dir"])
    if not lgbm_cfg.get("enabled", False):
        (out / "lgbm_status.json").write_text(json.dumps({"status": "disabled"}, indent=2), encoding="utf-8")
        print("LightGBM channel disabled; skip.")
        return

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        (out / "lgbm_status.json").write_text(
            json.dumps({"status": "missing_dependency", "package": "lightgbm"}, indent=2),
            encoding="utf-8",
        )
        print("LightGBM is not installed; skip LambdaRank channel.")
        return

    panel = load_panel(out / "panel.parquet")
    feat_cols = feature_columns(panel)
    if not feat_cols:
        raise ValueError("No numeric feature columns found for LightGBM.")

    relevance_bins = int(lgbm_cfg.get("relevance_bins", 101))
    train_df = prepare_ranker_frame(panel, feat_cols, None, cfg["train_end"], relevance_bins=relevance_bins)
    val_df = prepare_ranker_frame(panel, feat_cols, cfg["train_end"], cfg["val_end"], relevance_bins=relevance_bins)
    if train_df.empty or val_df.empty:
        raise ValueError("Empty LightGBM train/val data. Check train_end, val_end and labels.")
    relevance_min = int(min(train_df["relevance"].min(), val_df["relevance"].min()))
    relevance_max = int(max(train_df["relevance"].max(), val_df["relevance"].max()))
    print(
        f"LightGBM LambdaRank: train={train_df.shape}, val={val_df.shape}, "
        f"features={len(feat_cols)}, relevance={relevance_min}..{relevance_max}/{relevance_bins - 1}"
    )

    wandb_run = init_wandb(
        cfg,
        job_type="lgbm",
        extra_config={
            "script": "10_train_lgbm.py",
            "lgbm_train_rows": int(train_df.shape[0]),
            "lgbm_val_rows": int(val_df.shape[0]),
            "n_features": len(feat_cols),
            "relevance_bins": relevance_bins,
        },
    )
    model = train_lambdarank(train_df, val_df, feat_cols, lgbm_cfg)
    model_path = out / "lgbm_model.txt"
    model.save_model(model_path)
    pred = predict_ranker(model, val_df, feat_cols)
    pred.to_csv(out / "lgbm_val_predictions.csv", index=False)
    save_feature_importance(model, feat_cols, out / "lgbm_feature_importance.csv")

    meta = {
        "model_path": str(model_path),
        "feat_cols": feat_cols,
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
        "best_score": getattr(model, "best_score", {}),
        "config": lgbm_cfg,
    }
    (out / "lgbm_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "lgbm_status.json").write_text(json.dumps({"status": "ok"}, indent=2), encoding="utf-8")
    best_iter = int(getattr(model, "best_iteration", 0) or 0)
    wandb_log(wandb_run, {"lgbm/best_iteration": best_iter})
    wandb_summary_update(wandb_run, {"lgbm/best_iteration": best_iter})
    if cfg.get("wandb", {}).get("log_artifacts", True):
        wandb_log_artifact(wandb_run, model_path, name="stock-dl-lgbm-model", artifact_type="model")
        wandb_log_artifact(wandb_run, out / "lgbm_val_predictions.csv", name="stock-dl-lgbm-val-predictions", artifact_type="predictions")
        wandb_log_artifact(wandb_run, out / "lgbm_feature_importance.csv", name="stock-dl-lgbm-feature-importance", artifact_type="metrics")
    finish_wandb(wandb_run)
    print(f"Saved LightGBM model: {model_path}")


if __name__ == "__main__":
    main()
