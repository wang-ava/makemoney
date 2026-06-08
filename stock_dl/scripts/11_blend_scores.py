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
from src.data.labels import attach_expected_labels
from src.metrics.ic import daily_ic, ic_summary
from src.utils.wandb_utils import finish_wandb, init_wandb, wandb_log, wandb_log_artifact, wandb_summary_update


def _rank_by_day(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("trade_date")[col].rank(pct=True).fillna(0.5)


def _score_ic(df: pd.DataFrame, col: str) -> dict:
    scores = df.rename(columns={col: "value"})[["trade_date", "ts_code", "value"]]
    labels = df.rename(columns={"label": "value"})[["trade_date", "ts_code", "value"]]
    return ic_summary(daily_ic(scores, labels))


def _score_topk(df: pd.DataFrame, col: str, top_k: int) -> dict[str, float]:
    rows = []
    work = df.dropna(subset=[col, "label"]).copy()
    for _, g in work.groupby("trade_date"):
        if g.empty:
            continue
        k = min(max(1, int(top_k)), len(g))
        top = g.nlargest(k, col)
        universe_return = float(g["label"].mean())
        top_return = float(top["label"].mean())
        rows.append(
            {
                "topk_return": top_return,
                "topk_excess": top_return - universe_return,
                "topk_win_rate": float((top["label"] > 0).mean()),
            }
        )
    if not rows:
        return {"topk_return": 0.0, "topk_excess": 0.0, "topk_win_rate": 0.0}
    daily = pd.DataFrame(rows)
    return {
        "topk_return": float(daily["topk_return"].mean()),
        "topk_excess": float(daily["topk_excess"].mean()),
        "topk_win_rate": float(daily["topk_win_rate"].mean()),
    }


def _selection_value(stats: dict, topk_stats: dict, metric: str) -> float:
    metric = metric.lower()
    if metric in {"topk_excess", "val_topk_excess"}:
        value = topk_stats.get("topk_excess", 0.0)
    elif metric in {"topk_return", "val_topk_return"}:
        value = topk_stats.get("topk_return", 0.0)
    elif metric in {"topk_win_rate", "val_topk_win_rate"}:
        value = topk_stats.get("topk_win_rate", 0.0)
    elif metric in {"ic", "ic_mean", "val_ic"}:
        value = stats.get("ic_mean", 0.0)
    elif metric in {"icir", "val_icir"}:
        value = stats.get("icir", 0.0)
    else:
        raise ValueError(
            f"Unsupported ensemble.selection_metric={metric!r}; "
            "expected topk_excess, topk_return, topk_win_rate, ic_mean, or icir"
        )
    return float(value) if np.isfinite(value) else -np.inf


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
    panel = load_panel(out / "panel.parquet")
    deep, label_check = attach_expected_labels(
        deep,
        panel,
        label_mode=cfg.get("label_mode", "close_to_next_close"),
        label_horizon=cfg.get("label_horizon", 1),
        tradable_label_filter=cfg.get("tradable_label_filter", True),
        label_limit_up_pct=cfg.get("label_limit_up_pct", 9.5),
    )
    label_validation = label_check.to_dict() if label_check is not None else None

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
        meta = {"best_alpha": 1.0, "reason": reason, "label_validation": label_validation}
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
        meta = {
            "best_alpha": 1.0,
            "reason": "deep/lgbm predictions have no overlap",
            "label_validation": label_validation,
        }
        (out / "blend_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("No overlap between deep and LightGBM predictions; final score uses deep model only.")
        return

    merged["rank_deep"] = _rank_by_day(merged, "score_deep")
    merged["rank_lgbm"] = _rank_by_day(merged, "score_lgbm")

    ensemble_cfg = cfg.get("ensemble", {})
    alpha_grid = ensemble_cfg.get("alpha_grid", [0.3, 0.4, 0.5, 0.6, 0.7])
    selection_metric = str(ensemble_cfg.get("selection_metric", "topk_excess"))
    top_k = int(ensemble_cfg.get("topk_for_metric", cfg.get("train", {}).get("topk_for_metric", 50)))
    rows = []
    best_alpha = float(alpha_grid[0])
    best_objective = -np.inf
    best_stats = {}
    for alpha in alpha_grid:
        alpha = float(alpha)
        col = f"blend_{alpha:.2f}"
        merged[col] = alpha * merged["rank_deep"] + (1.0 - alpha) * merged["rank_lgbm"]
        stats = _score_ic(merged, col)
        topk_stats = _score_topk(merged, col, top_k)
        objective = _selection_value(stats, topk_stats, selection_metric)
        row = {
            "alpha": alpha,
            "selection_metric": selection_metric,
            "selection_objective": objective,
            "top_k": top_k,
            **stats,
            **topk_stats,
        }
        rows.append(row)
        if objective > best_objective:
            best_objective = objective
            best_alpha = alpha
            best_stats = row

    merged["score"] = best_alpha * merged["rank_deep"] + (1.0 - best_alpha) * merged["rank_lgbm"]
    final_cols = ["trade_date", "ts_code", "score", "label", "score_deep", "score_lgbm", "rank_deep", "rank_lgbm"]
    merged[final_cols].to_csv(out / "val_predictions_blend.csv", index=False)
    merged[final_cols].to_csv(out / "val_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "blend_alpha_search.csv", index=False)
    meta = {
        "best_alpha": best_alpha,
        "selection_metric": selection_metric,
        "selection_objective": best_objective,
        "top_k": top_k,
        "ic_mean": best_stats.get("ic_mean", 0.0),
        "topk_excess": best_stats.get("topk_excess", 0.0),
        "label_validation": label_validation,
        "alpha_search": rows,
    }
    (out / "blend_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    wandb_run = init_wandb(cfg, job_type="blend", extra_config={"script": "11_blend_scores.py"})
    payload = {
        "blend/best_alpha": float(best_alpha),
        "blend/selection_objective": float(best_objective),
        "blend/ic_mean": float(best_stats.get("ic_mean", 0.0)),
        "blend/topk_excess": float(best_stats.get("topk_excess", 0.0)),
        "blend/top_k": int(top_k),
    }
    wandb_log(wandb_run, payload)
    wandb_summary_update(wandb_run, payload)
    if cfg.get("wandb", {}).get("log_artifacts", True):
        wandb_log_artifact(wandb_run, out / "val_predictions_blend.csv", name="stock-dl-blend-val-predictions", artifact_type="predictions")
        wandb_log_artifact(wandb_run, out / "blend_alpha_search.csv", name="stock-dl-blend-alpha-search", artifact_type="metrics")
    finish_wandb(wandb_run)
    print(
        f"Hybrid blend selected alpha={best_alpha:.2f}, "
        f"{selection_metric}={best_objective:.6f}, "
        f"top{top_k}_excess={best_stats.get('topk_excess', 0.0):.6f}, "
        f"val IC={best_stats.get('ic_mean', 0.0):.6f}"
    )


if __name__ == "__main__":
    main()
