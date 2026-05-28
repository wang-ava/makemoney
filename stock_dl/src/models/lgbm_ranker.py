from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def prepare_ranker_frame(
    panel: pd.DataFrame,
    feat_cols: list[str],
    start_exclusive: str | None,
    end_inclusive: str,
    relevance_bins: int = 101,
) -> pd.DataFrame:
    relevance_bins = int(relevance_bins)
    if relevance_bins < 2:
        raise ValueError("relevance_bins must be at least 2 for LambdaRank.")

    df = panel.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    mask = df["label"].notna() & (df["trade_date"] <= str(end_inclusive))
    if start_exclusive is not None:
        mask &= df["trade_date"] > str(start_exclusive)
    cols = ["trade_date", "ts_code", "label", *feat_cols]
    out = df.loc[mask, cols].copy()
    out = out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    base = out[["trade_date", "ts_code", "label"]].copy()
    feature_values = out[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    max_relevance = relevance_bins - 1

    def _to_relevance(labels: pd.Series) -> pd.Series:
        if labels.size <= 1:
            return pd.Series(np.zeros(labels.size, dtype=np.int16), index=labels.index)
        rank = labels.rank(method="average", ascending=True) - 1.0
        relevance = (rank / (labels.size - 1) * max_relevance).round()
        return relevance.clip(0, max_relevance).astype(np.int16)

    relevance = out.groupby("trade_date", sort=False)["label"].transform(_to_relevance)
    return pd.concat(
        [base, feature_values, pd.DataFrame({"relevance": relevance.astype(np.int16)})],
        axis=1,
    ).copy()


def group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("trade_date", sort=False).size().astype(int).tolist()


def train_lambdarank(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feat_cols: list[str],
    cfg: dict[str, Any],
):
    import lightgbm as lgb

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": cfg.get("ndcg_eval_at", [10, 20, 30]),
        "num_leaves": int(cfg.get("num_leaves", 63)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "feature_fraction": float(cfg.get("feature_fraction", 0.8)),
        "bagging_fraction": float(cfg.get("bagging_fraction", 0.8)),
        "bagging_freq": int(cfg.get("bagging_freq", 5)),
        "min_data_in_leaf": int(cfg.get("min_data_in_leaf", 50)),
        "lambda_l2": float(cfg.get("lambda_l2", 1.0)),
        "verbose": -1,
        "seed": int(cfg.get("seed", 42)),
    }
    max_label = int(max(train_df["relevance"].max(), val_df["relevance"].max()))
    label_gain = cfg.get("label_gain")
    if label_gain is None:
        relevance_bins = max(int(cfg.get("relevance_bins", 101)), max_label + 1)
        label_gain = list(range(relevance_bins))
    if len(label_gain) <= max_label:
        raise ValueError(
            f"LightGBM label_gain has {len(label_gain)} entries, but relevance labels go up to {max_label}."
        )
    params["label_gain"] = label_gain
    if int(cfg.get("num_threads", 0)) > 0:
        params["num_threads"] = int(cfg["num_threads"])

    train_set = lgb.Dataset(
        train_df[feat_cols],
        label=train_df["relevance"],
        group=group_sizes(train_df),
        feature_name=feat_cols,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        val_df[feat_cols],
        label=val_df["relevance"],
        group=group_sizes(val_df),
        feature_name=feat_cols,
        reference=train_set,
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=int(cfg.get("log_period", 20)))]
    early_stopping_rounds = int(cfg.get("early_stopping_rounds", 50))
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=True))
    return lgb.train(
        params,
        train_set,
        num_boost_round=int(cfg.get("num_boost_round", 500)),
        valid_sets=[train_set, valid_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )


def predict_ranker(model, df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df[["trade_date", "ts_code", "label"]].copy()
    out["score_lgbm"] = model.predict(df[feat_cols], num_iteration=getattr(model, "best_iteration", None))
    out["score"] = out["score_lgbm"]
    return out


def save_feature_importance(model, feat_cols: list[str], path: Path) -> None:
    imp = pd.DataFrame(
        {
            "feature": feat_cols,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    imp.to_csv(path, index=False)
