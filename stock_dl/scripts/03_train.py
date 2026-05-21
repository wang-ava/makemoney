#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import DateBatchSampler, StockWindowDataset, load_panel
from src.data.features import feature_columns
from src.metrics.ic import daily_ic, ic_summary
from src.models.factory import build_model
from src.models.losses import composite_signal_loss, ic_loss
from src.utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    wandb_log,
    wandb_log_artifact,
    wandb_summary_update,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_dataset(model, loader, device):
    model.eval()
    preds, labels, codes, dates = [], [], [], []
    for x, _, raw, c, d in loader:
        x = x.to(device)
        p = model(x).cpu().numpy()
        preds.extend(p.tolist())
        labels.extend(raw.numpy().tolist())
        codes.extend(c)
        dates.extend(d)
    return pd.DataFrame({
        "trade_date": dates,
        "ts_code": codes,
        "score": preds,
        "label": labels,
    })


def get_scheduler(opt, cfg, num_training_steps):
    warmup_epochs = cfg["train"].get("warmup_epochs", 0)
    warmup_steps = warmup_epochs * num_training_steps // cfg["train"]["epochs"]

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    panel_path = Path(cfg["output_dir"]) / "panel.parquet"
    panel = load_panel(panel_path)
    for required_col in ("ts_code", "trade_date", "label"):
        if required_col not in panel.columns:
            raise ValueError(f"{panel_path} is missing required column: {required_col}")
    feat_cols = feature_columns(panel)
    if not feat_cols:
        raise ValueError("No numeric feature columns found. Check panel construction and feature_columns().")
    print(f"Loaded panel: {panel.shape[0]} rows, {panel.shape[1]} columns")
    print(f"Using {len(feat_cols)} features")

    model, flatten = build_model(cfg["model"], n_features=len(feat_cols), seq_len=cfg["seq_len"])
    target_col = cfg["train"].get("target_col", "label_cs_z")
    train_ds = StockWindowDataset(
        panel,
        feat_cols,
        cfg["seq_len"],
        "train",
        cfg["train_end"],
        cfg["val_end"],
        target_col=target_col,
        flatten=flatten,
    )
    val_ds = StockWindowDataset(
        panel,
        feat_cols,
        cfg["seq_len"],
        "val",
        cfg["train_end"],
        cfg["val_end"],
        target_col=target_col,
        flatten=flatten,
    )
    print(f"train samples={len(train_ds)}, val samples={len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Empty train/val dataset. Check start_date, train_end, val_end, seq_len, "
            "and whether labels are available after feature construction."
        )

    num_workers = int(cfg["train"].get("num_workers", 0))
    if cfg["train"].get("date_batch", False):
        sampler = DateBatchSampler(
            train_ds.sample_dates,
            max_batch_size=cfg["train"].get("batch_size", 2048),
            shuffle=True,
            seed=cfg["train"]["seed"],
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    wandb_run = init_wandb(
        cfg,
        job_type="train",
        extra_config={
            "script": "03_train.py",
            "n_features": len(feat_cols),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "n_parameters": n_params,
            "device": str(device),
        },
    )
    if wandb_run is not None:
        watch_mode = cfg.get("wandb", {}).get("watch", "gradients")
        if watch_mode:
            try:
                import wandb

                wandb.watch(model, log=watch_mode, log_freq=int(cfg.get("wandb", {}).get("watch_log_freq", 100)))
            except Exception as exc:
                print(f"W&B watch failed; continue training: {exc}")
        wandb_log(
            wandb_run,
            {
                "data/panel_rows": int(panel.shape[0]),
                "data/panel_cols": int(panel.shape[1]),
                "data/n_features": len(feat_cols),
                "data/train_samples": len(train_ds),
                "data/val_samples": len(val_ds),
                "model/n_parameters": n_params,
            },
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    loss_name = cfg["train"].get("loss", "huber").lower()
    if loss_name in {"ic", "ic_loss", "pearson_ic"}:
        loss_fn = ic_loss
    elif loss_name == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.SmoothL1Loss(beta=float(cfg["train"].get("huber_beta", 0.5)))

    num_training_steps = len(train_loader) * cfg["train"]["epochs"]
    scheduler = get_scheduler(opt, cfg, num_training_steps)

    early_metric = cfg["train"].get("early_stop_metric", "val_ic")
    best_metric = -float("inf") if early_metric == "val_ic" else float("inf")
    patience = 0
    out_dir = Path(cfg["output_dir"])
    ckpt = out_dir / "model.pt"

    label_smoothing = 0.0 if loss_name in {"ic", "ic_loss", "pearson_ic"} else cfg["train"].get("label_smoothing", 0.0)
    focal_gamma = cfg["train"].get("focal_gamma", 0.0)

    history = []
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr_losses = []
        tr_base, tr_rank, tr_dir = [], [], []
        for x, y, raw, _, _ in train_loader:
            x, y, raw = x.to(device), y.to(device), raw.to(device)
            opt.zero_grad()
            pred = model(x)
            loss, loss_parts = composite_signal_loss(
                pred,
                y,
                raw,
                loss_fn,
                rank_weight=float(cfg["train"].get("rank_loss_weight", 0.0)),
                direction_weight=float(cfg["train"].get("direction_loss_weight", 0.0)),
                rank_top_frac=float(cfg["train"].get("rank_top_frac", 0.2)),
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
            )
            loss.backward()
            if cfg["train"].get("grad_clip", 0):
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            opt.step()
            scheduler.step()
            tr_losses.append(loss.item())
            tr_base.append(loss_parts["base_loss"])
            tr_rank.append(loss_parts["rank_loss"])
            tr_dir.append(loss_parts["direction_loss"])

        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y, _, _, _ in val_loader:
                x, y = x.to(device), y.to(device)
                va_losses.append(loss_fn(model(x), y).item())

        tr_m = float(np.mean(tr_losses))
        va_m = float(np.mean(va_losses)) if va_losses else tr_m
        val_pred = predict_dataset(model, val_loader, device)
        scores = val_pred.rename(columns={"score": "value"})[["trade_date", "ts_code", "value"]]
        labels = val_pred.rename(columns={"label": "value"})[["trade_date", "ts_code", "value"]]
        ic_stats = ic_summary(daily_ic(scores, labels))
        val_ic = ic_stats["ic_mean"]
        lr = float(opt.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": tr_m,
                "train_base_loss": float(np.mean(tr_base)) if tr_base else tr_m,
                "train_rank_loss": float(np.mean(tr_rank)) if tr_rank else 0.0,
                "train_direction_loss": float(np.mean(tr_dir)) if tr_dir else 0.0,
                "val_loss": va_m,
                "val_ic": val_ic,
                "val_icir": ic_stats["icir"],
                "lr": lr,
            }
        )
        wandb_log(
            wandb_run,
            {
                "epoch": epoch + 1,
                "train/loss": tr_m,
                "train/base_loss": float(np.mean(tr_base)) if tr_base else tr_m,
                "train/rank_loss": float(np.mean(tr_rank)) if tr_rank else 0.0,
                "train/direction_loss": float(np.mean(tr_dir)) if tr_dir else 0.0,
                "val/loss": va_m,
                "val/ic_mean": val_ic,
                "val/icir": ic_stats["icir"],
                "train/lr": lr,
            },
            step=epoch + 1,
        )
        print(
            f"epoch {epoch+1}: train={tr_m:.6f} val={va_m:.6f} "
            f"rank={np.mean(tr_rank) if tr_rank else 0:.4f} "
            f"val_ic={val_ic:.5f} val_icir={ic_stats['icir']:.3f} lr={lr:.2e}"
        )

        current = val_ic if early_metric == "val_ic" else va_m
        improved = current > best_metric if early_metric == "val_ic" else current < best_metric

        if improved:
            best_metric = current
            patience = 0
            wandb_summary_update(
                wandb_run,
                {
                    "best_metric": float(best_metric),
                    "best_epoch": epoch + 1,
                    "best_val_ic": float(val_ic),
                    "best_val_icir": float(ic_stats["icir"]),
                },
            )
            torch.save({
                "model_state": model.state_dict(),
                "feat_cols": feat_cols,
                "seq_len": cfg["seq_len"],
                "model_cfg": cfg["model"],
                "target_col": target_col,
                "best_metric": best_metric,
                "best_epoch": epoch + 1,
            }, ckpt)
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print("Early stop")
                break

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_data["model_state"])

    val_pred = predict_dataset(model, val_loader, device)
    val_pred.to_csv(out_dir / "val_predictions_deep.csv", index=False)
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)

    meta = {
        "feat_cols": feat_cols,
        "history": history,
        "best_metric": best_metric,
        "early_stop_metric": early_metric,
        "target_col": target_col,
        "model": cfg["model"],
        "loss": loss_name,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    wandb_summary_update(
        wandb_run,
        {
            "final/best_metric": float(best_metric),
            "final/history_epochs": len(history),
            "final/target_col": target_col,
            "final/loss": loss_name,
        },
    )
    if cfg.get("wandb", {}).get("log_artifacts", True):
        wandb_log_artifact(wandb_run, ckpt, name="stock-dl-model", artifact_type="model")
        wandb_log_artifact(wandb_run, out_dir / "train_meta.json", name="stock-dl-train-meta", artifact_type="metadata")
        wandb_log_artifact(wandb_run, out_dir / "val_predictions_deep.csv", name="stock-dl-val-predictions-deep", artifact_type="predictions")
    finish_wandb(wandb_run)
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
