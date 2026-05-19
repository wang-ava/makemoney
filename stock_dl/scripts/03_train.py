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
from src.models.losses import composite_signal_loss


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

    panel = load_panel(Path(cfg["output_dir"]) / "panel.parquet")
    feat_cols = feature_columns(panel)
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
            num_workers=cfg["train"].get("num_workers", 0),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"].get("num_workers", 0),
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    if cfg["train"].get("loss", "huber").lower() == "mse":
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

    label_smoothing = cfg["train"].get("label_smoothing", 0.0)
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

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state"])

    val_pred = predict_dataset(model, val_loader, device)
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)

    meta = {
        "feat_cols": feat_cols,
        "history": history,
        "best_metric": best_metric,
        "early_stop_metric": early_metric,
        "target_col": target_col,
        "model": cfg["model"],
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
