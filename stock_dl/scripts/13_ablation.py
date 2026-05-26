#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataset import load_panel, save_panel
from src.data.features import feature_columns


VARIANTS = [
    "full",
    "no_gru",
    "no_transformer",
    "no_attention",
    "no_rank_loss",
    "no_direction_loss",
    "short_seq",
    "fewer_features",
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def run_script(script: str, config_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--config", str(config_path)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def mutate_cfg(base_cfg: dict[str, Any], variant: str, out_dir: Path, max_epochs: int | None) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["output_dir"] = str(out_dir)
    cfg.setdefault("wandb", {})["enabled"] = False
    cfg.setdefault("wandb", {})["log_artifacts"] = False
    cfg.setdefault("lgbm", {})["enabled"] = False
    cfg.setdefault("strategy", {})["auto_tune"] = False

    if max_epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(max_epochs)
        cfg["train"]["early_stop_patience"] = min(int(cfg["train"].get("early_stop_patience", max_epochs)), int(max_epochs))

    model = cfg.setdefault("model", {})
    train = cfg.setdefault("train", {})
    if variant == "no_gru":
        model["name"] = "temporal_attention"
    elif variant == "no_transformer":
        model["name"] = "gru_attention"
    elif variant == "no_attention":
        model.clear()
        model.update({"name": "mlp", "hidden_dims": [128, 64], "dropout": 0.2})
    elif variant == "no_rank_loss":
        train["rank_loss_weight"] = 0.0
    elif variant == "no_direction_loss":
        train["direction_loss_weight"] = 0.0
    elif variant == "short_seq":
        cfg["seq_len"] = max(5, int(cfg.get("seq_len", 20)) // 2)
    elif variant == "fewer_features":
        pass
    elif variant != "full":
        raise ValueError(f"Unknown ablation variant: {variant}")
    return cfg


def prepare_panel(shared_panel: Path, variant_dir: Path, variant: str) -> None:
    variant_panel = variant_dir / "panel.parquet"
    variant_dir.mkdir(parents=True, exist_ok=True)
    if variant_panel.exists():
        return
    shutil.copy2(shared_panel, variant_panel)
    if variant != "fewer_features":
        return

    panel = load_panel(variant_panel)
    feat_cols = feature_columns(panel)
    keep_features = feat_cols[: max(1, len(feat_cols) // 2)]
    keep_cols = [c for c in panel.columns if c not in feat_cols or c in keep_features]
    save_panel(panel[keep_cols], variant_panel)


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)) and pd.notna(value):
            flat[name] = float(value)
    return flat


def collect_result(variant: str, out_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": variant}
    ic_fp = out_dir / "ic_summary.json"
    bt_fp = out_dir / "backtest_metrics.json"
    meta_fp = out_dir / "train_meta.json"
    if ic_fp.exists():
        row.update(flatten_metrics(json.loads(ic_fp.read_text(encoding="utf-8"))))
    if bt_fp.exists():
        row.update(flatten_metrics(json.loads(bt_fp.read_text(encoding="utf-8")), "backtest"))
    if meta_fp.exists():
        meta = json.loads(meta_fp.read_text(encoding="utf-8"))
        row["best_metric"] = float(meta.get("best_metric", 0.0))
        row["epochs"] = len(meta.get("history", []))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/quick.yaml"))
    parser.add_argument("--variants", default=",".join(VARIANTS), help="逗号分隔的消融变体")
    parser.add_argument("--max-epochs", type=int, default=None, help="覆盖训练 epoch，便于快速试跑")
    parser.add_argument("--force", action="store_true", help="重新训练已存在结果的变体")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_cfg = read_yaml(config_path)
    resolved_cfg = load_config(config_path)
    base_cfg["data_dir"] = resolved_cfg["data_dir"]

    ablation_dir = Path(resolved_cfg["output_dir"]) / "ablation"
    config_dir = ablation_dir / "_configs"
    shared_dir = ablation_dir / "_shared_panel"
    shared_cfg = json.loads(json.dumps(base_cfg))
    shared_cfg["output_dir"] = str(shared_dir)
    shared_cfg.setdefault("wandb", {})["enabled"] = False
    shared_config_path = config_dir / "_shared_panel.yaml"
    write_yaml(shared_config_path, shared_cfg)

    shared_panel = shared_dir / "panel.parquet"
    if args.force and shared_panel.exists():
        shared_panel.unlink()
    if not shared_panel.exists():
        run_script("01_build_panel.py", shared_config_path)

    rows = []
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    for variant in requested:
        out_dir = ablation_dir / variant
        cfg = mutate_cfg(base_cfg, variant, out_dir, args.max_epochs)
        cfg_path = config_dir / f"{variant}.yaml"
        write_yaml(cfg_path, cfg)
        if args.force and out_dir.exists():
            for fp in ["model.pt", "val_predictions.csv", "daily_ic.csv", "ic_summary.json", "backtest_metrics.json"]:
                target = out_dir / fp
                if target.exists():
                    target.unlink()
        prepare_panel(shared_panel, out_dir, variant)

        if args.force or not ((out_dir / "ic_summary.json").exists() and (out_dir / "backtest_metrics.json").exists()):
            run_script("03_train.py", cfg_path)
            run_script("04_eval_ic.py", cfg_path)
            run_script("05_backtest.py", cfg_path)
        rows.append(collect_result(variant, out_dir))

    result = pd.DataFrame(rows)
    if "ic_mean" in result.columns:
        result = result.sort_values("ic_mean", ascending=False)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(ablation_dir / "ablation_results.csv", index=False)
    (ablation_dir / "ablation_results.json").write_text(
        json.dumps(result.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if "full" in set(result["variant"]) and "ic_mean" in result.columns:
        full_ic = float(result.loc[result["variant"] == "full", "ic_mean"].iloc[0])
        result["delta_ic_vs_full"] = result["ic_mean"] - full_ic
    show_cols = [c for c in ["variant", "ic_mean", "icir", "direction_accuracy", "backtest/total_return", "backtest/sharpe", "delta_ic_vs_full"] if c in result.columns]
    print(result[show_cols].to_string(index=False))
    print(f"Saved ablation results under {ablation_dir}")


if __name__ == "__main__":
    main()
