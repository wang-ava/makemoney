#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_TUNE_DIR = "outputs_tuning"
DEFAULT_BASE_CONFIG = "configs/server_8h.yaml"
REGISTRY_NAME = "trial_registry.json"
RESULTS_NAME = "tuning_results.csv"
BEST_JSON_NAME = "best_params.json"
BEST_YAML_NAME = "best_params.yaml"
BEST_CONFIG_NAME = "best_config.yaml"


CURATED_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "gt96_ic_rank_balanced",
        "overrides": {
            "model": {"name": "gru_transformer", "d_model": 96, "num_layers": 1, "dim_feedforward": 192, "head_hidden": 96, "gru_hidden": 48, "dropout": 0.18},
            "train": {"batch_size": 1536, "lr": 5e-4, "weight_decay": 2e-4, "target_col": "label_rank", "loss": "ic", "date_batch": True, "rank_loss_weight": 0.08, "direction_loss_weight": 0.03, "warmup_epochs": 2},
            "lgbm": {"num_boost_round": 260, "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 80, "lambda_l2": 2.0, "feature_fraction": 0.75},
        },
    },
    {
        "name": "gt128_ic_rank",
        "overrides": {
            "model": {"name": "gru_transformer", "d_model": 128, "num_layers": 1, "dim_feedforward": 256, "head_hidden": 128, "gru_hidden": 64, "dropout": 0.20},
            "train": {"batch_size": 1024, "lr": 3e-4, "weight_decay": 1e-4, "target_col": "label_rank", "loss": "ic", "date_batch": True, "rank_loss_weight": 0.10, "direction_loss_weight": 0.05, "warmup_epochs": 3},
            "lgbm": {"num_boost_round": 320, "num_leaves": 63, "learning_rate": 0.04, "min_data_in_leaf": 60, "lambda_l2": 1.5, "feature_fraction": 0.80},
        },
    },
    {
        "name": "gt64_ic_fast",
        "overrides": {
            "model": {"name": "gru_transformer", "d_model": 64, "num_layers": 1, "dim_feedforward": 128, "head_hidden": 64, "gru_hidden": 32, "dropout": 0.15},
            "train": {"batch_size": 2048, "lr": 8e-4, "weight_decay": 2e-4, "target_col": "label_rank", "loss": "ic", "date_batch": True, "rank_loss_weight": 0.06, "direction_loss_weight": 0.02, "warmup_epochs": 1},
            "lgbm": {"num_boost_round": 240, "num_leaves": 31, "learning_rate": 0.06, "min_data_in_leaf": 100, "lambda_l2": 2.0, "feature_fraction": 0.75},
        },
    },
    {
        "name": "gt96_huber_csz",
        "overrides": {
            "model": {"name": "gru_transformer", "d_model": 96, "num_layers": 1, "dim_feedforward": 192, "head_hidden": 96, "gru_hidden": 48, "dropout": 0.20},
            "train": {"batch_size": 1536, "lr": 5e-4, "weight_decay": 2e-4, "target_col": "label_cs_z", "loss": "huber", "date_batch": False, "rank_loss_weight": 0.02, "direction_loss_weight": 0.02, "label_smoothing": 0.03, "warmup_epochs": 2},
            "lgbm": {"num_boost_round": 260, "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 80, "lambda_l2": 2.0, "feature_fraction": 0.75},
        },
    },
    {
        "name": "gt96_two_layer_ic",
        "overrides": {
            "model": {"name": "gru_transformer", "d_model": 96, "num_layers": 2, "dim_feedforward": 192, "head_hidden": 96, "gru_hidden": 48, "dropout": 0.22},
            "train": {"batch_size": 1024, "lr": 3e-4, "weight_decay": 1e-4, "target_col": "label_rank", "loss": "ic", "date_batch": True, "rank_loss_weight": 0.08, "direction_loss_weight": 0.03, "warmup_epochs": 3},
            "lgbm": {"num_boost_round": 300, "num_leaves": 31, "learning_rate": 0.04, "min_data_in_leaf": 80, "lambda_l2": 2.0, "feature_fraction": 0.80},
        },
    },
    {
        "name": "gru_attn96_ic",
        "overrides": {
            "model": {"name": "gru_attention", "d_model": 96, "num_layers": 2, "head_hidden": 96, "dropout": 0.18},
            "train": {"batch_size": 1536, "lr": 6e-4, "weight_decay": 2e-4, "target_col": "label_rank", "loss": "ic", "date_batch": True, "rank_loss_weight": 0.08, "direction_loss_weight": 0.03, "warmup_epochs": 2},
            "lgbm": {"num_boost_round": 260, "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 80, "lambda_l2": 2.0, "feature_fraction": 0.75},
        },
    },
    {
        "name": "bilstm96_huber",
        "overrides": {
            "model": {"name": "bilstm_attention", "d_model": 96, "num_layers": 2, "head_hidden": 96, "dropout": 0.18},
            "train": {"batch_size": 1536, "lr": 5e-4, "weight_decay": 2e-4, "target_col": "label_cs_z", "loss": "huber", "date_batch": False, "rank_loss_weight": 0.02, "direction_loss_weight": 0.02, "label_smoothing": 0.03, "warmup_epochs": 2},
            "lgbm": {"num_boost_round": 260, "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 80, "lambda_l2": 2.0, "feature_fraction": 0.75},
        },
    },
    {
        "name": "temporal_cnn96_huber",
        "overrides": {
            "model": {"name": "temporal_cnn", "d_model": 96, "num_layers": 4, "kernel_size": 3, "head_hidden": 96, "dropout": 0.18},
            "train": {"batch_size": 2048, "lr": 8e-4, "weight_decay": 2e-4, "target_col": "label_cs_z", "loss": "huber", "date_batch": False, "rank_loss_weight": 0.02, "direction_loss_weight": 0.02, "label_smoothing": 0.03, "warmup_epochs": 1},
            "lgbm": {"num_boost_round": 260, "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 80, "lambda_l2": 2.0, "feature_fraction": 0.75},
        },
    },
    {
        "name": "inception96_huber",
        "overrides": {
            "model": {"name": "inception_time", "d_model": 96, "num_inception_blocks": 3, "head_hidden": 96, "dropout": 0.18},
            "train": {"batch_size": 2048, "lr": 8e-4, "weight_decay": 2e-4, "target_col": "label_cs_z", "loss": "huber", "date_batch": False, "rank_loss_weight": 0.02, "direction_loss_weight": 0.02, "label_smoothing": 0.03, "warmup_epochs": 1},
            "lgbm": {"num_boost_round": 260, "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 80, "lambda_l2": 2.0, "feature_fraction": 0.75},
        },
    },
    {
        "name": "gt128_huber_csz",
        "overrides": {
            "model": {"name": "gru_transformer", "d_model": 128, "num_layers": 1, "dim_feedforward": 256, "head_hidden": 128, "gru_hidden": 64, "dropout": 0.22},
            "train": {"batch_size": 1024, "lr": 4e-4, "weight_decay": 2e-4, "target_col": "label_cs_z", "loss": "huber", "date_batch": False, "rank_loss_weight": 0.03, "direction_loss_weight": 0.03, "label_smoothing": 0.05, "warmup_epochs": 2},
            "lgbm": {"num_boost_round": 320, "num_leaves": 63, "learning_rate": 0.04, "min_data_in_leaf": 60, "lambda_l2": 1.5, "feature_fraction": 0.80},
        },
    },
]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    atomic_write_text(path, text)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        else:
            flat[name] = value
    return flat


def candidate_hash(candidate: dict[str, Any]) -> str:
    payload = yaml.safe_dump(candidate["overrides"], sort_keys=True, allow_unicode=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(path: Path, registry: list[dict[str, Any]]) -> None:
    write_json(path, registry)


def active_registry(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stale_hours = float(os.environ.get("TUNE_STALE_HOURS", "10"))
    cutoff = time.time() - stale_hours * 3600
    for row in registry:
        if row.get("status") == "running" and float(row.get("heartbeat_ts", row.get("started_ts", 0))) < cutoff:
            row["status"] = "stale"
            row["finished_at"] = now_iso()
    return registry


class FileLock:
    def __init__(self, path: Path, timeout: int = 120, stale_after: int | None = 600):
        self.path = path
        self.timeout = timeout
        self.stale_after = stale_after
        self.fd: int | None = None
        self.token = f"{os.getpid()}-{uuid.uuid4().hex}"

    def __enter__(self):
        start = time.time()
        while True:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{self.token} {now_iso()}\n".encode("utf-8"))
                return self
            except FileExistsError:
                try:
                    lock_age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if self.stale_after is not None and lock_age > self.stale_after:
                    self.path.unlink(missing_ok=True)
                    continue
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Could not acquire tuning lock: {self.path}")
                time.sleep(2)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
        try:
            current = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        if current.startswith(self.token):
            self.path.unlink(missing_ok=True)


def rng_choice(rng: np.random.Generator, values: list[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def random_candidate(trial_number: int, completed: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed + trial_number * 9973)
    exploit = completed and rng.random() < 0.65
    if exploit:
        best = max(completed, key=lambda row: float(row.get("objective", -np.inf)))
        overrides = copy.deepcopy(best["candidate"]["overrides"])
        name = f"mutate_{best['trial_id']}_{trial_number:04d}"
    else:
        model_name = rng_choice(rng, ["gru_transformer", "gru_attention", "bilstm_attention", "temporal_cnn", "inception_time"])
        d_model = int(rng_choice(rng, [64, 96, 128]))
        loss_combo = rng_choice(
            rng,
            [
                ("label_rank", "ic", True, 0.06, 0.02),
                ("label_rank", "ic", True, 0.10, 0.05),
                ("label_cs_z", "huber", False, 0.02, 0.02),
                ("label_cs_z", "huber", False, 0.04, 0.03),
            ],
        )
        target_col, loss, date_batch, rank_weight, direction_weight = loss_combo
        model = {
            "name": model_name,
            "d_model": d_model,
            "num_layers": int(rng_choice(rng, [1, 2])),
            "dim_feedforward": d_model * 2,
            "head_hidden": d_model,
            "gru_hidden": max(d_model // 2, 1),
            "dropout": float(rng_choice(rng, [0.12, 0.15, 0.18, 0.22, 0.26])),
        }
        if model_name == "temporal_cnn":
            model["num_layers"] = int(rng_choice(rng, [3, 4, 5]))
            model["kernel_size"] = int(rng_choice(rng, [3, 5]))
        if model_name == "inception_time":
            model["num_inception_blocks"] = int(rng_choice(rng, [2, 3, 4]))
        overrides = {
            "model": model,
            "train": {
                "batch_size": int(rng_choice(rng, [1024, 1536, 2048])),
                "lr": float(rng_choice(rng, [2e-4, 3e-4, 5e-4, 8e-4])),
                "weight_decay": float(rng_choice(rng, [1e-4, 2e-4, 5e-4])),
                "target_col": target_col,
                "loss": loss,
                "date_batch": bool(date_batch),
                "rank_loss_weight": float(rank_weight),
                "direction_loss_weight": float(direction_weight),
                "label_smoothing": 0.0 if loss == "ic" else float(rng_choice(rng, [0.0, 0.03, 0.05])),
                "warmup_epochs": int(rng_choice(rng, [1, 2, 3])),
            },
            "lgbm": {
                "num_boost_round": int(rng_choice(rng, [220, 260, 320, 400])),
                "num_leaves": int(rng_choice(rng, [31, 47, 63])),
                "learning_rate": float(rng_choice(rng, [0.03, 0.04, 0.05, 0.06])),
                "min_data_in_leaf": int(rng_choice(rng, [50, 80, 120])),
                "lambda_l2": float(rng_choice(rng, [1.0, 2.0, 4.0])),
                "feature_fraction": float(rng_choice(rng, [0.70, 0.75, 0.85])),
            },
        }
        name = f"random_{trial_number:04d}"

    if exploit:
        train = overrides.setdefault("train", {})
        model = overrides.setdefault("model", {})
        lgbm = overrides.setdefault("lgbm", {})
        train["lr"] = float(np.clip(float(train.get("lr", 5e-4)) * rng_choice(rng, [0.7, 0.85, 1.15, 1.3]), 1e-4, 1.2e-3))
        train["weight_decay"] = float(rng_choice(rng, [1e-4, 2e-4, 5e-4]))
        if "dropout" in model:
            model["dropout"] = float(np.clip(float(model["dropout"]) + rng_choice(rng, [-0.04, -0.02, 0.02, 0.04]), 0.08, 0.32))
        train["rank_loss_weight"] = float(np.clip(float(train.get("rank_loss_weight", 0.06)) + rng_choice(rng, [-0.03, -0.01, 0.01, 0.03]), 0.0, 0.15))
        train["direction_loss_weight"] = float(np.clip(float(train.get("direction_loss_weight", 0.03)) + rng_choice(rng, [-0.02, 0.0, 0.02]), 0.0, 0.08))
        lgbm["num_leaves"] = int(rng_choice(rng, [31, 47, 63]))
        lgbm["min_data_in_leaf"] = int(rng_choice(rng, [50, 80, 120]))
        lgbm["lambda_l2"] = float(rng_choice(rng, [1.0, 2.0, 4.0]))

    return {"name": name, "overrides": overrides}


def next_candidate(trial_number: int, registry: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    attempted = {row.get("hash") for row in registry if row.get("status") in {"running", "complete", "failed"}}
    completed = [row for row in registry if row.get("status") == "complete" and "objective" in row]
    for candidate in CURATED_CANDIDATES:
        if candidate_hash(candidate) not in attempted:
            return candidate
    for offset in range(500):
        candidate = random_candidate(trial_number + offset, completed, seed=seed)
        if candidate_hash(candidate) not in attempted:
            return candidate
    raise RuntimeError("Could not find an unused tuning candidate.")


def make_trial_config(base_cfg: dict[str, Any], candidate: dict[str, Any], trial_id: str, trial_dir: Path, max_minutes: str | None) -> dict[str, Any]:
    cfg = deep_update(base_cfg, candidate["overrides"])
    cfg["output_dir"] = str(trial_dir.relative_to(ROOT))
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(os.environ.get("TUNE_EPOCHS", cfg["train"].get("epochs", 18)))
    if max_minutes:
        cfg["train"]["max_minutes"] = float(max_minutes)
    cfg.setdefault("wandb", {})
    cfg["wandb"]["run_name"] = f"stock-dl-tune-{trial_id}-{candidate['name']}"
    cfg["wandb"]["group"] = os.environ.get("WANDB_GROUP", "finalex-tune")
    cfg["wandb"]["watch"] = False
    tags = list(cfg["wandb"].get("tags", []))
    for tag in ["tune", trial_id, candidate["name"]]:
        if tag not in tags:
            tags.append(tag)
    cfg["wandb"]["tags"] = tags
    cfg["wandb"]["notes"] = f"Auto tuning trial {trial_id}: {candidate['name']}"
    return cfg


def build_shared_panel(base_cfg: dict[str, Any], tune_dir: Path, force: bool) -> Path:
    shared_dir = tune_dir / "shared_panel"
    panel_path = shared_dir / "panel.parquet"
    with FileLock(tune_dir / ".panel.lock", timeout=7 * 3600, stale_after=9 * 3600):
        if panel_path.exists() and not force:
            print(f"Shared panel exists: {panel_path}")
            return panel_path
        cfg = copy.deepcopy(base_cfg)
        cfg["output_dir"] = str(shared_dir.relative_to(ROOT))
        cfg_path = shared_dir / "shared_panel_config.yaml"
        write_yaml(cfg_path, cfg)
        run_step(["python", "scripts/01_build_panel.py", "--config", str(cfg_path.relative_to(ROOT))], "build shared panel")
    return panel_path


def link_panel(shared_panel: Path, trial_dir: Path) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    target = trial_dir / "panel.parquet"
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(shared_panel)
    except OSError:
        shutil.copy2(shared_panel, target)


def run_step(cmd: list[str], name: str, env: dict[str, str] | None = None) -> None:
    print(f"== {name} ==")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pick_backtest_core(metrics: dict[str, Any]) -> dict[str, float]:
    keys = ["total_return", "annual_return", "sharpe", "max_drawdown", "turnover", "daily_win_rate"]
    return {key: float(metrics.get(key, 0.0) or 0.0) for key in keys}


def collect_metrics(trial_id: str, candidate: dict[str, Any], config_path: Path, trial_dir: Path) -> dict[str, Any]:
    ic = read_json_or_empty(trial_dir / "ic_summary.json")
    bt = pick_backtest_core(read_json_or_empty(trial_dir / "backtest_metrics.json"))
    blend = read_json_or_empty(trial_dir / "blend_meta.json")
    train_meta = read_json_or_empty(trial_dir / "train_meta.json")
    lgbm_meta = read_json_or_empty(trial_dir / "lgbm_meta.json")
    history = train_meta.get("history", [])
    best_epoch = None
    if history:
        best_epoch = max(history, key=lambda row: float(row.get("val_ic", -np.inf))).get("epoch")
    row = {
        "trial_id": trial_id,
        "candidate_name": candidate["name"],
        "status": "complete",
        "objective": float(ic.get("ic_mean", 0.0) or 0.0),
        "ic_mean": float(ic.get("ic_mean", 0.0) or 0.0),
        "ic_std": float(ic.get("ic_std", 0.0) or 0.0),
        "icir": float(ic.get("icir", 0.0) or 0.0),
        "best_alpha": float(blend.get("best_alpha", 1.0) or 1.0),
        "best_epoch": int(best_epoch or 0),
        "lgbm_best_iteration": int(lgbm_meta.get("best_iteration", 0) or 0),
        "config_path": str(config_path.relative_to(ROOT)),
        "output_dir": str(trial_dir.relative_to(ROOT)),
        **bt,
    }
    row.update({f"param.{k}": v for k, v in flatten_dict(candidate["overrides"]).items()})
    return row


def append_results(tune_dir: Path, row: dict[str, Any]) -> None:
    path = tune_dir / RESULTS_NAME
    rows = []
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [old for old in rows if old.get("trial_id") != row["trial_id"]]
    rows.append({k: "" if v is None else v for k, v in row.items()})
    fieldnames: list[str] = []
    for item in rows:
        for key in item:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_completed_results(tune_dir: Path) -> list[dict[str, Any]]:
    path = tune_dir / RESULTS_NAME
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "complete"]


def update_best_files(tune_dir: Path) -> None:
    rows = load_completed_results(tune_dir)
    if not rows:
        return
    best = max(rows, key=lambda row: float(row.get("objective", 0.0) or 0.0))
    config_path = ROOT / best["config_path"]
    best_cfg = read_yaml(config_path)
    payload = {
        "selected_by": "max validation ic_mean",
        "updated_at": now_iso(),
        "best_trial": best,
        "best_config_path": best["config_path"],
        "best_output_dir": best["output_dir"],
        "best_overrides": {
            "model": best_cfg.get("model", {}),
            "train": best_cfg.get("train", {}),
            "lgbm": best_cfg.get("lgbm", {}),
            "ensemble": best_cfg.get("ensemble", {}),
            "strategy": best_cfg.get("strategy", {}),
        },
    }
    write_json(tune_dir / BEST_JSON_NAME, payload)
    write_yaml(tune_dir / BEST_YAML_NAME, payload)
    best_config = tune_dir / BEST_CONFIG_NAME
    tmp_config = best_config.with_name(f".{best_config.name}.{os.getpid()}.tmp")
    shutil.copy2(config_path, tmp_config)
    tmp_config.replace(best_config)
    best_link = tune_dir / "best_trial"
    if best_link.exists() or best_link.is_symlink():
        best_link.unlink()
    try:
        best_link.symlink_to(ROOT / best["output_dir"])
    except OSError:
        pass


def reserve_trial(tune_dir: Path, base_cfg: dict[str, Any], seed: int, max_minutes: str | None) -> tuple[str, dict[str, Any], Path, Path]:
    with FileLock(tune_dir / ".tune.lock"):
        registry_path = tune_dir / REGISTRY_NAME
        registry = active_registry(load_registry(registry_path))
        trial_number = max([int(str(row.get("trial_id", "trial_0000")).split("_")[-1]) for row in registry] or [0]) + 1
        trial_id = f"trial_{trial_number:04d}"
        candidate = next_candidate(trial_number, registry, seed)
        trial_dir = tune_dir / "trials" / trial_id
        config_path = trial_dir / "config.yaml"
        cfg = make_trial_config(base_cfg, candidate, trial_id, trial_dir, max_minutes)
        write_yaml(config_path, cfg)
        row = {
            "trial_id": trial_id,
            "status": "running",
            "candidate": candidate,
            "hash": candidate_hash(candidate),
            "config_path": str(config_path.relative_to(ROOT)),
            "output_dir": str(trial_dir.relative_to(ROOT)),
            "started_at": now_iso(),
            "started_ts": time.time(),
            "heartbeat_ts": time.time(),
        }
        registry.append(row)
        save_registry(registry_path, registry)
    return trial_id, candidate, config_path, trial_dir


def mark_trial(tune_dir: Path, trial_id: str, status: str, metrics: dict[str, Any] | None = None, error: str | None = None) -> None:
    with FileLock(tune_dir / ".tune.lock"):
        registry_path = tune_dir / REGISTRY_NAME
        registry = load_registry(registry_path)
        found = False
        for row in registry:
            if row.get("trial_id") == trial_id:
                found = True
                row["status"] = status
                row["finished_at"] = now_iso()
                row["heartbeat_ts"] = time.time()
                if metrics:
                    row["objective"] = metrics.get("objective")
                    row["metrics"] = {k: v for k, v in metrics.items() if not k.startswith("param.")}
                if error:
                    row["error"] = error[-2000:]
                break
        if not found:
            raise KeyError(f"Trial {trial_id} is not registered in {registry_path}")
        save_registry(registry_path, registry)
        if metrics:
            append_results(tune_dir, metrics)
            update_best_files(tune_dir)


def run_trial(trial_id: str, config_path: Path, shared_panel: Path, trial_dir: Path, run_lgbm: bool) -> None:
    link_panel(shared_panel, trial_dir)
    rel_config = str(config_path.relative_to(ROOT))
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = f"stockdl-tune-{trial_id}"
    run_step(["python", "scripts/03_train.py", "--config", rel_config], "train deep model", env=env)
    if run_lgbm:
        run_step(["python", "scripts/10_train_lgbm.py", "--config", rel_config], "train lgbm", env=env)
    run_step(["python", "scripts/11_blend_scores.py", "--config", rel_config], "blend scores", env=env)
    run_step(["python", "scripts/04_eval_ic.py", "--config", rel_config], "evaluate ic", env=env)
    run_step(["python", "scripts/05_backtest.py", "--config", rel_config], "backtest", env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=os.environ.get("TUNE_BASE_CONFIG", DEFAULT_BASE_CONFIG))
    parser.add_argument("--tune-dir", default=os.environ.get("TUNE_DIR", DEFAULT_TUNE_DIR))
    parser.add_argument("--trials", type=int, default=int(os.environ.get("TRIALS_PER_JOB", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TUNE_SEED", "42")))
    parser.add_argument("--max-minutes", default=os.environ.get("TRIAL_MAX_MINUTES"))
    parser.add_argument("--skip-lgbm", action="store_true", default=os.environ.get("TUNE_SKIP_LGBM", "0") == "1")
    parser.add_argument("--force-panel", action="store_true", default=os.environ.get("FORCE_TUNE_PANEL", "0") == "1")
    args = parser.parse_args()

    base_config_path = (ROOT / args.base_config).resolve()
    tune_dir = (ROOT / args.tune_dir).resolve()
    tune_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = read_yaml(base_config_path)

    shared_panel = build_shared_panel(base_cfg, tune_dir, force=args.force_panel)
    for _ in range(args.trials):
        trial_id, candidate, config_path, trial_dir = reserve_trial(tune_dir, base_cfg, args.seed, args.max_minutes)
        print(f"Reserved {trial_id}: {candidate['name']}")
        try:
            run_trial(trial_id, config_path, shared_panel, trial_dir, run_lgbm=not args.skip_lgbm)
            metrics = collect_metrics(trial_id, candidate, config_path, trial_dir)
            mark_trial(tune_dir, trial_id, "complete", metrics=metrics)
            print(f"Completed {trial_id}: objective/ic_mean={metrics['objective']:.6f}")
            best = read_json_or_empty(tune_dir / BEST_JSON_NAME).get("best_trial", {})
            if best:
                print(f"Current best: {best.get('trial_id')} {best.get('candidate_name')} ic_mean={float(best.get('ic_mean', 0.0)):.6f}")
        except Exception as exc:
            mark_trial(tune_dir, trial_id, "failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
