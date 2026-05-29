from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _looks_like_data_dir(path: Path) -> bool:
    return (path / "daily").is_dir() and (path / "basic.csv").exists()


def _latest_daily_date(data_dir: Path) -> str | None:
    daily_dir = data_dir / "daily"
    if not daily_dir.is_dir():
        return None
    dates = sorted(p.stem for p in daily_dir.glob("*.csv") if p.stem.isdigit())
    return dates[-1] if dates else None


def _resolve_auto_data_dir(root: Path, cfg: dict[str, Any]) -> Path:
    patterns = cfg.get("data_dir_glob") or [
        "../A股数据",           # A股数据优先
        "../documents-export-*",
        "documents-export-*",
    ]
    if isinstance(patterns, str):
        patterns = [patterns]

    candidates: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            matches = pattern_path.parent.glob(pattern_path.name)
        else:
            matches = root.glob(pattern)
        for match in matches:
            path = match.resolve()
            if path.is_dir() and _looks_like_data_dir(path):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "data_dir is auto, but no documents-export-* data folder was found. "
            "Put the synced course data next to the project, or set data_dir explicitly."
        )

    return max(candidates, key=lambda p: (_latest_daily_date(p) or "", p.stat().st_mtime))


def _resolve_data_dir(root: Path, cfg: dict[str, Any]) -> Path:
    raw_data_dir = str(cfg.get("data_dir", "auto"))
    if raw_data_dir.lower() in {"auto", "latest"}:
        return _resolve_auto_data_dir(root, cfg)

    data_dir = Path(raw_data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    return data_dir


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(path).resolve().parent.parent
    data_dir = _resolve_data_dir(root, cfg)
    cfg["data_dir"] = str(data_dir)

    latest_date = _latest_daily_date(data_dir)
    for key in ("end_date", "val_end"):
        value = str(cfg.get(key, "")).lower()
        if value in {"", "auto", "latest"}:
            if not latest_date:
                raise FileNotFoundError(f"Cannot infer {key}: no daily/*.csv files found in {data_dir}")
            cfg[key] = latest_date

    out_dir = Path(cfg.get("output_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    cfg["output_dir"] = str(out_dir)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg
