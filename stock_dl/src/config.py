from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(path).resolve().parent.parent
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    cfg["data_dir"] = str(data_dir)
    out_dir = Path(cfg.get("output_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    cfg["output_dir"] = str(out_dir)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg
