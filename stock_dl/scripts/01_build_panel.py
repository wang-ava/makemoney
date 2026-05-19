#!/usr/bin/env python3
"""合并 daily/metric/moneyflow 为 panel.parquet"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.features import add_features
from src.data.panel import build_panel
from src.data.dataset import save_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    panel = build_panel(
        cfg["data_dir"],
        cfg["start_date"],
        cfg["end_date"],
        use_metric=cfg["features"]["use_metric"],
        use_moneyflow=cfg["features"]["use_moneyflow"],
        use_market=cfg["features"].get("use_market", True),
        use_news=cfg["features"].get("use_news", False),
        universe=cfg["universe"],
    )
    panel = add_features(
        panel,
        cross_section_rank=cfg["features"]["cross_section_rank"],
        label_horizon=cfg.get("label_horizon", 1),
        fill_missing=cfg["features"].get("fill_missing", True),
    )

    out = Path(cfg["output_dir"]) / "panel.parquet"
    save_panel(panel, out)
    print(f"Saved panel: {out} shape={panel.shape}")


if __name__ == "__main__":
    main()
