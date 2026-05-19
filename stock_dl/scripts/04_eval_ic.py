#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.metrics.ic import daily_ic, ic_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    pred = pd.read_csv(out / "val_predictions.csv")
    scores = pred.rename(columns={"score": "value"})[["trade_date", "ts_code", "value"]]
    labels = pred.rename(columns={"label": "value"})[["trade_date", "ts_code", "value"]]

    ic_df = daily_ic(scores, labels)
    summary = ic_summary(ic_df)
    ic_df.to_csv(out / "daily_ic.csv", index=False)
    (out / "ic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("IC summary:", summary)


if __name__ == "__main__":
    main()
