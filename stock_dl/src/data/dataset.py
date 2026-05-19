from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class StockWindowDataset(Dataset):
  """Sliding windows with in-window standardization to avoid look-ahead leakage."""

  def __init__(
      self,
      panel: pd.DataFrame,
      feat_cols: list[str],
      seq_len: int,
      split: str,
      train_end: str,
      val_end: str,
      target_col: str = "label_cs_z",
      raw_label_col: str = "label",
      flatten: bool = True,
  ):
      self.feat_cols = feat_cols
      self.seq_len = seq_len
      self.flatten = flatten
      self.features: list[np.ndarray] = []
      self.targets: list[np.ndarray] = []
      self.raw_labels: list[np.ndarray] = []
      self.codes: list[str] = []
      self.dates: list[list[str]] = []
      self.samples: list[tuple[int, int]] = []
      self.sample_dates: list[str] = []

      if target_col not in panel.columns:
          target_col = raw_label_col

      df = panel.copy()
      for code, g in df.groupby("ts_code", sort=False):
          g = g.sort_values("trade_date")
          dates = g["trade_date"].tolist()
          feats = g[feat_cols].astype(np.float32).values
          targets = g[target_col].astype(np.float32).values
          raw_labels = g[raw_label_col].astype(np.float32).values

          group_id = len(self.features)
          self.features.append(feats)
          self.targets.append(targets)
          self.raw_labels.append(raw_labels)
          self.codes.append(str(code))
          self.dates.append(dates)

          for i in range(seq_len - 1, len(g)):
              d = dates[i]
              if not (np.isfinite(targets[i]) and np.isfinite(raw_labels[i])):
                  continue
              if split == "train" and d > train_end:
                  continue
              if split == "val" and (d <= train_end or d > val_end):
                  continue
              self.samples.append((group_id, i))
              self.sample_dates.append(str(d))

  def __len__(self) -> int:
      return len(self.samples)

  def __getitem__(self, idx: int):
      group_id, i = self.samples[idx]
      feats = self.features[group_id]
      window = feats[i - self.seq_len + 1 : i + 1].copy()
      # 只用窗口内部统计量，避免全量标准化暗含未来信息。
      mu = np.nanmean(window, axis=0, keepdims=True)
      std = np.nanstd(window, axis=0, keepdims=True) + 1e-6
      window = (window - mu) / std
      window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
      if self.flatten:
          window = window.reshape(-1)
      y = float(self.targets[group_id][i])
      raw = float(self.raw_labels[group_id][i])
      code = self.codes[group_id]
      date = self.dates[group_id][i]
      return (
          torch.from_numpy(window),
          torch.tensor(y, dtype=torch.float32),
          torch.tensor(raw, dtype=torch.float32),
          code,
          date,
      )


class DateBatchSampler(Sampler[list[int]]):
    """Yield batches grouped by trade_date for cross-sectional ranking losses."""

    def __init__(
        self,
        sample_dates: list[str],
        max_batch_size: int = 2048,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.sample_dates = sample_dates
        self.max_batch_size = max(1, int(max_batch_size))
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        groups: dict[str, list[int]] = {}
        for idx, d in enumerate(sample_dates):
            groups.setdefault(str(d), []).append(idx)
        self.groups = groups
        self.keys = sorted(groups)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self.keys)
        if self.shuffle:
            rng.shuffle(keys)
        for d in keys:
            idxs = np.array(self.groups[d], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.max_batch_size):
                batch = idxs[start : start + self.max_batch_size].tolist()
                if len(batch) > 1:
                    yield batch
        self.epoch += 1

    def __len__(self) -> int:
        return sum((len(v) + self.max_batch_size - 1) // self.max_batch_size for v in self.groups.values())


def save_panel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)


def load_panel(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
