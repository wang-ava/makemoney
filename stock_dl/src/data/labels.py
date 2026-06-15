from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_CLOSE_TO_NEXT_CLOSE = {"close_to_next_close", "close_to_future_close", "close_close"}
_NEXT_OPEN_TO_NEXT_CLOSE = {"next_open_to_next_close", "next_open_to_future_close", "open_close"}
_CLOSE_TO_NEXT_OPEN = {"close_to_next_open", "overnight_gap"}


@dataclass(frozen=True)
class LabelValidationResult:
    mode: str
    horizon: int
    rows_compared: int
    bad_rows: int
    bad_ratio: float
    mean_abs_diff: float
    max_abs_diff: float
    nan_mismatch_rows: int
    close_to_next_close_mean_abs_diff: float | None = None

    @property
    def ok(self) -> bool:
        return self.bad_rows == 0

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "mode": self.mode,
            "horizon": self.horizon,
            "rows_compared": self.rows_compared,
            "bad_rows": self.bad_rows,
            "bad_ratio": self.bad_ratio,
            "mean_abs_diff": self.mean_abs_diff,
            "max_abs_diff": self.max_abs_diff,
            "nan_mismatch_rows": self.nan_mismatch_rows,
            "close_to_next_close_mean_abs_diff": self.close_to_next_close_mean_abs_diff,
        }


def normalize_label_mode(label_mode: str | None) -> str:
    mode = str(label_mode or "close_to_next_close").lower()
    if mode in _CLOSE_TO_NEXT_CLOSE:
        return "close_to_next_close"
    if mode in _NEXT_OPEN_TO_NEXT_CLOSE:
        return "next_open_to_next_close"
    if mode in _CLOSE_TO_NEXT_OPEN:
        return "close_to_next_open"
    raise ValueError(
        "Unsupported label_mode="
        f"{label_mode!r}; expected close_to_next_close, next_open_to_next_close, "
        "or close_to_next_open"
    )


def validate_checkpoint_label_config(
    ckpt: dict,
    *,
    label_mode: str | None,
    label_horizon: int = 1,
    require_metadata: bool = True,
) -> None:
    ckpt_mode = ckpt.get("label_mode")
    ckpt_horizon = ckpt.get("label_horizon")
    if ckpt_mode is None or ckpt_horizon is None:
        if require_metadata:
            raise ValueError(
                "Checkpoint is missing label_mode/label_horizon metadata. "
                "Retrain the model with the current code before generating trading orders."
            )
        return

    expected_mode = normalize_label_mode(label_mode)
    actual_mode = normalize_label_mode(str(ckpt_mode))
    expected_horizon = max(int(label_horizon), 1)
    actual_horizon = max(int(ckpt_horizon), 1)
    if actual_mode != expected_mode or actual_horizon != expected_horizon:
        raise ValueError(
            "Checkpoint label config mismatch: "
            f"checkpoint={actual_mode}/horizon={actual_horizon}, "
            f"config={expected_mode}/horizon={expected_horizon}. "
            "Use the matching config or retrain the model."
        )


def _working_price_frame(panel: pd.DataFrame, require_label: bool = False) -> pd.DataFrame:
    cols = ["ts_code", "trade_date", "close"]
    if "open" in panel.columns:
        cols.append("open")
    if "vol" in panel.columns:
        cols.append("vol")
    if "amount" in panel.columns:
        cols.append("amount")
    if require_label:
        cols.append("label")

    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing required label columns: {', '.join(missing)}")

    df = panel.loc[:, cols].copy()
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values(["ts_code", "trade_date"])


def calculate_label_series(
    panel: pd.DataFrame,
    *,
    label_mode: str | None,
    label_horizon: int = 1,
    tradable_label_filter: bool = True,
    label_limit_up_pct: float = 9.5,
) -> pd.Series:
    """Calculate the configured raw label and return it aligned to panel.index."""
    mode = normalize_label_mode(label_mode)
    df = _working_price_frame(panel, require_label=False)
    by_code = df.groupby("ts_code", sort=False)
    horizon = max(int(label_horizon), 1)
    close = df["close"].replace(0, np.nan)
    future_close = by_code["close"].shift(-horizon)

    if mode == "close_to_next_close":
        label = future_close / close - 1.0
    elif mode == "next_open_to_next_close":
        if "open" not in df.columns:
            raise ValueError("label_mode=next_open_to_next_close requires an open column")
        entry_open = by_code["open"].shift(-1)
        label = future_close / entry_open.replace(0, np.nan) - 1.0
        if tradable_label_filter:
            invalid = (~np.isfinite(entry_open)) | (entry_open <= 0)
            invalid |= (~np.isfinite(future_close)) | (future_close <= 0)
            if "vol" in df.columns:
                invalid |= by_code["vol"].shift(-1).fillna(0) <= 0
            if "amount" in df.columns:
                invalid |= by_code["amount"].shift(-1).fillna(0) <= 0
            open_gap = entry_open / close - 1.0
            invalid |= open_gap >= float(label_limit_up_pct) / 100.0
            label = label.mask(invalid)
    else:
        if "open" not in df.columns:
            raise ValueError("label_mode=close_to_next_open requires an open column")
        label = by_code["open"].shift(-1) / close - 1.0

    return label.reindex(panel.index)


def validate_panel_labels(
    panel: pd.DataFrame,
    *,
    label_mode: str | None,
    label_horizon: int = 1,
    tradable_label_filter: bool = True,
    label_limit_up_pct: float = 9.5,
    tolerance: float = 1e-8,
    max_bad_ratio: float = 1e-6,
    raise_on_error: bool = True,
) -> LabelValidationResult:
    if "label" not in panel.columns:
        raise ValueError("Panel is missing required column: label")

    mode = normalize_label_mode(label_mode)
    expected = calculate_label_series(
        panel,
        label_mode=mode,
        label_horizon=label_horizon,
        tradable_label_filter=tradable_label_filter,
        label_limit_up_pct=label_limit_up_pct,
    )
    actual = pd.to_numeric(panel["label"], errors="coerce")

    finite_expected = np.isfinite(expected)
    finite_actual = np.isfinite(actual)
    finite_both = finite_expected & finite_actual
    abs_diff = (actual[finite_both] - expected[finite_both]).abs()
    diff_bad = abs_diff > float(tolerance)
    nan_mismatch = finite_expected ^ finite_actual
    rows_compared = int(finite_both.sum())
    bad_rows = int(diff_bad.sum() + nan_mismatch.sum())
    bad_ratio = float(bad_rows / max(len(panel), 1))

    close_to_next_close_mean_abs_diff: float | None = None
    if mode != "close_to_next_close":
        close_label = calculate_label_series(
            panel,
            label_mode="close_to_next_close",
            label_horizon=label_horizon,
            tradable_label_filter=False,
            label_limit_up_pct=label_limit_up_pct,
        )
        alt_mask = np.isfinite(actual) & np.isfinite(close_label)
        if bool(alt_mask.any()):
            close_to_next_close_mean_abs_diff = float((actual[alt_mask] - close_label[alt_mask]).abs().mean())

    result = LabelValidationResult(
        mode=mode,
        horizon=max(int(label_horizon), 1),
        rows_compared=rows_compared,
        bad_rows=bad_rows,
        bad_ratio=bad_ratio,
        mean_abs_diff=float(abs_diff.mean()) if len(abs_diff) else 0.0,
        max_abs_diff=float(abs_diff.max()) if len(abs_diff) else 0.0,
        nan_mismatch_rows=int(nan_mismatch.sum()),
        close_to_next_close_mean_abs_diff=close_to_next_close_mean_abs_diff,
    )

    if raise_on_error and bad_ratio > float(max_bad_ratio):
        hint = ""
        if close_to_next_close_mean_abs_diff is not None and close_to_next_close_mean_abs_diff <= float(tolerance):
            hint = (
                " The stored label matches close_to_next_close instead; rebuild panel.parquet "
                "with the current label_mode before training or evaluating."
            )
        raise ValueError(
            "Panel label mismatch: "
            f"expected {mode} horizon={result.horizon}, "
            f"bad_rows={result.bad_rows}/{len(panel)} ({result.bad_ratio:.2%}), "
            f"mean_abs_diff={result.mean_abs_diff:.6g}, max_abs_diff={result.max_abs_diff:.6g}."
            f"{hint}"
        )

    return result


def expected_label_frame(
    panel: pd.DataFrame,
    *,
    label_mode: str | None,
    label_horizon: int = 1,
    tradable_label_filter: bool = True,
    label_limit_up_pct: float = 9.5,
    label_col: str = "label",
) -> pd.DataFrame:
    label = calculate_label_series(
        panel,
        label_mode=label_mode,
        label_horizon=label_horizon,
        tradable_label_filter=tradable_label_filter,
        label_limit_up_pct=label_limit_up_pct,
    )
    out = panel.loc[:, ["trade_date", "ts_code"]].copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out[label_col] = label
    return out


def attach_expected_labels(
    pred: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    label_mode: str | None,
    label_horizon: int = 1,
    tradable_label_filter: bool = True,
    label_limit_up_pct: float = 9.5,
    label_col: str = "label",
) -> tuple[pd.DataFrame, LabelValidationResult | None]:
    """Attach labels recalculated from panel prices, replacing stale prediction labels."""
    result: LabelValidationResult | None = None
    if "label" in panel.columns:
        result = validate_panel_labels(
            panel,
            label_mode=label_mode,
            label_horizon=label_horizon,
            tradable_label_filter=tradable_label_filter,
            label_limit_up_pct=label_limit_up_pct,
            raise_on_error=False,
        )

    labels = expected_label_frame(
        panel,
        label_mode=label_mode,
        label_horizon=label_horizon,
        tradable_label_filter=tradable_label_filter,
        label_limit_up_pct=label_limit_up_pct,
        label_col=label_col,
    )
    work = pred.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    if label_col in work.columns:
        work = work.drop(columns=[label_col])
    merged = work.merge(labels, on=["trade_date", "ts_code"], how="left")
    return merged, result
