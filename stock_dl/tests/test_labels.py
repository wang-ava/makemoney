from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.labels import attach_expected_labels, calculate_label_series, validate_panel_labels


def _sample_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": ["20260101", "20260102", "20260105", "20260101", "20260102", "20260105"],
            "open": [10.0, 12.0, 12.5, 20.0, 19.0, 21.0],
            "close": [11.0, 12.1, 12.8, 18.0, 20.9, 21.2],
            "vol": [1000, 1000, 1000, 1000, 1000, 1000],
            "amount": [10000, 12000, 12500, 20000, 19000, 21000],
        }
    )


class LabelValidationTest(unittest.TestCase):
    def test_detects_close_to_close_panel_when_next_open_label_expected(self) -> None:
        panel = _sample_panel()
        panel["label"] = calculate_label_series(
            panel,
            label_mode="close_to_next_close",
            label_horizon=1,
            tradable_label_filter=False,
        )

        with self.assertRaisesRegex(ValueError, "matches close_to_next_close"):
            validate_panel_labels(
                panel,
                label_mode="next_open_to_next_close",
                label_horizon=1,
                tradable_label_filter=True,
                max_bad_ratio=0.0,
            )

    def test_accepts_matching_next_open_label(self) -> None:
        panel = _sample_panel()
        panel["label"] = calculate_label_series(
            panel,
            label_mode="next_open_to_next_close",
            label_horizon=1,
            tradable_label_filter=True,
        )

        result = validate_panel_labels(
            panel,
            label_mode="next_open_to_next_close",
            label_horizon=1,
            tradable_label_filter=True,
            max_bad_ratio=0.0,
        )

        self.assertEqual(result.bad_rows, 0)

    def test_attach_expected_labels_replaces_prediction_label(self) -> None:
        panel = _sample_panel()
        panel["label"] = calculate_label_series(
            panel,
            label_mode="close_to_next_close",
            label_horizon=1,
            tradable_label_filter=False,
        )
        pred = panel[["trade_date", "ts_code", "label"]].copy()
        pred["score"] = range(len(pred))

        fixed, _ = attach_expected_labels(
            pred,
            panel,
            label_mode="next_open_to_next_close",
            label_horizon=1,
            tradable_label_filter=True,
        )
        expected = calculate_label_series(
            panel,
            label_mode="next_open_to_next_close",
            label_horizon=1,
            tradable_label_filter=True,
        )

        pd.testing.assert_series_equal(fixed["label"], expected.reset_index(drop=True), check_names=False)


if __name__ == "__main__":
    unittest.main()
