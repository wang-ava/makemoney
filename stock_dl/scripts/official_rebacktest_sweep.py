#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.risk import attach_buyable_flag  # noqa: E402
from src.config import load_config  # noqa: E402


PANEL_COLS = [
    "trade_date",
    "ts_code",
    "open",
    "close",
    "pct_chg",
    "amount",
    "rank_ret_5d",
    "ret_5d",
    "hs300_idx_vol20",
    "sh_idx_vol20",
    "volatility_20d",
    "list_age_years",
    "circ_mv",
    "turnover_rate",
    "ret_1d",
]


def parse_grid(text: str, cast) -> list:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def load_scores(out_dir: Path, cfg: dict, scheme: str) -> pd.DataFrame:
    pred = pd.read_csv(out_dir / "val_predictions.csv")
    pred["trade_date"] = pred["trade_date"].astype(str)
    pred = pred[
        (pred["trade_date"] > str(cfg["train_end"]))
        & (pred["trade_date"] <= str(cfg["val_end"]))
    ].copy()
    return pred[["trade_date", "ts_code", "score"]].assign(scheme=scheme)


def curve_recent_return(eq: pd.DataFrame, window: int) -> float:
    if not isinstance(eq, pd.DataFrame) or len(eq) < window:
        return float("nan")
    start = float(eq["equity"].iloc[-window])
    end = float(eq["equity"].iloc[-1])
    return end / start - 1.0 if start > 0 else float("nan")


def run_one(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    base_strategy: dict,
    scheme: str,
    mode: str,
    n_hold: int,
    k_trade: int,
    target_position: float | None,
    momentum: bool | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    strategy = base_strategy.copy()
    strategy["n_hold"] = int(n_hold)
    strategy["k_trade"] = int(k_trade)

    if mode == "fixed":
        if target_position is None:
            raise ValueError("fixed mode requires target_position")
        strategy.update(
            {
                "adaptive_hold": False,
                "adaptive_hold_multi_indicator": False,
                "dynamic_k": False,
                "dynamic_position": False,
                "target_position_ratio": float(target_position),
                "min_position_ratio": 0.0,
                "max_position_ratio": 1.0,
                "cash_reserve_ratio": float(1.0 - target_position),
                "cash_reserve": float(1.0 - target_position),
            }
        )
        if momentum is not None:
            strategy["momentum_filter"] = bool(momentum)
    elif mode == "official_dynamic":
        # Keep the production config's adaptive hold, dynamic_k, dynamic_position, and filters.
        target_position = None
        momentum = bool(strategy.get("momentum_filter", False))
    else:
        raise ValueError(f"unknown mode: {mode}")

    result = run_backtest(
        scores,
        prices,
        n_hold=int(n_hold),
        k_trade=int(k_trade),
        initial_cash=float(strategy.get("initial_cash", 1_000_000)),
        cost_rate=float(strategy.get("cost_rate", 0.0003)),
        slippage=float(strategy.get("slippage", 0.0005)),
        use_long_short=bool(strategy.get("use_long_short", False)),
        short_ratio=float(strategy.get("short_ratio", 0.5)),
        strategy_cfg=strategy,
        cash_reserve_ratio=float(strategy.get("cash_reserve_ratio", 0.0)),
    )
    metrics = dict(result.get("metrics", {}))
    eq = result.get("equity_curve")
    metrics.update(
        {
            "scheme": scheme,
            "mode": mode,
            "n_hold": int(n_hold),
            "k_trade": int(k_trade),
            "target_position": target_position,
            "momentum_filter": momentum,
            "ret_last_10": curve_recent_return(eq, 10),
            "ret_last_20": curve_recent_return(eq, 20),
            "ret_last_60": curve_recent_return(eq, 60),
            "last_equity": float(eq["equity"].iloc[-1]) if isinstance(eq, pd.DataFrame) and not eq.empty else np.nan,
            "n_days": int(len(eq)) if isinstance(eq, pd.DataFrame) else 0,
        }
    )
    if isinstance(eq, pd.DataFrame):
        eq = eq.assign(
            scheme=scheme,
            mode=mode,
            n_hold=int(n_hold),
            k_trade=int(k_trade),
            target_position=target_position,
            momentum_filter=momentum,
        )
    return metrics, eq


def objective(row: pd.Series) -> float:
    # Practical ranking: reward Sharpe and total return, penalize drawdown/turnover, require recent health.
    sharpe = float(row.get("sharpe", 0.0) or 0.0)
    total = float(row.get("total_return", 0.0) or 0.0)
    dd = float(row.get("max_drawdown", 0.0) or 0.0)
    turnover = float(row.get("turnover", 0.0) or 0.0)
    recent60 = float(row.get("ret_last_60", 0.0) or 0.0)
    return sharpe + 1.5 * total + dd - 0.003 * turnover + 2.0 * recent60


def add_benchmarks(results: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    data_dir = Path(cfg.get("data_dir", ""))
    if str(data_dir) == "auto" or not data_dir.exists():
        data_dir = ROOT.parent / "A股数据"
    bench_map = {
        "bench_sh_return": "000001.SH.csv",
        "bench_hs300_return": "000300.SH.csv",
        "bench_cyb_return": "399006.SZ.csv",
    }
    start = str(results["start_date"].iloc[0])
    end = str(results["end_date"].iloc[0])
    for col, name in bench_map.items():
        fp = data_dir / "market" / name
        if not fp.exists():
            continue
        bench = pd.read_csv(fp, usecols=["trade_date", "close"])
        bench["trade_date"] = bench["trade_date"].astype(str)
        bench = bench[bench["trade_date"].between(start, end)].sort_values("trade_date")
        if len(bench) >= 2:
            results[col] = float(bench["close"].iloc[-1] / bench["close"].iloc[0] - 1.0)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-panel official engine strategy sweep.")
    parser.add_argument("--config-a", default=str(ROOT / "configs/server_8h_scheme_a_short.yaml"))
    parser.add_argument("--config-b", default=str(ROOT / "configs/server_8h_scheme_b_short.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs_server8h_scheme_a_short" / "trading_guides" / "20260605",
    )
    parser.add_argument("--n-grid", default="20,30,40,50,70,100")
    parser.add_argument("--k-grid", default="1,2,5,10,25")
    parser.add_argument("--fixed-n-grid", default="25,40,70,100")
    parser.add_argument("--fixed-k-grid", default="1,2,5,25")
    parser.add_argument("--target-grid", default="0.70,0.82,0.90")
    parser.add_argument("--modes", default="official_dynamic,fixed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_a = load_config(args.config_a)
    cfg_b = load_config(args.config_b)
    out_a = ROOT / cfg_a["output_dir"]
    out_b = ROOT / cfg_b["output_dir"]
    panel_path = out_a / "panel.parquet"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading full panel columns...")
    panel = pd.read_parquet(panel_path, columns=PANEL_COLS)
    panel["trade_date"] = panel["trade_date"].astype(str)
    prices = panel[["trade_date", "ts_code", "open", "close", "pct_chg"]].copy()

    all_scores = pd.concat(
        [load_scores(out_a, cfg_a, "A"), load_scores(out_b, cfg_b, "B")],
        ignore_index=True,
    )
    all_scores["trade_date"] = all_scores["trade_date"].astype(str)
    all_scores = all_scores[
        all_scores["trade_date"].isin(panel["trade_date"].unique())
    ].copy()

    print("Attaching full risk filters...")
    filtered_by_scheme: dict[str, pd.DataFrame] = {}
    for scheme, scores in all_scores.groupby("scheme"):
        cfg = cfg_a if scheme == "A" else cfg_b
        filtered = attach_buyable_flag(scores.drop(columns=["scheme"]), panel, cfg["strategy"])
        filtered_by_scheme[scheme] = filtered
        print(
            f"  {scheme}: rows={len(filtered):,}, dates={filtered['trade_date'].nunique()}, "
            f"buyable={filtered['buyable'].mean():.2%}"
        )

    n_grid = parse_grid(args.n_grid, int)
    k_grid = parse_grid(args.k_grid, int)
    fixed_n_grid = parse_grid(args.fixed_n_grid, int)
    fixed_k_grid = parse_grid(args.fixed_k_grid, int)
    target_grid = parse_grid(args.target_grid, float)
    modes = {x.strip() for x in args.modes.split(",") if x.strip()}

    scenarios: list[tuple[str, int, int, float | None, bool | None]] = []
    if "official_dynamic" in modes:
        for n in n_grid:
            for k in k_grid:
                if k <= n:
                    scenarios.append(("official_dynamic", n, k, None, None))
    if "fixed" in modes:
        for n in fixed_n_grid:
            for k in fixed_k_grid:
                if k > n:
                    continue
                for target in target_grid:
                    for momentum in [False, True]:
                        scenarios.append(("fixed", n, k, target, momentum))

    result_rows: list[dict[str, Any]] = []
    best_curve: pd.DataFrame | None = None
    best_obj = -float("inf")
    total = len(scenarios) * 2
    idx = 0
    for scheme in ["A", "B"]:
        cfg = cfg_a if scheme == "A" else cfg_b
        for mode, n, k, target, momentum in scenarios:
            idx += 1
            print(f"[{idx}/{total}] {scheme} {mode} n={n} k={k} target={target} momentum={momentum}", flush=True)
            metrics, curve = run_one(
                filtered_by_scheme[scheme],
                prices,
                cfg["strategy"],
                scheme,
                mode,
                n,
                k,
                target,
                momentum,
            )
            result_rows.append(metrics)
            obj = objective(pd.Series(metrics))
            if obj > best_obj and isinstance(curve, pd.DataFrame) and not curve.empty:
                best_obj = obj
                best_curve = curve

    results = pd.DataFrame(result_rows)
    results["start_date"] = all_scores["trade_date"].min()
    results["end_date"] = all_scores["trade_date"].max()
    results["objective"] = results.apply(objective, axis=1)
    results = add_benchmarks(results, cfg_a)
    results = results.sort_values(["objective", "sharpe", "total_return"], ascending=False)

    result_path = output_dir / "official_full_rebacktest_sweep_results.csv"
    summary_path = output_dir / "official_full_rebacktest_summary.json"
    curve_path = output_dir / "official_full_rebacktest_best_curve.csv"
    results.to_csv(result_path, index=False, encoding="utf-8-sig")
    if best_curve is not None:
        best_curve.to_csv(curve_path, index=False, encoding="utf-8-sig")
    summary = {
        "result_path": str(result_path),
        "best_curve_path": str(curve_path),
        "n_results": int(len(results)),
        "top20_by_objective": results.head(20).replace({np.nan: None}).to_dict(orient="records"),
        "top20_by_sharpe": results.sort_values(["sharpe", "total_return"], ascending=False)
        .head(20)
        .replace({np.nan: None})
        .to_dict(orient="records"),
        "top20_by_total_return": results.sort_values(["total_return", "sharpe"], ascending=False)
        .head(20)
        .replace({np.nan: None})
        .to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {curve_path}")
    print(f"Saved: {summary_path}")
    print(results.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
