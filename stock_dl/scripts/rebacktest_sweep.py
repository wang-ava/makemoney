#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    scheme: str
    n_hold: int
    k_trade: int
    target_position: float
    use_momentum: bool


def normalize_code(ts_code: str) -> str:
    return str(ts_code).strip().upper()


def load_daily_prices(data_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    daily_dir = data_dir / "daily"
    for fp in sorted(daily_dir.glob("*.csv")):
        date = fp.stem
        if date < start_date or date > end_date:
            continue
        df = pd.read_csv(
            fp,
            usecols=["ts_code", "trade_date", "open", "close", "pct_chg", "amount"],
        )
        df["trade_date"] = df["trade_date"].astype(str)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No daily price files in {daily_dir} for {start_date}-{end_date}")
    out = pd.concat(frames, ignore_index=True)
    out["ts_code"] = out["ts_code"].map(normalize_code)
    out = out.sort_values(["ts_code", "trade_date"])
    out["ret_5d"] = out.groupby("ts_code")["close"].pct_change(5)
    out["rank_ret_5d"] = out.groupby("trade_date")["ret_5d"].rank(pct=True)
    return out


def load_predictions(out_dir: Path, cfg: dict, scheme: str) -> pd.DataFrame:
    pred = pd.read_csv(out_dir / "val_predictions.csv")
    pred["trade_date"] = pred["trade_date"].astype(str)
    pred["ts_code"] = pred["ts_code"].map(normalize_code)
    pred = pred[
        (pred["trade_date"] > str(cfg["train_end"]))
        & (pred["trade_date"] <= str(cfg["val_end"]))
    ].copy()
    pred["scheme"] = scheme
    if "score" not in pred.columns:
        raise ValueError(f"{out_dir}/val_predictions.csv missing score column")
    return pred[["scheme", "trade_date", "ts_code", "score"]]


def attach_light_filters(scores: pd.DataFrame, daily: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    ctx = daily[["trade_date", "ts_code", "pct_chg", "amount", "rank_ret_5d", "ret_5d"]].copy()
    out = scores.merge(ctx, on=["trade_date", "ts_code"], how="left")

    if not strategy.get("risk_filter", True):
        out["buyable"] = True
        return out

    max_abs_pct = float(strategy.get("max_abs_pct_chg", 9.5))
    min_amount_q = float(strategy.get("min_amount_quantile", 0.05))
    min_amount_abs = float(strategy.get("min_avg_amount_abs", 0.0))
    out["amount_cut"] = out.groupby("trade_date")["amount"].transform(
        lambda s: s.quantile(min_amount_q)
    )
    out["buyable"] = (
        out["pct_chg"].abs().fillna(999.0).lt(max_abs_pct)
        & out["amount"].fillna(0.0).ge(out["amount_cut"].fillna(np.inf))
        & out["amount"].fillna(0.0).ge(min_amount_abs)
    )

    prefixes = tuple(str(p) for p in strategy.get("exclude_code_prefixes", []) if str(p))
    if prefixes:
        raw = out["ts_code"].str.split(".", regex=False).str[0]
        out["buyable"] &= ~raw.str.startswith(prefixes)
    return out


def build_lookup(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    open_px = daily.pivot(index="trade_date", columns="ts_code", values="open").sort_index()
    close_px = daily.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    pct = daily.pivot(index="trade_date", columns="ts_code", values="pct_chg").sort_index()
    return open_px, close_px, pct


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1.0).min())


def metric_block(curve: pd.DataFrame, initial_cash: float, turnover: float) -> dict[str, Any]:
    eq = pd.concat([pd.Series([initial_cash]), curve["equity"].reset_index(drop=True)])
    ret = eq.pct_change().dropna()
    if curve.empty or ret.empty:
        return {}
    total_return = float(curve["equity"].iloc[-1] / initial_cash - 1.0)
    annual_return = float((curve["equity"].iloc[-1] / initial_cash) ** (252 / len(curve)) - 1.0)
    sharpe = float(ret.mean() / (ret.std(ddof=1) + 1e-12) * math.sqrt(252))
    dd = max_drawdown(eq)
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "daily_win_rate": float((ret > 0).mean()),
        "turnover": float(turnover / initial_cash),
        "ret_last_10": float(curve["equity"].iloc[-1] / curve["equity"].iloc[-10] - 1.0)
        if len(curve) >= 10
        else np.nan,
        "ret_last_20": float(curve["equity"].iloc[-1] / curve["equity"].iloc[-20] - 1.0)
        if len(curve) >= 20
        else np.nan,
        "ret_last_60": float(curve["equity"].iloc[-1] / curve["equity"].iloc[-60] - 1.0)
        if len(curve) >= 60
        else np.nan,
    }


def run_scenario(
    scenario: Scenario,
    score_by_date: dict[str, pd.DataFrame],
    trade_dates: list[str],
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    pct_chg: pd.DataFrame,
    strategy: dict,
    initial_cash: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cost = float(strategy.get("cost_rate", 0.0003)) + float(strategy.get("slippage", 0.0005))
    limit_pct = float(strategy.get("max_abs_pct_chg", 9.5))
    min_momentum = float(strategy.get("min_momentum_rank", 0.35))

    cash = initial_cash
    holdings: dict[str, float] = {}
    turnover = 0.0
    rows: list[dict[str, Any]] = []

    next_date: dict[str, str] = {}
    for i, d in enumerate(trade_dates[:-1]):
        next_date[d] = trade_dates[i + 1]

    for signal_date in trade_dates:
        if signal_date not in next_date or signal_date not in score_by_date:
            continue
        exec_date = next_date[signal_date]
        day = score_by_date[signal_date]
        if scenario.use_momentum and "rank_ret_5d" in day.columns:
            filtered = day[day["rank_ret_5d"].fillna(0.5).ge(min_momentum)]
            if not filtered.empty:
                day = filtered
        buyable = day[day["buyable"].astype(bool)].sort_values("score", ascending=False)
        all_scores = day.set_index("ts_code")["score"]
        if buyable.empty or all_scores.empty:
            continue

        # Sell worst held names, plus any excess if a smaller n_hold is selected.
        held_scored = [c for c in holdings if c in all_scores.index]
        extra_sells = max(0, len(holdings) - scenario.n_hold)
        sell_n = min(len(held_scored), max(scenario.k_trade, extra_sells))
        sell_codes = (
            all_scores.loc[held_scored].sort_values().head(sell_n).index.tolist()
            if sell_n > 0
            else []
        )

        executed_sells = 0
        for code in sell_codes:
            if exec_date not in open_px.index or code not in open_px.columns:
                continue
            pct = pct_chg.at[exec_date, code] if code in pct_chg.columns else np.nan
            if np.isfinite(pct) and pct <= -limit_pct:
                continue
            p = open_px.at[exec_date, code]
            if not np.isfinite(p) or p <= 0:
                continue
            shares = holdings.pop(code, 0.0)
            gross = shares * p
            cash += gross * (1.0 - cost)
            turnover += gross
            executed_sells += 1

        if not holdings:
            buy_n = scenario.n_hold
        else:
            buy_n = max(executed_sells, scenario.n_hold - len(holdings))
        buy_codes = [
            c
            for c in buyable["ts_code"].tolist()
            if c not in holdings
        ][:buy_n]

        mv_open = 0.0
        for code, shares in holdings.items():
            if exec_date in open_px.index and code in open_px.columns:
                p = open_px.at[exec_date, code]
                if np.isfinite(p):
                    mv_open += shares * p
        equity_open = cash + mv_open
        reserve_cash = max(0.0, equity_open * (1.0 - scenario.target_position))
        spendable = max(0.0, cash - reserve_cash)
        per = spendable / max(len(buy_codes), 1)

        for code in buy_codes:
            if exec_date not in open_px.index or code not in open_px.columns:
                continue
            pct = pct_chg.at[exec_date, code] if code in pct_chg.columns else np.nan
            if np.isfinite(pct) and pct >= limit_pct:
                continue
            p = open_px.at[exec_date, code]
            if not np.isfinite(p) or p <= 0 or per <= 0:
                continue
            spend = min(per, cash)
            shares = spend * (1.0 - cost) / p
            holdings[code] = holdings.get(code, 0.0) + shares
            cash -= spend
            turnover += spend

        mv_close = 0.0
        missing = 0
        for code, shares in holdings.items():
            if exec_date in close_px.index and code in close_px.columns:
                p = close_px.at[exec_date, code]
                if np.isfinite(p):
                    mv_close += shares * p
                else:
                    missing += 1
            else:
                missing += 1
        equity = cash + mv_close
        rows.append(
            {
                "trade_date": exec_date,
                "signal_date": signal_date,
                "equity": equity,
                "cash": cash,
                "market_value": mv_close,
                "position_ratio": mv_close / equity if equity > 0 else np.nan,
                "n_positions": len(holdings),
                "missing_prices": missing,
            }
        )

    curve = pd.DataFrame(rows)
    metrics = metric_block(curve, initial_cash, turnover)
    metrics.update(
        {
            "scheme": scenario.scheme,
            "n_hold": scenario.n_hold,
            "k_trade": scenario.k_trade,
            "target_position": scenario.target_position,
            "use_momentum": scenario.use_momentum,
            "last_equity": float(curve["equity"].iloc[-1]) if not curve.empty else np.nan,
            "median_position_ratio": float(curve["position_ratio"].median()) if not curve.empty else np.nan,
            "n_days": int(len(curve)),
        }
    )
    return metrics, curve


def add_benchmark_metrics(rows: pd.DataFrame, data_dir: Path, initial_cash: float) -> pd.DataFrame:
    bench_files = {
        "bench_sh_total_return": "000001.SH.csv",
        "bench_hs300_total_return": "000300.SH.csv",
        "bench_cyb_total_return": "399006.SZ.csv",
    }
    for col, name in bench_files.items():
        fp = data_dir / "market" / name
        if not fp.exists():
            continue
        bench = pd.read_csv(fp, usecols=["trade_date", "close"])
        bench["trade_date"] = bench["trade_date"].astype(str)
        bench = bench.sort_values("trade_date")
        bench = bench[bench["trade_date"].between(rows["start_date"].iloc[0], rows["end_date"].iloc[0])]
        if len(bench) >= 2:
            value = float(bench["close"].iloc[-1] / bench["close"].iloc[0] - 1.0)
            rows[col] = value
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight independent strategy sweep backtest.")
    parser.add_argument("--config-a", default=str(ROOT / "configs/server_8h_scheme_a_short.yaml"))
    parser.add_argument("--config-b", default=str(ROOT / "configs/server_8h_scheme_b_short.yaml"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_server8h_scheme_a_short" / "trading_guides" / "20260605")
    parser.add_argument("--n-grid", default="10,20,25,30,40,50,70,100")
    parser.add_argument("--k-grid", default="1,2,5,10,25")
    parser.add_argument("--target-grid", default="0.70,0.82,0.90")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_a = load_config(args.config_a)
    cfg_b = load_config(args.config_b)
    data_dir = Path(cfg_a["data_dir"])
    if str(data_dir) == "auto":
        data_dir = PROJECT_ROOT / "A股数据"

    out_a = ROOT / cfg_a["output_dir"]
    out_b = ROOT / cfg_b["output_dir"]
    pred_a = load_predictions(out_a, cfg_a, "A")
    pred_b = load_predictions(out_b, cfg_b, "B")
    all_pred = pd.concat([pred_a, pred_b], ignore_index=True)
    start = all_pred["trade_date"].min()
    end = "20260605"
    daily = load_daily_prices(data_dir, start, end)

    strategy = cfg_a["strategy"].copy()
    scored = attach_light_filters(all_pred, daily, strategy)
    trade_dates = sorted(set(daily["trade_date"].astype(str)) & set(scored["trade_date"].astype(str)))
    open_px, close_px, pct_chg = build_lookup(daily)

    by_scheme_date: dict[tuple[str, str], pd.DataFrame] = {}
    for (scheme, date), group in scored.groupby(["scheme", "trade_date"], sort=False):
        by_scheme_date[(scheme, date)] = group.sort_values("score", ascending=False)

    n_grid = [int(x) for x in args.n_grid.split(",") if x.strip()]
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
    target_grid = [float(x) for x in args.target_grid.split(",") if x.strip()]
    scenarios = [
        Scenario(scheme, n, k, target, momentum)
        for scheme in ["A", "B"]
        for n in n_grid
        for k in k_grid
        for target in target_grid
        for momentum in [False, True]
        if k <= n
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    best_curve: pd.DataFrame | None = None
    best_key: tuple[float, float, float] | None = None

    print(f"Running {len(scenarios)} scenarios over {len(trade_dates)} signal dates...")
    for idx, scenario in enumerate(scenarios, start=1):
        score_by_date = {
            d: by_scheme_date[(scenario.scheme, d)]
            for d in trade_dates
            if (scenario.scheme, d) in by_scheme_date
        }
        metrics, curve = run_scenario(
            scenario,
            score_by_date,
            trade_dates,
            open_px,
            close_px,
            pct_chg,
            strategy,
            float(strategy.get("initial_cash", 1_000_000)),
        )
        # Prefer robust live candidates: high Sharpe, positive recent 20/60, controlled drawdown.
        rank_key = (
            float(metrics.get("sharpe", -999.0)),
            float(metrics.get("ret_last_60", -999.0)),
            float(metrics.get("total_return", -999.0)),
        )
        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_curve = curve.assign(
                scheme=scenario.scheme,
                n_hold=scenario.n_hold,
                k_trade=scenario.k_trade,
                target_position=scenario.target_position,
                use_momentum=scenario.use_momentum,
            )
        all_rows.append(metrics)
        if idx % 25 == 0 or idx == len(scenarios):
            print(f"  finished {idx}/{len(scenarios)}")

    results = pd.DataFrame(all_rows)
    results["start_date"] = start
    results["end_date"] = end
    results = add_benchmark_metrics(results, data_dir, float(strategy.get("initial_cash", 1_000_000)))
    results = results.sort_values(
        ["sharpe", "ret_last_60", "total_return"],
        ascending=[False, False, False],
    )

    result_path = args.output_dir / "rebacktest_sweep_results.csv"
    best_curve_path = args.output_dir / "rebacktest_best_equity_curve.csv"
    summary_path = args.output_dir / "rebacktest_summary.json"
    results.to_csv(result_path, index=False, encoding="utf-8-sig")
    if best_curve is not None:
        best_curve.to_csv(best_curve_path, index=False, encoding="utf-8-sig")
    summary = {
        "result_path": str(result_path),
        "best_curve_path": str(best_curve_path),
        "start_date": start,
        "end_date": end,
        "n_scenarios": len(scenarios),
        "top10": results.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "by_scheme_best": results.sort_values(["scheme", "sharpe"], ascending=[True, False])
        .groupby("scheme")
        .head(1)
        .replace({np.nan: None})
        .to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {best_curve_path}")
    print(f"Saved: {summary_path}")
    print(results.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
