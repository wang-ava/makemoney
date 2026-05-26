from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_percentile(series: pd.Series, lookback: int = 250) -> pd.Series:
    values = series.astype(float)
    out = []
    for i, value in enumerate(values):
        start = max(0, i - lookback + 1)
        window = values.iloc[start : i + 1].dropna()
        if not np.isfinite(value) or window.empty:
            out.append(0.5)
        else:
            out.append(float((window <= value).mean()))
    return pd.Series(out, index=series.index)


def _choose_adaptive_n(base_n: int, vol_pct: float | None, strategy_cfg: dict) -> int:
    if not strategy_cfg.get("adaptive_hold", False) or vol_pct is None:
        return int(base_n)
    min_n = int(strategy_cfg.get("adaptive_min_hold", max(1, base_n // 2)))
    max_n = int(strategy_cfg.get("adaptive_max_hold", max(base_n, base_n + base_n // 2)))
    low = float(strategy_cfg.get("adaptive_low_vol_pct", 0.2))
    high = float(strategy_cfg.get("adaptive_high_vol_pct", 0.8))
    if vol_pct >= high:
        return max_n
    if vol_pct <= low:
        return min_n
    return int(base_n)


def _filter_momentum(day: pd.DataFrame, buy_scores: pd.Series, strategy_cfg: dict) -> pd.Series:
    if not strategy_cfg.get("momentum_filter", False):
        return buy_scores
    col = strategy_cfg.get("momentum_rank_col", "rank_ret_5d")
    if col not in day.columns:
        return buy_scores
    threshold = float(strategy_cfg.get("min_momentum_rank", 0.2))
    keep = day[day[col].fillna(0.5) >= threshold].set_index("ts_code")
    filtered = buy_scores[buy_scores.index.isin(keep.index)]
    return filtered if not filtered.empty else buy_scores


def _choose_dynamic_k(day_scores: pd.Series, buy_scores: pd.Series, holdings: dict[str, float], base_k: int, strategy_cfg: dict) -> int:
    if not strategy_cfg.get("dynamic_k", False) or not holdings:
        return int(base_k)
    held = [c for c in holdings if c in day_scores.index]
    candidates = buy_scores.drop(index=[c for c in holdings if c in buy_scores.index], errors="ignore")
    if not held or candidates.empty:
        return int(base_k)
    gap = float(candidates.max() - day_scores.loc[held].min())
    high = float(strategy_cfg.get("score_gap_high", 0.10))
    low = float(strategy_cfg.get("score_gap_low", 0.02))
    if gap > high:
        return int(base_k + int(strategy_cfg.get("dynamic_k_step", 2)))
    if gap < low:
        return max(1, int(base_k) - int(strategy_cfg.get("dynamic_k_step", 1)))
    return int(base_k)


def _score_confidence(buy_scores: pd.Series, n_long: int, strategy_cfg: dict) -> float:
    scores = buy_scores.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(scores) < 2:
        return 0.5
    top_n = min(max(3, int(n_long)), len(scores))
    spread = float(scores.nlargest(top_n).mean() - scores.median())
    std = float(scores.std(ddof=0))
    if not np.isfinite(spread) or not np.isfinite(std) or std <= 1e-12:
        return 0.5
    z_spread = spread / std
    low = float(strategy_cfg.get("position_score_z_low", 0.4))
    high = float(strategy_cfg.get("position_score_z_high", 1.2))
    if high <= low:
        return 0.5
    return float(np.clip((z_spread - low) / (high - low), 0.0, 1.0))


def choose_target_position(
    buy_scores: pd.Series,
    n_long: int,
    strategy_cfg: dict,
    vol_pct: float | None = None,
    cash_reserve_ratio: float = 0.0,
) -> tuple[float, float]:
    """Choose a daily target long position ratio, bounded by the 80% floor."""
    legacy_target = 1.0 - float(cash_reserve_ratio)
    min_position = float(strategy_cfg.get("min_position_ratio", legacy_target))
    max_position = float(strategy_cfg.get("max_position_ratio", 1.0))
    min_position = float(np.clip(min_position, 0.0, 1.0))
    max_position = float(np.clip(max(max_position, min_position), min_position, 1.0))

    if not strategy_cfg.get("dynamic_position", False):
        target = float(strategy_cfg.get("target_position_ratio", legacy_target))
        return float(np.clip(target, min_position, max_position)), 0.5

    target_floor = max(
        min_position,
        float(strategy_cfg.get("min_target_position_ratio", min_position + float(strategy_cfg.get("position_floor_buffer", 0.0)))),
    )
    target_floor = float(np.clip(target_floor, min_position, max_position))
    base_position = float(strategy_cfg.get("base_position_ratio", (min_position + max_position) / 2.0))
    confidence = _score_confidence(buy_scores, n_long, strategy_cfg)
    vol_score = 0.5 if vol_pct is None or not np.isfinite(vol_pct) else 1.0 - float(np.clip(vol_pct, 0.0, 1.0))
    target = (
        base_position
        + float(strategy_cfg.get("position_confidence_weight", 0.08)) * (confidence - 0.5)
        + float(strategy_cfg.get("position_vol_weight", 0.12)) * (vol_score - 0.5)
    )
    return float(np.clip(target, target_floor, max_position)), confidence


def _market_value(holdings: dict[str, float], prices: pd.DataFrame, date: str) -> float:
    value = 0.0
    for code, shares in holdings.items():
        p = prices.at[date, code] if code in prices.columns else np.nan
        if np.isfinite(p):
            value += shares * p
    return float(value)


def run_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    n_hold: int = 10,
    k_trade: int = 2,
    initial_cash: float = 1_000_000.0,
    cost_rate: float = 0.0003,
    slippage: float = 0.0005,
    use_long_short: bool = False,
    short_ratio: float = 0.5,
    strategy_cfg: dict | None = None,
    cash_reserve_ratio: float = 0.0,
) -> dict:
    """
    scores: trade_date, ts_code, score
    prices: trade_date, ts_code, open, close
    策略：用 d 日盘后 signal，在下一交易日开盘执行；收盘记账。

    Enhanced with optional long-short strategy and advanced risk management.
    """
    signal_dates = sorted(scores["trade_date"].astype(str).unique())
    price_dates = sorted(prices["trade_date"].astype(str).unique())
    if len(signal_dates) < 2 or len(price_dates) < 2:
        return {"equity_curve": [], "metrics": {}}

    strategy_cfg = strategy_cfg or {}
    cash_reserve_ratio = float(np.clip(cash_reserve_ratio, 0.0, 0.95))
    position_floor = float(strategy_cfg.get("min_position_ratio", 1.0 - cash_reserve_ratio))
    holdings: dict[str, float] = {}
    short_holdings: dict[str, float] = {}
    cash = initial_cash
    short_cash = initial_cash * short_ratio if use_long_short else 0.0
    equity_curve = []
    total_turnover = 0.0
    long_buy_dates: dict[str, str] = {}
    trade_blocks = {
        "t1_sell_blocked": 0,
        "limit_buy_blocked": 0,
        "limit_sell_blocked": 0,
    }

    px = prices.copy()
    px["trade_date"] = px["trade_date"].astype(str)
    close_pivot = px.pivot(index="trade_date", columns="ts_code", values="close")
    open_col = "open" if "open" in px.columns else "close"
    open_pivot = px.pivot(index="trade_date", columns="ts_code", values=open_col)
    pct_chg_pivot = px.pivot(index="trade_date", columns="ts_code", values="pct_chg") if "pct_chg" in px.columns else None
    limit_pct = float(strategy_cfg.get("limit_pct_chg", strategy_cfg.get("max_abs_pct_chg", 9.5)))
    enforce_t1 = bool(strategy_cfg.get("enforce_t1", True))

    def can_trade(code: str, date: str, side: str) -> bool:
        if pct_chg_pivot is None or date not in pct_chg_pivot.index or code not in pct_chg_pivot.columns:
            return True
        pct = pct_chg_pivot.at[date, code]
        if not np.isfinite(pct):
            return True
        if side == "buy" and pct >= limit_pct:
            trade_blocks["limit_buy_blocked"] += 1
            return False
        if side == "sell" and pct <= -limit_pct:
            trade_blocks["limit_sell_blocked"] += 1
            return False
        return True

    def can_sell_long(code: str, date: str) -> bool:
        if enforce_t1 and long_buy_dates.get(code) == date:
            trade_blocks["t1_sell_blocked"] += 1
            return False
        return can_trade(code, date, "sell")

    vol_pct_by_date: dict[str, float] = {}
    if strategy_cfg.get("adaptive_hold", False) or strategy_cfg.get("dynamic_position", False):
        vol_col = strategy_cfg.get("market_vol_col")
        if not vol_col:
            for candidate in ["hs300_idx_vol20", "sh_idx_vol20", "volatility_20d"]:
                if candidate in scores.columns:
                    vol_col = candidate
                    break
        if vol_col and vol_col in scores.columns:
            vol_by_date = scores.groupby("trade_date")[vol_col].median().sort_index()
            pct = _rolling_percentile(vol_by_date, int(strategy_cfg.get("adaptive_lookback", 250)))
            vol_pct_by_date = pct.to_dict()

    next_date_by_signal = {}
    p_idx = 0
    for d in signal_dates:
        while p_idx < len(price_dates) and price_dates[p_idx] <= d:
            p_idx += 1
        if p_idx < len(price_dates):
            next_date_by_signal[d] = price_dates[p_idx]

    for d in signal_dates:
        if d not in next_date_by_signal:
            continue
        exec_date = next_date_by_signal[d]
        day = scores[scores["trade_date"] == d].copy()
        day_scores = day.set_index("ts_code")["score"]
        if "buyable" in day.columns:
            buy_scores = day[day["buyable"].astype(bool)].set_index("ts_code")["score"]
        else:
            buy_scores = day_scores
        buy_scores = _filter_momentum(day, buy_scores, strategy_cfg)
        if day_scores.empty or buy_scores.empty:
            continue

        cost = cost_rate + slippage
        n_long = _choose_adaptive_n(n_hold, vol_pct_by_date.get(d), strategy_cfg)
        target_position, position_confidence = choose_target_position(
            buy_scores,
            n_long,
            strategy_cfg,
            vol_pct_by_date.get(d),
            cash_reserve_ratio,
        )
        n_short = int(n_hold * short_ratio) if use_long_short else 0
        day_k = min(
            max(1, _choose_dynamic_k(day_scores, buy_scores, holdings, k_trade, strategy_cfg)),
            max(n_long, 1),
        )

        if not holdings:
            picks = buy_scores.nlargest(n_long).index.tolist()
            available_cash = cash * target_position
            per = available_cash / max(len(picks), 1)
            for code in picks:
                if not can_trade(code, exec_date, "buy"):
                    continue
                p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                if np.isfinite(p) and p > 0:
                    spend = min(per, cash)
                    holdings[code] = spend * (1 - cost) / p
                    long_buy_dates[code] = exec_date
                    cash -= spend
                    total_turnover += spend

            if use_long_short and n_short > 0:
                short_picks = buy_scores.nsmallest(n_short).index.tolist()
                per_short = short_cash / max(len(short_picks), 1)
                for code in short_picks:
                    if not can_trade(code, exec_date, "sell"):
                        continue
                    p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                    if np.isfinite(p) and p > 0:
                        short_holdings[code] = per_short * (1 - cost) / p
                        short_cash -= per_short
                        total_turnover += per_short
        else:
            held = [c for c in holdings if c in day_scores.index]
            if held:
                extra_sells = max(0, len(holdings) - n_long)
                sell_n = min(len(held), max(day_k, extra_sells))
                sell_codes = day_scores.loc[held].nsmallest(sell_n).index.tolist()
            else:
                sell_codes = []

            executed_rebalance_sells = 0
            for code in sell_codes:
                if not can_sell_long(code, exec_date):
                    continue
                p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                if np.isfinite(p) and p > 0:
                    gross = holdings.pop(code) * p
                    long_buy_dates.pop(code, None)
                    cash += gross * (1 - cost)
                    total_turnover += gross
                    executed_rebalance_sells += 1

            if strategy_cfg.get("dynamic_position_sell_down", True):
                mv_open = _market_value(holdings, open_pivot, exec_date)
                long_equity_open = cash + mv_open
                target_cash = (1.0 - target_position) * long_equity_open
                held_after_sell = [c for c in holdings if c in day_scores.index]
                extra_sell_candidates = [
                    c for c in day_scores.loc[held_after_sell].sort_values().index.tolist()
                    if c not in sell_codes
                ] if held_after_sell else []
                for code in extra_sell_candidates:
                    if cash >= target_cash:
                        break
                    if not can_sell_long(code, exec_date):
                        continue
                    p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                    if np.isfinite(p) and p > 0:
                        gross = holdings.pop(code) * p
                        long_buy_dates.pop(code, None)
                        cash += gross * (1 - cost)
                        total_turnover += gross

            post_sell_count = len(holdings)
            buy_n = max(executed_rebalance_sells, n_long - post_sell_count)
            buy_codes = buy_scores.nlargest(n_long + buy_n).index.tolist()
            buy_codes = [c for c in buy_codes if c not in holdings][:buy_n]

            if buy_codes:
                mv_open = _market_value(holdings, open_pivot, exec_date)
                long_equity_open = cash + mv_open
                target_cash = (1.0 - target_position) * long_equity_open
                available_cash = max(0.0, cash - target_cash)
                per = available_cash / len(buy_codes)
                for code in buy_codes:
                    if not can_trade(code, exec_date, "buy"):
                        continue
                    p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                    if np.isfinite(p) and p > 0:
                        spend = min(per, cash)
                        holdings[code] = holdings.get(code, 0) + spend * (1 - cost) / p
                        long_buy_dates[code] = exec_date
                        cash -= spend
                        total_turnover += spend

            if use_long_short and n_short > 0 and short_holdings:
                short_held = [c for c in short_holdings if c in day_scores.index]
                short_sell_codes = day_scores.loc[short_held].nlargest(day_k).index.tolist()
                short_buy_codes = buy_scores.nsmallest(n_short + day_k).index.tolist()
                short_buy_codes = [c for c in short_buy_codes if c not in short_holdings][:day_k]

                for code in short_sell_codes:
                    if not can_trade(code, exec_date, "buy"):
                        continue
                    p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                    if np.isfinite(p) and p > 0:
                        gross = short_holdings.pop(code) * p
                        short_cash += gross * (1 - cost)
                        total_turnover += gross

                if short_buy_codes:
                    per_short = short_cash / len(short_buy_codes)
                    for code in short_buy_codes:
                        if not can_trade(code, exec_date, "sell"):
                            continue
                        p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                        if np.isfinite(p) and p > 0:
                            spend = min(per_short, short_cash)
                            short_holdings[code] = short_holdings.get(code, 0) + spend * (1 - cost) / p
                            short_cash -= spend
                            total_turnover += spend

        mv = 0.0
        for code, sh in holdings.items():
            p = close_pivot.at[exec_date, code] if code in close_pivot.columns else np.nan
            if np.isfinite(p):
                mv += sh * p

        short_mv = 0.0
        for code, sh in short_holdings.items():
            p = close_pivot.at[exec_date, code] if code in close_pivot.columns else np.nan
            if np.isfinite(p):
                short_mv += sh * p

        total_equity = mv + cash + short_mv + short_cash
        long_equity = mv + cash
        position_ratio = mv / long_equity if long_equity > 0 else 0.0

        equity_curve.append(
            {
                "trade_date": exec_date,
                "signal_date": d,
                "equity": total_equity,
                "long_equity": long_equity,
                "short_equity": short_mv + short_cash,
                "cash": cash,
                "position_ratio": position_ratio,
                "target_position_ratio": target_position,
                "position_confidence": position_confidence,
                "market_vol_percentile": vol_pct_by_date.get(d),
                "short_cash": short_cash,
                "n_positions": len(holdings),
                "n_short_positions": len(short_holdings),
                "target_n_hold": n_long,
                "day_k_trade": day_k,
            }
        )

    eq = pd.DataFrame(equity_curve)
    if len(eq) < 2:
        return {"equity_curve": eq, "metrics": {}}

    initial_equity = initial_cash * (1.0 + short_ratio) if use_long_short else initial_cash
    equity_for_metrics = pd.concat(
        [pd.Series([initial_equity]), eq["equity"].reset_index(drop=True)],
        ignore_index=True,
    )
    ret = equity_for_metrics.pct_change().dropna()
    total_return = eq["equity"].iloc[-1] / initial_equity - 1
    ann = (eq["equity"].iloc[-1] / initial_equity) ** (252 / len(eq)) - 1
    sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)
    dd = (equity_for_metrics / equity_for_metrics.cummax() - 1).min()

    long_return = eq["long_equity"].iloc[-1] / initial_cash - 1
    short_return = 0.0
    if use_long_short:
        initial_short_equity = initial_cash * short_ratio
        short_return = eq["short_equity"].iloc[-1] / initial_short_equity - 1 if initial_short_equity > 0 else 0.0

    return {
        "equity_curve": eq,
        "metrics": {
            "total_return": float(total_return),
            "annual_return": float(ann),
            "sharpe": float(sharpe),
            "max_drawdown": float(dd),
            "daily_win_rate": float((ret > 0).mean()),
            "turnover": float(total_turnover / initial_cash),
            "long_return": float(long_return),
            "short_return": float(short_return),
            "min_position_ratio": float(eq["position_ratio"].min()),
            "median_position_ratio": float(eq["position_ratio"].median()),
            "max_target_position_ratio": float(eq["target_position_ratio"].max()),
            "mean_target_position_ratio": float(eq["target_position_ratio"].mean()),
            "min_target_position_ratio": float(eq["target_position_ratio"].min()),
            "position_floor": float(position_floor),
            "position_floor_breach_days": int((eq["position_ratio"] + 1e-9 < position_floor).sum()),
            "dynamic_position": bool(strategy_cfg.get("dynamic_position", False)),
            **trade_blocks,
        },
    }
