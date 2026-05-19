from __future__ import annotations

import numpy as np
import pandas as pd


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

    holdings: dict[str, float] = {}
    short_holdings: dict[str, float] = {}
    cash = initial_cash
    short_cash = initial_cash * short_ratio
    equity_curve = []
    total_turnover = 0.0

    px = prices.copy()
    px["trade_date"] = px["trade_date"].astype(str)
    close_pivot = px.pivot(index="trade_date", columns="ts_code", values="close")
    open_col = "open" if "open" in px.columns else "close"
    open_pivot = px.pivot(index="trade_date", columns="ts_code", values=open_col)

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
        if day_scores.empty or buy_scores.empty:
            continue

        cost = cost_rate + slippage
        n_long = n_hold
        n_short = int(n_hold * short_ratio) if use_long_short else 0

        if not holdings:
            picks = buy_scores.nlargest(n_long).index.tolist()
            per = cash / max(len(picks), 1)
            for code in picks:
                p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                if np.isfinite(p) and p > 0:
                    spend = min(per, cash)
                    holdings[code] = spend * (1 - cost) / p
                    cash -= spend
                    total_turnover += spend

            if use_long_short and n_short > 0:
                short_picks = buy_scores.nsmallest(n_short).index.tolist()
                per_short = short_cash / max(len(short_picks), 1)
                for code in short_picks:
                    p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                    if np.isfinite(p) and p > 0:
                        short_holdings[code] = per_short * (1 - cost) / p
                        short_cash -= per_short
                        total_turnover += per_short
        else:
            held = [c for c in holdings if c in day_scores.index]
            if held:
                sell_codes = day_scores.loc[held].nsmallest(k_trade).index.tolist()
            else:
                sell_codes = []
            buy_codes = buy_scores.nlargest(n_long + k_trade).index.tolist()
            buy_codes = [c for c in buy_codes if c not in holdings][:k_trade]

            for code in sell_codes:
                p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                if np.isfinite(p) and p > 0:
                    gross = holdings.pop(code) * p
                    cash += gross * (1 - cost)
                    total_turnover += gross

            if buy_codes:
                per = cash / len(buy_codes)
                for code in buy_codes:
                    p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                    if np.isfinite(p) and p > 0:
                        spend = min(per, cash)
                        holdings[code] = holdings.get(code, 0) + spend * (1 - cost) / p
                        cash -= spend
                        total_turnover += spend

            if use_long_short and n_short > 0 and short_holdings:
                short_held = [c for c in short_holdings if c in day_scores.index]
                short_sell_codes = day_scores.loc[short_held].nlargest(k_trade).index.tolist()
                short_buy_codes = buy_scores.nsmallest(n_short + k_trade).index.tolist()
                short_buy_codes = [c for c in short_buy_codes if c not in short_holdings][:k_trade]

                for code in short_sell_codes:
                    p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                    if np.isfinite(p) and p > 0:
                        gross = short_holdings.pop(code) * p
                        short_cash += gross * (1 - cost)
                        total_turnover += gross

                if short_buy_codes:
                    per_short = short_cash / len(short_buy_codes)
                    for code in short_buy_codes:
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

        equity_curve.append(
            {
                "trade_date": exec_date,
                "signal_date": d,
                "equity": total_equity,
                "long_equity": mv + cash,
                "short_equity": short_mv + short_cash,
                "cash": cash,
                "short_cash": short_cash,
                "n_positions": len(holdings),
                "n_short_positions": len(short_holdings),
            }
        )

    eq = pd.DataFrame(equity_curve)
    if len(eq) < 2:
        return {"equity_curve": eq, "metrics": {}}

    ret = eq["equity"].pct_change().dropna()
    total_return = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
    ann = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (252 / len(eq)) - 1
    sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()

    return {
        "equity_curve": eq,
        "metrics": {
            "total_return": float(total_return),
            "annual_return": float(ann),
            "sharpe": float(sharpe),
            "max_drawdown": float(dd),
            "daily_win_rate": float((ret > 0).mean()),
            "turnover": float(total_turnover / initial_cash),
            "long_return": float(eq["long_equity"].iloc[-1] / eq["long_equity"].iloc[0] - 1) if len(eq) > 1 else 0.0,
            "short_return": float(eq["short_equity"].iloc[-1] / eq["short_equity"].iloc[0] - 1) if len(eq) > 1 and use_long_short else 0.0,
        },
    }
