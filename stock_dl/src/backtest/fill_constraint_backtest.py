"""
成交量约束回测模块

对 run_backtest 进行增强:
1. 基于历史日均成交量估算成交率
2. 根据实际成交率调整持仓
3. 记录详细的成交约束统计
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np

from .engine import run_backtest as original_run_backtest
from .fill_rate import (
    estimate_daily_volume_stats,
    estimate_fill_rate,
    estimate_fill_rate_by_amount,
    get_market_condition,
    MarketCondition,
    FillRateConfig,
    DEFAULT_CONFIG,
)


def run_backtest_with_fill_constraint(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    panel: Optional[pd.DataFrame] = None,
    n_hold: int = 10,
    k_trade: int = 2,
    initial_cash: float = 1_000_000.0,
    cost_rate: float = 0.0003,
    slippage: float = 0.0005,
    use_long_short: bool = False,
    short_ratio: float = 0.5,
    strategy_cfg: dict | None = None,
    cash_reserve_ratio: float = 0.0,
    use_fill_constraint: bool = True,
    fill_rate_config: FillRateConfig = DEFAULT_CONFIG,
    volume_window: int = 20,
) -> dict:
    """
    带成交量约束的回测

    新增参数:
    - use_fill_constraint: 是否启用成交约束
    - fill_rate_config: 成交率配置
    - volume_window: 计算日均成交量的窗口天数

    Returns: 与 original_run_backtest 相同, 新增以下字段:
        - fill_rate_stats: {
            'avg_fill_rate': 平均成交率,
            'avg_market_fill_rate': 按市场环境的平均成交率,
            'underfilled_days': 未足额成交天数,
            'avg_position_achievement': 平均仓位达成率,
          }
    """
    strategy_cfg = strategy_cfg or {}

    # 如果不启用成交约束, 直接调用原函数
    if not use_fill_constraint or panel is None:
        result = original_run_backtest(
            scores=scores,
            prices=prices,
            n_hold=n_hold,
            k_trade=k_trade,
            initial_cash=initial_cash,
            cost_rate=cost_rate,
            slippage=slippage,
            use_long_short=use_long_short,
            short_ratio=short_ratio,
            strategy_cfg=strategy_cfg,
            cash_reserve_ratio=cash_reserve_ratio,
        )
        # 添加空的成交率统计
        result["metrics"]["fill_rate_stats"] = {
            "enabled": False,
            "avg_fill_rate": 1.0,
        }
        return result

    # 预处理: 计算每只股票的日均成交量
    print(f"  计算日均成交量 (窗口={volume_window}天)...")
    panel_copy = panel.copy()
    panel_copy["trade_date"] = panel_copy["trade_date"].astype(str)

    # 单位修正: vol=手数->股数, amount=万元->元
    if "vol" in panel_copy.columns:
        panel_copy["vol"] = panel_copy["vol"] * 100
    if "amount" in panel_copy.columns:
        panel_copy["amount"] = panel_copy["amount"] * 10000  # 万元转元

    volume_stats = estimate_daily_volume_stats(panel_copy, window=volume_window)
    volume_stats_dict = volume_stats.set_index("ts_code").to_dict("index")

    # 记录成交率统计
    fill_rate_records = []
    position_achievement_records = []

    # 预处理价格数据
    px = prices.copy()
    px["trade_date"] = px["trade_date"].astype(str)

    # 获取涨跌停数据
    limit_pct = float(strategy_cfg.get("limit_pct_chg", 9.5))
    if "pct_chg" in px.columns:
        pct_chg_pivot = px.pivot(index="trade_date", columns="ts_code", values="pct_chg")
    else:
        pct_chg_pivot = None

    # 获取成交量数据
    if "vol" in px.columns:
        vol_pivot = px.pivot(index="trade_date", columns="ts_code", values="vol")
    else:
        vol_pivot = None

    # 准备市场环境数据
    market_conditions = {}
    if "pct_chg" in px.columns:
        for date in px["trade_date"].unique():
            day_data = px[px["trade_date"] == date]
            limit_up_count = (day_data["pct_chg"] >= limit_pct).sum()
            if limit_up_count > 100:
                market_conditions[date] = MarketCondition.HOT
            elif limit_up_count > 50:
                market_conditions[date] = MarketCondition.WARM
            else:
                market_conditions[date] = MarketCondition.NORMAL

    def get_fill_rate(
        code: str,
        target_shares: float,
        price: float,
        date: str,
        is_buy: bool = True,
    ) -> float:
        """获取单只股票的成交率"""
        # 获取日均成交量
        vol_info = volume_stats_dict.get(code, {})
        daily_vol = vol_info.get("avg_volume", 0)
        daily_amount = vol_info.get("avg_amount", 0)

        if daily_vol <= 0:
            return 0.8  # 默认80%

        # 获取涨跌停状态
        is_limit_up = False
        is_limit_down = False
        if pct_chg_pivot is not None and date in pct_chg_pivot.index and code in pct_chg_pivot.columns:
            pct = pct_chg_pivot.at[date, code]
            if np.isfinite(pct):
                is_limit_up = pct >= limit_pct
                is_limit_down = pct <= -limit_pct

        # 获取市场环境
        market_cond = market_conditions.get(date, MarketCondition.NORMAL)

        # 估算成交率 (使用金额方式, 更稳定)
        target_amount = target_shares * price
        fill_rate = estimate_fill_rate_by_amount(
            target_amount=target_amount,
            avg_daily_amount=daily_amount,
            market_condition=market_cond,
            is_limit_up=is_limit_up,
            is_limit_down=is_limit_down,
            config=fill_rate_config,
        )

        return fill_rate

    # ========== 修改后的回测逻辑 ==========

    signal_dates = sorted(scores["trade_date"].astype(str).unique())
    price_dates = sorted(px["trade_date"].astype(str).unique())

    if len(signal_dates) < 2:
        result = original_run_backtest(
            scores=scores,
            prices=prices,
            n_hold=n_hold,
            k_trade=k_trade,
            initial_cash=initial_cash,
            cost_rate=cost_rate,
            slippage=slippage,
            use_long_short=use_long_short,
            short_ratio=short_ratio,
            strategy_cfg=strategy_cfg,
            cash_reserve_ratio=cash_reserve_ratio,
        )
        result["metrics"]["fill_rate_stats"] = {"enabled": True, "avg_fill_rate": 0.8}
        return result

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

    close_pivot = px.pivot(index="trade_date", columns="ts_code", values="close")
    open_col = "open" if "open" in px.columns else "close"
    open_pivot = px.pivot(index="trade_date", columns="ts_code", values=open_col)

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
        if side == "sell" and pct < -limit_pct:
            trade_blocks["limit_sell_blocked"] += 1
            return False
        return True

    def can_sell_long(code: str, date: str) -> bool:
        if enforce_t1 and long_buy_dates.get(code) == date:
            trade_blocks["t1_sell_blocked"] += 1
            return False
        return can_trade(code, date, "sell")

    from .engine import (
        _filter_momentum,
        _choose_adaptive_n,
        _choose_adaptive_n_multi,
        _choose_dynamic_k,
        _score_confidence,
        choose_target_position,
        _market_value,
        _rolling_percentile,
    )

    vol_pct_by_date: dict[str, float] = {}
    ic_trend_by_date: dict[str, float] = {}
    use_multi_indicator = strategy_cfg.get("adaptive_hold_multi_indicator", False)

    if strategy_cfg.get("adaptive_hold", False):
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

        if use_multi_indicator and "ic" in scores.columns:
            ic_by_date = scores.groupby("trade_date")["ic"].mean().sort_index()
            if len(ic_by_date) >= 20:
                ic_trend = ic_by_date.diff(20).dropna()
                ic_trend_by_date = {d: float(v) for d, v in ic_trend.items()}

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

        target_position, position_confidence = choose_target_position(
            buy_scores,
            n_hold,
            strategy_cfg,
            vol_pct_by_date.get(d),
            cash_reserve_ratio,
        )

        if use_multi_indicator:
            ic_trend = ic_trend_by_date.get(d)
            n_long = _choose_adaptive_n_multi(
                n_hold, vol_pct_by_date.get(d), ic_trend,
                position_confidence, strategy_cfg
            )
        else:
            n_long = _choose_adaptive_n(n_hold, vol_pct_by_date.get(d), strategy_cfg)
        n_short = int(n_hold * short_ratio) if use_long_short else 0
        day_k = min(
            max(1, _choose_dynamic_k(day_scores, buy_scores, holdings, k_trade, strategy_cfg)),
            max(n_long, 1),
        )

        # 记录当日成交率
        day_fill_rates = []

        if not holdings:
            picks = buy_scores.nlargest(n_long).index.tolist()
            available_cash = cash * target_position
            per = available_cash / max(len(picks), 1)

            for code in picks:
                if not can_trade(code, exec_date, "buy"):
                    continue
                p = open_pivot.at[exec_date, code] if code in open_pivot.columns else np.nan
                if np.isfinite(p) and p > 0 and p > 0:
                    # 计算目标股数和成交率
                    target_shares = min(per, cash) / (p * (1 + cost))
                    fill_rate = get_fill_rate(code, target_shares, p, exec_date, is_buy=True)
                    actual_shares = target_shares * fill_rate

                    holdings[code] = holdings.get(code, 0) + actual_shares
                    long_buy_dates[code] = exec_date
                    cash -= actual_shares * p * (1 - cost)
                    total_turnover += actual_shares * p

                    day_fill_rates.append(fill_rate)

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
                    # 卖出不受成交率限制, 卖多少算多少
                    gross = holdings.pop(code) * p
                    long_buy_dates.pop(code, None)
                    cash += gross * (1 - cost)
                    total_turnover += gross
                    executed_rebalance_sells += 1
                    day_fill_rates.append(1.0)  # 卖出按100%算

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
                        target_shares = min(per, cash) / (p * (1 + cost))
                        fill_rate = get_fill_rate(code, target_shares, p, exec_date, is_buy=True)
                        actual_shares = target_shares * fill_rate

                        holdings[code] = holdings.get(code, 0) + actual_shares
                        long_buy_dates[code] = exec_date
                        cash -= actual_shares * p * (1 - cost)
                        total_turnover += actual_shares * p

                        day_fill_rates.append(fill_rate)

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

        # 计算当日成交率统计
        if day_fill_rates:
            avg_fill_rate = np.mean(day_fill_rates)
            fill_rate_records.append({
                "date": exec_date,
                "avg_fill_rate": avg_fill_rate,
                "min_fill_rate": np.min(day_fill_rates),
                "max_fill_rate": np.max(day_fill_rates),
                "trade_count": len(day_fill_rates),
            })

        # 计算仓位达成率
        target_value = n_long * (initial_cash / n_long)
        actual_value = sum(holdings.get(code, 0) * close_pivot.at[exec_date, code]
                          if code in close_pivot.columns else 0
                          for code in buy_scores.nlargest(n_long).index.tolist() if code in holdings)
        position_achievement = actual_value / (target_value + 1e-9) if target_value > 0 else 1.0
        position_achievement_records.append({
            "date": exec_date,
            "position_achievement": position_achievement,
        })

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
                "day_fill_rate": fill_rate_records[-1]["avg_fill_rate"] if fill_rate_records else 1.0,
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

    # 计算成交率统计
    if fill_rate_records:
        fill_df = pd.DataFrame(fill_rate_records)
        fill_rate_stats = {
            "enabled": True,
            "avg_fill_rate": float(fill_df["avg_fill_rate"].mean()),
            "min_fill_rate": float(fill_df["avg_fill_rate"].min()),
            "underfilled_days": int((fill_df["avg_fill_rate"] < 0.9).sum()),
            "avg_trade_count_per_day": float(fill_df["trade_count"].mean()),
        }
    else:
        fill_rate_stats = {
            "enabled": True,
            "avg_fill_rate": 1.0,
            "min_fill_rate": 1.0,
            "underfilled_days": 0,
            "avg_trade_count_per_day": 0.0,
        }

    if position_achievement_records:
        pos_df = pd.DataFrame(position_achievement_records)
        fill_rate_stats["avg_position_achievement"] = float(pos_df["position_achievement"].mean())
        fill_rate_stats["min_position_achievement"] = float(pos_df["position_achievement"].min())

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
            "fill_rate_stats": fill_rate_stats,
        },
    }
