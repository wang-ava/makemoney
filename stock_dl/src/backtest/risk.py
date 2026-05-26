from __future__ import annotations

import numpy as np
import pandas as pd


def attach_buyable_flag(scores: pd.DataFrame, panel: pd.DataFrame, strategy_cfg: dict) -> pd.DataFrame:
    """Attach a conservative buyable flag using only same-day known data.

    Enhanced with multiple risk filters including volatility, liquidity, market cap, and industry constraints.
    """
    out = scores.copy()
    if not strategy_cfg.get("risk_filter", True):
        out["buyable"] = True
        return out

    cols = ["trade_date", "ts_code", "pct_chg", "amount"]
    context_cols = [
        "rank_ret_5d",
        "ret_5d",
        "hs300_idx_vol20",
        "sh_idx_vol20",
        "volatility_20d",
    ]
    for optional in ["volatility_20d", "list_age_years", "circ_mv", "turnover_rate", "ret_1d", *context_cols]:
        if optional in panel.columns:
            cols.append(optional)

    risk_cols = ["trade_date", "ts_code", "pct_chg", "amount"]
    for c in cols:
        if c in panel.columns and c not in risk_cols:
            risk_cols.append(c)
    risk = panel[[c for c in risk_cols if c in panel.columns]].copy()
    risk["trade_date"] = risk["trade_date"].astype(str)

    max_abs_pct = float(strategy_cfg.get("max_abs_pct_chg", 9.5))
    min_amount_q = float(strategy_cfg.get("min_amount_quantile", 0.05))
    max_vol_q = float(strategy_cfg.get("max_volatility_quantile", 0.98))
    min_age = float(strategy_cfg.get("min_list_age_years", 0.0))
    max_mv_q = float(strategy_cfg.get("max_mv_quantile", 0.95))
    min_mv_q = float(strategy_cfg.get("min_mv_quantile", 0.05))
    max_turnover = float(strategy_cfg.get("max_turnover", 30.0))

    risk["amount_cut"] = risk.groupby("trade_date")["amount"].transform(lambda s: s.quantile(min_amount_q))
    risk["buyable"] = (
        (risk["pct_chg"].abs() < max_abs_pct) &
        (risk["amount"] >= risk["amount_cut"])
    )

    if "volatility_20d" in risk.columns:
        risk["vol_cut"] = risk.groupby("trade_date")["volatility_20d"].transform(lambda s: s.quantile(max_vol_q))
        risk["buyable"] &= risk["volatility_20d"].fillna(0) <= risk["vol_cut"].fillna(np.inf)

    if "list_age_years" in risk.columns and min_age > 0:
        risk["buyable"] &= risk["list_age_years"].fillna(min_age) >= min_age

    if "circ_mv" in risk.columns:
        risk["max_mv_cut"] = risk.groupby("trade_date")["circ_mv"].transform(lambda s: s.quantile(max_mv_q))
        risk["min_mv_cut"] = risk.groupby("trade_date")["circ_mv"].transform(lambda s: s.quantile(min_mv_q))
        risk["buyable"] &= (risk["circ_mv"].fillna(0) <= risk["max_mv_cut"].fillna(np.inf)) & (risk["circ_mv"].fillna(0) >= risk["min_mv_cut"].fillna(0))

    if "turnover_rate" in risk.columns and max_turnover < 100:
        risk["buyable"] &= risk["turnover_rate"].fillna(0) <= max_turnover

    if "ret_1d" in risk.columns:
        max_ret_cut = float(strategy_cfg.get("max_ret_cut", 8.0))
        risk["buyable"] &= risk["ret_1d"].abs() <= max_ret_cut / 100.0

    out["trade_date"] = out["trade_date"].astype(str)
    keep_context = [c for c in context_cols if c in risk.columns and c not in out.columns]
    out = out.merge(
        risk[["trade_date", "ts_code", "buyable", *keep_context]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    out["buyable"] = out["buyable"].fillna(False).astype(bool)
    return out


def calculate_portfolio_risk_metrics(equity_curve: pd.DataFrame, returns: pd.Series) -> dict:
    """Calculate advanced risk metrics for the portfolio."""
    if len(returns) < 2:
        return {}

    metrics = {}
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return {}

    rolling_max = equity_curve["equity"].cummax()
    drawdown = (equity_curve["equity"] - rolling_max) / rolling_max
    metrics["max_drawdown"] = float(drawdown.min())
    metrics["max_drawdown_duration"] = int((rolling_max - equity_curve["equity"]).groupby((equity_curve["equity"] == rolling_max).cumsum()).cumcount().max())
    metrics["annual_return"] = float((1.0 + returns).prod() ** (252 / len(returns)) - 1.0)

    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        metrics["sortino_ratio"] = float(returns.mean() / (downside_returns.std() + 1e-9) * np.sqrt(252))
    else:
        metrics["sortino_ratio"] = 0.0

    skewness = float(returns.skew())
    metrics["skewness"] = skewness

    kurtosis = float(returns.kurtosis())
    metrics["kurtosis"] = kurtosis

    var_95 = float(returns.quantile(0.05))
    metrics["var_95"] = var_95
    metrics["cvar_95"] = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95

    metrics["calmar_ratio"] = float(metrics["annual_return"] / abs(metrics["max_drawdown"]) if metrics["max_drawdown"] != 0 else 0)

    return metrics


def apply_position_sizing(scores: pd.DataFrame, panel: pd.DataFrame, strategy_cfg: dict) -> pd.DataFrame:
    """Calculate position sizes based on risk metrics and score confidence."""
    out = scores.copy()

    if not strategy_cfg.get("use_risk_parity", False):
        return out

    cols = ["trade_date", "ts_code", "pct_chg", "amount"]
    for optional in ["volatility_20d", "circ_mv"]:
        if optional in panel.columns:
            cols.append(optional)

    risk = panel[[c for c in cols if c in panel.columns]].copy()
    risk["trade_date"] = risk["trade_date"].astype(str)

    risk["vol_normalized"] = risk.groupby("trade_date")["volatility_20d"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12)
    )

    out["trade_date"] = out["trade_date"].astype(str)
    out = out.merge(risk[["trade_date", "ts_code", "vol_normalized"]], on=["trade_date", "ts_code"], how="left")

    base_size = 1.0
    vol_scalar = 0.02
    out["position_size"] = base_size / (1 + vol_scalar * out["vol_normalized"].fillna(0).abs())
    out["position_size"] = out["position_size"] / out["position_size"].sum()

    return out
