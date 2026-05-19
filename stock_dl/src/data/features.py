from __future__ import annotations

import numpy as np
import pandas as pd


RANK_COLS = [
    "pct_chg",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "mom_20d",
    "volatility_20d",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "net_mf_amount",
    "mf_ratio",
    "circ_mv",
    "total_mv",
    "amount_ma20_ratio",
    "rsi_14",
    "macd_hist",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "cci_14",
    "williams_r",
    "obv",
    "atr_14",
    "momentum_10",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window // 2).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window // 2).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 3)).mean()


def _cross_section_z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("trade_date")[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    return (df[col] - mean) / (std + 1e-12)


def _kdj_group(high, low, close, n=9, m1=3, m2=3):
    """Calculate KDJ for a single stock."""
    lowest_low = low.rolling(window=n, min_periods=1).min()
    highest_high = high.rolling(window=n, min_periods=1).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-12) * 100
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def _cci(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(window=window, min_periods=window // 2).mean()
    mad = tp.rolling(window=window, min_periods=window // 2).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    cci = (tp - sma) / (mad * 0.015 + 1e-12)
    return cci


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Williams %R indicator."""
    highest_high = high.rolling(window=window, min_periods=1).max()
    lowest_low = low.rolling(window=window, min_periods=1).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-12)
    return wr


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume using vectorized operations."""
    close_diff = close.diff()
    direction = pd.Series(0, index=close.index)
    direction[close_diff > 0] = 1
    direction[close_diff < 0] = -1
    direction.iloc[0] = 0
    obv = (direction * volume).cumsum()
    return obv


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window // 2).mean()
    return atr


def _momentum(close: pd.Series, window: int = 10) -> pd.Series:
    """Calculate Momentum indicator."""
    return close.diff(window)


def _roc(close: pd.Series, window: int = 12) -> pd.Series:
    """Calculate Rate of Change."""
    return (close - close.shift(window)) / (close.shift(window) + 1e-12) * 100


def add_features(
    panel: pd.DataFrame,
    cross_section_rank: bool = True,
    label_horizon: int = 1,
    fill_missing: bool = True,
) -> pd.DataFrame:
    df = panel.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    by_code = df.groupby("ts_code", sort=False)
    close_g = by_code["close"]

    df["ret_1d"] = close_g.pct_change()
    for w in (2, 3, 5, 10, 20, 60):
        df[f"ret_{w}d"] = close_g.pct_change(w)
        ma = close_g.transform(lambda s, win=w: s.rolling(win, min_periods=max(2, win // 2)).mean())
        df[f"ma{w}_gap"] = df["close"] / ma.replace(0, np.nan) - 1.0

    df["mom_20d"] = df["ret_5d"] - df["ret_20d"]
    df["intraday_ret"] = df["close"] / df["open"].replace(0, np.nan) - 1.0
    df["overnight_gap"] = df["open"] / df["pre_close"].replace(0, np.nan) - 1.0
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"].replace(0, np.nan)
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"].replace(0, np.nan)
    if "vwap" in df.columns:
        df["vwap_gap"] = df["close"] / df["vwap"].replace(0, np.nan) - 1.0

    ret_g = by_code["ret_1d"]
    for w in (5, 10, 20):
        df[f"volatility_{w}d"] = ret_g.transform(lambda s, win=w: s.rolling(win, min_periods=max(3, win // 2)).std())
        df[f"ret_mean_{w}d"] = ret_g.transform(lambda s, win=w: s.rolling(win, min_periods=max(3, win // 2)).mean())

    df["log_vol"] = np.log1p(df["vol"].clip(lower=0))
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    for w in (5, 20):
        vol_ma = by_code["vol"].transform(lambda s, win=w: s.rolling(win, min_periods=max(2, win // 2)).mean())
        amount_ma = by_code["amount"].transform(lambda s, win=w: s.rolling(win, min_periods=max(2, win // 2)).mean())
        df[f"vol_ma{w}_ratio"] = df["vol"] / vol_ma.replace(0, np.nan)
        df[f"amount_ma{w}_ratio"] = df["amount"] / amount_ma.replace(0, np.nan)

    df["rsi_14"] = close_g.transform(_rsi)
    ema12 = close_g.transform(lambda s: _ema(s, 12))
    ema26 = close_g.transform(lambda s: _ema(s, 26))
    df["macd"] = ema12 - ema26
    df["macd_signal"] = by_code["macd"].transform(lambda s: _ema(s, 9))
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    ma20 = close_g.transform(lambda s: s.rolling(20, min_periods=10).mean())
    std20 = close_g.transform(lambda s: s.rolling(20, min_periods=10).std())
    df["boll_pos"] = (df["close"] - ma20) / (2 * std20 + 1e-12)
    df["boll_width"] = (2 * std20) / (ma20 + 1e-12)
    df["limit_up_flag"] = (df["pct_chg"] >= 9.5).astype(float)
    df["limit_down_flag"] = (df["pct_chg"] <= -9.5).astype(float)

    if "net_mf_amount" in df.columns and "amount" in df.columns:
        denom = (df["amount"] / 10).replace(0, np.nan)
        df["mf_ratio"] = df["net_mf_amount"] / denom
        df["mf_ratio_5d"] = by_code["mf_ratio"].transform(lambda s: s.rolling(5, min_periods=3).mean())

    flow_cols = ["lg", "elg"]
    if all(f"buy_{c}_amount" in df.columns and f"sell_{c}_amount" in df.columns for c in flow_cols):
        smart_buy = df["buy_lg_amount"].fillna(0) + df["buy_elg_amount"].fillna(0)
        smart_sell = df["sell_lg_amount"].fillna(0) + df["sell_elg_amount"].fillna(0)
        df["smart_money_ratio"] = (smart_buy - smart_sell) / (df["amount"] / 10).replace(0, np.nan)

        active_buy = df["buy_lg_amount"].fillna(0) + df["buy_md_amount"].fillna(0)
        active_sell = df["sell_lg_amount"].fillna(0) + df["sell_md_amount"].fillna(0)
        df["active_money_ratio"] = (active_buy - active_sell) / (df["amount"] / 10 + 1e-12)

    for c in [
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "circ_mv",
        "total_mv",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "dv_ratio",
        "dv_ttm",
        "news_count",
        "news_len_mean",
    ]:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0).fillna(0))
    if "pe_ttm" in df.columns:
        df["ep_ttm"] = 1.0 / df["pe_ttm"].replace(0, np.nan)
    if "pb" in df.columns:
        df["bp"] = 1.0 / df["pb"].replace(0, np.nan)
    if {"free_share", "total_share"}.issubset(df.columns):
        df["free_share_ratio"] = df["free_share"] / df["total_share"].replace(0, np.nan)
    if "list_date" in df.columns:
        trade_dt = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        list_dt = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
        df["list_age_years"] = (trade_dt - list_dt).dt.days / 365.25

    if {"high", "low", "close"}.issubset(df.columns) and {"vol"}.issubset(df.columns):
        def _calc_indicators(g):
            high = g["high"]
            low = g["low"]
            close = g["close"]
            vol = g["vol"]

            k, d, j = _kdj_group(high, low, close)
            g["kdj_k"] = k
            g["kdj_d"] = d
            g["kdj_j"] = j
            g["kdj_diff"] = g["kdj_k"] - g["kdj_d"]
            g["kdj金叉"] = ((g["kdj_k"] > g["kdj_d"]) & (g["kdj_k"].shift(1) <= g["kdj_d"].shift(1))).astype(float)
            g["kdj死叉"] = ((g["kdj_k"] < g["kdj_d"]) & (g["kdj_k"].shift(1) >= g["kdj_d"].shift(1))).astype(float)

            g["cci_14"] = _cci(close, high, low, window=14)
            g["cci_20"] = _cci(close, high, low, window=20)

            g["williams_r"] = _williams_r(high, low, close)

            g["obv"] = _obv(close, vol.fillna(0))
            g["obv_ma10"] = g["obv"].rolling(10, min_periods=5).mean()
            g["obv_gap"] = g["obv"] / g["obv_ma10"].replace(0, np.nan) - 1.0

            g["atr_14"] = _atr(high, low, close, window=14)

            g["momentum_10"] = _momentum(close, window=10)
            g["roc_12"] = _roc(close, window=12)

            for c in ["kdj_k", "kdj_d", "kdj_j", "cci_14", "williams_r", "atr_14", "momentum_10"]:
                g[f"{c}_ma5"] = g[c].rolling(5, min_periods=3).mean()

            return g

        df = df.groupby("ts_code", sort=False).apply(_calc_indicators).reset_index(drop=True)

    if cross_section_rank:
        for c in RANK_COLS:
            if c in df.columns:
                df[f"rank_{c}"] = df.groupby("trade_date")[c].rank(pct=True)
                df[f"z_{c}"] = _cross_section_z(df, c)
        if "industry" in df.columns:
            for c in ["ret_5d", "ret_20d", "turnover_rate", "circ_mv", "mf_ratio"]:
                if c in df.columns:
                    df[f"ind_rank_{c}"] = df.groupby(["trade_date", "industry"])[c].rank(pct=True)

    df = df.copy()
    horizon = max(int(label_horizon), 1)
    df["label"] = close_g.shift(-horizon) / df["close"] - 1.0
    df["label_direction"] = (df["label"] > 0).astype(float)
    df["label_rank"] = df.groupby("trade_date")["label"].rank(pct=True) - 0.5
    df["label_cs_z"] = _cross_section_z(df, "label").clip(-5, 5)

    if fill_missing:
        feats = feature_columns(df)
        df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
        med = df.groupby("trade_date")[feats].transform("median")
        df[feats] = df[feats].fillna(med).fillna(0.0)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "ts_code",
        "trade_date",
        "name",
        "label",
        "label_cs_z",
        "label_rank",
        "label_direction",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pretrade_date",
        "industry",
        "area",
        "cnspell",
        "market",
        "list_date",
        "act_name",
        "act_ent_type",
        "symbol",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
            cols.append(c)
    return cols
