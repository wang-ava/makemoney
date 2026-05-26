from __future__ import annotations

import numpy as np
import pandas as pd


RANK_COLS = [
    "pct_chg",
    "ret_1d",
    "ret_2d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "mom_20d",
    "vol_ratio",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "net_mf_amount",
    "net_mf_ratio",
    "mf_ratio",
    "mf_momentum_3d",
    "mf_ratio_5d",
    "circ_mv",
    "total_mv",
    "amount_ma5_ratio",
    "amount_ma20_ratio",
    "rsi_6",
    "rsi_14",
    "macd",
    "macd_hist",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "kdj_diff",
    "cci_14",
    "cci_20",
    "williams_r",
    "obv",
    "obv_gap",
    "atr_14",
    "atr_norm_14",
    "momentum_10",
    "roc_10",
    "roc_12",
    # ---- 新增特征 (增强版) ----
    # 技术指标
    "stochastic_k",
    "stochastic_d",
    "adx",
    "plus_di",
    "minus_di",
    "supertrend",
    "rsi_9",
    "rsi_21",
    "psy",
    # 成交量
    "vwap_gap",
    "volume_burst",
    "volume_persistence",
    "vol_change_ratio",
    # 资金流
    "mf_acceleration",
    "smart_money_ratio",
    "active_money_ratio",
    # 趋势
    "trend_strength",
    "ma_bull_count",
    "price_position",
    # 波动率
    "downside_vol",
    "upside_vol",
    # 估值
    "ep_ttm",
    "bp",
    # 规模
    "log_circ_mv",
    "log_total_mv",
    "free_share_ratio",
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


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, m: int = 3) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator %K and %D.

    Returns:
        tuple: (slow_k, slow_d)
    """
    lowest_low = low.rolling(window=n, min_periods=1).min()
    highest_high = high.rolling(window=n, min_periods=1).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-12) * 100
    slow_k = rsv.rolling(window=m, min_periods=1).mean()
    slow_d = slow_k.rolling(window=m, min_periods=1).mean()
    return slow_k, slow_d


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX (Average Directional Index), +DI and -DI.

    Returns:
        tuple: (adx, plus_di, minus_di)
    """
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.copy()
    minus_dm = low_diff.copy()
    plus_dm[high_diff <= low_diff] = 0
    minus_dm[low_diff <= high_diff] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=window, min_periods=window // 2).mean()
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window // 2).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window // 2).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.rolling(window=window, min_periods=window // 2).mean()

    return adx, plus_di, minus_di


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Calculate Supertrend indicator and direction.

    Returns:
        tuple: (supertrend_value, direction)
            direction: 1 for uptrend, -1 for downtrend
    """
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period // 2).mean()

    hl_avg = (high + low) / 2
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr

    supertrend = pd.Series(0.0, index=close.index)
    direction = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        if pd.isna(supertrend.iloc[i - 1]):
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] > upper_band.iloc[i - 1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i - 1]
            direction.iloc[i] = direction.iloc[i - 1]

    return supertrend, direction


def _downside_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate downside volatility (only negative returns)."""
    downside = returns.copy()
    downside[downside > 0] = 0
    return downside.rolling(window=window, min_periods=max(5, window // 2)).std()


def _upside_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate upside volatility (only positive returns)."""
    upside = returns.copy()
    upside[upside < 0] = 0
    return upside.rolling(window=window, min_periods=max(5, window // 2)).std()


def _volume_burst(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volume burst: (vol - MA) / std."""
    vol_ma = volume.rolling(window=window, min_periods=max(5, window // 2)).mean()
    vol_std = volume.rolling(window=window, min_periods=max(5, window // 2)).std()
    return (volume - vol_ma) / (vol_std + 1e-12)


def _volume_persistence(volume: pd.Series, window: int = 5) -> pd.Series:
    """Calculate volume persistence: consecutive up/down days."""
    vol_change = volume.pct_change()
    persistence = pd.Series(0, index=volume.index)

    for i in range(window, len(volume)):
        recent = vol_change.iloc[i - window + 1:i + 1]
        positive_count = (recent > 0).sum()
        negative_count = (recent < 0).sum()
        persistence.iloc[i] = positive_count if positive_count > negative_count else -negative_count

    return persistence


def _psychological_line(close: pd.Series, window: int = 12) -> pd.Series:
    """Calculate Psychological Line indicator."""
    up_days = (close.diff() > 0).astype(int)
    return up_days.rolling(window=window, min_periods=max(3, window // 2)).sum() / window * 100


def _price_position(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate price position within historical range."""
    highest = high.rolling(window=window, min_periods=max(5, window // 2)).max()
    lowest = low.rolling(window=window, min_periods=max(5, window // 2)).min()
    return (close - lowest) / (highest - lowest + 1e-12)


def _ma_bull_count(close: pd.Series, periods: list = None) -> pd.Series:
    """Count how many moving averages are in ascending order."""
    if periods is None:
        periods = [5, 10, 20, 60]
    bull_count = pd.Series(0, index=close.index)

    for i in range(1, len(close)):
        count = 0
        for p in periods:
            if i >= p:
                ma_current = close.iloc[i]
                ma_past = close.iloc[i - p]
                if pd.notna(ma_past) and pd.notna(ma_current):
                    if ma_current > ma_past:
                        count += 1
        bull_count.iloc[i] = count

    return bull_count


def add_features(
    panel: pd.DataFrame,
    cross_section_rank: bool = True,
    label_horizon: int = 1,
    fill_missing: bool = True,
) -> pd.DataFrame:
    df = panel.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if fill_missing:
        raw_fill_cols = [
            c
            for c in [
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "vol",
                "amount",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "ps",
                "ps_ttm",
                "total_mv",
                "circ_mv",
            ]
            if c in df.columns
        ]
        if raw_fill_cols:
            df[raw_fill_cols] = df.groupby("ts_code", sort=False)[raw_fill_cols].ffill(limit=5)

    by_code = df.groupby("ts_code", sort=False)
    close_g = by_code["close"]

    if "vol" in df.columns:
        df["suspended_flag"] = (df["vol"].fillna(0) <= 0).astype(float)

    df["ret_1d"] = close_g.pct_change()
    for w in (2, 3, 5, 10, 20, 60):
        df[f"ret_{w}d"] = close_g.pct_change(w)
        ma = close_g.transform(lambda s, win=w: s.rolling(win, min_periods=max(2, win // 2)).mean())
        df[f"ma{w}_gap"] = df["close"] / ma.replace(0, np.nan) - 1.0

    df["mom_20d"] = df["ret_5d"] - df["ret_20d"]
    df["intraday_ret"] = df["close"] / df["open"].replace(0, np.nan) - 1.0
    df["overnight_gap"] = df["open"] / df["pre_close"].replace(0, np.nan) - 1.0
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"].replace(0, np.nan)
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"].replace(0, np.nan)
    df["upper_shadow_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (hl_range + 1e-12)
    df["lower_shadow_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (hl_range + 1e-12)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (hl_range + 1e-12)
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["amount_per_vol"] = df["amount"] / df["vol"].replace(0, np.nan)
    if "vwap" in df.columns:
        df["vwap_gap"] = df["close"] / df["vwap"].replace(0, np.nan) - 1.0
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

    ret_g = by_code["ret_1d"]
    for w in (5, 10, 20):
        df[f"volatility_{w}d"] = ret_g.transform(lambda s, win=w: s.rolling(win, min_periods=max(3, win // 2)).std())
        df[f"ret_mean_{w}d"] = ret_g.transform(lambda s, win=w: s.rolling(win, min_periods=max(3, win // 2)).mean())
    if {"volatility_5d", "volatility_20d"}.issubset(df.columns):
        df["vol_ratio"] = df["volatility_5d"] / df["volatility_20d"].replace(0, np.nan)

    df["log_vol"] = np.log1p(df["vol"].clip(lower=0))
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    for w in (5, 20):
        vol_ma = by_code["vol"].transform(lambda s, win=w: s.rolling(win, min_periods=max(2, win // 2)).mean())
        amount_ma = by_code["amount"].transform(lambda s, win=w: s.rolling(win, min_periods=max(2, win // 2)).mean())
        df[f"vol_ma{w}_ratio"] = df["vol"] / vol_ma.replace(0, np.nan)
        df[f"amount_ma{w}_ratio"] = df["amount"] / amount_ma.replace(0, np.nan)
    if "vol_ma5_ratio" in df.columns:
        df["volume_ratio_5d"] = df["vol_ma5_ratio"]
    if "vol_ma20_ratio" in df.columns:
        df["volume_ratio_20d"] = df["vol_ma20_ratio"]
    if {"volume_ratio_5d", "volume_ratio_20d"}.issubset(df.columns):
        df["turnover_change"] = df["volume_ratio_5d"] / df["volume_ratio_20d"].replace(0, np.nan)

    df["rsi_6"] = close_g.transform(lambda s: _rsi(s, 6))
    df["rsi_14"] = close_g.transform(_rsi)
    ema5 = close_g.transform(lambda s: _ema(s, 5))
    ema20 = close_g.transform(lambda s: _ema(s, 20))
    df["ema_5_20_ratio"] = ema5 / ema20.replace(0, np.nan)
    ema12 = close_g.transform(lambda s: _ema(s, 12))
    ema26 = close_g.transform(lambda s: _ema(s, 26))
    df["macd"] = ema12 - ema26
    df["macd_signal"] = by_code["macd"].transform(lambda s: _ema(s, 9))
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    ma20 = close_g.transform(lambda s: s.rolling(20, min_periods=10).mean())
    std20 = close_g.transform(lambda s: s.rolling(20, min_periods=10).std())
    df["boll_pos"] = (df["close"] - ma20) / (2 * std20 + 1e-12)
    df["boll_width"] = (2 * std20) / (ma20 + 1e-12)
    lower_band = ma20 - 2 * std20
    upper_band = ma20 + 2 * std20
    df["bollinger_pos"] = (df["close"] - lower_band) / (upper_band - lower_band + 1e-12)
    df["limit_up_flag"] = (df["pct_chg"] >= 9.5).astype(float)
    df["limit_down_flag"] = (df["pct_chg"] <= -9.5).astype(float)

    if "net_mf_amount" in df.columns and "amount" in df.columns:
        denom = (df["amount"] / 10).replace(0, np.nan)
        df["mf_ratio"] = df["net_mf_amount"] / denom
        df["net_mf_ratio"] = df["mf_ratio"]
        df["mf_ratio_5d"] = by_code["mf_ratio"].transform(lambda s: s.rolling(5, min_periods=3).mean())
        df["mf_momentum_3d"] = by_code["net_mf_amount"].transform(lambda s: s.rolling(3, min_periods=2).sum())
        df["mf_reversal"] = df["net_mf_amount"] - by_code["net_mf_amount"].transform(
            lambda s: s.rolling(5, min_periods=3).mean()
        )

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
        def _calc_indicators(g, code):
            g = g.copy()
            g["ts_code"] = str(code)
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
            g["atr_norm_14"] = g["atr_14"] / close.replace(0, np.nan)

            g["momentum_10"] = _momentum(close, window=10)
            g["roc_10"] = _roc(close, window=10)
            g["roc_12"] = _roc(close, window=12)
            x = np.arange(5, dtype=float)
            x_centered = x - x.mean()
            denom = float((x_centered ** 2).sum())
            g["obv_slope_5d"] = g["obv"].rolling(5, min_periods=5).apply(
                lambda y: float(np.dot(x_centered, y - y.mean()) / (denom + 1e-12)),
                raw=False,
            )

            for c in ["kdj_k", "kdj_d", "kdj_j", "cci_14", "williams_r", "atr_14", "momentum_10"]:
                g[f"{c}_ma5"] = g[c].rolling(5, min_periods=3).mean()

            return g

        indicator_chunks = [
            _calc_indicators(g, code)
            for code, g in df.groupby("ts_code", sort=False)
        ]
        df = pd.concat(indicator_chunks, ignore_index=True)

    # ===== 新增特征：增强技术指标 =====
    # 多周期RSI
    df["rsi_9"] = close_g.transform(lambda s: _rsi(s, 9))
    df["rsi_21"] = close_g.transform(lambda s: _rsi(s, 21))

    # Stochastic指标 (使用整体计算避免groupby问题)
    if {"high", "low", "close", "vol"}.issubset(df.columns):
        for code, g in df.groupby("ts_code", sort=False):
            high = g["high"]
            low = g["low"]
            close = g["close"]
            vol = g["vol"]

            # Stochastic
            sk, sd = _stochastic(high, low, close, n=14, m=3)
            df.loc[g.index, "stochastic_k"] = sk.values
            df.loc[g.index, "stochastic_d"] = sd.values
            df.loc[g.index, "stochastic_j"] = 3 * sk - 2 * sd

            # ADX
            adx, plus_di, minus_di = _adx(high, low, close, window=14)
            df.loc[g.index, "adx"] = adx.values
            df.loc[g.index, "plus_di"] = plus_di.values
            df.loc[g.index, "minus_di"] = minus_di.values

            # Supertrend
            st, direction = _supertrend(high, low, close, period=10, multiplier=3)
            df.loc[g.index, "supertrend"] = st.values
            df.loc[g.index, "supertrend_dir"] = direction.values

            # Psychological Line
            psy = _psychological_line(close, window=12)
            df.loc[g.index, "psy"] = psy.values

            # Volume Burst
            vb = _volume_burst(vol, window=20)
            df.loc[g.index, "volume_burst"] = vb.values

            # Volume Persistence
            vp = _volume_persistence(vol, window=5)
            df.loc[g.index, "volume_persistence"] = vp.values

            # Price Position
            pp = _price_position(high, low, close, window=20)
            df.loc[g.index, "price_position"] = pp.values

            # MA Bull Count
            bc = _ma_bull_count(close)
            df.loc[g.index, "ma_bull_count"] = bc.values

            # Downside/Upside Volatility
            ret = close.pct_change()
            ds_vol = _downside_volatility(ret, window=20)
            us_vol = _upside_volatility(ret, window=20)
            df.loc[g.index, "downside_vol"] = ds_vol.values
            df.loc[g.index, "upside_vol"] = us_vol.values

            # Golden Cross / Death Cross (MA5 vs MA20)
            ma5 = close.rolling(5, min_periods=3).mean()
            ma20 = close.rolling(20, min_periods=10).mean()
            ma10 = close.rolling(10, min_periods=5).mean()
            gc = ((ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))).astype(float)
            dc = ((ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1))).astype(float)
            df.loc[g.index, "golden_cross"] = gc.values
            df.loc[g.index, "death_cross"] = dc.values

            # Vol change ratio
            vol_change = vol.pct_change()
            vol_change_ratio = vol_change / (vol_change.rolling(5, min_periods=3).std() + 1e-12)
            df.loc[g.index, "vol_change_ratio"] = vol_change_ratio.values

            # MF acceleration (资金流加速度)
            if "net_mf_amount" in g.columns:
                mf = g["net_mf_amount"].fillna(0)
                mf_accel = mf.diff()
                df.loc[g.index, "mf_acceleration"] = mf_accel.values
                mf_ma5 = mf.rolling(5, min_periods=3).mean()
                df.loc[g.index, "mf_5d_ma"] = mf_ma5.values
                mf_ma10 = mf.rolling(10, min_periods=5).mean()
                df.loc[g.index, "mf_10d_ma"] = mf_ma10.values

    # ===== 新增特征：行业相对估值 =====
    if "industry" in df.columns and "pe_ttm" in df.columns:
        ind_pe_median = df.groupby(["trade_date", "industry"])["pe_ttm"].transform("median")
        df["industry_rel_pe"] = df["pe_ttm"] / (ind_pe_median.replace(0, np.nan) + 1e-12)

    if "industry" in df.columns and "pb" in df.columns:
        ind_pb_median = df.groupby(["trade_date", "industry"])["pb"].transform("median")
        df["industry_rel_pb"] = df["pb"] / (ind_pb_median.replace(0, np.nan) + 1e-12)

    # ===== 新增特征：规模因子 =====
    if "circ_mv" in df.columns:
        # 市值规模分位数
        df["mv_percentile"] = df.groupby("trade_date")["circ_mv"].rank(pct=True)

        # 规模中性因子 (剔除市值影响后的残差)
        log_mv = np.log1p(df["circ_mv"].clip(lower=0).fillna(0))
        mv_effect = df.groupby("trade_date")["pct_chg"].transform(lambda x: np.polyval(np.polyfit(log_mv.loc[x.index], x, 1), log_mv.loc[x.index]))
        df["size_neutral_ret"] = df["pct_chg"] - mv_effect

    # ===== 新增特征：趋势信号 =====
    # ADX趋势强度分位数
    if "adx" in df.columns:
        df["adx_percentile"] = df.groupby("trade_date")["adx"].rank(pct=True)

    # Trend persistence (趋势持续天数)
    if "supertrend_dir" in df.columns:
        df["trend_persistence"] = df.groupby("ts_code")["supertrend_dir"].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
        ) * df["supertrend_dir"]

    # ===== 新增特征：价值综合分 =====
    if {"ep_ttm", "bp", "ps"}.issubset(df.columns):
        # 标准化后求和
        ep_norm = df.groupby("trade_date")["ep_ttm"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
        bp_norm = df.groupby("trade_date")["bp"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
        df["value_score"] = (ep_norm + bp_norm) / 2

    # ===== 新增特征：动量加速度 =====
    if "momentum_10" in df.columns:
        df["momentum_accel"] = df.groupby("ts_code")["momentum_10"].diff()

    # ===== 新增特征：量价背离 =====
    if {"ret_5d", "volume_ratio_5d"}.issubset(df.columns):
        vol_ret_corr = df.groupby("ts_code").apply(
            lambda g: g["ret_5d"].rolling(10, min_periods=5).corr(g["volume_ratio_5d"])
        ).reset_index(level=0, drop=True)
        df["vol_price_divergence"] = vol_ret_corr

    # ===== 新增特征：收益分布特征 =====
    if "ret_1d" in df.columns:
        # 偏度
        df["ret_skewness"] = close_g.transform(lambda s: s.pct_change().rolling(20, min_periods=10).skew())
        # 峰度
        df["ret_kurtosis"] = close_g.transform(lambda s: s.pct_change().rolling(20, min_periods=10).kurt())

    # ===== 新增特征：波动率压缩 =====
    if {"volatility_5d", "volatility_20d"}.issubset(df.columns):
        df["vol_compression"] = df["volatility_5d"] / (df["volatility_20d"] + 1e-12)

    # ===== 新增特征：支撑阻力位 =====
    if "close" in df.columns:
        for code, g in df.groupby("ts_code", sort=False):
            close = g["close"]
            # 20日高低点
            high_20 = close.rolling(20, min_periods=10).max()
            low_20 = close.rolling(20, min_periods=10).min()
            # 支撑位置 (收盘价离20日低点的距离)
            df.loc[g.index, "support_position"] = (close - low_20) / (high_20 - low_20 + 1e-12)
            # 阻力位置 (收盘价离20日高点的距离)
            df.loc[g.index, "resistance_position"] = (high_20 - close) / (high_20 - low_20 + 1e-12)

    # ===== 新增特征：时间效应 =====
    if "trade_date" in df.columns:
        trade_dt = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df["month_of_year"] = trade_dt.dt.month.fillna(0).astype(np.int16)
        df["day_of_week"] = trade_dt.dt.dayofweek.fillna(-1).astype(np.int16)
        df["week_of_year"] = trade_dt.dt.isocalendar().week.fillna(0).astype(np.int16)
        # 财报季标识 (3, 4, 6, 7, 9, 10, 12, 1月)
        df["earnings_season"] = df["month_of_year"].isin([3, 4, 6, 7, 9, 10, 12, 1]).astype(np.int8)

    if cross_section_rank:
        derived: dict[str, pd.Series] = {}
        for c in RANK_COLS:
            if c in df.columns:
                derived[f"rank_{c}"] = df.groupby("trade_date")[c].rank(pct=True)
                derived[f"z_{c}"] = _cross_section_z(df, c)

        # 行业截面排名
        if "industry" in df.columns:
            for c in [
                "ret_5d", "ret_20d", "turnover_rate", "circ_mv", "mf_ratio",
                "rsi_14", "macd_hist", "kdj_k", "volume_ratio_5d",
                "stochastic_k", "adx", "supertrend_dir"
            ]:
                if c in df.columns:
                    derived[f"ind_rank_{c}"] = df.groupby(["trade_date", "industry"])[c].rank(pct=True)

        # 规模中性排名
        if "size_neutral_ret" in df.columns:
            derived["rank_size_neutral_ret"] = df.groupby("trade_date")["size_neutral_ret"].rank(pct=True)

        # 价值因子排名
        if "value_score" in df.columns:
            derived["rank_value_score"] = df.groupby("trade_date")["value_score"].rank(pct=True)

        # 趋势因子排名
        if "trend_persistence" in df.columns:
            derived["rank_trend_persistence"] = df.groupby("trade_date")["trend_persistence"].rank(pct=True)

        # 波动率排名
        if "vol_compression" in df.columns:
            derived["rank_vol_compression"] = df.groupby("trade_date")["vol_compression"].rank(pct=True)

        # 资金流排名
        if "mf_acceleration" in df.columns:
            derived["rank_mf_acceleration"] = df.groupby("trade_date")["mf_acceleration"].rank(pct=True)

        if derived:
            df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)

    df = df.copy()
    by_code = df.groupby("ts_code", sort=False)
    close_g = by_code["close"]
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
        # 新增排除列
        "month_of_year",
        "day_of_week",
        "week_of_year",
        "earnings_season",
        "supertrend_dir",
        "mf_5d_ma",
        "mf_10d_ma",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
            cols.append(c)
    return cols
