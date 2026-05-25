from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _list_trade_dates(daily_dir: Path, start: str, end: str) -> list[str]:
    dates = sorted(p.stem for p in daily_dir.glob("*.csv"))
    return [d for d in dates if start <= d <= end]


def _load_hs300_codes(data_dir: Path, as_of: str) -> set[str]:
    """Load HS300 constituents using the latest month not after ``as_of``."""
    iw_dir = data_dir / "index_weight"
    as_of_month = str(as_of)[:6]
    candidates = []
    for fp in iw_dir.glob("*_000300.SH.csv"):
        month = fp.name.split("_", 1)[0]
        if month <= as_of_month:
            candidates.append(fp)
    candidates = sorted(candidates, reverse=True)
    for fp in candidates:
        df = pd.read_csv(fp)
        if len(df) > 10 and "con_code" in df.columns:
            return set(df["con_code"].astype(str))

    # fallback: 用当日成交额前 300，保证没有未来成分股泄露。
    daily_fp = data_dir / "daily" / f"{as_of}.csv"
    if not daily_fp.exists():
        return set()
    df = pd.read_csv(daily_fp, usecols=["ts_code", "amount"])
    return set(df.nlargest(300, "amount")["ts_code"].astype(str))


def _load_market_features(data_dir: Path, start: str, end: str) -> pd.DataFrame:
    """Load and compute market index features.

    Features per index:
    - ret1/ret5/ret10/ret20: Returns at different horizons
    - ma_gap: Gap from moving average
    - volatility: Rolling volatility
    - amount_trend: Amount trend
    - trend_strength: ADX-like trend strength
    - market_sentiment: Composite market mood
    """
    rows: list[pd.DataFrame] = []
    index_map = {
        "000001.SH": "sh",      # 上证指数
        "000300.SH": "hs300",   # 沪深300
        "000016.SH": "sz50",    # 上证50
        "000905.SH": "zz500",   # 中证500
        "399006.SZ": "cyb",     # 创业板指
        "399005.SZ": "cye",     # 创业板综
    }
    for code, prefix in index_map.items():
        fp = data_dir / "market" / f"{code}.csv"
        if not fp.exists():
            continue
        m = pd.read_csv(fp, usecols=["trade_date", "close", "pct_chg", "amount", "high", "low"])
        m["trade_date"] = m["trade_date"].astype(str)
        m = m[(m["trade_date"] >= start) & (m["trade_date"] <= end)].copy()
        m = m.sort_values("trade_date")
        close = m["close"].replace(0, pd.NA)

        # 收益率
        m[f"{prefix}_idx_ret1"] = m["pct_chg"] / 100.0
        m[f"{prefix}_idx_ret5"] = close.pct_change(5)
        m[f"{prefix}_idx_ret10"] = close.pct_change(10)
        m[f"{prefix}_idx_ret20"] = close.pct_change(20)

        # 均线偏离
        m[f"{prefix}_idx_ma5_gap"] = close / close.rolling(5, min_periods=3).mean() - 1.0
        m[f"{prefix}_idx_ma20_gap"] = close / close.rolling(20, min_periods=5).mean() - 1.0
        m[f"{prefix}_idx_ma60_gap"] = close / close.rolling(60, min_periods=20).mean() - 1.0

        # 波动率
        m[f"{prefix}_idx_vol5"] = m[f"{prefix}_idx_ret1"].rolling(5, min_periods=3).std()
        m[f"{prefix}_idx_vol20"] = m[f"{prefix}_idx_ret1"].rolling(20, min_periods=5).std()
        m[f"{prefix}_idx_vol60"] = m[f"{prefix}_idx_ret1"].rolling(60, min_periods=20).std()

        # 成交量趋势
        m[f"{prefix}_idx_vol_ma5_ratio"] = m["amount"] / m["amount"].rolling(5, min_periods=3).mean()
        m[f"{prefix}_idx_vol_ma20_ratio"] = m["amount"] / m["amount"].rolling(20, min_periods=5).mean()

        # 指数特有特征
        m[f"{prefix}_idx_amount_log"] = np.log1p(m["amount"].clip(lower=0))

        # 布林带位置
        ma20 = close.rolling(20, min_periods=5).mean()
        std20 = close.rolling(20, min_periods=5).std()
        m[f"{prefix}_idx_boll_pos"] = (close - ma20) / (2 * std20 + 1e-12)

        # 高低点位置
        high_20 = m["high"].rolling(20, min_periods=5).max()
        low_20 = m["low"].rolling(20, min_periods=5).min()
        m[f"{prefix}_idx_high_low_pos"] = (close - low_20) / (high_20 - low_20 + 1e-12)

        cols = [
            "trade_date",
            f"{prefix}_idx_ret1",
            f"{prefix}_idx_ret5",
            f"{prefix}_idx_ret10",
            f"{prefix}_idx_ret20",
            f"{prefix}_idx_ma5_gap",
            f"{prefix}_idx_ma20_gap",
            f"{prefix}_idx_ma60_gap",
            f"{prefix}_idx_vol5",
            f"{prefix}_idx_vol20",
            f"{prefix}_idx_vol60",
            f"{prefix}_idx_vol_ma5_ratio",
            f"{prefix}_idx_vol_ma20_ratio",
            f"{prefix}_idx_boll_pos",
            f"{prefix}_idx_high_low_pos",
        ]
        rows.append(m[[c for c in cols if c in m.columns]])

    if not rows:
        return pd.DataFrame({"trade_date": []})

    out = rows[0]
    for item in rows[1:]:
        out = out.merge(item, on="trade_date", how="outer")

    # 计算市场情绪综合指标
    if "sh_idx_ret1" in out.columns and "cyb_idx_ret1" in out.columns:
        # 市场分化度
        out["market_divergence"] = out["sh_idx_ret1"] - out["cyb_idx_ret1"]
        # 市场趋势 (上证 + 创业板 平均)
        out["market_trend"] = (out["sh_idx_ret1"] + out["cyb_idx_ret1"]) / 2

    return out


def _load_news_features(data_dir: Path, dates: list[str]) -> pd.DataFrame:
    """Build enhanced market-wide news sentiment features.

    Features:
    - news_count: number of news articles
    - news_sentiment: (positive - negative) / count
    - news_len_mean: average news length
    - news_sentiment_smooth: 5-day smoothed sentiment
    - market_sentiment_index: composite market mood indicator
    """
    # 扩展情绪词典
    positive = (
        "增长", "上升", "利好", "突破", "回暖", "创新高", "增持", "盈利", "合作", "提振",
        "大涨", "涨停", "强势", "领涨", "反弹", "复苏", "景气", "高增长", "超预期", "业绩",
        "扭亏", "首亏", "预盈", "预增", "大幅增长", "持续增长", "稳定增长", "加速发展",
        "突破性", "历史新高", "市场看好", "资金流入", "外资流入", "机构增持"
    )
    negative = (
        "下降", "下跌", "风险", "亏损", "减持", "处罚", "违约", "调查", "承压", "放缓",
        "大跌", "跌停", "弱势", "领跌", "回调", "衰退", "低迷", "低增长", "不及预期", "业绩下滑",
        "首亏", "预亏", "预减", "大幅下降", "持续下滑", "危机", "不确定", "市场担忧", "资金流出",
        "外资流出", "机构减持", "黑天鹅", "暴雷", "造假", "退市风险", "债务危机", "流动性危机"
    )
    # 行业特定关键词
    industry_positive = ("政策利好", "行业扶持", "国产替代", "技术突破", "市场份额扩大", "订单饱满", "产能扩张")
    industry_negative = ("政策收紧", "行业整顿", "竞争加剧", "成本上升", "需求下滑", "产能过剩", "价格战")

    rows = []
    for d in dates:
        fp = news_dir / f"{d}.csv" if 'news_dir' in dir() else data_dir / "news" / f"{d}.csv"
        news_dir = data_dir / "news"
        fp = news_dir / f"{d}.csv"
        if not fp.exists():
            rows.append({
                "trade_date": d,
                "news_count": 0,
                "news_sentiment": 0.0,
                "news_len_mean": 0.0,
                "news_sentiment_smooth": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "industry_positive_count": 0,
                "industry_negative_count": 0,
            })
            continue
        news = pd.read_csv(fp, usecols=["title", "content"])
        text = (news["title"].fillna("") + " " + news["content"].fillna("")).astype(str)
        count = len(news)

        # 基础情绪
        pos = sum(text.str.contains(k, regex=False).sum() for k in positive)
        neg = sum(text.str.contains(k, regex=False).sum() for k in negative)

        # 行业情绪
        ind_pos = sum(text.str.contains(k, regex=False).sum() for k in industry_positive)
        ind_neg = sum(text.str.contains(k, regex=False).sum() for k in industry_negative)

        rows.append({
            "trade_date": d,
            "news_count": count,
            "news_sentiment": (pos - neg) / max(count, 1),
            "news_len_mean": float(text.str.len().mean()) if count else 0.0,
            "positive_count": pos,
            "negative_count": neg,
            "industry_positive_count": ind_pos,
            "industry_negative_count": ind_neg,
        })

    df = pd.DataFrame(rows)

    # 添加5日平滑情绪
    df = df.sort_values("trade_date")
    df["news_sentiment_smooth"] = df["news_sentiment"].rolling(5, min_periods=1).mean()

    # 添加市场情绪指数 (综合多个指标)
    if "news_count" in df.columns:
        df["news_intensity"] = df["news_count"] / df["news_count"].rolling(20, min_periods=5).mean()

    return df


def build_panel(
    data_dir: str | Path,
    start_date: str,
    end_date: str,
    use_metric: bool = True,
    use_moneyflow: bool = True,
    use_market: bool = True,
    use_news: bool = False,
    universe: str = "hs300",
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    daily_dir = data_dir / "daily"
    dates = _list_trade_dates(daily_dir, start_date, end_date)
    if not dates:
        raise FileNotFoundError(f"No daily files in [{start_date}, {end_date}]")

    basic = pd.read_csv(data_dir / "basic.csv", dtype=str)
    bj_codes = set(basic.loc[basic["market"] == "北交所", "ts_code"])
    basic_cols = ["ts_code", "name", "industry", "market", "list_date"]
    basic_info = basic[[c for c in basic_cols if c in basic.columns]].copy()

    hs300 = _load_hs300_codes(data_dir, dates[-1]) if universe == "hs300" else None
    market_feat = _load_market_features(data_dir, dates[0], dates[-1]) if use_market else None
    news_feat = _load_news_features(data_dir, dates) if use_news else None

    chunks: list[pd.DataFrame] = []
    for d in tqdm(dates, desc="load daily"):
        daily = pd.read_csv(daily_dir / f"{d}.csv")
        daily["trade_date"] = str(d)

        st_fp = data_dir / "stock_st" / f"{d}.csv"
        if st_fp.exists():
            st = pd.read_csv(st_fp, usecols=["ts_code"])
            st_set = set(st["ts_code"])
            daily = daily[~daily["ts_code"].isin(st_set)]

        daily = daily[~daily["ts_code"].isin(bj_codes)]
        if hs300 is not None:
            daily = daily[daily["ts_code"].isin(hs300)]
        daily = daily.merge(basic_info, on="ts_code", how="left")

        if use_metric:
            mf = data_dir / "metric" / f"{d}.csv"
            if mf.exists():
                metric = pd.read_csv(mf)
                drop_cols = [c for c in ["close", "trade_date"] if c in metric.columns]
                metric = metric.drop(columns=drop_cols, errors="ignore")
                daily = daily.merge(metric, on="ts_code", how="left")

        if use_moneyflow:
            mff = data_dir / "moneyflow" / f"{d}.csv"
            if mff.exists():
                flow = pd.read_csv(mff)
                flow = flow.drop(columns=["trade_date"], errors="ignore")
                daily = daily.merge(flow, on="ts_code", how="left")

        if market_feat is not None and not market_feat.empty:
            daily = daily.merge(market_feat[market_feat["trade_date"] == d], on="trade_date", how="left")

        if news_feat is not None and not news_feat.empty:
            daily = daily.merge(news_feat[news_feat["trade_date"] == d], on="trade_date", how="left")

        chunks.append(daily)

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return panel
