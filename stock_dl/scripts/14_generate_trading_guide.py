#!/usr/bin/env python3
"""
同花顺手动下单 - 动态持仓交易指南生成器 v3

与 engine.py 保持一致的完整实现：
1. 正确的波动率分位数计算 (_rolling_percentile)
2. Momentum 过滤
3. 动态 k_trade
4. T+1 检查
5. 涨跌停过滤
6. 动态目标仓位计算
7. 按分数动态分配权重

用法:
    python scripts/14_generate_trading_guide.py \
        --config configs/local_scheme_a.yaml \
        --holdings "000001.SZ:1000,600000.SH:500" \
        --portfolio-value 1000000
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.features import add_features, feature_columns
from src.data.panel import build_panel
from src.models.factory import build_model_from_checkpoint


# ============================================================================
# engine.py 中的核心函数（保持一致）
# ============================================================================

def _rolling_percentile(series: pd.Series, lookback: int = 250) -> pd.Series:
    """滚动分位数计算"""
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
    """动态持仓数量计算"""
    if not strategy_cfg.get("adaptive_hold", False) or vol_pct is None:
        return int(base_n)
    min_n = int(strategy_cfg.get("adaptive_min_hold", max(1, base_n // 2)))
    max_n = int(strategy_cfg.get("adaptive_max_hold", max(base_n, base_n + base_n // 2)))
    low = float(strategy_cfg.get("adaptive_low_vol_pct", 0.3))
    high = float(strategy_cfg.get("adaptive_high_vol_pct", 0.7))
    if vol_pct >= high:
        return max_n
    if vol_pct <= low:
        return min_n
    return int(base_n)


def _choose_adaptive_n_multi(
    base_n: int,
    vol_pct: float | None,
    ic_trend: float | None,
    confidence: float,
    strategy_cfg: dict
) -> int:
    """多指标自适应持仓数量计算"""
    if not strategy_cfg.get("adaptive_hold_multi_indicator", False):
        return _choose_adaptive_n(base_n, vol_pct, strategy_cfg)

    min_n = int(strategy_cfg.get("adaptive_min_hold", max(1, base_n // 2)))
    max_n = int(strategy_cfg.get("adaptive_max_hold", max(base_n, base_n + base_n // 2)))

    # 波动率得分 (高波动→少持、留更多现金；低波动→可适度多持)
    if vol_pct is not None and np.isfinite(vol_pct):
        vol_score = 1.0 - float(np.clip(vol_pct, 0.0, 1.0))
    else:
        vol_score = 0.5

    # IC趋势得分
    if ic_trend is not None and np.isfinite(ic_trend):
        ic_trend_score = float(np.clip((ic_trend + 0.1) / 0.2, 0.0, 1.0))
    else:
        ic_trend_score = 0.5

    # 置信度
    confidence_score = float(np.clip(confidence, 0.0, 1.0))

    # 加权
    vol_weight = float(strategy_cfg.get("adaptive_vol_weight", 0.4))
    ic_weight = float(strategy_cfg.get("adaptive_ic_weight", 0.3))
    conf_weight = float(strategy_cfg.get("adaptive_conf_weight", 0.3))

    n_hold_score = vol_weight * vol_score + ic_weight * ic_trend_score + conf_weight * confidence_score
    n_hold = int(round(min_n + n_hold_score * (max_n - min_n)))
    return int(np.clip(n_hold, min_n, max_n))


def _filter_momentum(day: pd.DataFrame, buy_scores: pd.Series, strategy_cfg: dict) -> pd.Series:
    """Momentum 过滤"""
    if not strategy_cfg.get("momentum_filter", False):
        return buy_scores
    col = strategy_cfg.get("momentum_rank_col", "rank_ret_5d")
    if col not in day.columns:
        return buy_scores
    threshold = float(strategy_cfg.get("min_momentum_rank", 0.2))
    keep = day[day[col].fillna(0.5) >= threshold].set_index("ts_code")
    filtered = buy_scores[buy_scores.index.isin(keep.index)]
    return filtered if not filtered.empty else buy_scores


def _choose_dynamic_k(day_scores: pd.Series, buy_scores: pd.Series, holdings: dict, base_k: int, strategy_cfg: dict) -> int:
    """动态换手数量（持仓质量驱动）

    核心思想：换手数不应该只看 base_k，应该跟随**当前持仓的"烂股数量"**动态调整
    - 持仓中有 N 只分数低于候选池中位数的股 → 至少换 N 只（这些都该被换掉）
    - 持仓质量都很高（无烂股）且分数差距小 → 减少换手
    - 候选质量特别好（分数远高于持仓）→ 多换手

    逻辑（按优先级）：
    1. 计算烂股数：held_score < candidate_median 的持仓股数量
    2. 计算好候选数：candidate_score > held_mean 的候选股数量
    3. k = max(base_k, 烂股数 + bad_buffer)
    4. 若好候选>=bad_min 且分数差距 > high_threshold → k 加 dynamic_k_step
    5. 若烂股=0 且分数差距 < low_threshold → k 减 1
    6. 若开启 score_gap_trigger 且没有烂股、分差不够，返回 0（不做噪音换手）
    7. 限制上限：k <= max(烂股数, 好候选数)（cfg 中 dynamic_k_max 可覆盖）
    """
    if not strategy_cfg.get("dynamic_k", False) or not holdings:
        return int(base_k)

    if strategy_cfg.get("dynamic_k_use_rank_scores", True):
        decision_scores = day_scores.rank(pct=True)
        decision_buy_scores = decision_scores.reindex(buy_scores.index).dropna()
    else:
        decision_scores = day_scores
        decision_buy_scores = buy_scores

    held = [c for c in holdings if c in decision_scores.index]
    candidates = decision_buy_scores.drop(index=[c for c in holdings if c in decision_buy_scores.index], errors="ignore")
    if not held or candidates.empty:
        return int(base_k)

    held_scores = decision_scores.loc[held]
    held_mean = float(held_scores.mean())
    held_min = float(held_scores.min())
    candidate_max = float(candidates.max())
    candidate_median = float(candidates.median())

    # === 持仓质量驱动：烂股数 ===
    # 持仓中分数 < 候选中位数的股 = "理论上该被换掉"的股
    bad_stocks_count = int((held_scores < candidate_median).sum())
    bad_min = int(strategy_cfg.get("dynamic_k_bad_min", 1))  # 烂股阈值（>=1 触发）
    bad_buffer = int(strategy_cfg.get("dynamic_k_bad_buffer", 0))  # 烂股基础上额外加几个

    # 起步：base_k
    k = int(base_k)

    # 1) 烂股数 >= bad_min → 必须换掉，k 至少 = 烂股数 + buffer
    if bad_stocks_count >= bad_min:
        k = max(k, bad_stocks_count + bad_buffer)

    # 2) 候选质量好 → 候选比持仓均分高很多 → 多换手
    high = float(strategy_cfg.get("score_gap_high", 0.10))
    low = float(strategy_cfg.get("score_gap_low", 0.02))
    gap = candidate_max - held_min
    good_candidates_count = int((candidates > held_mean).sum())

    if strategy_cfg.get("score_gap_trigger", False) and bad_stocks_count == 0:
        trigger_threshold = float(strategy_cfg.get("score_gap_trigger_threshold", low))
        if gap <= trigger_threshold:
            return 0

    if good_candidates_count >= bad_min and gap > high:
        k = k + int(strategy_cfg.get("dynamic_k_step", 2))

    # 3) 烂股 = 0 且分数差距小 → 减少换手（持仓已经很好了）
    if bad_stocks_count == 0 and gap < low:
        k = max(1, k - int(strategy_cfg.get("dynamic_k_step", 1)))

    # 4) 限制上限：理论换手上限 = max(烂股数, 好候选数)
    # 持仓 20-50 只，烂持仓时烂股数可达 50；好候选数可达 50
    # 用户可在 cfg 里设更严的上限
    max_k_default = max(bad_stocks_count, good_candidates_count)
    max_k_cfg = strategy_cfg.get("dynamic_k_max", None)
    max_k = int(max_k_cfg) if max_k_cfg is not None else max_k_default
    k = min(k, max_k)

    # 5) 有明确换手信号时至少为 1；score_gap_trigger 可在上面返回 0
    return max(1, k)


def _score_confidence(buy_scores: pd.Series, n_long: int, strategy_cfg: dict) -> float:
    """计算模型置信度"""
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
    """选择目标仓位（与 engine.py 一致）"""
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


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class HoldingStock:
    ts_code: str
    shares: int
    price: float = 0.0


@dataclass
class TargetStock:
    ts_code: str
    rank: int
    score: float
    weight: float
    target_shares: int
    target_value: float
    current_shares: int
    action: str  # buy, sell, hold
    action_shares: int
    buy_price: float
    sell_price: float
    liquidity_score: float
    risk_flag: str = ""


# ============================================================================
# 持仓解析
# ============================================================================

def parse_holdings(holdings_str: str) -> list[HoldingStock]:
    """解析持仓字符串

    输入格式: 代码:手数,代码:手数  （1手=100股）
    内部存储: HoldingStock.shares 统一为"股数"（× 100 转换）
    这样与 target_shares（= 100的整数倍 = 股数）保持单位一致
    """
    holdings = []
    if not holdings_str.strip():
        return holdings
    for item in holdings_str.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        code, shares_str = item.split(":")
        # 用户输入"手数" → 内部统一存"股数"
        shares = int(shares_str) * 100
        holdings.append(HoldingStock(ts_code=code.strip(), shares=shares))
    return holdings


# ============================================================================
# 分配策略
# ============================================================================

def calculate_weights(scores: pd.Series, n_hold: int, strategy: str = "score_weighted") -> pd.Series:
    top_n = scores.nlargest(n_hold)
    if strategy == "equal":
        return pd.Series(1.0 / len(top_n), index=top_n.index)
    elif strategy == "score_weighted":
        total = top_n.sum()
        if total == 0:
            return pd.Series(1.0 / len(top_n), index=top_n.index)
        weights = top_n / total
        return weights / weights.sum()
    elif strategy == "rank_weighted":
        n = len(top_n)
        return pd.Series({code: 2 * (n - i) / (n * (n + 1)) for i, code in enumerate(top_n.index)})
    else:
        powered = top_n ** 2
        return (powered / powered.sum())


def calculate_liquidity_score(panel: pd.DataFrame, ts_code: str, window: int = 20) -> float:
    try:
        data = panel[panel["ts_code"] == ts_code].sort_values("trade_date").tail(window)
        if len(data) < 5:
            return 0.5
        avg_amount = data["amount"].mean()
        std_amount = data["amount"].std()
        cv = std_amount / (avg_amount + 1)
        amount_score = min(1.0, avg_amount / 1e8)
        stability_score = max(0, 1 - cv)
        return (amount_score * 0.6 + stability_score * 0.4)
    except:
        return 0.5


# ============================================================================
# 核心函数
# ============================================================================

def score_latest(panel: pd.DataFrame, feat_cols: list, seq_len: int, ckpt: dict, device) -> pd.DataFrame:
    """对最新交易日数据进行预测打分"""
    import torch

    model, flatten = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    last_date = panel["trade_date"].max()
    rows = []
    for code, g in panel.groupby("ts_code"):
        g = g.sort_values("trade_date")
        if len(g) < seq_len or g["trade_date"].iloc[-1] != last_date:
            continue
        window = g[feat_cols].astype(np.float32).values[-seq_len:]
        if np.isnan(window).any():
            continue
        mu = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-6
        window = (window - mu) / std
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x_np = window.reshape(-1) if flatten else window
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        with torch.no_grad():
            score = float(model(x).cpu().item())
        rows.append({"trade_date": last_date, "ts_code": code, "score": score})

    return pd.DataFrame(rows)


def score_latest_lgbm(panel: pd.DataFrame, out: Path) -> pd.DataFrame:
    """对最新交易日数据用 LightGBM LambdaRank 打分。

    与 06_infer_orders.py 同名函数一致：仅当 lgbm_status.json 不存在或 status=='ok'
    且 lgbm_model.txt + lgbm_meta.json 都存在时才推理；任何缺失/异常返回空 DataFrame，
    blend 阶段自动降级为纯 DL。
    """
    model_path = out / "lgbm_model.txt"
    meta_path = out / "lgbm_meta.json"
    status_path = out / "lgbm_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status", "ok") != "ok":
            return pd.DataFrame()
    if not model_path.exists() or not meta_path.exists():
        return pd.DataFrame()
    try:
        import lightgbm as lgb
    except ImportError:
        return pd.DataFrame()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feat_cols = meta.get("feat_cols", [])
    if not feat_cols:
        return pd.DataFrame()
    last_date = panel["trade_date"].max()
    latest = panel[panel["trade_date"] == last_date].copy()
    if latest.empty:
        return pd.DataFrame()
    for col in feat_cols:
        if col not in latest.columns:
            latest[col] = 0.0
    latest[feat_cols] = latest[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    booster = lgb.Booster(model_file=str(model_path))
    latest["score_lgbm"] = booster.predict(latest[feat_cols], num_iteration=booster.best_iteration)
    return latest[["trade_date", "ts_code", "score_lgbm"]]


def blend_latest_scores(deep_scores: pd.DataFrame, lgbm_scores: pd.DataFrame, out: Path, cfg: dict) -> pd.DataFrame:
    """把 DL 分数与 LGBM 分数按 rank 加权融合。

    与 06_infer_orders.py 同名函数一致：alpha 优先从 blend_meta.json.best_alpha 读，
    其次回退到 cfg.ensemble.alpha；lgbm 禁用或缺失时直接返回纯 DL 分数（保持列结构）。
    """
    lgbm_enabled = cfg.get("lgbm", {}).get("enabled", True)

    # 纯 DL 模式：直接使用深度学习分数
    if lgbm_scores.empty or not lgbm_enabled:
        if deep_scores.empty:
            return pd.DataFrame()
        result = deep_scores.rename(columns={"score": "score_deep"})
        result["score"] = result["score_deep"]
        result["rank_deep"] = result.groupby("trade_date")["score_deep"].rank(pct=True)
        result["rank_lgbm"] = result["rank_deep"]  # 占位符
        result["score_lgbm"] = result["score_deep"]  # 占位符
        return result[["trade_date", "ts_code", "score", "score_deep", "score_lgbm", "rank_deep", "rank_lgbm"]]

    merged = deep_scores.rename(columns={"score": "score_deep"}).merge(
        lgbm_scores,
        on=["trade_date", "ts_code"],
        how="inner",
    )
    if merged.empty:
        return deep_scores
    meta_path = out / "blend_meta.json"
    if meta_path.exists():
        alpha = float(json.loads(meta_path.read_text(encoding="utf-8")).get("best_alpha", 0.6))
    else:
        alpha = float(cfg.get("ensemble", {}).get("alpha", 0.6))
    merged["rank_deep"] = merged.groupby("trade_date")["score_deep"].rank(pct=True)
    merged["rank_lgbm"] = merged.groupby("trade_date")["score_lgbm"].rank(pct=True)
    merged["score"] = alpha * merged["rank_deep"] + (1.0 - alpha) * merged["rank_lgbm"]
    return merged[["trade_date", "ts_code", "score", "score_deep", "score_lgbm", "rank_deep", "rank_lgbm"]]


def get_prices(
    panel: pd.DataFrame,
    codes: list,
    sell_slack: float = 0.002,
    buy_slack: float = 0.001,
    limit_pct: float = 0.10,
) -> dict:
    """获取价格信息

    改进点（实盘同花顺成交优化）：
    - 卖价 = prev_close * (1 - sell_slack)：让卖单快速成交（满仓时先卖后买场景）
    - 买价 = prev_close * (1 + buy_slack)：让买单优先成交但不过度溢价
    - 限制在涨跌停范围内（主板 ±limit_pct=10%）
    - sell_slack=0.002 意味着卖价低于市价 0.2%，买单优先匹配
    - buy_slack=0.001 意味着买价高于市价 0.1%，快速吃单

    Args:
        panel: 价格数据面板
        codes: 股票代码列表
        sell_slack: 卖出价相对收盘价的折让（默认 0.002 = 0.2%）
        buy_slack: 买入价相对收盘价的溢价（默认 0.001 = 0.1%）
        limit_pct: 涨跌停限制（主板 10%，创业板/科创板 20%）
    """
    last_date = panel["trade_date"].max()

    prices = {}
    for code in codes:
        # 直接用最新收盘价
        curr_row = panel[(panel["ts_code"] == code) & (panel["trade_date"].astype(str) == str(last_date))]
        if not curr_row.empty:
            prev_close = float(curr_row["close"].iloc[0])
        else:
            # 如果没有当天数据，尝试前一天
            prev_date = (pd.to_datetime(last_date, format="%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
            prev_row = panel[(panel["ts_code"] == code) & (panel["trade_date"].astype(str) == str(prev_date))]
            prev_close = float(prev_row["close"].iloc[0]) if not prev_row.empty else 0

        if prev_close > 0:
            # 计算带 slack 的建议价
            raw_sell = prev_close * (1 - sell_slack)
            raw_buy = prev_close * (1 + buy_slack)
            # 涨跌停限制（买单不能高于涨停价，卖单不能低于跌停价）
            upper_limit = round(prev_close * (1 + limit_pct), 2)
            lower_limit = round(prev_close * (1 - limit_pct), 2)
            sell_price = max(round(raw_sell, 2), lower_limit)
            buy_price = min(round(raw_buy, 2), upper_limit)
        else:
            sell_price, buy_price = 0.0, 0.0

        prices[code] = {
            "prev_close": prev_close,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "sell_slack_pct": sell_slack,
            "buy_slack_pct": buy_slack,
        }
    return prices


def infer_next_trading_date(panel: pd.DataFrame, last_date: str) -> str:
    """推断信号日后的下一交易日，优先使用数据日历，缺失时跳过周末。"""
    date_str = str(last_date)
    future_dates = sorted(
        d for d in panel["trade_date"].astype(str).unique()
        if d > date_str
    )
    if future_dates:
        return future_dates[0]

    dt = pd.to_datetime(date_str, format="%Y%m%d") + timedelta(days=1)
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.strftime("%Y%m%d")


def generate_trading_guide(
    scores: pd.DataFrame,
    panel: pd.DataFrame,
    current_holdings: list[HoldingStock],
    strategy_cfg: dict,
    n_hold: int,
    k_trade: int,
    target_position: float,
    portfolio_value: float,
    allocation_strategy: str,
    scheme_name: str,
    output_dir: Path,
    vol_pct: float = None,
    position_confidence: float = 0.5,
) -> tuple[str, pd.DataFrame, Path]:
    """生成完整的交易指南"""

    last_date = scores["trade_date"].max()
    next_trade_date = infer_next_trading_date(panel, str(last_date))
    trading_date = pd.to_datetime(next_trade_date, format="%Y%m%d").strftime("%Y-%m-%d")

    # 可买股票筛选
    if "buyable" in scores.columns:
        buyable = scores[scores["buyable"].astype(bool)].copy()
    else:
        buyable = scores.copy()

    # 过滤股票类型：排除 301(创业板)、688(科创板)、8开头(北交所)、003/001(主板需权限)
    import re
    exclude_pattern = re.compile(r'^(301|688|8|003|001)\d{3,}')
    buyable = buyable[~buyable['ts_code'].str.replace('.SZ|.SH', '', regex=False).str.match(exclude_pattern)]
    print(f"✓ 排除创业板/科创板/北交所股票后: {len(buyable)}只")

    # Momentum 过滤（与 engine.py 一致）
    day = scores[scores["trade_date"] == last_date].copy()
    buyable = _filter_momentum(day, buyable.set_index("ts_code")["score"], strategy_cfg)
    if hasattr(buyable, 'index'):
        buyable = buyable.reset_index()
    else:
        buyable = scores[scores["ts_code"].isin(buyable.index)].copy()

    # 再次过滤股票类型：保持与第一次过滤完全一致（避免 003/001 主板权限股在 momentum 过滤后回流）
    import re
    exclude_pattern = re.compile(r'^(301|688|8|003|001)\d{3,}')
    buyable = buyable[~buyable['ts_code'].str.replace('.SZ|.SH', '', regex=False).str.match(exclude_pattern)]
    print(f"✓ 排除创业板/科创板/北交所股票后: {len(buyable)}只")

    buyable = buyable.sort_values("score", ascending=False)

    # 计算权重
    weights = calculate_weights(buyable.set_index("ts_code")["score"], n_hold, allocation_strategy)

    # 目标持仓
    target_codes = buyable.head(n_hold)["ts_code"].tolist()
    available_value = portfolio_value * target_position

    # 当前持仓映射
    current_map = {h.ts_code: h for h in current_holdings}

    # 获取价格（带 slack，确保同花顺快速成交）
    all_codes = list(set(target_codes + [h.ts_code for h in current_holdings]))
    sell_slack = float(strategy_cfg.get("sell_slack", 0.002))
    buy_slack = float(strategy_cfg.get("buy_slack", 0.001))
    limit_pct = float(strategy_cfg.get("limit_pct", 0.10))
    prices = get_prices(panel, all_codes, sell_slack=sell_slack, buy_slack=buy_slack, limit_pct=limit_pct)
    # 补仓候选池：扩大价格表，提前获取 top-41 之后的候选价格（贪心/补仓用）
    _topup_prefetch = int(strategy_cfg.get("topup_prefetch_codes", 80))
    if _topup_prefetch > 0 and len(buyable) > n_hold:
        prefetch_codes = buyable.iloc[n_hold : n_hold + _topup_prefetch]["ts_code"].tolist()
        prefetch_codes = [c for c in prefetch_codes if c not in prices]
        if prefetch_codes:
            prices.update(get_prices(panel, prefetch_codes, sell_slack=sell_slack, buy_slack=buy_slack, limit_pct=limit_pct))

    # 构建目标持仓
    targets = []
    for rank, code in enumerate(target_codes, 1):
        score = float(buyable[buyable["ts_code"] == code]["score"].iloc[0])
        weight = float(weights.get(code, 0))
        target_value = available_value * weight
        price = prices.get(code, {}).get("prev_close", 0)
        target_shares = int(target_value / price // 100 * 100) if price > 0 else 0

        current = current_map.get(code)
        current_shares = current.shares if current else 0

        if current_shares == 0:
            action, action_shares = "buy", target_shares
        elif target_shares > current_shares:
            action, action_shares = "buy", target_shares - current_shares
        elif target_shares < current_shares:
            action, action_shares = "sell", current_shares - target_shares
        else:
            action, action_shares = "hold", 0

        # 流动性
        liq_score = calculate_liquidity_score(panel, code)

        # 风险提示
        risk = ""
        if price > 0 and target_value / price > 500000:
            risk = "⚠️ 大单，分2批"
        if liq_score < 0.3:
            risk = "⚠️ 流动性差"
        if price > 0 and target_shares > 0:
            avg_amt = panel[panel["ts_code"] == code]["amount"].mean()
            participation = (target_shares * price) / (avg_amt * 10000 + 1) if avg_amt > 0 else 0
            if participation > 0.1:
                risk = "⚠️ 参与率高，分2批"

        targets.append(TargetStock(
            ts_code=code,
            rank=rank,
            score=score,
            weight=weight,
            target_shares=target_shares,
            target_value=target_value,
            current_shares=current_shares,
            action=action,
            action_shares=action_shares,
            buy_price=prices.get(code, {}).get("buy_price", 0),
            sell_price=prices.get(code, {}).get("sell_price", 0),
            liquidity_score=liq_score,
            risk_flag=risk,
        ))

    # 备选股票
    backup_codes = buyable.iloc[n_hold:n_hold * 3]["ts_code"].tolist()
    backups = []
    for rank, code in enumerate(backup_codes, n_hold + 1):
        score = float(buyable[buyable["ts_code"] == code]["score"].iloc[0])
        liq_score = calculate_liquidity_score(panel, code)
        backups.append({
            "rank": rank,
            "ts_code": code,
            "score": score,
            "buy_price": prices.get(code, {}).get("buy_price", 0),
            "liquidity": "优" if liq_score > 0.6 else ("中" if liq_score > 0.3 else "差"),
            "替代": f"当{targets[rank - n_hold - 1].ts_code if rank - n_hold - 1 < len(targets) else '目标股'}买不到时使用",
        })

    # 分离买卖
    to_buy = [t for t in targets if t.action == "buy"]
    to_sell = [t for t in targets if t.action == "sell"]
    to_hold = [t for t in targets if t.action == "hold"]

    # 按权重排序
    to_buy.sort(key=lambda x: -x.weight)
    to_sell.sort(key=lambda x: x.weight)

    # === 候选外持仓清理（不受 dynamic_k 限制）===
    # 模式（由 sell_outsiders_mode 控制）：
    #   "all"  (默认，激进): 卖所有 不在 top n_hold 候选 的持仓（不论分数）
    #   "bad"  (保守):      只卖 持仓分 < 候选中位 且 不在 top n_hold 候选 的股
    # 目的：让现金回流给 top n_hold 候选股（解决"满仓无现金"问题）
    # 数据依据（方案 A）：IC=0.105, ICIR=0.945，模型预测力强 → 激进换到 top 41 收益高
    outsider_sells: list[TargetStock] = []
    sell_mode = strategy_cfg.get("sell_outsiders_mode", "all")
    if strategy_cfg.get("sell_outsiders", False) and current_holdings and sell_mode in ("all", "bad"):
        held_codes_all = [h.ts_code for h in current_holdings]
        _ds = buyable.set_index("ts_code")["score"]
        _target_set = set(target_codes)
        _candidates_only = _ds.drop(index=[c for c in held_codes_all if c in _ds.index], errors="ignore")
        _cand_median = float(_candidates_only.median()) if not _candidates_only.empty else 0.0

        # 候选外的所有持仓（不在 top n_hold 候选）
        _outsider_codes = set(held_codes_all) - _target_set
        # "bad" 模式：进一步筛 持仓分<候选中位
        if sell_mode == "bad":
            _bad_codes = set(_ds[_ds < _cand_median].index)
            _outsider_codes = _outsider_codes & _bad_codes

        for code in _outsider_codes:
            holding = next((h for h in current_holdings if h.ts_code == code), None)
            if holding is None or code not in prices:
                continue
            if any(t.ts_code == code for t in to_sell):
                continue  # 已在 to_sell 中（候选内 sell）
            _liq = calculate_liquidity_score(panel, code)
            _sell_px = prices.get(code, {}).get("sell_price", 0)
            held_score = float(_ds.get(code, 0.0))
            is_bad = bool(np.isfinite(_cand_median) and held_score < _cand_median)
            _risk = "🔄 烂股清理" if is_bad else "🔄 候选外清理"
            if _liq < 0.3:
                _risk += " ⚠️流动性差"
            outsider_sells.append(TargetStock(
                ts_code=code,
                rank=0,
                score=held_score,
                weight=0.0,
                target_shares=0,
                target_value=0.0,
                current_shares=holding.shares,
                action="sell",
                action_shares=holding.shares,
                buy_price=0.0,
                sell_price=_sell_px,
                liquidity_score=_liq,
                risk_flag=_risk,
            ))
        # 按当前持仓市值升序（小单优先卖，快回笼资金）
        outsider_sells.sort(key=lambda t: t.current_shares * t.sell_price)
        if strategy_cfg.get("debug_dynamic_k", False):
            mode_label = "全部候选外" if sell_mode == "all" else "烂股(score<候选中位)"
            print(f"  [debug] 候选外清理: {len(outsider_sells)} 只 (模式={mode_label}, 候选中位={_cand_median:.3f})")

    # 动态 k_trade（与 engine.py 一致）
    day_scores = buyable.set_index("ts_code")["score"]
    buy_scores = day_scores
    holdings_dict = {h.ts_code: h.shares for h in current_holdings}

    # 空仓建仓时，不受k_trade限制，一次性买够
    is_empty_portfolio = (len(current_holdings) == 0)
    if is_empty_portfolio:
        actual_k = len(to_buy)  # 一次性买完所有目标股票
    else:
        # 调试：打印 dynamic_k 决策依据
        if strategy_cfg.get("debug_dynamic_k", False):
            held_codes = [c for c in holdings_dict if c in day_scores.index]
            held_scores_dbg = day_scores.loc[held_codes] if held_codes else pd.Series(dtype=float)
            cand_dbg = buy_scores.drop(index=[c for c in holdings_dict if c in buy_scores.index], errors="ignore")
            if not held_scores_dbg.empty and not cand_dbg.empty:
                held_mean = float(held_scores_dbg.mean())
                held_min = float(held_scores_dbg.min())
                cand_med = float(cand_dbg.median())
                bad_n = int((held_scores_dbg < cand_med).sum())
                good_n = int((cand_dbg > held_mean).sum())
                gap = float(cand_dbg.max() - held_min)
                print(f"  [debug] 持仓 {len(held_codes)} 只, 均分 {held_mean:.3f}, 最低 {held_min:.3f}")
                print(f"  [debug] 候选 {len(cand_dbg)} 只, 中位 {cand_med:.3f}, 最高 {cand_dbg.max():.3f}")
                print(f"  [debug] 烂股数 (持仓分<候选中位) = {bad_n}, 好候选数 (候选>持仓均分) = {good_n}, gap = {gap:.3f}")
        actual_k = _choose_dynamic_k(day_scores, buy_scores, holdings_dict, k_trade, strategy_cfg)
        if strategy_cfg.get("debug_dynamic_k", False):
            print(f"  [debug] 决策 k = {actual_k} (base_k={k_trade})")

    actual_buy = to_buy[:actual_k]
    rebalance_sells = to_sell[:actual_k]
    if strategy_cfg.get("sell_outsiders_unlimited", False):
        limited_outsider_sells = outsider_sells
    else:
        remaining_sell_capacity = max(0, actual_k - len(rebalance_sells))
        max_outsider_sells_cfg = strategy_cfg.get("sell_outsiders_max", remaining_sell_capacity)
        max_outsider_sells = max(0, int(max_outsider_sells_cfg))
        limited_outsider_sells = outsider_sells[:max_outsider_sells]
    # 候选内卖出与候选外清理共同受 dynamic_k/显式上限约束，避免“清理”绕开换手控制
    actual_sell = rebalance_sells + limited_outsider_sells

    # 汇总卖/买金额（用于"先卖后买"资金时序）
    # 注：shares 字段语义统一为"股数"（parse_holdings 已 × 100）
    sell_total_value = sum(t.action_shares * t.sell_price for t in actual_sell)
    buy_total_value = sum(t.action_shares * t.buy_price for t in actual_buy)
    sell_total_shares = sum(t.action_shares for t in actual_sell)  # 股
    buy_total_shares = sum(t.action_shares for t in actual_buy)  # 股
    # 真实可用现金 = portfolio_value - 当前持仓总市值（满仓用户 ≈ 0）
    holdings_market_value = sum(
        h.shares * prices.get(h.ts_code, {}).get("prev_close", 0)
        for h in current_holdings
    )
    actual_cash = max(0.0, portfolio_value - holdings_market_value)
    cash_after_sell = actual_cash + sell_total_value
    cash_gap = cash_after_sell - buy_total_value  # 正=充足，负=需补现金/减买

    # 现金约束：实盘手动下单默认不补现金。若卖出后现金不足，按买入优先级从低到高删单。
    if strategy_cfg.get("enforce_cash_limit", True) and cash_gap < 0 and actual_buy:
        removed_for_cash: list[str] = []
        while actual_buy and cash_gap < 0:
            removed = actual_buy.pop()
            removed_amount = removed.action_shares * removed.buy_price
            buy_total_value -= removed_amount
            buy_total_shares -= removed.action_shares
            cash_gap += removed_amount
            removed_for_cash.append(removed.ts_code)
        if strategy_cfg.get("debug_dynamic_k", False) and removed_for_cash:
            print(
                f"  [debug-cash] 现金不足，删去 {len(removed_for_cash)} 个低优先级买单: "
                f"{removed_for_cash[:5]}{'…' if len(removed_for_cash) > 5 else ''}, "
                f"资金缺口: {cash_gap:+,.0f} 元"
            )

    # ============================================================================
    # 贪心放大买入：用尽卖出的现金（解决 "卖 30 只却只买 8 只" 的资金闲置问题）
    # ----------------------------------------------------------------------------
    # 触发条件：sell_outsiders 会把候选外持仓全卖 → 卖出会远超 actual_k
    # 若只用 actual_k 限制买入，会出现现金大量闲置、买入不足 → 收益浪费
    # 解决：贪心按 weight 顺序补买入候选，直到卖出资金用完（或到 n_hold 上限）
    # 关键：to_buy 已按 -weight 排序 → 下一只必然是剩余最优的，不打乱原优先级
    # ============================================================================
    amplify_buys = bool(strategy_cfg.get("amplify_buys", True))
    if amplify_buys and cash_gap > 0 and actual_k < len(to_buy):
        extra_added: list[str] = []
        for cand in to_buy[actual_k:]:
            cand_amount = cand.action_shares * cand.buy_price
            if cand_amount <= 0:
                continue
            if cand_amount > cash_gap:
                break  # 下一只的金额已超过剩余资金 → 停止
            actual_buy.append(cand)
            buy_total_value += cand_amount
            buy_total_shares += cand.action_shares
            cash_gap -= cand_amount
            extra_added.append(cand.ts_code)
            if cash_gap <= 0 or len(actual_buy) >= n_hold:
                break
        if strategy_cfg.get("debug_dynamic_k", False) and extra_added:
            print(f"  [debug] 贪心放大新增买入: {len(extra_added)} 只 (新增: {extra_added[:5]}{'…' if len(extra_added) > 5 else ''}), 资金缺口: {cash_gap:+,.0f} 元")
        # 同步给 to_buy 用于后续展示（不影响交易）
        # （实际清单以 actual_buy 为准）

    # ============================================================================
    # 加仓（hand-expand）：用剩余 cash_gap 扩大已买入股票的每只手数
    # ----------------------------------------------------------------------------
    # 场景：贪心放大已用完 to_buy 所有可买候选，但 cash_gap 仍有 200K+
    # 用户要求：扩大每个股票的手数（不要让策略变差 = 多买已选中的优质股）
    # 解决：按 weight 等比扩大 actual_buy 中各只股票的手数（最少 1 手 / 100 股）
    # ============================================================================
    hand_expand_enabled = bool(strategy_cfg.get("hand_expand_enabled", True))
    if hand_expand_enabled and cash_gap > 0 and actual_buy:
        expand_threshold = float(strategy_cfg.get("hand_expand_min_gap_pct", 0.01)) * portfolio_value
        # 按 weight 比例分摊剩余资金
        expand_pool = [t for t in actual_buy if t.action_shares > 0 and t.buy_price > 0]
        if cash_gap > expand_threshold and expand_pool:
            sum_w = sum(t.weight for t in expand_pool)
            if sum_w > 0:
                if strategy_cfg.get("debug_dynamic_k", False):
                    print(f"  [debug-expand] 进入加仓块: cash_gap={cash_gap:,.0f}, 阈值={expand_threshold:,.0f}, 候选={len(expand_pool)}")
                expanded_count = 0
                for t in expand_pool:
                    if cash_gap <= 0:
                        break
                    extra_value = (t.weight / sum_w) * cash_gap
                    extra_shares = int(extra_value / t.buy_price // 100) * 100
                    if extra_shares <= 0:
                        continue
                    # 不超过 cash_gap
                    if extra_shares * t.buy_price > cash_gap:
                        extra_shares = int(cash_gap / t.buy_price // 100) * 100
                        if extra_shares <= 0:
                            continue
                    t.action_shares += extra_shares
                    t.target_shares += extra_shares
                    t.target_value += extra_shares * t.buy_price
                    extra_amount = extra_shares * t.buy_price
                    buy_total_value += extra_amount
                    buy_total_shares += extra_shares
                    cash_gap -= extra_amount
                    expanded_count += 1
                if strategy_cfg.get("debug_dynamic_k", False):
                    print(f"  [debug-expand] 加仓完成: 扩大 {expanded_count} 只, 剩余资金: {cash_gap:+,.0f} 元")

    # ============================================================================
    # 补仓（top-up）：用尽 target_position 应有的仓位
    # ----------------------------------------------------------------------------
    # 场景：top-41 候选的 target_value 总和 < target_position × portfolio 时
    #       （如 30 只 outsider 释放 700K 但 top-41 只够填 678K）
    # 解决：从候选池第 n_hold+1 名起，按分数比例分摊剩余资金
    # 上限：n_hold + topup_max_extra（避免过度分散）
    # 阈值：cash_gap > topup_min_gap_pct × portfolio 才补（小额不补，避免频繁补仓）
    # ============================================================================
    topup_enabled = bool(strategy_cfg.get("topup_enabled", True))
    if topup_enabled and cash_gap > 0 and len(buyable) > n_hold:
        topup_min_gap_pct = float(strategy_cfg.get("topup_min_gap_pct", 0.02))
        topup_max_extra = int(strategy_cfg.get("topup_max_extra", 15))
        topup_min_liquidity = float(strategy_cfg.get("topup_min_liquidity", 0.3))
        topup_min_score_pct = float(strategy_cfg.get("topup_min_score_pct", 0.5))

        if strategy_cfg.get("debug_dynamic_k", False):
            print(f"  [debug-topup] 进入补仓块: cash_gap={cash_gap:,.0f}, 阈值={topup_min_gap_pct * portfolio_value:,.0f}, buyable_size={len(buyable)}")

        if cash_gap > topup_min_gap_pct * portfolio_value and topup_max_extra > 0:
            # 候选池第 n_hold+1 名起
            filler_pool = buyable.iloc[n_hold:].copy()
            # 分数下限：top-41 中位 × 比例（避免太弱候选拉低收益）
            if not filler_pool.empty and len(filler_pool) > 0:
                top41_median_score = float(buyable.iloc[: min(n_hold, len(buyable))]["score"].median())
                top41_min_score = float(buyable.iloc[: min(n_hold, len(buyable))]["score"].min())
                score_floor = top41_median_score * topup_min_score_pct
                filler_pool = filler_pool[filler_pool["score"] >= score_floor]
                if strategy_cfg.get("debug_dynamic_k", False):
                    print(f"  [debug-topup] top41_median={top41_median_score:.3f}, top41_min={top41_min_score:.3f}, score_floor={score_floor:.3f}")
                    print(f"  [debug-topup] 补仓候选池(原 {len(buyable.iloc[n_hold:])} 只) → 分数过滤后 {len(filler_pool)} 只")
                    if len(filler_pool) > 0:
                        print(f"  [debug-topup] 补仓候选分数范围: {filler_pool['score'].min():.3f} ~ {filler_pool['score'].max():.3f}")
                    # 打印分数分位点
                    raw = buyable.iloc[n_hold:]["score"]
                    if len(raw) > 0:
                        print(f"  [debug-topup] rank42+ 原始分数分位: 0%={raw.min():.3f}, 25%={raw.quantile(0.25):.3f}, 50%={raw.median():.3f}, 75%={raw.quantile(0.75):.3f}, 90%={raw.quantile(0.9):.3f}, 99%={raw.quantile(0.99):.3f}")

            existing_codes = {t.ts_code for t in actual_buy} | {t.ts_code for t in actual_sell} | {h.ts_code for h in current_holdings}
            valid_fillers: list[tuple[str, float, float, float]] = []  # (code, score, buy_price, liq)
            liq_pass = liq_fail = existing_excluded = px_fail = 0
            for _, row in filler_pool.iterrows():
                code = row["ts_code"]
                if code in existing_codes:
                    existing_excluded += 1
                    continue
                px_info = prices.get(code, {})
                px = px_info.get("buy_price", 0)
                if px <= 0:
                    px_fail += 1
                    continue
                liq = calculate_liquidity_score(panel, code)
                if liq < topup_min_liquidity:
                    liq_fail += 1
                    continue
                liq_pass += 1
                valid_fillers.append((code, float(row["score"]), px, liq))
                if len(valid_fillers) >= topup_max_extra:
                    break
            if strategy_cfg.get("debug_dynamic_k", False):
                print(f"  [debug-topup] 过滤统计: 现有持仓排除={existing_excluded}, 价格失败={px_fail}, 流动性失败={liq_fail}, 合格={liq_pass}")

            if valid_fillers:
                sum_filler_scores = sum(s for _, s, _, _ in valid_fillers)
                topup_added: list[str] = []
                if sum_filler_scores > 0:
                    for code, score, buy_px, liq in valid_fillers:
                        if cash_gap <= buy_px * 100:  # 剩余资金不足买 1 手
                            break
                        target_value = (score / sum_filler_scores) * cash_gap
                        target_shares = int(target_value / buy_px // 100) * 100
                        if target_shares <= 0:
                            target_shares = 100  # 至少 1 手
                        actual_value = target_shares * buy_px
                        if actual_value > cash_gap:
                            target_shares = int(cash_gap / buy_px // 100) * 100
                            actual_value = target_shares * buy_px
                            if target_shares <= 0:
                                break
                        actual_buy.append(TargetStock(
                            ts_code=code,
                            rank=n_hold + len(topup_added) + 1,
                            score=score,
                            weight=0.0,
                            target_shares=target_shares,
                            target_value=actual_value,
                            current_shares=0,
                            action="buy",
                            action_shares=target_shares,
                            buy_price=buy_px,
                            sell_price=px_info.get("sell_price", buy_px),
                            liquidity_score=liq,
                            risk_flag="🆕 补仓（非 top-41 候选）",
                        ))
                        topup_added.append(code)
                        buy_total_value += actual_value
                        buy_total_shares += target_shares
                        cash_gap -= actual_value
                        if cash_gap <= 0 or len(actual_buy) >= n_hold + topup_max_extra:
                            break
                if strategy_cfg.get("debug_dynamic_k", False) and topup_added:
                    print(f"  [debug] 补仓新增: {len(topup_added)} 只 (代码: {topup_added[:5]}{'…' if len(topup_added) > 5 else ''}), 资金缺口: {cash_gap:+,.0f} 元")

    # 构建文本
    guide = f"""
================================================================================
                     同花顺手动下单交易指南
================================================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%M')}
数据日期: {last_date}
交易日期: {trading_date}
方案: {scheme_name}

--------------------------------------------------------------------------------
                           【策略参数】
--------------------------------------------------------------------------------
  持仓目标: {n_hold} 只
  目标仓位: {target_position:.0%}
  模型置信度: {position_confidence:.2%}
  市场波动分位: {(f"{vol_pct:.2%}") if vol_pct is not None else 'N/A'}
  账户总额: {portfolio_value:,.0f} 元
  可用资金: {available_value:,.0f} 元
  动态换手: {actual_k} 只 (基准: {k_trade}){'  | 贪心放大: +' + str(len([t for t in actual_buy if t.risk_flag == '']) - actual_k) + ' 只' if len([t for t in actual_buy if t.risk_flag == '']) > actual_k else ''}{'  | 补仓: +' + str(len([t for t in actual_buy if t.risk_flag == '🆕 补仓（非 top-41 候选）'])) + ' 只' if any(t.risk_flag == '🆕 补仓（非 top-41 候选）' for t in actual_buy) else ''}
  分配策略: {allocation_strategy}
  卖出折让: -{sell_slack*100:.2f}% | 买入溢价: +{buy_slack*100:.2f}% (确保同花顺快速成交)

================================================================================
                         【执行流程 - 先卖后买】
================================================================================
  1) 09:30-09:35 集合竞价结束后，按【卖出清单】挂卖单（推荐挂单价已含 0.2% 折让）
  2) 09:35-09:45 卖单成交后，等待资金即时到账
  3) 09:35-10:00 按【买入清单】挂买单（推荐挂单价已含 0.1% 溢价，优先吃单）
  4) 若目标股涨停无法买入：使用【备选股票替换表】中的备选股

  资金时序：
  - 卖出总额: {sell_total_value:,.0f} 元 ({sell_total_shares:,} 股)
  - 买入总额: {buy_total_value:,.0f} 元 ({buy_total_shares:,} 股)
  - 卖出后可用: {cash_after_sell:,.0f} 元
  - 资金缺口: {cash_gap:+,.0f} 元 ({'✅ 充足' if cash_gap >= 0 else '⚠️ 需补现金/减买'})

================================================================================
                         【卖出清单 - 优先执行】
================================================================================
操作顺序: 先卖出，后买入

"""

    if not actual_sell:
        guide += "  ✓ 无需卖出\n\n"
    else:
        guide += f"  {'优先级':<4} {'股票代码':<12} {'卖出手数':<8} {'收盘价':<8} {'推荐挂单价':<11} {'折让':<7} {'风险提示':<15}\n"
        guide += "  " + "-" * 80 + "\n"
        for i, t in enumerate(actual_sell, 1):
            prev_close = prices.get(t.ts_code, {}).get("prev_close", 0)
            slack_pct = prices.get(t.ts_code, {}).get("sell_slack_pct", 0.002)
            slack_label = f"-{slack_pct*100:.2f}%"
            # shares 内部存股数，输出给同花顺"手数" = 股数 // 100
            hands = t.action_shares // 100
            guide += f"  {i:<4} {t.ts_code:<12} {hands:<8} {prev_close:<8.2f} {t.sell_price:<11.2f} {slack_label:<7} {t.risk_flag:<15}\n"

    guide += f"""
================================================================================
                         【买入清单 - 后执行】
================================================================================
"""

    if not actual_buy:
        guide += "  ✓ 无需买入\n\n"
    else:
        guide += f"  {'优先级':<4} {'股票代码':<12} {'买入手数':<8} {'收盘价':<8} {'推荐挂单价':<11} {'溢价':<7} {'预估金额':<12} {'风险提示':<15}\n"
        guide += "  " + "-" * 100 + "\n"
        for i, t in enumerate(actual_buy, 1):
            est_amount = t.action_shares * t.buy_price  # 股数 × 元/股 = 元
            prev_close = prices.get(t.ts_code, {}).get("prev_close", 0)
            slack_pct = prices.get(t.ts_code, {}).get("buy_slack_pct", 0.001)
            slack_label = f"+{slack_pct*100:.2f}%"
            hands = t.action_shares // 100
            guide += f"  {i:<4} {t.ts_code:<12} {hands:<8} {prev_close:<8.2f} {t.buy_price:<11.2f} {slack_label:<7} {est_amount:<12,.0f} {t.risk_flag:<15}\n"

    # 备选股票池（包含完整价格）
    all_backup_prices = {}
    for code in set(backups[i]['ts_code'] for i in range(len(backups))):
        if code in prices:
            all_backup_prices[code] = prices[code]
        else:
            # 从panel获取备选股票价格
            code_data = panel[panel['ts_code'] == code].sort_values('trade_date')
            if not code_data.empty:
                prev_close = float(code_data['close'].iloc[-1])
                all_backup_prices[code] = {
                    'prev_close': prev_close,
                    'buy_price': round(prev_close * 1.002, 2),
                    'sell_price': round(prev_close * 0.998, 2),
                }

    # 构建备选替换映射
    backup_map = {}
    backup_idx = 0
    for t in targets:
        if t.action == 'buy' and backup_idx < len(backups):
            backup = backups[backup_idx]
            backup_code = backup['ts_code']
            backup_price = all_backup_prices.get(backup_code, {}).get('buy_price', 0)
            # 计算备选股票的手数（根据原股票的目标金额分配）
            if backup_price > 0:
                backup_shares = int(t.target_value / backup_price // 100 * 100)
            else:
                backup_shares = 0
            backup_map[t.ts_code] = {
                'original_code': t.ts_code,
                'original_price': t.buy_price,
                'original_shares': t.target_shares,
                'backup_code': backup_code,
                'backup_price': backup_price,
                'backup_shares': backup_shares,
            }
            backup_idx += 1

    # 备选替换表
    guide += f"""
================================================================================
                         【备选股票替换表】
================================================================================
说明：如果目标股票无法买入（涨停/无法交易），按下方表格替换

  {'原股票':<12} {'原价':<8} {'原手数':<8} │ {'备选股票':<12} {'备选价':<8} {'备选手数':<8}
  """ + "-" * 60 + "\n"

    for orig_code, info in backup_map.items():
        # shares 字段是股数，输出给同花顺"手数" = // 100
        orig_hands = info['original_shares'] // 100
        backup_hands = info['backup_shares'] // 100
        guide += f"  {info['original_code']:<12} {info['original_price']:<8.2f} {orig_hands:<8} │ {info['backup_code']:<12} {info['backup_price']:<8.2f} {backup_hands:<8}\n"

    # 备选股票池
    guide += f"""
================================================================================
                         【备选股票池完整信息】
================================================================================

  {'序号':<4} {'股票代码':<12} {'分数':<10} {'建议买入价':<10} {'可买手数':<10} {'流动性':<8}
  """ + "-" * 60 + "\n"

    for i, b in enumerate(backups[:20], 1):
        backup_code = b['ts_code']
        backup_price = all_backup_prices.get(backup_code, {}).get('buy_price', 0)
        # 计算可买手数
        if backup_price > 0:
            available_for_backup = int(portfolio_value * 0.02 / backup_price // 100 * 100) // 100  # 预留2%资金给每个备选，输出"手数"
        else:
            available_for_backup = 0
        guide += f"  {i:<4} {backup_code:<12} {b['score']:<10.4f} {backup_price:<10.2f} {available_for_backup:<10} {b['liquidity']:<8}\n"

    # 操作格式
    guide += f"""
================================================================================
                         【同花顺操作格式】
================================================================================
【卖出操作】（先执行）
  证券代码            卖出价格        卖出数量
"""

    if not actual_sell:
        guide += "  (无需卖出)\n"
    else:
        for t in actual_sell:
            hands = t.action_shares // 100
            guide += f"  {t.ts_code:<16} {t.sell_price:<14.2f} {hands:>10}\n"

    guide += """
【买入操作】（后执行）
  证券代码            买入价格        买入数量
"""

    if not actual_buy:
        guide += "  (无需买入)\n"
    else:
        for t in actual_buy:
            hands = t.action_shares // 100
            guide += f"  {t.ts_code:<16} {t.buy_price:<14.2f} {hands:>10}\n"

    actual_buy_by_code = {t.ts_code: t for t in actual_buy}
    actual_sell_by_code = {t.ts_code: t for t in actual_sell}

    # 完整持仓
    guide += f"""
================================================================================
                         【完整持仓目标】
================================================================================
  说明：本表是目标仓位总览；真正下单以【卖出清单/买入清单】和 CSV 中“本次执行”列为准。

  {'执行':<8} {'股票代码':<12} {'权重':<10} {'目标手数':<8} {'当前手数':<8} {'本次买卖':<8} {'目标动作':<8}
"""

    guide += "  " + "-" * 85 + "\n"

    for t in targets:
        if t.ts_code in actual_buy_by_code:
            exec_action = "buy"
            exec_need = f"+{actual_buy_by_code[t.ts_code].action_shares // 100}"
        elif t.ts_code in actual_sell_by_code:
            exec_action = "sell"
            exec_need = f"-{actual_sell_by_code[t.ts_code].action_shares // 100}"
        else:
            exec_action = "skip" if t.action in ("buy", "sell") else "hold"
            exec_need = "0"
        # shares 字段是股数，输出给同花顺"手数" = // 100
        tgt_hands = t.target_shares // 100
        cur_hands = t.current_shares // 100
        guide += f"  {exec_action:<8} {t.ts_code:<12} {t.weight:<10.2%} {tgt_hands:<8} {cur_hands:<8} {exec_need:<8} {t.action:<8}\n"

    outsider_rows_for_full = [t for t in actual_sell if t.ts_code not in {x.ts_code for x in targets}]
    for t in outsider_rows_for_full:
        guide += f"  {'sell':<8} {t.ts_code:<12} {'0.00%':<10} {0:<8} {t.current_shares // 100:<8} {'-' + str(t.action_shares // 100):<8} {'outsider':<8}\n"

    guide += f"""
================================================================================
                         【快速检查清单】
================================================================================
□ 确认卖出股票后资金可用
□ 确认买入股票有足够资金
□ 检查是否有涨停/跌停股需要特殊处理
□ 准备好备选股票代码
□ 大单确认分批下单策略

================================================================================
"""

    # 保存文件
    date_dir = output_dir / "trading_guides" / last_date
    date_dir.mkdir(parents=True, exist_ok=True)

    guide_path = date_dir / "trading_guide.txt"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide)

    # 保存CSV（shares 内部存股数，输出给同花顺"手数" = // 100）
    orders = []
    for t in actual_sell:
        orders.append({
            "操作": "卖出",
            "股票代码": t.ts_code,
            "卖出手数": t.action_shares // 100,
            "当前手数": t.current_shares // 100,
            "建议卖出价": t.sell_price,
            "预估金额": t.action_shares * t.sell_price,
            "风险提示": t.risk_flag,
        })
    for t in actual_buy:
        orders.append({
            "操作": "买入",
            "股票代码": t.ts_code,
            "买入手数": t.action_shares // 100,
            "目标手数": t.target_shares // 100,
            "建议买入价": t.buy_price,
            "预估金额": t.action_shares * t.buy_price,
            "风险提示": t.risk_flag,
        })

    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(date_dir / "trading_orders.csv", index=False, encoding="utf-8-sig")

    if backups:
        backups_df = pd.DataFrame(backups)
        backups_df.to_csv(date_dir / "backup_stocks.csv", index=False, encoding="utf-8-sig")

    # 完整持仓：操作/需买卖手数表示“本次实际执行”；目标动作列保留理想目标差异。
    full = []
    for t in targets:
        target_action = t.action
        target_delta_hands = t.action_shares // 100
        if t.ts_code in actual_buy_by_code:
            exec_action = "buy"
            exec_delta_hands = actual_buy_by_code[t.ts_code].action_shares // 100
            execution_status = "本次执行"
        elif t.ts_code in actual_sell_by_code:
            exec_action = "sell"
            exec_delta_hands = actual_sell_by_code[t.ts_code].action_shares // 100
            execution_status = "本次执行"
        else:
            exec_action = "skip" if target_action in ("buy", "sell") else "hold"
            exec_delta_hands = 0
            execution_status = "暂不执行" if target_action in ("buy", "sell") else "继续持有"
        full.append({
            "排名": t.rank,
            "股票代码": t.ts_code,
            "分数": t.score,
            "权重": f"{t.weight:.2%}",
            "目标手数": t.target_shares // 100,
            "当前手数": t.current_shares // 100,
            "操作": exec_action,
            "需买卖手数": exec_delta_hands,
            "执行状态": execution_status,
            "目标操作": target_action,
            "目标需买卖手数": target_delta_hands,
            "建议买入价": t.buy_price,
            "建议卖出价": t.sell_price,
            "流动性": "优" if t.liquidity_score > 0.6 else ("中" if t.liquidity_score > 0.3 else "差"),
            "风险提示": t.risk_flag,
        })
    for t in outsider_rows_for_full:
        full.append({
            "排名": 0,
            "股票代码": t.ts_code,
            "分数": t.score,
            "权重": "0.00%",
            "目标手数": 0,
            "当前手数": t.current_shares // 100,
            "操作": "sell",
            "需买卖手数": t.action_shares // 100,
            "执行状态": "本次执行",
            "目标操作": "outsider_sell",
            "目标需买卖手数": t.action_shares // 100,
            "建议买入价": t.buy_price,
            "建议卖出价": t.sell_price,
            "流动性": "优" if t.liquidity_score > 0.6 else ("中" if t.liquidity_score > 0.3 else "差"),
            "风险提示": t.risk_flag,
        })

    full_df = pd.DataFrame(full)
    full_df.to_csv(date_dir / "full_positions.csv", index=False, encoding="utf-8-sig")

    # 配置信息
    config_info = {
        "data_date": last_date,
        "trading_date": trading_date,
        "scheme": scheme_name,
        "portfolio_value": portfolio_value,
        "target_position": target_position,
        "n_hold": len(targets),
        "n_buy": len(actual_buy),
        "n_sell": len(actual_sell),
        "n_target_buy": len(to_buy),
        "n_target_sell": len(to_sell),
        "n_outsider_sell_candidates": len(outsider_sells),
        "n_outsider_sell_executed": len(limited_outsider_sells),
        "dynamic_k": actual_k,
        "base_k": k_trade,
        "vol_pct": vol_pct,
        "position_confidence": position_confidence,
        "allocation_strategy": allocation_strategy,
    }
    with open(date_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    return guide, orders_df, date_dir


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="同花顺手动下单交易指南生成器")
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--holdings", default="", help="格式: 代码:手数,代码:手数")
    parser.add_argument("--portfolio-value", type=float, default=1000000)
    parser.add_argument("--scheme-name", default="方案A(纯DL)")
    parser.add_argument("--allocation", default="score_weighted",
                       choices=["equal", "score_weighted", "rank_weighted", "power"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  同花顺手动下单交易指南生成器 v3")
    print(f"{'='*60}")
    print(f"配置: {args.config}")
    print(f"方案: {args.scheme_name}")
    print()

    # 加载模型
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = output_dir / "model.pt"

    if not ckpt_path.exists():
        print(f"❌ 错误: 模型文件不存在: {ckpt_path}")
        print("请先运行训练脚本")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"✓ 模型加载成功")

    # 构建面板
    end = cfg["end_date"]
    end_dt = datetime.strptime(end, "%Y%m%d")
    start = (end_dt - timedelta(days=250)).strftime("%Y%m%d")

    panel = build_panel(
        cfg["data_dir"],
        start_date=start,
        end_date=end,
        use_metric=cfg["features"]["use_metric"],
        use_moneyflow=cfg["features"]["use_moneyflow"],
        use_market=cfg["features"].get("use_market", True),
        use_news=cfg["features"].get("use_news", False),
        universe=cfg["universe"],
    )
    panel = add_features(
        panel,
        cross_section_rank=cfg["features"]["cross_section_rank"],
        label_horizon=cfg.get("label_horizon", 1),
        fill_missing=cfg["features"].get("fill_missing", True),
    )

    feat_cols = ckpt.get("feat_cols") or feature_columns(panel)
    missing = [c for c in feat_cols if c not in panel.columns]
    if missing:
        panel = panel.assign(**{c: 0.0 for c in missing})
    panel = panel.copy()

    print(f"✓ 数据面板构建成功: {len(panel)} 行")

    # 预测
    scores = score_latest(panel, feat_cols, ckpt["seq_len"], ckpt, device)

    # LightGBM 通道（仅 cfg.lgbm.enabled=True 且本地有 lgbm_model.txt+lgbm_meta.json 时启用）
    lgbm_enabled = cfg.get("lgbm", {}).get("enabled", False)
    if lgbm_enabled:
        lgbm_scores = score_latest_lgbm(panel, output_dir)
        if lgbm_scores.empty:
            print("⚠ LGBM 通道启用但本地缺 lgbm_model.txt / lgbm_meta.json 或 lightgbm 未安装，降级为纯 DL")
        else:
            print(f"✓ LGBM 打分完成: {len(lgbm_scores)} 只股票")
    else:
        lgbm_scores = pd.DataFrame()
        print("• LGBM 通道未启用（纯 DL 方案）")

    # 与 DL 分数 rank 融合（alpha 来自 blend_meta.json.best_alpha 或 cfg.ensemble.alpha）
    scores = blend_latest_scores(scores, lgbm_scores, output_dir, cfg)
    if scores.empty:
        print("❌ 错误: 无法生成预测分数（DL/LGBM 融合后为空）")
        sys.exit(1)
    if lgbm_enabled and not lgbm_scores.empty:
        meta_path = output_dir / "blend_meta.json"
        if meta_path.exists():
            best_alpha = float(json.loads(meta_path.read_text(encoding="utf-8")).get("best_alpha", 0.6))
        else:
            best_alpha = float(cfg.get("ensemble", {}).get("alpha", 0.6))
        print(f"✓ 分数融合完成: alpha={best_alpha:.2f}（DL 权重），1-alpha={1-best_alpha:.2f}（LGBM 权重）")

    # 添加 buyable 标记
    from src.backtest.risk import attach_buyable_flag
    scores = attach_buyable_flag(scores, panel, cfg["strategy"])

    if scores.empty:
        print("❌ 错误: 无法生成预测分数")
        sys.exit(1)

    print(f"✓ 预测完成: {len(scores)} 只股票")

    # 解析持仓
    holdings = parse_holdings(args.holdings)
    print(f"✓ 当前持仓: {len(holdings)} 只股票")

    # 策略参数
    strategy_cfg = cfg["strategy"]
    base_n_hold = strategy_cfg.get("n_hold", 30)
    k_trade = strategy_cfg.get("k_trade", 1)
    cash_reserve_ratio = float(strategy_cfg.get("cash_reserve_ratio", 0.0))

    # ========== 市场波动分位（与 engine.py 一致）==========
    vol_pct = None
    if strategy_cfg.get("adaptive_hold", False):
        vol_col = strategy_cfg.get("market_vol_col")
        if not vol_col:
            for candidate in ["sh_idx_vol20", "hs300_idx_vol20", "volatility_20d"]:
                if candidate in scores.columns:
                    vol_col = candidate
                    break

        if vol_col and vol_col in scores.columns:
            vol_by_date = scores.groupby("trade_date")[vol_col].median().sort_index()
            if len(vol_by_date) > 0:
                vol_pct = float(_rolling_percentile(vol_by_date, int(strategy_cfg.get("adaptive_lookback", 250))).iloc[-1])
                print(f"✓ 市场波动分位: {vol_pct:.2%}")

    # 用 buyable + momentum 后的候选池估计置信度；先用 base_n_hold 估一次，再决定 n_hold
    latest_day = scores[scores["trade_date"] == scores["trade_date"].max()].copy()
    if "buyable" in latest_day.columns:
        buy_s = latest_day[latest_day["buyable"].astype(bool)].set_index("ts_code")["score"]
    else:
        buy_s = latest_day.set_index("ts_code")["score"]
    buy_s = _filter_momentum(latest_day, buy_s, strategy_cfg)

    _, preliminary_confidence = choose_target_position(
        buy_s, base_n_hold, strategy_cfg, vol_pct, cash_reserve_ratio
    )

    # ========== 动态持仓数量（与 engine.py 一致）==========
    if strategy_cfg.get("adaptive_hold", False):
        if strategy_cfg.get("adaptive_hold_multi_indicator", False):
            n_hold = _choose_adaptive_n_multi(base_n_hold, vol_pct, None, preliminary_confidence, strategy_cfg)
        else:
            n_hold = _choose_adaptive_n(base_n_hold, vol_pct, strategy_cfg)
    else:
        n_hold = base_n_hold

    print(f"✓ 持仓数量: {n_hold} (基准: {base_n_hold})")

    # 动态目标仓位
    target_position, position_confidence = choose_target_position(
        buy_s, n_hold, strategy_cfg, vol_pct, cash_reserve_ratio
    )

    print(f"✓ 目标仓位: {target_position:.0%}")
    print(f"✓ 置信度: {position_confidence:.2f}")

    # 生成指南
    guide, orders_df, date_dir = generate_trading_guide(
        scores=scores,
        panel=panel,
        current_holdings=holdings,
        strategy_cfg=strategy_cfg,
        n_hold=n_hold,
        k_trade=k_trade,
        target_position=target_position,
        portfolio_value=args.portfolio_value,
        allocation_strategy=args.allocation,
        scheme_name=f"{args.scheme_name} ({args.allocation})",
        output_dir=output_dir,
        vol_pct=vol_pct,
        position_confidence=position_confidence,
    )

    print(guide)

    print(f"\n✅ 文件已保存:")
    print(f"   {date_dir}/")
    print(f"   ├── trading_guide.txt")
    print(f"   ├── trading_orders.csv")
    print(f"   ├── backup_stocks.csv")
    print(f"   ├── full_positions.csv")
    print(f"   └── config.json")


if __name__ == "__main__":
    main()
