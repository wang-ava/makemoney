"""
成交量约束交易模块

提供:
1. 历史成交率估算
2. 智能下单执行器
3. 回测成交约束增强
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np


class MarketCondition(Enum):
    """市场环境枚举"""
    HOT = "hot"           # 火热 (涨停家数 > 100)
    WARM = "warm"         # 偏热 (涨停家数 50-100)
    NORMAL = "normal"     # 正常
    COLD = "cold"         # 冷淡 (跌多涨少)


@dataclass
class FillRateConfig:
    """成交率配置"""
    # 参与率阈值 (目标股数 / 日均成交量)
    participation_tiers: list[tuple[float, float]] = None  # (threshold, fill_rate)

    # 市场环境影响系数
    market_hot_multiplier: float = 0.7
    market_warm_multiplier: float = 0.85
    market_normal_multiplier: float = 1.0
    market_cold_multiplier: float = 1.1  # 冷市场反而容易买

    # 涨跌停限制
    allow_limit_up_fill: float = 0.95   # 涨停时买入成交率
    allow_limit_down_sell: float = 0.0  # 跌停时无法卖出

    def __post_init__(self):
        if self.participation_tiers is None:
            # 默认参与率分级: (参与率阈值, 成交率)
            self.participation_tiers = [
                (0.01, 0.95),   # 1%以下: 几乎必成交
                (0.05, 0.90),   # 5%以下: 高成交率
                (0.10, 0.75),   # 10%以下: 较好成交率
                (0.15, 0.60),   # 15%以下: 中等成交率
                (0.20, 0.45),   # 20%以下: 较低成交率
                (0.30, 0.30),   # 30%以下: 低成交率
                (1.00, 0.20),   # 30%以上: 很低成交率
            ]


# 默认配置
DEFAULT_CONFIG = FillRateConfig()


def estimate_daily_volume_stats(panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    估算每只股票的历史日均成交量

    Args:
        panel: 包含日线数据的panel (需包含ts_code, trade_date, vol, amount)
        window: 统计窗口天数

    Returns:
        DataFrame with columns: ts_code, avg_volume, volume_std, avg_amount
    """
    stats = []

    for code, group in panel.groupby("ts_code"):
        if "vol" not in group.columns or "amount" not in group.columns:
            continue

        # 取最近window天的数据
        recent = group.tail(window)

        # 排除涨跌停日 (这些日子成交量异常)
        if "pct_chg" in group.columns:
            normal_days = recent[(recent["pct_chg"] > -9.5) & (recent["pct_chg"] < 9.5)]
        else:
            normal_days = recent

        if len(normal_days) < 5:
            normal_days = recent  # 数据不足时使用全部

        stats.append({
            "ts_code": code,
            "avg_volume": normal_days["vol"].mean(),
            "volume_std": normal_days["vol"].std(),
            "avg_amount": normal_days["amount"].mean(),
            "median_volume": normal_days["vol"].median(),
            "min_volume": normal_days["vol"].min(),
            "max_volume": normal_days["vol"].max(),
            "data_days": len(normal_days),
        })

    return pd.DataFrame(stats)


def get_market_condition(panel: pd.DataFrame, date: str) -> MarketCondition:
    """
    根据当日涨跌停家数判断市场环境

    Args:
        panel: 日线数据
        date: 交易日期

    Returns:
        MarketCondition enum
    """
    day_data = panel[panel["trade_date"] == date]

    if "pct_chg" not in day_data.columns:
        return MarketCondition.NORMAL

    limit_up = (day_data["pct_chg"] >= 9.5).sum()
    limit_down = (day_data["pct_chg"] <= -9.5).sum()

    total = len(day_data)
    if total == 0:
        return MarketCondition.NORMAL

    limit_up_ratio = limit_up / total

    if limit_up > 100 or limit_up_ratio > 0.05:
        return MarketCondition.HOT
    elif limit_up > 50 or limit_up_ratio > 0.02:
        return MarketCondition.WARM
    else:
        return MarketCondition.NORMAL


def estimate_fill_rate(
    target_shares: float,
    daily_volume: float,
    price: float,
    market_condition: MarketCondition = MarketCondition.NORMAL,
    is_limit_up: bool = False,
    is_limit_down: bool = False,
    config: FillRateConfig = DEFAULT_CONFIG,
) -> float:
    """
    估算成交率

    Args:
        target_shares: 目标买入/卖出股数
        daily_volume: 日均成交量 (股数)
        price: 限价
        market_condition: 市场环境
        is_limit_up: 是否涨停
        is_limit_down: 是否跌停
        config: 成交率配置

    Returns:
        成交率 (0.0 ~ 1.0)
    """
    # 涨跌停限制
    if is_limit_up and target_shares > 0:  # 买入涨停
        return config.allow_limit_up_fill

    if is_limit_down and target_shares < 0:  # 卖出跌停
        return config.allow_limit_down_sell

    # 计算参与率
    if daily_volume <= 0:
        return 0.5  # 默认50%

    participation = abs(target_shares) / daily_volume

    # 分层查表获取基础成交率
    base_fill_rate = 0.5
    for threshold, fill_rate in config.participation_tiers:
        if participation <= threshold:
            base_fill_rate = fill_rate
            break

    # 应用市场环境系数
    market_multiplier = {
        MarketCondition.HOT: config.market_hot_multiplier,
        MarketCondition.WARM: config.market_warm_multiplier,
        MarketCondition.NORMAL: config.market_normal_multiplier,
        MarketCondition.COLD: config.market_cold_multiplier,
    }.get(market_condition, 1.0)

    # 涨停时买入困难, 跌停时卖出困难 (反向)
    if is_limit_up and target_shares > 0:
        market_multiplier *= 0.5
    if is_limit_down and target_shares < 0:
        market_multiplier *= 0.0  # 跌停卖出几乎不可能

    final_fill_rate = base_fill_rate * market_multiplier

    # 成交率限制在 [0, 1] 范围内
    return float(np.clip(final_fill_rate, 0.0, 1.0))


def estimate_fill_rate_by_amount(
    target_amount: float,
    avg_daily_amount: float,
    market_condition: MarketCondition = MarketCondition.NORMAL,
    is_limit_up: bool = False,
    is_limit_down: bool = False,
    config: FillRateConfig = DEFAULT_CONFIG,
) -> float:
    """
    基于金额估算成交率 (推荐使用这个, 金额更稳定)

    Args:
        target_amount: 目标成交金额
        avg_daily_amount: 日均成交额
        market_condition: 市场环境
        is_limit_up: 是否涨停
        is_limit_down: 是否跌停

    Returns:
        成交率 (0.0 ~ 1.0)
    """
    if avg_daily_amount <= 0:
        return 0.5

    # 参与率用金额计算
    participation = target_amount / avg_daily_amount

    # 分层查表
    base_fill_rate = 0.5
    for threshold, fill_rate in config.participation_tiers:
        if participation <= threshold:
            base_fill_rate = fill_rate
            break

    # 市场系数
    market_multiplier = {
        MarketCondition.HOT: config.market_hot_multiplier,
        MarketCondition.WARM: config.market_warm_multiplier,
        MarketCondition.NORMAL: config.market_normal_multiplier,
        MarketCondition.COLD: config.market_cold_multiplier,
    }.get(market_condition, 1.0)

    # 涨跌停限制
    if is_limit_up and target_amount > 0:
        market_multiplier *= 0.5
    if is_limit_down and target_amount < 0:
        market_multiplier *= 0.0

    return float(np.clip(base_fill_rate * market_multiplier, 0.0, 1.0))


class SmartExecutor:
    """
    智能下单执行器

    支持:
    - 分批下单
    - 追价重试
    - 成交率反馈
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        fill_rate_config: FillRateConfig = DEFAULT_CONFIG,
    ):
        self.config = config or {}
        self.fill_rate_config = fill_rate_config

        # 执行参数
        self.max_retry = self.config.get("max_retry", 3)
        self.price_improvement = self.config.get("price_improvement", 0.002)  # 0.2%
        self.wait_seconds = self.config.get("wait_seconds", 30)
        self.batch_ratio = self.config.get("batch_ratio", 0.5)  # 第一批下单比例

        # 模拟模式 (回测时为True)
        self.simulation_mode = self.config.get("simulation_mode", True)

        # 成交记录
        self.execution_log: list[dict] = []

    def execute_buy(
        self,
        stock_code: str,
        target_shares: int,
        limit_price: float,
        market_condition: MarketCondition = MarketCondition.NORMAL,
        daily_volume: float = 0,
    ) -> dict:
        """
        执行买入

        策略:
        1. 先下一半的量, 价格提高0.2% (激进单)
        2. 剩下一半在目标价限价单
        3. 如果未成交, 改价继续
        4. 直到成交或达到最大重试次数

        Returns:
            {
                'filled_shares': 实际成交股数,
                'avg_price': 成交均价,
                'fill_rate': 成交比例,
                'executions': [{'batch': 1, 'price': x, 'shares': y}, ...]
            }
        """
        if self.simulation_mode:
            # 模拟模式: 直接用成交率估算
            fill_rate = estimate_fill_rate(
                target_shares, daily_volume, limit_price,
                market_condition, is_limit_up=False, config=self.fill_rate_config
            )

            filled_shares = int(target_shares * fill_rate)
            avg_price = limit_price * (1 + self.price_improvement * 0.3)  # 模拟滑点

            return {
                "filled_shares": filled_shares,
                "avg_price": avg_price,
                "fill_rate": fill_rate,
                "stock": stock_code,
                "target_shares": target_shares,
                "executions": [{
                    "batch": 1,
                    "price": avg_price,
                    "shares": filled_shares,
                    "fill_rate": fill_rate,
                }],
            }

        # 实盘模式: 需要对接券商API
        return self._execute_real_buy(stock_code, target_shares, limit_price)

    def execute_sell(
        self,
        stock_code: str,
        target_shares: int,
        limit_price: float,
        market_condition: MarketCondition = MarketCondition.NORMAL,
        daily_volume: float = 0,
        is_limit_down: bool = False,
    ) -> dict:
        """
        执行卖出

        策略:
        1. 检查是否跌停, 跌停则无法卖出
        2. 先下一半在略低价格
        3. 剩下一半正常限价
        """
        if is_limit_down:
            return {
                "filled_shares": 0,
                "avg_price": 0,
                "fill_rate": 0.0,
                "stock": stock_code,
                "target_shares": target_shares,
                "reason": "limit_down",
                "executions": [],
            }

        if self.simulation_mode:
            fill_rate = estimate_fill_rate(
                -target_shares, daily_volume, limit_price,
                market_condition, is_limit_down=False, config=self.fill_rate_config
            )

            filled_shares = int(target_shares * fill_rate)
            avg_price = limit_price * (1 - self.price_improvement * 0.3)

            return {
                "filled_shares": filled_shares,
                "avg_price": avg_price,
                "fill_rate": fill_rate,
                "stock": stock_code,
                "target_shares": target_shares,
                "executions": [{
                    "batch": 1,
                    "price": avg_price,
                    "shares": filled_shares,
                    "fill_rate": fill_rate,
                }],
            }

        return self._execute_real_sell(stock_code, target_shares, limit_price)

    def _execute_real_buy(self, stock_code, target_shares, limit_price) -> dict:
        """实盘买入 (需要对接券商API)"""
        raise NotImplementedError("需要对接券商API实现")

    def _execute_real_sell(self, stock_code, target_shares, limit_price) -> dict:
        """实盘卖出 (需要对接券商API)"""
        raise NotImplementedError("需要对接券商API实现")

    def get_execution_summary(self) -> pd.DataFrame:
        """获取执行摘要"""
        if not self.execution_log:
            return pd.DataFrame()
        return pd.DataFrame(self.execution_log)


# 便捷函数: 一行代码估算成交率
def quick_fill_rate(
    target_shares: int,
    daily_volume: int,
    market: str = "normal",
) -> float:
    """
    快速估算成交率 (便捷接口)

    Args:
        target_shares: 目标股数
        daily_volume: 日均成交量
        market: 市场环境 ("hot", "warm", "normal", "cold")

    Example:
        >>> quick_fill_rate(100000, 5000000, "normal")
        0.90
    """
    condition = MarketCondition(market.lower())
    return estimate_fill_rate(target_shares, daily_volume, 0, condition)
