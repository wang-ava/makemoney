#!/usr/bin/env python3
"""
智能下单执行器

功能:
1. 读取订单列表
2. 分批下单 + 追价重试
3. 记录执行结果
4. 支持模拟模式和实盘模式

用法:
    # 模拟模式 (回测)
    python scripts/smart_executor.py --orders orders_20260529.csv --mode simulation

    # 实盘模式 (需要对接券商API)
    python scripts/smart_executor.py --orders orders_20260529.csv --mode live --broker xq

    # 查看执行报告
    python scripts/smart_executor.py --report execution_report.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.fill_rate import (
    FillRateConfig,
    MarketCondition,
    estimate_fill_rate_by_amount,
    DEFAULT_CONFIG,
)
from src.trading.ths_client import (
    ThsClient,
    ThsOrder,
    normalize_stock_code,
    round_lot_shares,
)


class ExecutionStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Order:
    """订单"""
    ts_code: str
    side: str  # 'buy' or 'sell'
    price: float  # 限价
    shares: int  # 股数 (100的倍数)
    order_id: Optional[str] = None
    status: str = "pending"
    filled_shares: int = 0
    avg_price: float = 0.0
    submit_time: Optional[str] = None
    fill_time: Optional[str] = None
    retry_count: int = 0


@dataclass
class ExecutionResult:
    """执行结果"""
    ts_code: str
    side: str
    target_shares: int
    filled_shares: int
    fill_rate: float
    avg_price: float
    total_cost: float
    executions: list[dict]  # 每次下单的详情
    duration_seconds: float
    status: str  # success, partial, failed
    reason: Optional[str] = None


class LiveBrokerAdapter:
    """
    实盘券商适配器基类

    需要根据实际券商API实现具体方法
    """

    def __init__(self, config: dict):
        self.config = config

    def connect(self) -> bool:
        """连接券商"""
        raise NotImplementedError("需要实现券商连接")

    def submit_order(self, ts_code: str, side: str, price: float, shares: int) -> str:
        """
        提交订单

        Returns:
            order_id: 订单ID

        Raises:
            ConnectionError: 连接失败
            ValueError: 参数错误
        """
        raise NotImplementedError("需要实现下单方法")

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        raise NotImplementedError("需要实现撤单方法")

    def get_order_status(self, order_id: str) -> dict:
        """
        查询订单状态

        Returns:
            {
                'status': 'pending|submitted|partial|filled|canceled|rejected',
                'filled_shares': int,
                'avg_price': float,
            }
        """
        raise NotImplementedError("需要实现查询方法")

    def disconnect(self):
        """断开连接"""
        pass


class XQAdapter(LiveBrokerAdapter):
    """XQ Broker (迅投) 适配器示例"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.api = None  # 需要初始化XQ API

    def connect(self) -> bool:
        # 实现XQ连接
        raise NotImplementedError("需要实现XQ API连接")

    def submit_order(self, ts_code: str, side: str, price: float, shares: int) -> str:
        raise NotImplementedError("需要实现XQ下单")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("需要实现XQ撤单")

    def get_order_status(self, order_id: str) -> dict:
        raise NotImplementedError("需要实现XQ查询")


class ThsBrokerAdapter(LiveBrokerAdapter):
    """同花顺模拟盘适配器。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.client: ThsClient | None = None
        self.order_dates: dict[str, str] = {}
        self.order_targets: dict[str, dict] = {}
        self.post_submit_delay = float(config.get("post_submit_delay", 0.8))

    def connect(self) -> bool:
        self.client = ThsClient()
        self.client.get_fund()
        return True

    def _require_client(self) -> ThsClient:
        if self.client is None:
            raise ConnectionError("THS adapter is not connected")
        return self.client

    @staticmethod
    def _side_label(side: str) -> str:
        return "买入" if side.lower() == "buy" else "卖出"

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        try:
            return int(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            return default

    def _match_order(
        self,
        orders: list[ThsOrder],
        known_ids: set[str],
        ts_code: str,
        side: str,
        price: float,
        shares: int,
    ) -> ThsOrder | None:
        plain_code = normalize_stock_code(ts_code)
        side_label = self._side_label(side)
        candidates = []
        for order in orders:
            if order.order_id in known_ids:
                continue
            if normalize_stock_code(order.stock_code) != plain_code:
                continue
            if order.side != side_label:
                continue
            if self._to_int(order.amount) != shares:
                continue
            if abs(self._to_float(order.price) - price) > 0.011:
                continue
            candidates.append(order)
        if not candidates:
            return None
        return sorted(candidates, key=lambda o: (o.order_date, o.time, o.order_id))[-1]

    def submit_order(self, ts_code: str, side: str, price: float, shares: int) -> str:
        client = self._require_client()
        shares = round_lot_shares(shares)
        before_ids = {order.order_id for order in client.get_orders()}
        result = client.submit_order(ts_code, side, price, shares)
        time.sleep(self.post_submit_delay)
        after_orders = client.get_orders()
        matched = self._match_order(after_orders, before_ids, ts_code, side, result.price, shares)

        if matched is None:
            order_id = f"ths-immediate-{normalize_stock_code(ts_code)}-{int(time.time() * 1000)}"
            self.order_targets[order_id] = {
                "ts_code": ts_code,
                "side": side,
                "price": result.price,
                "shares": shares,
                "immediate": True,
                "payload": result.payload,
            }
            return order_id

        self.order_dates[matched.order_id] = matched.order_date
        self.order_targets[matched.order_id] = {
            "ts_code": ts_code,
            "side": side,
            "price": result.price,
            "shares": shares,
            "immediate": False,
            "payload": result.payload,
        }
        return matched.order_id

    def cancel_order(self, order_id: str) -> bool:
        target = self.order_targets.get(order_id, {})
        if target.get("immediate"):
            return True
        client = self._require_client()
        order_date = self.order_dates.get(order_id)
        client.cancel_order(order_id, order_date)
        return True

    def get_order_status(self, order_id: str) -> dict:
        target = self.order_targets.get(order_id, {})
        target_shares = self._to_int(target.get("shares"))
        target_price = self._to_float(target.get("price"))

        if target.get("immediate"):
            return {
                "status": "filled",
                "filled_shares": target_shares,
                "avg_price": target_price,
            }

        client = self._require_client()
        open_order = next((order for order in client.get_orders() if order.order_id == order_id), None)
        if open_order is None:
            return {
                "status": "filled",
                "filled_shares": target_shares,
                "avg_price": target_price,
            }

        status_text = open_order.status or ""
        open_amount = self._to_int(open_order.amount)
        if "废" in status_text or "拒" in status_text:
            status = "rejected"
            filled = 0
        elif "部" in status_text:
            status = "partial"
            filled = max(0, target_shares - open_amount)
        else:
            status = "submitted"
            filled = 0
        return {
            "status": status,
            "filled_shares": filled,
            "avg_price": self._to_float(open_order.price, target_price),
        }


class SmartOrderExecutor:
    """
    智能订单执行器

    策略:
    1. 分两批下单: 激进单(价格改善) + 正常单
    2. 等待成交
    3. 未完全成交时, 撤单重试
    4. 最多重试N次
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        fill_rate_config: FillRateConfig = DEFAULT_CONFIG,
        broker_adapter: Optional[LiveBrokerAdapter] = None,
    ):
        self.config = config or {}
        self.fill_rate_config = fill_rate_config
        self.broker = broker_adapter

        # 执行参数
        self.max_retry = self.config.get("max_retry", 3)
        self.price_improvement = self.config.get("price_improvement", 0.002)  # 0.2%
        self.wait_seconds = self.config.get("wait_seconds", 30)
        self.batch_ratio = self.config.get("batch_ratio", 0.5)  # 第一批比例
        self.simulation_mode = self.config.get("simulation_mode", True)

        # 执行记录
        self.execution_log: list[ExecutionResult] = []

    def execute_order(self, order: Order, market_condition: MarketCondition = MarketCondition.NORMAL) -> ExecutionResult:
        """
        执行单个订单

        策略:
        1. 计算目标成交金额
        2. 第一批: 50%量, 价格改善0.2%
        3. 等待N秒
        4. 检查成交情况
        5. 未成交部分继续, 或改价重试
        """
        start_time = time.time()
        executions = []
        total_filled = 0
        total_cost = 0

        if self.simulation_mode:
            # 模拟模式
            return self._execute_simulation(order, market_condition)

        # 实盘模式
        return self._execute_live(order, market_condition)

    def _execute_simulation(self, order: Order, market_condition: MarketCondition) -> ExecutionResult:
        """模拟执行"""
        start_time = time.time()

        # 获取日均成交额 (需要从外部数据源获取)
        # 这里用默认值估算
        avg_daily_amount = self.config.get("avg_daily_amount", 1_000_000_000)  # 默认10亿

        target_amount = order.shares * order.price
        fill_rate = estimate_fill_rate_by_amount(
            target_amount=target_amount,
            avg_daily_amount=avg_daily_amount,
            market_condition=market_condition,
            is_limit_up=False,
            is_limit_down=False,
            config=self.fill_rate_config,
        )

        filled_shares = int(order.shares * fill_rate)
        # 考虑滑点
        if order.side == "buy":
            avg_price = order.price * (1 + self.price_improvement * 0.5)
        else:
            avg_price = order.price * (1 - self.price_improvement * 0.5)

        total_cost = filled_shares * avg_price
        duration = time.time() - start_time

        status = "success" if fill_rate >= 0.95 else ("partial" if fill_rate >= 0.5 else "failed")
        reason = None
        if fill_rate < 0.5:
            reason = f"成交率过低: {fill_rate:.1%}"

        result = ExecutionResult(
            ts_code=order.ts_code,
            side=order.side,
            target_shares=order.shares,
            filled_shares=filled_shares,
            fill_rate=fill_rate,
            avg_price=avg_price,
            total_cost=total_cost,
            executions=[{
                "batch": 1,
                "price": avg_price,
                "shares": filled_shares,
                "fill_rate": fill_rate,
                "mode": "simulation",
            }],
            duration_seconds=duration,
            status=status,
            reason=reason,
        )

        self.execution_log.append(result)
        return result

    def _execute_live(self, order: Order, market_condition: MarketCondition) -> ExecutionResult:
        """实盘执行"""
        if self.broker is None:
            raise RuntimeError("live mode requires a broker adapter")
        start_time = time.time()
        executions = []
        remaining_shares = order.shares
        total_filled = 0
        total_cost = 0

        for retry in range(self.max_retry + 1):
            # 计算下单价格和数量
            if retry == 0:
                # 第一批: 激进单
                batch_shares = round_lot_shares(max(100, int(remaining_shares * self.batch_ratio)))
                price_adj = self.price_improvement
            else:
                # 后续批次: 正常单
                batch_shares = round_lot_shares(remaining_shares)
                price_adj = self.price_improvement * (0.5 + retry * 0.2)  # 越来越激进
            if batch_shares <= 0:
                break
            batch_shares = min(batch_shares, remaining_shares)

            if order.side == "buy":
                batch_price = order.price * (1 + price_adj)
            else:
                batch_price = order.price * (1 - price_adj)

            # 向上取整到最小单位 (A股最小1分钱)
            batch_price = round(batch_price, 2)

            try:
                order_id = self.broker.submit_order(
                    order.ts_code, order.side, batch_price, batch_shares
                )
            except Exception as e:
                executions.append({
                    "batch": retry + 1,
                    "price": batch_price,
                    "shares": batch_shares,
                    "status": "failed",
                    "error": str(e),
                })
                continue

            # 等待成交
            time.sleep(self.wait_seconds)

            # 查询状态
            status_info = self.broker.get_order_status(order_id)

            if status_info["status"] == "filled":
                filled = status_info["filled_shares"]
                avg_px = status_info["avg_price"]
                total_filled += filled
                total_cost += filled * avg_px
                remaining_shares -= filled
                executions.append({
                    "batch": retry + 1,
                    "order_id": order_id,
                    "price": batch_price,
                    "target_shares": batch_shares,
                    "filled_shares": filled,
                    "avg_price": avg_px,
                    "status": "filled",
                })

                if remaining_shares <= 0:
                    break

            elif status_info["status"] == "partial":
                filled = status_info["filled_shares"]
                avg_px = status_info["avg_price"]
                total_filled += filled
                total_cost += filled * avg_px
                remaining_shares -= filled

                # 撤单重试
                self.broker.cancel_order(order_id)
                executions.append({
                    "batch": retry + 1,
                    "order_id": order_id,
                    "price": batch_price,
                    "target_shares": batch_shares,
                    "filled_shares": filled,
                    "avg_price": avg_px,
                    "status": "partial",
                })

            else:
                # 撤单重试
                self.broker.cancel_order(order_id)
                executions.append({
                    "batch": retry + 1,
                    "order_id": order_id,
                    "price": batch_price,
                    "target_shares": batch_shares,
                    "filled_shares": 0,
                    "status": "no_fill",
                })

        duration = time.time() - start_time
        fill_rate = total_filled / order.shares if order.shares > 0 else 0
        status = "success" if fill_rate >= 0.95 else ("partial" if fill_rate >= 0.5 else "failed")
        reason = None
        if fill_rate < 0.5:
            reason = f"成交率过低: {fill_rate:.1%}"

        result = ExecutionResult(
            ts_code=order.ts_code,
            side=order.side,
            target_shares=order.shares,
            filled_shares=total_filled,
            fill_rate=fill_rate,
            avg_price=total_cost / total_filled if total_filled > 0 else 0,
            total_cost=total_cost,
            executions=executions,
            duration_seconds=duration,
            status=status,
            reason=reason,
        )

        self.execution_log.append(result)
        return result

    def execute_batch(self, orders: list[Order], market_condition: MarketCondition = MarketCondition.NORMAL) -> list[ExecutionResult]:
        """批量执行订单"""
        results = []
        for order in orders:
            print(f"  Executing {order.side.upper()} {order.ts_code}: {order.shares} shares @ {order.price}")
            result = self.execute_order(order, market_condition)
            results.append(result)
            print(f"    -> Filled: {result.filled_shares}/{result.target_shares} ({result.fill_rate:.1%})")

        return results

    def get_summary(self) -> dict:
        """获取执行摘要"""
        if not self.execution_log:
            return {}

        df = pd.DataFrame([{
            "ts_code": r.ts_code,
            "side": r.side,
            "target_shares": r.target_shares,
            "filled_shares": r.filled_shares,
            "fill_rate": r.fill_rate,
            "avg_price": r.avg_price,
            "total_cost": r.total_cost,
            "duration": r.duration_seconds,
            "status": r.status,
        } for r in self.execution_log])

        return {
            "total_orders": len(df),
            "total_target_shares": int(df["target_shares"].sum()),
            "total_filled_shares": int(df["filled_shares"].sum()),
            "overall_fill_rate": df["filled_shares"].sum() / df["target_shares"].sum() if df["target_shares"].sum() > 0 else 0,
            "avg_fill_rate": df["fill_rate"].mean(),
            "success_count": (df["status"] == "success").sum(),
            "partial_count": (df["status"] == "partial").sum(),
            "failed_count": (df["status"] == "failed").sum(),
            "total_cost": df["total_cost"].sum(),
            "avg_duration": df["duration"].mean(),
        }

    def save_report(self, filepath: str):
        """保存执行报告"""
        df = pd.DataFrame([{
            "ts_code": r.ts_code,
            "side": r.side,
            "target_shares": r.target_shares,
            "filled_shares": r.filled_shares,
            "fill_rate": f"{r.fill_rate:.2%}",
            "avg_price": f"{r.avg_price:.2f}",
            "total_cost": f"{r.total_cost:.2f}",
            "duration_seconds": f"{r.duration_seconds:.1f}",
            "status": r.status,
            "reason": r.reason or "",
        } for r in self.execution_log])

        df.to_csv(filepath, index=False)
        print(f"Report saved to: {filepath}")

        # 同时保存详细执行记录
        detail_file = filepath.replace(".csv", "_detail.csv")
        detail_rows = []
        for r in self.execution_log:
            for exec_detail in r.executions:
                detail_rows.append({
                    "ts_code": r.ts_code,
                    "side": r.side,
                    "batch": exec_detail.get("batch", ""),
                    "price": exec_detail.get("price", ""),
                    "shares": exec_detail.get("target_shares", ""),
                    "filled": exec_detail.get("filled_shares", ""),
                    "exec_status": exec_detail.get("status", ""),
                })
        pd.DataFrame(detail_rows).to_csv(detail_file, index=False)
        print(f"Detail saved to: {detail_file}")


def load_orders(filepath: str) -> list[Order]:
    """加载订单"""
    df = pd.read_csv(filepath)
    orders = []

    def cell(row: pd.Series, name: str, default=0):
        value = row.get(name, default)
        if pd.isna(value):
            return default
        return value

    for _, row in df.iterrows():
        if {"ts_code", "side", "price", "shares"}.issubset(df.columns):
            ts_code = str(cell(row, "ts_code", ""))
            side = str(cell(row, "side", "")).lower()
            price = float(cell(row, "price", 0))
            shares = round_lot_shares(cell(row, "shares", 0))
        elif {"操作", "股票代码"}.issubset(df.columns):
            action = str(cell(row, "操作", ""))
            ts_code = str(cell(row, "股票代码", ""))
            if "买" in action:
                side = "buy"
                price = float(cell(row, "建议买入价", 0))
                shares = round_lot_shares(cell(row, "买入股数", 0))
            elif "卖" in action:
                side = "sell"
                price = float(cell(row, "建议卖出价", 0))
                shares = round_lot_shares(cell(row, "卖出股数", 0))
            else:
                continue
        else:
            raise ValueError("orders file must contain ts_code/side/price/shares or 操作/股票代码 columns")
        if shares <= 0 or price <= 0 or side not in {"buy", "sell"}:
            continue
        orders.append(Order(ts_code=ts_code, side=side, price=price, shares=shares))
    return orders


def main():
    parser = argparse.ArgumentParser(description="智能下单执行器")
    parser.add_argument("--orders", "-o", help="订单文件路径")
    parser.add_argument("--mode", "-m", choices=["simulation", "live"], default="simulation",
                        help="执行模式: simulation(模拟) 或 live(实盘)")
    parser.add_argument("--output", "-out", help="输出报告路径")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--broker", "-b", choices=["ths", "xq", "gf", "ht"], default="ths",
                        help="券商: ths(同花顺模拟盘), xq(迅投), gf(广发), ht(华泰)")
    parser.add_argument("--confirm-live", action="store_true",
                        help="实盘/模拟盘委托确认开关；live 模式必须显式提供")
    parser.add_argument("--report", "-r", help="查看已有报告")
    args = parser.parse_args()

    # 查看报告模式
    if args.report:
        df = pd.read_csv(args.report)
        print(df.to_string())
        return

    # 加载配置
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # 加载订单
    if not args.orders:
        print("Error: --orders is required")
        return

    orders = load_orders(args.orders)
    print(f"Loaded {len(orders)} orders from {args.orders}")

    if args.mode == "live" and not args.confirm_live:
        print("Refusing live mode without --confirm-live. Use simulation first, then rerun with explicit confirmation.")
        return

    broker_adapter: LiveBrokerAdapter | None = None
    if args.mode == "live":
        if args.broker == "ths":
            broker_adapter = ThsBrokerAdapter(config)
        elif args.broker == "xq":
            broker_adapter = XQAdapter(config)
        else:
            raise NotImplementedError(f"broker {args.broker} is not implemented")
        broker_adapter.connect()

    config["simulation_mode"] = args.mode == "simulation"

    # 创建执行器
    executor = SmartOrderExecutor(
        config=config,
        broker_adapter=broker_adapter,
    )

    # 执行订单
    print(f"\nExecuting {len(orders)} orders in {args.mode} mode...")
    results = executor.execute_batch(orders)

    # 打印摘要
    summary = executor.get_summary()
    print("\n" + "="*50)
    print("EXECUTION SUMMARY")
    print("="*50)
    print(f"Total Orders: {summary['total_orders']}")
    print(f"Overall Fill Rate: {summary['overall_fill_rate']:.2%}")
    print(f"Success: {summary['success_count']}, Partial: {summary['partial_count']}, Failed: {summary['failed_count']}")
    print(f"Total Cost: {summary['total_cost']:,.2f}")

    # 保存报告
    if args.output:
        executor.save_report(args.output)
    else:
        default_output = f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        executor.save_report(default_output)


if __name__ == "__main__":
    main()
