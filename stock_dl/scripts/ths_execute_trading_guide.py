#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import (  # noqa: E402
    ThsClient,
    ThsPosition,
    normalize_stock_code,
    round_lot_shares,
)


@dataclass
class PlannedOrder:
    action: str
    stock_code: str
    stock_name: str
    shares: int
    order_price: float
    source: str
    gdzh: str | None = None
    status: str = "planned"
    message: str = ""


def to_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def to_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def position_map(positions: list[ThsPosition]) -> dict[str, ThsPosition]:
    return {normalize_stock_code(position.stock_code): position for position in positions}


def discover_market_gdzh(positions: list[ThsPosition]) -> dict[str, str]:
    market_gdzh: dict[str, str] = {}
    for position in positions:
        code = normalize_stock_code(position.stock_code)
        if not position.gdzh:
            continue
        if code.startswith(("6", "9")) or str(position.market_code) == "2":
            market_gdzh.setdefault("sh", position.gdzh)
        else:
            market_gdzh.setdefault("sz", position.gdzh)
    return market_gdzh


def gdzh_for_buy(stock_code: str, default_gdzh: str | None, market_gdzh: dict[str, str]) -> str | None:
    if stock_code.startswith(("6", "9")):
        return market_gdzh.get("sh") or default_gdzh
    return market_gdzh.get("sz") or default_gdzh


def build_orders(
    orders_csv: Path,
    positions: list[ThsPosition],
    default_gdzh: str | None,
    market_gdzh: dict[str, str],
) -> tuple[list[PlannedOrder], list[PlannedOrder]]:
    df = pd.read_csv(orders_csv)
    required = {"操作", "股票代码", "卖出股数", "建议卖出价", "买入股数", "建议买入价"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trading guide csv missing columns: {sorted(missing)}")

    positions_by_code = position_map(positions)
    sell_orders: list[PlannedOrder] = []
    buy_orders: list[PlannedOrder] = []

    for _, row in df.iterrows():
        action_label = str(row.get("操作", "")).strip()
        stock_code = normalize_stock_code(str(row.get("股票代码", "")).strip())
        if action_label == "卖出":
            shares = round_lot_shares(to_int(row.get("卖出股数")))
            price = to_float(row.get("建议卖出价"))
            if shares <= 0 or price <= 0:
                continue
            position = positions_by_code.get(stock_code)
            gdzh = position.gdzh if position else None
            stock_name = position.stock_name if position else ""
            order = PlannedOrder(
                action="sell",
                stock_code=stock_code,
                stock_name=stock_name,
                shares=shares,
                order_price=price,
                source=str(orders_csv),
                gdzh=gdzh,
            )
            available = to_int(position.available) if position else 0
            if available < shares:
                order.status = "failed"
                order.message = f"available shares {available} < planned sell {shares}"
            sell_orders.append(order)
        elif action_label == "买入":
            shares = round_lot_shares(to_int(row.get("买入股数")))
            price = to_float(row.get("建议买入价"))
            if shares <= 0 or price <= 0:
                continue
            buy_orders.append(
                PlannedOrder(
                    action="buy",
                    stock_code=stock_code,
                    stock_name="",
                    shares=shares,
                    order_price=price,
                    source=str(orders_csv),
                    gdzh=gdzh_for_buy(stock_code, default_gdzh, market_gdzh),
                )
            )

    return sell_orders, buy_orders


def filter_orders(orders: list[PlannedOrder], include_codes: set[str]) -> list[PlannedOrder]:
    if not include_codes:
        return orders
    return [order for order in orders if order.stock_code in include_codes]


def print_plan(sell_orders: list[PlannedOrder], buy_orders: list[PlannedOrder]) -> None:
    sell_value = sum(order.shares * order.order_price for order in sell_orders if order.status != "failed")
    buy_value = sum(order.shares * order.order_price for order in buy_orders if order.status != "failed")
    print(f"Sell orders: {len(sell_orders)}, estimated value={sell_value:,.2f}")
    print(f"Buy orders: {len(buy_orders)}, estimated value={buy_value:,.2f}")
    print("Sells:")
    for order in sell_orders:
        msg = f" [{order.status}: {order.message}]" if order.status == "failed" else ""
        print(
            f"  sell {order.stock_code:<6} {order.stock_name:<12} "
            f"{order.shares:>6} @ {order.order_price:.2f}{msg}"
        )
    print("Buys:")
    for order in buy_orders:
        msg = f" [{order.status}: {order.message}]" if order.status == "failed" else ""
        print(f"  buy  {order.stock_code:<6} {order.shares:>6} @ {order.order_price:.2f}{msg}")


def write_report(orders: list[PlannedOrder], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "action",
                "stock_code",
                "stock_name",
                "shares",
                "order_price",
                "source",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        for order in orders:
            writer.writerow(
                {
                    "action": order.action,
                    "stock_code": order.stock_code,
                    "stock_name": order.stock_name,
                    "shares": order.shares,
                    "order_price": f"{order.order_price:.2f}",
                    "source": order.source,
                    "status": order.status,
                    "message": order.message,
                }
            )
    print(f"Report saved: {output}")


def submit_orders(client: ThsClient, orders: list[PlannedOrder], delay: float) -> list[PlannedOrder]:
    for order in orders:
        if order.status == "failed":
            continue
        try:
            result = client.submit_order(
                order.stock_code,
                order.action,
                order.order_price,
                order.shares,
                gdzh=order.gdzh,
            )
            order.status = "submitted"
            order.message = f"amount={result.amount}, price={result.price:.3f}"
        except Exception as exc:
            order.status = "failed"
            order.message = f"{type(exc).__name__}: {exc}"
        if delay > 0:
            time.sleep(delay)
    return orders


def wait_for_sells(client: ThsClient, sell_orders: list[PlannedOrder], timeout: float, interval: float) -> bool:
    if timeout <= 0:
        return True
    target_codes = {order.stock_code for order in sell_orders if order.status == "submitted"}
    if not target_codes:
        return True

    deadline = time.time() + timeout
    while True:
        open_orders = client.get_orders()
        open_sells = [
            order
            for order in open_orders
            if normalize_stock_code(order.stock_code) in target_codes and "卖" in order.side
        ]
        if not open_sells:
            print("Sell orders are no longer open; continuing to buys.", flush=True)
            return True

        remaining = deadline - time.time()
        if remaining <= 0:
            print(
                f"Sell wait timeout with {len(open_sells)} sell orders still open.",
                flush=True,
            )
            return False
        sleep_seconds = min(interval, remaining)
        print(f"Waiting for {len(open_sells)} open sell orders: {sleep_seconds:.0f}s", flush=True)
        time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a generated THS trading guide CSV.")
    parser.add_argument("--orders-csv", type=Path, required=True)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--skip-sells", action="store_true")
    parser.add_argument("--skip-buys", action="store_true")
    parser.add_argument(
        "--include-codes",
        default="",
        help="Comma-separated plain or ts_code stock codes to submit; empty means all planned orders.",
    )
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--wait-for-sells", type=float, default=0.0)
    parser.add_argument("--wait-interval", type=float, default=30.0)
    parser.add_argument(
        "--allow-buys-with-open-sells",
        action="store_true",
        help="Continue submitting buys even if submitted sell orders are still open after --wait-for-sells.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or (
        ROOT / "outputs" / f"execute_trading_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    try:
        client = ThsClient()
        positions = client.get_positions()
        gdzh = client.discover_gdzh()
        market_gdzh = discover_market_gdzh(positions)
        sell_orders, buy_orders = build_orders(args.orders_csv, positions, gdzh, market_gdzh)
        if args.skip_sells:
            sell_orders = []
        if args.skip_buys:
            buy_orders = []
        if args.include_codes:
            include_codes = {
                normalize_stock_code(code.strip()) for code in args.include_codes.split(",") if code.strip()
            }
            sell_orders = filter_orders(sell_orders, include_codes)
            buy_orders = filter_orders(buy_orders, include_codes)
    except Exception as exc:
        print(f"Build failed: {type(exc).__name__}: {exc}")
        return 1

    print_plan(sell_orders, buy_orders)
    all_orders = sell_orders + buy_orders
    if not args.confirm:
        write_report(all_orders, output)
        print("Dry run only. Re-run with --confirm to submit.")
        return 0

    submit_orders(client, sell_orders, args.delay)
    submitted_sells = sum(1 for order in sell_orders if order.status == "submitted")
    if sell_orders and submitted_sells == 0:
        print("No sell orders were submitted; skipping buys.", flush=True)
        write_report(all_orders, output)
        failed = sum(1 for order in sell_orders if order.status == "failed")
        print(f"Submitted: 0, failed: {failed}")
        return 1

    sells_cleared = wait_for_sells(client, sell_orders, args.wait_for_sells, args.wait_interval)
    if not sells_cleared and not args.allow_buys_with_open_sells:
        for order in buy_orders:
            if order.status != "failed":
                order.status = "skipped"
                order.message = "sell orders still open; buys skipped"
        write_report(all_orders, output)
        print("Buys skipped because submitted sell orders were still open.")
        return 1

    submit_orders(client, buy_orders, args.delay)
    write_report(all_orders, output)
    submitted = sum(1 for order in all_orders if order.status == "submitted")
    failed = sum(1 for order in all_orders if order.status == "failed")
    print(f"Submitted: {submitted}, failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
