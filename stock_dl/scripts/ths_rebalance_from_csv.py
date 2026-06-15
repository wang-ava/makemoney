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
    ThsAPIError,
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
    reference_price: float
    order_price: float
    source: str
    order_id: str = ""
    status: str = "planned"
    message: str = ""


def to_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def ts_to_plain_code(ts_code: str) -> str:
    return normalize_stock_code(str(ts_code).strip())


def build_sell_orders(client: ThsClient, sell_offset: float) -> list[PlannedOrder]:
    orders: list[PlannedOrder] = []
    for position in client.get_positions():
        stock_code = normalize_stock_code(position.stock_code)
        available = to_int(position.available)
        shares = round_lot_shares(available)
        if shares <= 0:
            continue

        stock_info = client.query_stock(stock_code, command="cmd_wt_maichu")
        close_price = to_float(stock_info.get("cur_price"))
        if close_price <= 0:
            orders.append(
                PlannedOrder(
                    action="sell",
                    stock_code=stock_code,
                    stock_name=position.stock_name,
                    shares=shares,
                    reference_price=0.0,
                    order_price=0.0,
                    source="current_position",
                    status="failed",
                    message="missing close/current price",
                )
            )
            continue

        orders.append(
            PlannedOrder(
                action="sell",
                stock_code=stock_code,
                stock_name=position.stock_name,
                shares=shares,
                reference_price=close_price,
                order_price=round(max(0.01, close_price - sell_offset), 2),
                source="current_position",
            )
        )
    return orders


def build_buy_orders(orders_csv: Path, buy_offset: float) -> list[PlannedOrder]:
    df = pd.read_csv(orders_csv)
    required = {"side", "ts_code", "reference_price", "estimated_shares"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"orders csv missing columns: {sorted(missing)}")

    planned: list[PlannedOrder] = []
    for _, row in df.iterrows():
        side = str(row.get("side", "")).lower()
        buyable = row.get("buyable", True)
        if side != "buy" or buyable is False:
            continue

        shares = round_lot_shares(to_int(row.get("estimated_shares")))
        reference_price = to_float(row.get("reference_price"))
        if shares <= 0 or reference_price <= 0:
            continue

        planned.append(
            PlannedOrder(
                action="buy",
                stock_code=ts_to_plain_code(row.get("ts_code")),
                stock_name="",
                shares=shares,
                reference_price=reference_price,
                order_price=round(reference_price + buy_offset, 2),
                source=str(orders_csv),
            )
        )
    return planned


def discover_market_gdzh(client: ThsClient) -> dict[str, str]:
    market_gdzh: dict[str, str] = {}
    for position in client.get_positions():
        code = normalize_stock_code(position.stock_code)
        if not position.gdzh:
            continue
        if code.startswith(("6", "9")) or str(position.market_code) == "2":
            market_gdzh.setdefault("sh", position.gdzh)
        else:
            market_gdzh.setdefault("sz", position.gdzh)
    return market_gdzh


def gdzh_for_order(order: PlannedOrder, default_gdzh: str | None, market_gdzh: dict[str, str]) -> str | None:
    if order.action == "buy" and order.stock_code.startswith(("6", "9")):
        return market_gdzh.get("sh") or default_gdzh
    if order.action == "buy":
        return market_gdzh.get("sz") or default_gdzh
    return default_gdzh


def submit_orders(
    client: ThsClient,
    orders: list[PlannedOrder],
    gdzh: str | None,
    delay: float,
    market_gdzh: dict[str, str] | None = None,
) -> list[PlannedOrder]:
    market_gdzh = market_gdzh or {}
    for order in orders:
        if order.status == "failed":
            continue
        try:
            result = client.submit_order(
                order.stock_code,
                order.action,
                order.order_price,
                order.shares,
                gdzh=gdzh_for_order(order, gdzh, market_gdzh),
            )
            order.status = "submitted"
            order.message = f"amount={result.amount}, price={result.price:.3f}"
        except Exception as exc:
            order.status = "failed"
            order.message = f"{type(exc).__name__}: {exc}"
        if delay > 0:
            time.sleep(delay)
    return orders


def wait_for_orders_to_clear(
    client: ThsClient,
    submitted_orders: list[PlannedOrder],
    timeout: float,
    interval: float,
) -> None:
    if timeout <= 0:
        return

    target_codes = {
        order.stock_code
        for order in submitted_orders
        if order.action == "sell" and order.status == "submitted"
    }
    if not target_codes:
        return

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
            return

        remaining = deadline - time.time()
        if remaining <= 0:
            print(
                f"Sell wait timeout with {len(open_sells)} sell orders still open; continuing to buys.",
                flush=True,
            )
            return

        sleep_seconds = min(interval, remaining)
        print(f"Waiting for {len(open_sells)} open sell orders: {sleep_seconds:.0f}s", flush=True)
        time.sleep(sleep_seconds)


def print_plan(sell_orders: list[PlannedOrder], buy_orders: list[PlannedOrder]) -> None:
    sell_value = sum(order.shares * order.order_price for order in sell_orders if order.order_price > 0)
    buy_value = sum(order.shares * order.order_price for order in buy_orders if order.order_price > 0)
    print(f"Sell orders: {len(sell_orders)}, estimated value={sell_value:,.2f}")
    print(f"Buy orders: {len(buy_orders)}, estimated value={buy_value:,.2f}")
    print("Sells:")
    for order in sell_orders:
        print(
            f"  sell {order.stock_code:<6} {order.stock_name:<12} "
            f"{order.shares:>6} @ {order.order_price:.2f} "
            f"(ref {order.reference_price:.2f})"
        )
    print("Buys:")
    for order in buy_orders:
        print(
            f"  buy  {order.stock_code:<6} {order.shares:>6} @ {order.order_price:.2f} "
            f"(ref {order.reference_price:.2f})"
        )


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
                "reference_price",
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
                    "reference_price": f"{order.reference_price:.2f}",
                    "order_price": f"{order.order_price:.2f}",
                    "source": order.source,
                    "status": order.status,
                    "message": order.message,
                }
            )
    print(f"Report saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sell all current THS positions and buy CSV recommendations."
    )
    parser.add_argument("--orders-csv", type=Path, required=True)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--skip-sells", action="store_true")
    parser.add_argument("--skip-buys", action="store_true")
    parser.add_argument(
        "--include-codes",
        default="",
        help="Comma-separated plain or ts_code stock codes to submit; empty means all planned orders.",
    )
    parser.add_argument("--sell-offset", type=float, default=0.10)
    parser.add_argument("--buy-offset", type=float, default=0.10)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--wait-for-sells", type=float, default=0.0)
    parser.add_argument("--wait-interval", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or (
        ROOT / "outputs" / f"rebalance_from_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    try:
        client = ThsClient()
        gdzh = client.discover_gdzh()
        market_gdzh = discover_market_gdzh(client)
        sell_orders = [] if args.skip_sells else build_sell_orders(client, args.sell_offset)
        buy_orders = [] if args.skip_buys else build_buy_orders(args.orders_csv, args.buy_offset)
        if args.include_codes:
            include_codes = {
                normalize_stock_code(code.strip()) for code in args.include_codes.split(",") if code.strip()
            }
            sell_orders = [order for order in sell_orders if order.stock_code in include_codes]
            buy_orders = [order for order in buy_orders if order.stock_code in include_codes]
    except Exception as exc:
        print(f"Build failed: {type(exc).__name__}: {exc}")
        return 1

    print_plan(sell_orders, buy_orders)
    if not args.confirm:
        write_report(sell_orders + buy_orders, output)
        print("Dry run only. Re-run with --confirm to submit.")
        return 0

    submit_orders(client, sell_orders, gdzh, args.delay, market_gdzh)
    submitted_sells = sum(1 for order in sell_orders if order.status == "submitted")
    if sell_orders and submitted_sells == 0:
        print("No sell orders were submitted; skipping buys.", flush=True)
        write_report(sell_orders + buy_orders, output)
        failed = sum(1 for order in sell_orders if order.status == "failed")
        print(f"Submitted: 0, failed: {failed}")
        return 1

    wait_for_orders_to_clear(client, sell_orders, args.wait_for_sells, args.wait_interval)
    submit_orders(client, buy_orders, gdzh, args.delay, market_gdzh)
    all_orders = sell_orders + buy_orders
    write_report(all_orders, output)

    submitted = sum(1 for order in all_orders if order.status == "submitted")
    failed = sum(1 for order in all_orders if order.status == "failed")
    print(f"Submitted: {submitted}, failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
