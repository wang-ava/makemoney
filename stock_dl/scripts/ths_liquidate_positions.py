#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
class LiquidationRow:
    stock_code: str
    stock_name: str
    balance: int
    available: int
    sellable: int
    current_price: float
    order_price: float
    gdzh: str
    status: str
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


def resolve_sell_price(
    client: ThsClient,
    stock_code: str,
    discount: float,
    price_source: str,
) -> tuple[float, float]:
    stock_info = client.query_stock(stock_code, command="cmd_wt_maichu")
    current_price = to_float(stock_info.get("cur_price"))
    if current_price <= 0:
        raise ThsAPIError(f"stock {stock_code} has no valid current price")
    if price_source == "sell1":
        base_price = to_float(stock_info.get("wudang", {}).get("sell_price", [""])[0])
    elif price_source == "buy1":
        base_price = to_float(stock_info.get("wudang", {}).get("buy_price", [""])[0])
    else:
        base_price = current_price
    if base_price <= 0:
        raise ThsAPIError(f"stock {stock_code} has no valid {price_source} price")
    order_price = round(max(0.01, base_price * (1 - discount)), 2)
    return current_price, order_price


def make_row(
    client: ThsClient,
    position: ThsPosition,
    discount: float,
    price_source: str,
) -> LiquidationRow:
    stock_code = normalize_stock_code(position.stock_code)
    balance = to_int(position.balance)
    available = to_int(position.available)
    sellable = round_lot_shares(available)
    current_price = 0.0
    order_price = 0.0
    status = "ready" if sellable > 0 else "skip"
    message = ""

    if sellable > 0:
        try:
            current_price, order_price = resolve_sell_price(
                client, stock_code, discount, price_source
            )
        except Exception as exc:
            status = "failed"
            message = f"price query failed: {type(exc).__name__}: {exc}"
    elif available > 0:
        message = "available shares are below one 100-share lot"
    else:
        message = "no available shares"

    return LiquidationRow(
        stock_code=stock_code,
        stock_name=position.stock_name,
        balance=balance,
        available=available,
        sellable=sellable,
        current_price=current_price,
        order_price=order_price,
        gdzh=position.gdzh,
        status=status,
        message=message,
    )


def build_plan(client: ThsClient, discount: float, price_source: str) -> list[LiquidationRow]:
    positions = client.get_positions()
    return [make_row(client, position, discount, price_source) for position in positions]


def get_price_source_value(client: ThsClient, stock_code: str, price_source: str) -> float:
    stock_info = client.query_stock(stock_code, command="cmd_wt_maichu")
    if price_source == "sell1":
        return to_float(stock_info.get("wudang", {}).get("sell_price", [""])[0])
    if price_source == "buy1":
        return to_float(stock_info.get("wudang", {}).get("buy_price", [""])[0])
    return to_float(stock_info.get("cur_price"))


def wait_until_price_available(
    client: ThsClient,
    price_source: str,
    timeout: float,
    interval: float,
) -> None:
    if timeout <= 0 or price_source == "current":
        return

    deadline = time.time() + timeout
    while True:
        positions = client.get_positions()
        probe = next(
            (position for position in positions if round_lot_shares(to_int(position.available)) > 0),
            None,
        )
        if probe is None:
            return

        stock_code = normalize_stock_code(probe.stock_code)
        price = get_price_source_value(client, stock_code, price_source)
        if price > 0:
            print(f"{price_source} is available on {stock_code}: {price:.3f}", flush=True)
            return

        remaining = deadline - time.time()
        if remaining <= 0:
            raise ThsAPIError(f"{price_source} did not become available before timeout")
        sleep_seconds = min(interval, remaining)
        print(
            f"{price_source} unavailable on {stock_code}; waiting {sleep_seconds:.0f}s...",
            flush=True,
        )
        time.sleep(sleep_seconds)


def print_plan(rows: list[LiquidationRow], discount: float) -> None:
    print(f"Liquidation plan ({len(rows)} positions, discount={discount:.4%})")
    print("-" * 96)
    print(f"{'code':<8} {'name':<12} {'balance':>10} {'available':>10} {'sell':>10} {'cur_px':>10} {'order_px':>10} status")
    for row in rows:
        print(
            f"{row.stock_code:<8} {row.stock_name:<12} {row.balance:>10} "
            f"{row.available:>10} {row.sellable:>10} {row.current_price:>10.2f} "
            f"{row.order_price:>10.2f} {row.status}"
            + (f" - {row.message}" if row.message else "")
        )


def write_report(rows: list[LiquidationRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stock_code",
                "stock_name",
                "balance",
                "available",
                "sellable",
                "current_price",
                "order_price",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stock_code": row.stock_code,
                    "stock_name": row.stock_name,
                    "balance": row.balance,
                    "available": row.available,
                    "sellable": row.sellable,
                    "current_price": f"{row.current_price:.2f}",
                    "order_price": f"{row.order_price:.2f}",
                    "status": row.status,
                    "message": row.message,
                }
            )
    print(f"Report saved: {output}")


def execute_plan(client: ThsClient, rows: list[LiquidationRow], delay: float) -> list[LiquidationRow]:
    for row in rows:
        if row.sellable <= 0 or row.status != "ready":
            continue
        try:
            result = client.submit_order(
                row.stock_code,
                "sell",
                row.order_price,
                row.sellable,
                gdzh=row.gdzh or None,
            )
            row.status = "submitted"
            row.message = f"amount={result.amount}, price={result.price:.3f}"
        except Exception as exc:
            row.status = "failed"
            row.message = f"{type(exc).__name__}: {exc}"
        if delay > 0:
            time.sleep(delay)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sell all currently available THS simulated positions."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually submit sell orders. Without this flag the script only prints a dry-run plan.",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.0,
        help="Limit-price discount from current price, e.g. 0.002 means current price * 0.998.",
    )
    parser.add_argument(
        "--price-source",
        choices=["current", "sell1", "buy1"],
        default="current",
        help="Price source for sell orders: current price, sell1 ask, or buy1 bid.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds to wait between submitted orders.",
    )
    parser.add_argument(
        "--wait-for-price",
        type=float,
        default=0.0,
        help="Wait up to this many seconds for the selected price source to become available.",
    )
    parser.add_argument(
        "--wait-interval",
        type=float,
        default=30.0,
        help="Seconds between price-source checks while waiting.",
    )
    parser.add_argument(
        "--cancel-open-orders",
        action="store_true",
        help="Cancel existing open orders before building the liquidation plan. Only runs with --confirm.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV report path. Defaults to stock_dl/outputs/liquidation_YYYYMMDD_HHMMSS.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.discount < 0:
        print("Error: --discount must be non-negative")
        return 2

    output = args.output or (
        ROOT
        / "outputs"
        / f"liquidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    try:
        client = ThsClient()
        if args.cancel_open_orders:
            if not args.confirm:
                print("Dry run: --cancel-open-orders requested, but no orders were cancelled.")
            else:
                cancelled = client.cancel_all_orders()
                print(f"Cancelled {len(cancelled)} open orders.")

        wait_until_price_available(
            client,
            args.price_source,
            args.wait_for_price,
            args.wait_interval,
        )
        rows = build_plan(client, args.discount, args.price_source)
    except ThsAPIError as exc:
        print(f"THS liquidation failed: {exc}")
        return 1
    except Exception as exc:
        print(f"THS liquidation failed: {type(exc).__name__}: {exc}")
        return 1

    print_plan(rows, args.discount)
    ready = [row for row in rows if row.status == "ready" and row.sellable > 0]
    print(f"Ready sell orders: {len(ready)}")

    if not args.confirm:
        write_report(rows, output)
        print("Dry run only. Re-run with --confirm to submit.")
        return 0

    rows = execute_plan(client, rows, args.delay)
    write_report(rows, output)

    submitted = sum(1 for row in rows if row.status == "submitted")
    failed = sum(1 for row in rows if row.status == "failed")
    print(f"Submitted: {submitted}, failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
