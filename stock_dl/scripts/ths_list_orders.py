#!/usr/bin/env python3
"""List currently-open sell orders so we can decide which to re-quote."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import ThsClient  # noqa: E402


def main() -> int:
    client = ThsClient()
    orders = client.get_orders()
    print(f"# open orders: {len(orders)}")
    print("order_id | side | code | name | price | amount | status | date")
    for o in orders:
        print(
            f"{o.order_id} | {o.side} | {o.stock_code} | {o.stock_name} | "
            f"{o.price} | {o.amount} | {o.status} | {o.order_date}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
