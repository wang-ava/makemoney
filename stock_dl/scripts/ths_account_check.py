#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import ThsAPIError, ThsClient


def main() -> int:
    try:
        client = ThsClient()
        fund = client.get_fund()
        positions = client.get_positions()
        orders = client.get_orders()
        gdzh = client.discover_gdzh()
    except ThsAPIError as exc:
        print(f"THS check failed: {exc}")
        return 1
    except Exception as exc:
        print(f"THS check failed: {type(exc).__name__}: {exc}")
        return 1

    print("THS cookie check: OK")
    print(f"Available cash: {fund.get('kyje', '')}")
    print(f"Total asset: {fund.get('zzc', fund.get('zcz', ''))}")
    print(f"Positions: {len(positions)}")
    print(f"Open orders: {len(orders)}")
    print(f"GDZH: {gdzh or 'not found; set THS_GDZH in .env before buying from empty position'}")

    for position in positions[:10]:
        print(
            f"- {position.stock_code} {position.stock_name}: "
            f"balance={position.balance}, available={position.available}, gdzh={position.gdzh}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
