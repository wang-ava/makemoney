#!/usr/bin/env python3
"""Cancel stuck 300619 buy, re-quote at fresh ask1, and add 600123 backup
to push position ratio to ~90%."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import ThsClient  # noqa: E402


def main() -> int:
    client = ThsClient()

    # Step 1: cancel stuck 300619 buy
    print("=== Step 1: cancel stuck 300619 ===")
    try:
        client.cancel_order("6033197890", "20260610")
        print("  canceled 6033197890")
    except Exception as exc:
        print(f"  cancel failed: {exc}")

    time.sleep(1.5)

    # Step 2: re-quote 300619 at ask1 (500 shares)
    print("\n=== Step 2: re-quote 300619 @ ask1 ===")
    d = client.query_stock("300619", "cmd_wt_mairu")
    w = d["wudang"]
    ask1 = float(w["sell_price"][0])
    print(f"  cur={d['cur_price']:.2f} ask1={ask1:.2f}({w['sell_amount'][0]})")
    try:
        r = client.buy("300619", 500, price=ask1)
        print(f"  resubmit 300619 500 @ {ask1} -> amount={r.amount} price={r.price:.3f}")
    except Exception as exc:
        print(f"  resubmit failed: {exc}")

    time.sleep(0.3)

    # Step 3: add 600123 backup to push toward 90% position
    print("\n=== Step 3: add 600123 backup ===")
    d2 = client.query_stock("600123", "cmd_wt_mairu")
    w2 = d2["wudang"]
    ask1_600123 = float(w2["sell_price"][0])
    ask1_amt_600123 = float(w2["sell_amount"][0])
    print(f"  cur={d2['cur_price']:.2f} ask1={ask1_600123:.2f}({w2['sell_amount'][0]})")
    # buy 4600 shares (within ask1_amt of 31000)
    try:
        r2 = client.buy("600123", 4600, price=ask1_600123)
        print(f"  submit 600123 4600 @ {ask1_600123} -> amount={r2.amount} price={r2.price:.3f}")
    except Exception as exc:
        print(f"  submit failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
