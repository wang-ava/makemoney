#!/usr/bin/env python3
"""Cancel 11 stuck sell orders and re-quote at the current bid1 (or lower
if bid1 is too thin to absorb the full position).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import (  # noqa: E402
    ThsAPIError,
    ThsClient,
    normalize_stock_code,
)


# (order_id, order_date, code, name, original_shares)
STUCK = [
    ("6032841622", "20260610", "600933", "爱柯迪", 1400),
    ("6032833629", "20260610", "002195", "岩山科技", 4000),
    ("6032844474", "20260610", "000551", "创元科技", 800),
    ("6032803959", "20260610", "300017", "网宿科技", 1000),
    ("6032840770", "20260610", "002283", "天润工业", 1800),
    ("6032820796", "20260610", "600575", "淮河能源", 5200),
    ("6032788813", "20260610", "300018", "中元股份", 1900),
    ("6032834033", "20260610", "603686", "福龙马", 1700),
    ("6032842421", "20260610", "603127", "昭衍新药", 1200),
    ("6032800763", "20260610", "603520", "司太立", 7500),
    ("6032834381", "20260610", "002284", "亚太股份", 7600),
]


def main() -> int:
    client = ThsClient()
    print("=== Step 1: cancel 11 stuck sells ===")
    for oid, od, code, name, shares in STUCK:
        try:
            client.cancel_order(oid, od)
            print(f"  canceled {oid} {code} {name}")
        except ThsAPIError as exc:
            print(f"  cancel FAILED {oid} {code}: {exc}")
        time.sleep(0.2)

    time.sleep(2.0)

    print("\n=== Step 2: re-query 5-level book and re-quote at aggressive bid ===")
    print("code | name | orig | bid1 | bid1_amt | bid2 | bid2_amt | new_price | new_shares")
    re_quote: list[tuple[str, str, int, float]] = []
    for oid, od, code, name, orig_shares in STUCK:
        try:
            info = client.query_stock(code, "cmd_wt_maichu")
        except ThsAPIError as exc:
            print(f"  query FAILED {code}: {exc}")
            continue
        bid_prices = info["wudang"]["buy_price"]
        bid_amounts = info["wudang"]["buy_amount"]
        bids = [(float(p) if p else 0.0, float(a) if a else 0.0) for p, a in zip(bid_prices, bid_amounts)]

        # walk down bids until we accumulate enough to absorb orig_shares
        cumulative = 0
        chosen_price = None
        chosen_idx = 0
        for idx, (p, a) in enumerate(bids):
            if p <= 0:
                continue
            cumulative += a
            if cumulative >= orig_shares:
                chosen_price = p
                chosen_idx = idx
                break
        if chosen_price is None:
            # bids too thin, use the lowest bid we saw (max concession)
            nonzero = [(p, a) for p, a in bids if p > 0]
            if not nonzero:
                print(f"  NO BIDS for {code}, skipping")
                continue
            chosen_price = nonzero[-1][0]
            chosen_idx = len(nonzero) - 1

        # aggressive: for illiquid names, drop 0.01 below chosen bid to ensure fill
        if chosen_idx >= 2:
            aggressive_price = round(chosen_price - 0.01, 2)
        else:
            aggressive_price = chosen_price

        # determine remaining shares: for 002284 (partial), need current balance
        re_shares = orig_shares
        if code == "002284":
            positions = client.get_positions()
            for p in positions:
                if normalize_stock_code(p.stock_code) == code:
                    re_shares = int(float(p.balance))
                    break

        re_quote.append((code, name, re_shares, aggressive_price))
        b1p, b1a = bids[0] if bids else (0, 0)
        b2p, b2a = bids[1] if len(bids) > 1 else (0, 0)
        print(
            f"  {code} {name} | orig={orig_shares} | "
            f"bid1={b1p:.2f}({b1a:.0f}) bid2={b2p:.2f}({b2a:.0f}) | "
            f"new={aggressive_price:.2f} shares={re_shares}"
        )
        time.sleep(0.15)

    print("\n=== Step 3: submit re-quoted sells ===")
    for code, name, shares, price in re_quote:
        try:
            r = client.sell(code, shares, price=price)
            print(f"  resubmit {code} {name} {shares} @ {price} -> amount={r.amount} price={r.price:.3f}")
        except ThsAPIError as exc:
            print(f"  resubmit FAILED {code}: {exc}")
        time.sleep(0.3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
