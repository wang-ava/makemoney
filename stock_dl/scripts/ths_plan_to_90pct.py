#!/usr/bin/env python3
"""Pull ask1 for the 21 buys + 2 backups, sized to reach ~90% position."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import ThsClient  # noqa: E402


# Main 21 buys — guide shares
BUYS = [
    ("002388", 17400,  5.45),
    ("300727",  2100, 32.19),
    ("600009",  1800, 23.72),
    ("601018", 12200,  3.30),
    ("300560",  2200, 17.65),
    ("002250",  2800, 13.83),
    ("002236",  2300, 16.59),
    ("000796",  7600,  4.45),
    ("600765",  2400, 13.37),
    ("601216",  5700,  5.10),
    ("002163",  2700,  9.44),
    ("002350",  1700, 14.59),
    ("002152",  2400,  9.91),
    ("601611",  2100, 10.92),
    ("300619",   500, 39.32),
    ("601717",  1200, 17.71),
    ("600958",  2300,  8.82),
    ("002937",   500, 35.37),
    ("601106",  4700,  3.54),
    ("600372",  1500, 10.81),
    ("601901",  2400,  6.44),
]

# Backup candidates — re-quote, pick cheap ones to fill the gap
BACKUPS = [
    ("002157", 8600,  3.04),  # ~26K
    ("601618", 6100,  2.65),  # ~16K
    ("600123", 4600,  6.96),  # ~32K
]


def main() -> int:
    client = ThsClient()

    # also pull current account state
    fund = client.get_fund()
    cash = float(fund.get("kyje", 0))
    total_asset = float(fund.get("zzc", 0) or fund.get("zcz", 0))
    position_value = total_asset - cash
    target_position = 0.90 * total_asset
    need_to_buy = target_position - position_value
    print(f"# account: cash={cash:.2f}, total={total_asset:.2f}, pos={position_value:.2f}")
    print(f"# target 90%: pos_target={target_position:.2f}, need_to_buy={need_to_buy:.2f}\n")

    print("## 21 main buys — fresh ask1")
    print("code | guide_shares | guide_price | ask1 | ask1_amt | cost @ask1 | new_shares | new_cost")
    main_total = 0.0
    main_rows = []
    for code, shares, guide_p in BUYS:
        try:
            d = client.query_stock(code, "cmd_wt_mairu")
        except Exception as exc:
            print(f"  {code} | QUERY FAILED: {type(exc).__name__}: {exc} — skipping")
            time.sleep(0.2)
            continue
        if "wudang" not in d:
            print(f"  {code} | UNEXPECTED RESPONSE: {d} — skipping")
            time.sleep(0.2)
            continue
        w = d["wudang"]
        ask1 = float(w["sell_price"][0]) if w["sell_price"][0] else 0.0
        ask1_amt = float(w["sell_amount"][0]) if w["sell_amount"][0] else 0.0
        cost = shares * ask1
        main_total += cost
        # determine executable shares: if ask1_amt < shares, step up to ask2
        new_shares = shares
        if ask1_amt < shares and ask1_amt > 0:
            cum = ask1_amt
            for i in range(1, 5):
                p = float(w["sell_price"][i]) if w["sell_price"][i] else 0.0
                a = float(w["sell_amount"][i]) if w["sell_amount"][i] else 0.0
                if p <= 0 or a <= 0:
                    break
                cum += a
                if cum >= shares:
                    break
            new_shares = int(cum // 100 * 100)
        new_cost = new_shares * ask1
        main_rows.append((code, new_shares, ask1))
        print(
            f"  {code} | {shares:>6} @ {guide_p:.2f} | ask1={ask1:.2f}({ask1_amt:.0f}) | "
            f"cost={cost:.0f} | new_shares={new_shares} cost={new_cost:.0f}"
        )
        time.sleep(0.15)

    # recalc total with new_shares
    main_total = sum(s * p for _, s, p in main_rows)
    print(f"\n  main 21 buys subtotal @ fresh ask1 = {main_total:.0f}")

    gap = need_to_buy - main_total
    print(f"\n  need to spend {need_to_buy:.0f}, main covers {main_total:.0f}, gap = {gap:.0f}")

    if gap > 5000:
        print("\n## Backups to close the gap")
        backup_rows = []
        for code, shares, guide_p in BACKUPS:
            try:
                d = client.query_stock(code, "cmd_wt_mairu")
            except Exception as exc:
                print(f"  {code} | QUERY FAILED: {type(exc).__name__}: {exc} — skipping")
                time.sleep(0.2)
                continue
            if "wudang" not in d:
                print(f"  {code} | UNEXPECTED RESPONSE: {d} — skipping")
                time.sleep(0.2)
                continue
            w = d["wudang"]
            ask1 = float(w["sell_price"][0]) if w["sell_price"][0] else 0.0
            ask1_amt = float(w["sell_amount"][0]) if w["sell_amount"][0] else 0.0
            cost = shares * ask1
            cum = ask1_amt
            new_shares = shares
            for i in range(1, 5):
                p = float(w["sell_price"][i]) if w["sell_price"][i] else 0.0
                a = float(w["sell_amount"][i]) if w["sell_amount"][i] else 0.0
                if p <= 0 or a <= 0:
                    break
                cum += a
                if cum >= shares:
                    break
            new_shares = int(cum // 100 * 100)
            new_cost = new_shares * ask1
            backup_rows.append((code, new_shares, ask1, new_cost))
            print(f"  {code} | {shares:>6} @ {guide_p:.2f} | ask1={ask1:.2f}({ask1_amt:.0f}) | "
                  f"new_shares={new_shares} cost={new_cost:.0f}")
            time.sleep(0.15)
            if sum(c for _, _, _, c in backup_rows) >= gap:
                break

        # final plan
        print("\n## FINAL EXECUTION PLAN")
        print("## Main 21 buys")
        for code, s, p in main_rows:
            print(f"buy {code} {s} @ {p:.2f} (cost={s*p:.0f})")
        print("## Backups to fill gap")
        running = main_total
        for code, s, p, c in backup_rows:
            if running >= need_to_buy:
                break
            print(f"buy {code} {s} @ {p:.2f} (cost={c:.0f})")
            running += c
        print(f"\n  final projected total buy = {running:.0f}, target = {need_to_buy:.0f}, cash_left = {cash - running:.0f}")
    else:
        print("\n  gap is small enough; main 21 alone is fine.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
