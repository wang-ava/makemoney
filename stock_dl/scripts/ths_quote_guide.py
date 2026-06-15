#!/usr/bin/env python3
"""Pull real-time 5-level order book (买一卖一) for every stock in the
20260609 scheme A trading guide, then print actionable limit-price ideas.

Outputs a markdown table to stdout that the operator can paste into the
THS manual-order screen.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import ThsAPIError, ThsClient  # noqa: E402


# (side, code, shares, guide_price) — guide price is the "建议卖出价" or "建议买入价"
SELLS = [
    # rebalance sells (4): partial trim, still hold the rest
    ("sell", "600933", 1400, 14.98),
    ("sell", "002195", 4000, 7.30),
    ("sell", "000551",  800, 15.01),
    ("sell", "601608", 2900, 5.34),
    # outsiders (19): full close — current_shares field ignored, we sell all available
    ("sell", "600475",  900, 14.54),
    ("sell", "300017", 1000, 14.31),
    ("sell", "600133", 2200,  8.09),
    ("sell", "002283", 1800, 10.14),
    ("sell", "601678", 3700,  5.13),
    ("sell", "600575", 5200,  3.80),
    ("sell", "603070", 1300, 16.48),
    ("sell", "300018", 1900, 12.11),
    ("sell", "603686", 1700, 14.64),
    ("sell", "000581", 1400, 18.14),
    ("sell", "603156",  700, 39.09),
    ("sell", "603505", 2200, 17.04),
    ("sell", "603127", 1200, 31.40),
    ("sell", "002277", 6100,  6.71),
    ("sell", "300576",  900, 46.14),
    ("sell", "002939", 8100,  7.90),
    ("sell", "603520", 7500,  9.34),
    ("sell", "002284", 7600, 10.96),
    ("sell", "002272", 4800, 19.91),
]

BUYS = [
    ("buy", "002388", 17400,  5.45),
    ("buy", "300727",  2100, 32.19),
    ("buy", "600009",  1800, 23.72),
    ("buy", "601018", 12200,  3.30),
    ("buy", "300560",  2200, 17.65),
    ("buy", "002250",  2800, 13.83),
    ("buy", "002236",  2300, 16.59),
    ("buy", "000796",  7600,  4.45),
    ("buy", "600765",  2400, 13.37),
    ("buy", "601216",  5700,  5.10),
    ("buy", "002163",  2700,  9.44),
    ("buy", "002350",  1700, 14.59),
    ("buy", "002152",  2400,  9.91),
    ("buy", "601611",  2100, 10.92),
    ("buy", "300619",   500, 39.32),
    ("buy", "601717",  1200, 17.71),
    ("buy", "600958",  2300,  8.82),
    ("buy", "002937",   500, 35.37),
    ("buy", "601106",  4700,  3.54),
    ("buy", "600372",  1500, 10.81),
    ("buy", "601901",  2400,  6.44),
]

BACKUPS = [
    ("300708",  8.70, 10900),
    ("002131",  5.55, 12200),
    ("000922", 12.38,  3500),
    ("002317", 21.84,  1800),
    ("000815", 15.34,  2600),
    ("600601", 12.38,  3100),
    ("603991",105.20,   300),
    ("300067",  5.34,  6400),
    ("600704",  4.69,  7000),
    ("002886", 27.98,  1000),
    ("002157",  3.04,  8600),
    ("603127", 31.43,   800),
    ("600733",  5.97,  3900),
    ("603660", 11.55,  2000),
    ("300377", 13.21,  1700),
    ("600862", 17.33,  1200),
    ("300448",  7.34,  2700),
    ("600901",  6.52,  2900),
    ("601808", 12.55,  1300),
    ("601618",  2.65,  6100),
]


def fmt_price(value: str) -> str:
    if not value:
        return "—"
    try:
        f = float(value)
        if f == 0:
            return "0"
        return f"{f:.3f}".rstrip("0").rstrip(".")
    except ValueError:
        return value


def fmt_amount(value: str, shares_unit: int = 100) -> str:
    if not value:
        return "—"
    try:
        f = float(value)
    except ValueError:
        return value
    hands = f / shares_unit
    return f"{f:>10.0f}股 ({hands:>5.0f}手)"


def query_one(client: ThsClient, code: str, command: str):
    """Return dict with cur_price, buy1-5, sell1-5, amounts."""
    try:
        info = client.query_stock(code, command=command)
    except ThsAPIError as exc:
        return {"code": code, "error": str(exc)}
    w = info["wudang"]
    return {
        "code": code,
        "cur": info["cur_price"],
        "can_buy": info.get("can_buy", 0),
        "buy_prices": [float(p) if p else 0.0 for p in w["buy_price"]],
        "buy_amounts": [float(a) if a else 0.0 for a in w["buy_amount"]],
        "sell_prices": [float(p) if p else 0.0 for p in w["sell_price"]],
        "sell_amounts": [float(a) if a else 0.0 for a in w["sell_amount"]],
    }


def main() -> int:
    client = ThsClient()
    print(f"# THS 实时盘口 — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- 卖出清单 ---
    print("## 1) 卖出清单（先执行）\n")
    print("说明：bid=买一（你按这个挂卖能立即成交），ask=卖一（你按这个要排队）。")
    print("推荐挂单价 = bid1 - 0.01（或 bid1 价位的对手方更紧），以确保 09:35 前成交。\n")
    print("| # | 代码 | 现价 | 买一价 | 买一量 | 卖一价 | 卖一量 | 建议价 | 指南价 | 偏离 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, (side, code, shares, guide) in enumerate(SELLS, 1):
        d = query_one(client, code, "cmd_wt_maichu")
        if "error" in d:
            print(f"| {i} | {code} | err: {d['error']} | | | | | | {guide:.2f} | |")
            continue
        bid1, ask1 = d["buy_prices"][0], d["sell_prices"][0]
        bid1_amt = d["buy_amounts"][0]
        ask1_amt = d["sell_amounts"][0]
        # recommended sell price: hit the bid1 to ensure fill (most aggressive)
        rec = bid1 if bid1 > 0 else guide
        gap = (rec - guide) / guide * 100 if guide > 0 else 0
        print(
            f"| {i} | {code} | {d['cur']:.2f} | {bid1:.2f} | {bid1_amt:.0f} | "
            f"{ask1:.2f} | {ask1_amt:.0f} | **{rec:.2f}** | {guide:.2f} | {gap:+.2f}% |"
        )
        time.sleep(0.15)

    # --- 买入清单 ---
    print("\n## 2) 买入清单（卖出资金到账后执行）\n")
    print("说明：ask=卖一（你按这个挂买能立即吃单）。")
    print("推荐挂单价 = ask1（直接吃卖一保证成交），若 ask1 较远可挂 ask1 - 0.01 等回补。\n")
    print("| # | 代码 | 现价 | 买一价 | 买一量 | 卖一价 | 卖一量 | 建议价 | 指南价 | 偏离 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, (side, code, shares, guide) in enumerate(BUYS, 1):
        d = query_one(client, code, "cmd_wt_mairu")
        if "error" in d:
            print(f"| {i} | {code} | err: {d['error']} | | | | | | {guide:.2f} | |")
            continue
        bid1, ask1 = d["buy_prices"][0], d["sell_prices"][0]
        bid1_amt = d["buy_amounts"][0]
        ask1_amt = d["sell_amounts"][0]
        # recommended buy price: lift the ask1 to ensure fill
        rec = ask1 if ask1 > 0 else guide
        gap = (rec - guide) / guide * 100 if guide > 0 else 0
        print(
            f"| {i} | {code} | {d['cur']:.2f} | {bid1:.2f} | {bid1_amt:.0f} | "
            f"{ask1:.2f} | {ask1_amt:.0f} | **{rec:.2f}** | {guide:.2f} | {gap:+.2f}% |"
        )
        time.sleep(0.15)

    # --- 备选池 ---
    print("\n## 3) 备选股票池（仅在主线无法买入时启动）\n")
    print("| # | 代码 | 现价 | 买一价 | 买一量 | 卖一价 | 卖一量 | 建议价 | 指南价 | 偏离 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, (code, guide, shares) in enumerate(BACKUPS, 1):
        d = query_one(client, code, "cmd_wt_mairu")
        if "error" in d:
            print(f"| {i} | {code} | err: {d['error']} | | | | | | {guide:.2f} | |")
            continue
        bid1, ask1 = d["buy_prices"][0], d["sell_prices"][0]
        bid1_amt = d["buy_amounts"][0]
        ask1_amt = d["sell_amounts"][0]
        rec = ask1 if ask1 > 0 else guide
        gap = (rec - guide) / guide * 100 if guide > 0 else 0
        print(
            f"| {i} | {code} | {d['cur']:.2f} | {bid1:.2f} | {bid1_amt:.0f} | "
            f"{ask1:.2f} | {ask1_amt:.0f} | {rec:.2f} | {guide:.2f} | {gap:+.2f}% |"
        )
        time.sleep(0.15)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
