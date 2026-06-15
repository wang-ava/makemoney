#!/usr/bin/env python3
"""用买一/卖一价重挂卖单 + 提交12笔失败买单"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import ThsClient, normalize_stock_code

# 7笔未成交卖单
PENDING_SELLS = [
    ("002388.SZ", "002388", 17100, 5.27, "6035205030"),
    ("600372.SH", "600372", 1500, 10.72, "6035205031"),
    ("002152.SZ", "002152", 2400, 9.88, "6035171686"),
    ("000796.SZ", "000796", 7600, 4.48, "6035094005"),
    ("300560.SZ", "300560", 2200, 16.59, "6035172635"),
    ("000551.SZ", "000551", 5400, 15.07, "6035206222"),
    ("002272.SZ", "002272", 4400, 20.58, "6035217770"),
]

# 12笔失败买单（按金额降序）
FAILED_BUYS = [
    ("603020.SH", "603020", 3200, 11.97),
    ("600598.SH", "600598", 2400, 13.97),
    ("000581.SZ", "000581", 1500, 17.88),
    ("002277.SZ", "002277", 3800, 6.61),
    ("601108.SH", "601108", 3100, 7.50),
    ("300518.SZ", "300518", 700, 28.22),
    ("600970.SH", "600970", 2200, 9.02),
    ("002590.SZ", "002590", 1700, 11.23),
    ("300649.SZ", "300649", 600, 26.50),
    ("300243.SZ", "300243", 700, 19.10),
    ("600320.SH", "600320", 3100, 4.39),
    ("002380.SZ", "002380", 300, 37.45),
]


def fetch_bid_ask(client, stock_code, level=1):
    """取买一/卖一价"""
    try:
        info = client.query_stock(stock_code)
        wudang = info.get("wudang", {})
        bid = wudang.get("buy_price", [""] * 5)[level - 1]
        ask = wudang.get("sell_price", [""] * 5)[level - 1]
        return float(bid) if bid else 0.0, float(ask) if ask else 0.0, float(info.get("cur_price", 0))
    except Exception as exc:
        print(f"  ⚠️ 取{stock_code}行情失败: {exc}")
        return 0.0, 0.0, 0.0


def main():
    client = ThsClient()
    print("=" * 60)
    print("阶段1: 查询7只待重挂卖单的买一价")
    print("=" * 60)
    new_sell_prices = {}
    for ts_code, code6, shares, old_price, order_id in PENDING_SELLS:
        bid, ask, cur = fetch_bid_ask(client, code6)
        # 卖价用 买一价（取当前最高bid，hit the bid），如买一为0则用现价-0.1%
        if bid > 0:
            sell_price = bid  # 买一价
        elif cur > 0:
            sell_price = round(cur * 0.999, 2)  # 现价折让0.1%
        else:
            sell_price = old_price  # 回退
        new_sell_prices[order_id] = (code6, shares, sell_price, old_price, bid, ask, cur)
        print(f"  {ts_code}: 现价{cur} 买一{bid} 卖一{ask} → 改挂 {sell_price} (原{old_price})")

    print()
    print("=" * 60)
    print("阶段2: 撤掉7笔未成交卖单")
    print("=" * 60)
    for order_id, (code6, shares, new_px, old_px, bid, ask, cur) in new_sell_prices.items():
        try:
            r = client.cancel_order(order_id, "20260610")
            print(f"  ✅ 撤单 {order_id} ({code6}) - {r}")
        except Exception as e:
            print(f"  ⚠️ 撤单 {order_id} ({code6}) 失败: {e}")

    print()
    print("=" * 60)
    print("阶段3: 用买一价重新挂卖单")
    print("=" * 60)
    sell_submitted = 0
    for order_id, (code6, shares, new_px, old_px, bid, ask, cur) in new_sell_prices.items():
        try:
            r = client.sell(code6, shares, new_px)
            print(f"  ✅ 卖出 {code6} {shares}股 @{new_px} - {r.payload.get('ret_msg', '')}")
            sell_submitted += 1
        except Exception as e:
            print(f"  ❌ {code6} 卖出 {shares}股 @{new_px} 失败: {e}")

    print()
    print("=" * 60)
    print("阶段4: 查询12只失败买单的卖一价（市价）")
    print("=" * 60)
    new_buy_prices = {}
    for ts_code, code6, shares, old_price in FAILED_BUYS:
        bid, ask, cur = fetch_bid_ask(client, code6)
        # 买价用 卖一价（lift the offer），快速成交
        if ask > 0:
            buy_price = ask
        elif cur > 0:
            buy_price = round(cur * 1.001, 2)
        else:
            buy_price = old_price
        new_buy_prices[ts_code] = (code6, shares, buy_price, old_price, bid, ask, cur)
        print(f"  {ts_code}: 现价{cur} 买一{bid} 卖一{ask} → 改挂 {buy_price} (原{old_price})")

    print()
    print("=" * 60)
    print("阶段5: 提交12笔买单（用卖一价）")
    print("=" * 60)
    buy_submitted = 0
    buy_failed = 0
    for ts_code, (code6, shares, buy_px, old_px, bid, ask, cur) in new_buy_prices.items():
        try:
            r = client.buy(code6, shares, buy_px)
            print(f"  ✅ {ts_code} 买入 {shares}股 @{buy_px} - {r.payload.get('ret_msg', '')}")
            buy_submitted += 1
        except Exception as e:
            print(f"  ❌ {ts_code} 买入 {shares}股 @{buy_px} 失败: {str(e)[:80]}")
            buy_failed += 1

    print()
    print("=" * 60)
    print(f"汇总: 卖单重挂 {sell_submitted}/7 成功, 买单提交 {buy_submitted}/12 成功（{buy_failed} 失败）")
    print("=" * 60)


if __name__ == "__main__":
    main()
