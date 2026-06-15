#!/usr/bin/env python3
"""等资金到账后自动重试12笔失败买单的轮询脚本"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.ths_client import ThsClient, normalize_stock_code

# 12笔失败买单（按金额降序，便于现金不足时优先提交大单）
RETRY_BUYS = [
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
TOTAL_NEEDED = sum(shares * price for _, _, shares, price in RETRY_BUYS)
CASH_THRESHOLD = 270000  # 等可用资金 >= 27万才动手（259823 + 1万缓冲）
POLL_INTERVAL = 30  # 秒
MAX_WAIT_MIN = 60


def main() -> int:
    client = ThsClient()
    start_ts = time.time()
    attempt = 0
    while True:
        attempt += 1
        elapsed_min = (time.time() - start_ts) / 60
        if elapsed_min > MAX_WAIT_MIN:
            print(f"⏰ 超过最大等待 {MAX_WAIT_MIN} 分钟，停止轮询。")
            return 1

        try:
            fund = client.get_fund()
        except Exception as exc:
            print(f"[{attempt}] 获取资金失败: {exc}")
            time.sleep(POLL_INTERVAL)
            continue

        cash = float(fund.get("kyje", 0))
        frozen = float(fund.get("djje", 0))
        asset = float(fund.get("zzc", 0))
        print(f"[{attempt}] {time.strftime('%H:%M:%S')} | 可用:{cash:,.2f} 冻结:{frozen:,.2f} 总资产:{asset:,.2f} | 需要:{TOTAL_NEEDED:,.2f}")

        if cash >= CASH_THRESHOLD:
            print(f"\n✅ 资金到账（{cash:,.2f} >= {CASH_THRESHOLD:,.0f}），开始按金额降序提交 {len(RETRY_BUYS)} 笔买单…\n")
            submitted = failed = 0
            for ts_code, code6, shares, price in RETRY_BUYS:
                amt = shares * price
                if cash < amt:
                    print(f"  ⏭️  跳过 {ts_code} {shares}股 @{price} ({amt:,.0f}元) - 现金不足")
                    failed += 1
                    continue
                try:
                    r = client.buy(code6, shares, price)
                    print(f"  ✅ {ts_code} {shares}股 @{price} - {r.payload}")
                    submitted += 1
                    cash -= amt
                except Exception as e:
                    msg = str(e)
                    print(f"  ❌ {ts_code} {shares}股 @{price} - {msg[:100]}")
                    if "余额不够" in msg or "资金" in msg:
                        # 余额不够，停止后续尝试
                        print(f"  ⏹️ 资金不足，停止")
                        failed += len(RETRY_BUYS) - submitted - failed
                        break
                    failed += 1
            print(f"\n📊 提交完成: 成功 {submitted}, 失败 {failed}")
            return 0 if failed == 0 else 1

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
