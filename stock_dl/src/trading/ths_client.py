from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ThsAPIError(RuntimeError):
    """Raised when the THS simulated trading API returns an unusable response."""


@dataclass
class ThsOrder:
    time: str
    order_id: str
    order_date: str
    stock_code: str
    stock_name: str
    price: str
    amount: str
    side: str
    status: str
    market_type: str


@dataclass
class ThsTradeResult:
    action: str
    stock_code: str
    price: float
    amount: int
    payload: dict[str, Any]


@dataclass
class ThsPosition:
    stock_code: str
    stock_name: str
    balance: str
    available: str
    amount: str
    price: str
    profit_loss: str
    profit_loss_ratio: str
    market_type: str
    gdzh: str
    market_code: str


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY='value' lines without adding a python-dotenv dependency."""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def normalize_stock_code(ts_code: str) -> str:
    """Convert 600000.SH / 000001.SZ into the six digit THS stock code."""
    code = str(ts_code).strip().upper()
    if "." in code:
        code = code.split(".", 1)[0]
    if not code.isdigit():
        raise ValueError(f"Invalid stock code: {ts_code}")
    return code.zfill(6)


def round_lot_shares(shares: int | float) -> int:
    """A-share order quantity rounded down to 100-share lots."""
    return int(float(shares) // 100 * 100)


class ThsClient:
    def __init__(
        self,
        cookie: str | None = None,
        gdzh: str | None = None,
        user_agent: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        load_env_file()
        self.cookie = cookie or os.getenv("THS_COOKIE", "").strip()
        self.gdzh = (gdzh or os.getenv("THS_GDZH", "")).strip() or None
        self.timeout = timeout

        if not self.cookie:
            raise ThsAPIError("Missing THS_COOKIE in .env")

        self.session = requests.Session()
        self.headers = {
            "host": "mncg.10jqka.com.cn",
            "origin": "https://mncg.10jqka.com.cn",
            "referer": "https://mncg.10jqka.com.cn/cgiwt/index/index",
            "cookie": self.cookie,
            "user-agent": user_agent
            or os.getenv("THS_USER_AGENT", "").strip()
            or "Mozilla/5.0",
        }
        self.base_url = "https://mncg.10jqka.com.cn/cgiwt/delegate"

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                response = self.session.post(
                    url,
                    headers=self.headers,
                    data=data,
                    timeout=self.timeout,
                )
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == 3:
                    raise
                time.sleep(0.8 * (attempt + 1))
        else:
            raise ThsAPIError(f"THS request failed: {last_exc}")
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise ThsAPIError("THS returned non-JSON; the cookie may be expired.") from exc
        if not isinstance(payload, dict):
            raise ThsAPIError("THS returned an unexpected response shape.")
        return payload

    @staticmethod
    def _assert_success(payload: dict[str, Any], action: str) -> None:
        error_code = payload.get("errorcode")
        if error_code is None:
            return
        if str(error_code) in {"0", ""}:
            return
        message = payload.get("errormsg") or payload.get("message") or payload
        raise ThsAPIError(f"THS {action} failed: {message}")

    def _updateclass(self, command: str, update_class: str) -> dict[str, Any]:
        payload = self._post(
            "updateclass/",
            {"type": command, "updateClass": update_class},
        )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ThsAPIError(f"THS updateclass failed: {payload}")
        return result

    def get_fund(self) -> dict[str, Any]:
        result = self._updateclass("cmd_wt_mairu", "qryzijin|")
        return result["qryzijin"]["result"]["data"]

    def get_positions(self) -> list[ThsPosition]:
        result = self._updateclass("cmd_wt_mairu", "qryzijin|qryChicang|")
        rows = result["qryChicang"]["result"].get("list", [])
        positions = [
            ThsPosition(
                stock_code=row.get("d_2102", ""),
                stock_name=row.get("d_2103", ""),
                balance=row.get("d_2117", ""),
                available=row.get("d_2121", ""),
                amount=row.get("d_2164", ""),
                price=row.get("d_2122", ""),
                profit_loss=row.get("d_2147", ""),
                profit_loss_ratio=row.get("d_3616", ""),
                market_type=row.get("d_2108", ""),
                gdzh=row.get("d_2106", ""),
                market_code=row.get("d_2167", ""),
            )
            for row in rows
        ]
        if positions and not self.gdzh:
            self.gdzh = positions[0].gdzh
        return positions

    def get_orders(self) -> list[ThsOrder]:
        result = self._updateclass("cmd_qu_chedan", "qryzijin|qryChedan|")
        rows = result["qryChedan"]["result"].get("list", [])
        return [
            ThsOrder(
                time=row.get("d_2140", ""),
                order_id=row.get("d_2135", ""),
                order_date=row.get("d_2139", ""),
                stock_code=row.get("d_2102", ""),
                stock_name=row.get("d_2103", ""),
                price=row.get("d_2127", ""),
                amount=row.get("d_2126", ""),
                side=row.get("d_2109", ""),
                status=row.get("d_2105", ""),
                market_type=row.get("d_2108", ""),
            )
            for row in rows
        ]

    def query_stock(self, stock_code: str, command: str = "cmd_wt_mairu") -> dict[str, Any]:
        stock_code = normalize_stock_code(stock_code)
        payload = self._post(
            "qrystock/",
            {"type": command, "stockcode": stock_code},
        )
        try:
            info = payload["result"]["data"]
        except KeyError as exc:
            raise ThsAPIError(f"THS stock query failed: {payload}") from exc
        return {
            "cur_price": float(info["st_price"]),
            "can_buy": int(info.get("canbuy", 0)),
            "market_code": info["mkcode"],
            "wudang": {
                "buy_price": [info.get(f"mrjw{i}", "") for i in range(1, 6)],
                "buy_amount": [info.get(f"mrsl{i}", "") for i in range(1, 6)],
                "sell_price": [info.get(f"mcjw{i}", "") for i in range(1, 6)],
                "sell_amount": [info.get(f"mcsl{i}", "") for i in range(1, 6)],
            },
            "raw": info,
        }

    def _find_position(self, stock_code: str) -> ThsPosition | None:
        stock_code = normalize_stock_code(stock_code)
        for position in self.get_positions():
            if normalize_stock_code(position.stock_code) == stock_code:
                return position
        return None

    def _resolve_gdzh(self, stock_code: str, side: str, gdzh: str | None = None) -> str:
        if gdzh:
            return gdzh
        if side == "sell":
            position = self._find_position(stock_code)
            if position and position.gdzh:
                return position.gdzh
        discovered = self.discover_gdzh()
        if discovered:
            return discovered
        raise ThsAPIError("Missing THS_GDZH; set it in .env before buying from an empty account")

    def submit_order(
        self,
        stock_code: str,
        side: str,
        price: float | None,
        amount: int,
        gdzh: str | None = None,
    ) -> ThsTradeResult:
        """Submit a simulated THS buy/sell order.

        This follows the public ths_simulated_API request shape:
        /delegate/tradestock/ with type, mkcode, gdzh, stockcode, price, amount.
        """
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        stock_code = normalize_stock_code(stock_code)
        amount = round_lot_shares(amount)
        if amount <= 0:
            raise ValueError("amount must be at least 100 shares")

        command = "cmd_wt_mairu" if side == "buy" else "cmd_wt_maichu"
        stock_info = self.query_stock(stock_code, command=command)
        order_price = float(stock_info["cur_price"] if price is None else price)
        if order_price <= 0:
            raise ValueError("price must be positive")

        payload = self._post(
            "tradestock/",
            {
                "type": command,
                "mkcode": stock_info["market_code"],
                "gdzh": self._resolve_gdzh(stock_code, side, gdzh),
                "stockcode": stock_code,
                "price": f"{order_price:.3f}",
                "amount": amount,
            },
        )
        self._assert_success(payload, f"{side} {stock_code}")
        return ThsTradeResult(
            action=side,
            stock_code=stock_code,
            price=order_price,
            amount=amount,
            payload=payload,
        )

    def buy(self, stock_code: str, amount: int, price: float | None = None) -> ThsTradeResult:
        return self.submit_order(stock_code, "buy", price, amount)

    def sell(self, stock_code: str, amount: int, price: float | None = None) -> ThsTradeResult:
        return self.submit_order(stock_code, "sell", price, amount)

    def cancel_order(self, order_id: str, order_date: str | None = None) -> dict[str, Any]:
        if not order_date:
            for order in self.get_orders():
                if order.order_id == str(order_id):
                    order_date = order.order_date
                    break
        if not order_date:
            raise ThsAPIError(f"Cannot find order_date for order_id={order_id}")

        payload = self._post(
            "cancelDelegated/",
            {"htbh": str(order_id), "wtrq": str(order_date)},
        )
        self._assert_success(payload, f"cancel {order_id}")
        return payload

    def cancel_all_orders(self) -> list[dict[str, Any]]:
        results = []
        for order in self.get_orders():
            results.append(self.cancel_order(order.order_id, order.order_date))
        return results

    def discover_gdzh(self) -> str | None:
        if self.gdzh:
            return self.gdzh
        self.get_positions()
        return self.gdzh
