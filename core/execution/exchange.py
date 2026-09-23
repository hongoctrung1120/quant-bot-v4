"""Exchange adapter interface and simulated exchange."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ExchangeAdapter(ABC):
    """Abstract exchange interface."""

    @abstractmethod
    async def get_balance(self) -> dict[str, float]:
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str) -> dict:
        ...

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> dict:
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        ...

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> dict:
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> list[dict]:
        ...


class SimulatedExchange(ExchangeAdapter):
    """In-memory simulated exchange for backtest/paper."""

    def __init__(self, initial_balance: float = 100000.0) -> None:
        self._balance = {"USDT": initial_balance}
        self._positions: list[dict] = []
        self._orders: dict[str, dict] = {}

    async def get_balance(self) -> dict[str, float]:
        return dict(self._balance)

    async def get_positions(self) -> list[dict]:
        return list(self._positions)

    async def get_orderbook(self, symbol: str) -> dict:
        return {"bids": [], "asks": [], "symbol": symbol}

    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> dict:
        order_id = f"sim-{len(self._orders)}"
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "price": price,
            "status": "open",
        }
        self._orders[order_id] = order
        return order

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order:
            order["status"] = "cancelled"
            return True
        return False

    async def get_order(self, symbol: str, order_id: str) -> dict:
        return self._orders.get(order_id, {})

    async def get_open_orders(self, symbol: str) -> list[dict]:
        return [
            o for o in self._orders.values()
            if o.get("symbol") == symbol and o.get("status") == "open"
        ]
