"""Simulated execution engine for backtesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from backtest.costs import CostConfig, TransactionCostModel
from backtest.fills import Fill, FillSimulator
from core.models.order import OrderIntent, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class SimulatedOrder:
    intent: OrderIntent
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_fee: float = 0.0


class SimulatedExecutionEngine:
    """Event-driven simulated execution with costs and partial fills."""

    def __init__(
        self,
        cost_config: Optional[CostConfig] = None,
        partial_fills: bool = True,
        fill_probability: float = 0.95,
        seed: int = 42,
    ) -> None:
        self._costs = TransactionCostModel(cost_config or CostConfig())
        self._fills = FillSimulator(partial_fills, fill_probability, seed)
        self._orders: dict[str, SimulatedOrder] = {}
        self._fill_history: list[Fill] = []

    @property
    def fill_history(self) -> list[Fill]:
        return list(self._fill_history)

    async def submit(self, intent: OrderIntent) -> str:
        self._orders[intent.order_id] = SimulatedOrder(
            intent=intent,
            status=OrderStatus.SUBMITTED,
        )
        return intent.order_id

    async def cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def process_intent(
        self,
        intent: OrderIntent,
        market_price: float,
    ) -> Optional[Fill]:
        """Synchronously process order for backtesting."""
        fill_price, fee = self._costs.total_cost(
            market_price, intent.quantity, intent.side.value
        )
        fill = self._fills.simulate(intent, market_price, fill_price, fee)
        if fill is None:
            order = self._orders.get(intent.order_id)
            if order:
                order.status = OrderStatus.REJECTED
            return None

        order = self._orders.setdefault(
            intent.order_id,
            SimulatedOrder(intent=intent),
        )
        order.filled_quantity += fill.quantity
        order.avg_fill_price = fill.price
        order.total_fee += fill.fee
        order.status = fill.status
        self._fill_history.append(fill)
        return fill

    def cancel_all_pending(self) -> int:
        count = 0
        for order in self._orders.values():
            if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                order.status = OrderStatus.CANCELLED
                count += 1
        return count
