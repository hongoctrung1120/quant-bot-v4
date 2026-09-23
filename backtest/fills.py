"""Fill simulation for backtesting."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

from core.models.order import OrderIntent, OrderSide, OrderStatus


@dataclass(frozen=True)
class Fill:
    """Simulated order fill."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.FILLED


class FillSimulator:
    """Simulate order fills with partial fill support."""

    def __init__(
        self,
        partial_fills_enabled: bool = True,
        fill_probability: float = 0.95,
        seed: int = 42,
    ) -> None:
        self._partial = partial_fills_enabled
        self._fill_prob = fill_probability
        self._rng = random.Random(seed)

    def simulate(
        self,
        intent: OrderIntent,
        market_price: float,
        fill_price: float,
        fee: float,
    ) -> Fill | None:
        if self._rng.random() > self._fill_prob:
            return None

        quantity = intent.quantity
        status = OrderStatus.FILLED

        if self._partial and self._rng.random() < 0.2:
            quantity = intent.quantity * self._rng.uniform(0.5, 0.95)
            status = OrderStatus.PARTIALLY_FILLED

        partial_fee = fee * (quantity / intent.quantity)

        return Fill(
            order_id=intent.order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            price=fill_price,
            fee=partial_fee,
            timestamp=intent.timestamp,
            status=status,
        )
