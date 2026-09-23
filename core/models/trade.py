"""Trade-level market data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Trade:
    """Immutable raw trade record.

    Raw data is never modified after creation. All transformations
    occur downstream in bar construction and feature engines.
    """

    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: TradeSide
    trade_id: str
    exchange: str
    maker_taker: Optional[str] = None
    exchange_timestamp: Optional[datetime] = None
    local_timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}")
        if self.quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.quantity}")
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not self.trade_id:
            raise ValueError("trade_id cannot be empty")

    @property
    def dollar_value(self) -> float:
        return self.price * self.quantity
