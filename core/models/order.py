"""Order intent and order state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class OrderIntent:
    """Pre-execution order specification.

    Created by position sizing engine. Consumed by execution engine.
    Strategy engine must NEVER create orders directly.
    """

    symbol: str
    side: OrderSide
    quantity: float
    entry_type: OrderType
    timestamp: datetime
    strategy_id: str
    allocation_id: str
    risk_id: str
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def entry_reference(self) -> Optional[float]:
        """Backward-compatible entry price reference."""
        return self.limit_price

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive: {self.quantity}")
