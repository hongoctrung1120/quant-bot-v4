"""Strategy signal data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Signal:
    """Strategy output signal. Does NOT contain position size."""

    symbol: str
    direction: SignalDirection
    confidence: float
    timestamp: datetime
    strategy_id: str
    regime: str
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_reference: Optional[float] = None
    stop_reference: Optional[float] = None
    target_reference: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1]: {self.confidence}"
            )
