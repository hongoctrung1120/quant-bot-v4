"""Unified bar data model for time and dollar bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Union


class BarType(str, Enum):
    TIME = "TIME"
    DOLLAR = "DOLLAR"
    VOLUME = "VOLUME"


@dataclass(frozen=True)
class Bar:
    """Common bar interface consumed by all downstream modules."""

    timestamp: datetime
    start_timestamp: datetime
    end_timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float
    trade_count: int
    bar_type: BarType
    bar_threshold: Union[float, str]

    buy_volume: Optional[float] = None
    sell_volume: Optional[float] = None
    buy_dollar_volume: Optional[float] = None
    sell_dollar_volume: Optional[float] = None
    vwap: Optional[float] = None
    order_flow_imbalance: Optional[float] = None

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(
                f"high ({self.high}) must be >= low ({self.low})"
            )
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open ({self.open}) outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close ({self.close}) outside [low, high]")
        if self.volume < 0:
            raise ValueError(f"volume must be non-negative: {self.volume}")
        if self.dollar_volume < 0:
            raise ValueError(
                f"dollar_volume must be non-negative: {self.dollar_volume}"
            )
        if self.trade_count < 0:
            raise ValueError(
                f"trade_count must be non-negative: {self.trade_count}"
            )

    @property
    def is_closed(self) -> bool:
        """A bar is closed when end_timestamp is set and >= start_timestamp."""
        return self.end_timestamp >= self.start_timestamp
