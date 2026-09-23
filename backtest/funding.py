"""Funding-cost accrual for perpetual-futures backtests."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass(frozen=True)
class FundingConfig:
    rate_pct: float = 0.01
    interval_hours: int = 8

class FundingModel:
    """Apply deterministic funding at fixed UTC intervals.

    Positive funding means longs pay shorts; negative funding means shorts pay longs.
    """
    def __init__(self, config: FundingConfig) -> None:
        if config.interval_hours <= 0:
            raise ValueError("funding interval_hours must be positive")
        self.config = config

    def due_events(self, previous: datetime | None, current: datetime):
        if previous is None:
            return []
        interval = timedelta(hours=self.config.interval_hours)
        first = previous.replace(minute=0, second=0, microsecond=0)
        events = []
        t = first + interval
        while t <= current:
            events.append(t)
            t += interval
        return events

    def cost(self, signed_quantity: float, mark_price: float) -> float:
        notional = abs(signed_quantity) * mark_price
        rate = self.config.rate_pct / 100.0
        return notional * rate if signed_quantity > 0 else -notional * rate
