"""Capital allocation result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass(frozen=True)
class AllocationResult:
    """Output of the capital allocation engine."""

    timestamp: datetime
    total_equity: float
    reserve_amount: float
    trading_capital: float
    asset_allocations: dict[str, float]
    strategy_allocations: dict[str, float]
    regime_adjustments: dict[str, float]
    allocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: str = "fixed"
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        for name, weight in self.asset_allocations.items():
            if weight < 0:
                raise ValueError(
                    f"Negative asset allocation for {name}: {weight}"
                )
        for name, weight in self.strategy_allocations.items():
            if weight < 0:
                raise ValueError(
                    f"Negative strategy allocation for {name}: {weight}"
                )
