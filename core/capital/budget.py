"""Capital budget computation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapitalBudget:
    """Portfolio capital layers."""

    total_equity: float
    reserve_amount: float
    trading_capital: float

    @staticmethod
    def from_equity(
        total_equity: float,
        reserve_pct: float,
        max_trading_capital_pct: float,
    ) -> CapitalBudget:
        reserve = total_equity * (reserve_pct / 100.0)
        max_trading = total_equity * (max_trading_capital_pct / 100.0)
        available = total_equity - reserve
        trading = min(available, max_trading)
        return CapitalBudget(
            total_equity=total_equity,
            reserve_amount=reserve,
            trading_capital=max(trading, 0.0),
        )
