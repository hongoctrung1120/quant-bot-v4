"""Transaction cost model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.05
    slippage_pct: float = 0.01
    spread_pct: float = 0.005


class TransactionCostModel:
    """Compute fees and slippage for simulated execution."""

    def __init__(self, config: CostConfig) -> None:
        self._config = config

    def apply_slippage(self, price: float, side: str) -> float:
        slip = self._config.slippage_pct / 100.0
        spread = self._config.spread_pct / 100.0
        total = slip + spread / 2
        if side.upper() == "BUY":
            return price * (1 + total)
        return price * (1 - total)

    def compute_fee(self, notional: float, is_maker: bool = False) -> float:
        rate = (
            self._config.maker_fee_pct if is_maker
            else self._config.taker_fee_pct
        )
        return notional * (rate / 100.0)

    def total_cost(self, price: float, quantity: float, side: str) -> tuple[float, float]:
        """Return (fill_price, fee)."""
        fill_price = self.apply_slippage(price, side)
        notional = fill_price * quantity
        fee = self.compute_fee(notional, is_maker=False)
        return fill_price, fee
