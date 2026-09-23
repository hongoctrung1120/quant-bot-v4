"""Transaction cost model: spread, slippage, commission, swap.

Spread is charged through the fill price (buy at ask, sell at bid) rather than
as a separate fee, so it must not also be added as a commission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from fx import sessions
from fx.instruments import FxInstrument, RateBook, pip_value_per_lot


@dataclass
class FxCostConfig:
    spread_stress_multiplier: float = 1.0
    session_spread_enabled: bool = True
    slippage_pips: float = 0.2
    commission_per_lot_per_side: float = 0.0
    swap_enabled: bool = True


class FxCostModel:
    def __init__(self, config: FxCostConfig | None = None) -> None:
        self.config = config or FxCostConfig()

    def spread_pips(self, instrument: FxInstrument, timestamp: datetime) -> float:
        spread = instrument.typical_spread_pips * self.config.spread_stress_multiplier
        if self.config.session_spread_enabled:
            spread *= sessions.spread_multiplier(timestamp)
        return spread

    def fill_price(
        self,
        instrument: FxInstrument,
        mid_price: float,
        is_buy: bool,
        timestamp: datetime,
        include_slippage: bool = True,
    ) -> float:
        """Buys fill at ask, sells at bid; slippage always moves against us."""
        half_spread = self.spread_pips(instrument, timestamp) / 2.0
        adverse = half_spread + (self.config.slippage_pips if include_slippage else 0.0)
        signed = adverse if is_buy else -adverse
        return instrument.offset_price(mid_price, signed)

    def implicit_cost(
        self,
        instrument: FxInstrument,
        mid_price: float,
        fill_price: float,
        lots: float,
        rates: RateBook,
    ) -> float:
        """Account-currency value of the spread and slippage paid on a fill.

        This is already embedded in the fill price, so it must be reported
        rather than charged again — but it is usually the dominant cost, and a
        report that shows zero costs because commission is zero is misleading.
        """
        pips = abs(fill_price - mid_price) / instrument.pip_size
        return pips * pip_value_per_lot(instrument, rates) * abs(lots)

    def commission(self, instrument: FxInstrument, lots: float) -> float:
        per_lot = (
            instrument.commission_per_lot_per_side
            or self.config.commission_per_lot_per_side
        )
        return abs(lots) * per_lot

    def swap_amount(
        self,
        instrument: FxInstrument,
        lots: float,
        timestamp: datetime,
        rates: RateBook,
    ) -> float:
        """Overnight financing in account currency. Negative is a debit."""
        if not self.config.swap_enabled:
            return 0.0
        pips = (
            instrument.swap_long_pips_per_day
            if lots > 0
            else instrument.swap_short_pips_per_day
        )
        pip_value = pip_value_per_lot(instrument, rates)
        return pips * pip_value * abs(lots) * sessions.swap_multiplier(timestamp)
