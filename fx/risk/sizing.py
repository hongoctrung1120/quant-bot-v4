"""Pip-based position sizing with broker lot granularity.

The critical rule: when the smallest tradable lot would risk more than the
configured budget, the trade is REFUSED, not rounded up. Silently taking an
oversized position is how a $100 account ends up risking 2% per trade while
its config claims 0.5%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fx.instruments import FxInstrument, RateBook, pip_value_per_lot
from fx.risk.exposure import projected_currency_exposure
from fx.risk.governor import RiskLimits


@dataclass
class SizingResult:
    lots: float = 0.0
    risk_amount: float = 0.0
    stop_pips: float = 0.0
    pip_value_per_lot: float = 0.0
    min_lot_risk_pct: float = 0.0
    rejected: Optional[str] = None
    capped_by: str = ""

    @property
    def accepted(self) -> bool:
        return self.rejected is None and self.lots > 0


class FxPositionSizer:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def size(
        self,
        instrument: FxInstrument,
        entry_price: float,
        stop_price: float,
        equity: float,
        rates: RateBook,
        is_long: bool,
        risk_scaler: float = 1.0,
        free_margin: float = 0.0,
        leverage: float = 100.0,
        current_exposure: Optional[dict[str, float]] = None,
        current_currency_risk: Optional[dict[str, float]] = None,
        gross_exposure: float = 0.0,
    ) -> SizingResult:
        if equity <= 0:
            return SizingResult(rejected="no_equity")

        stop_pips = instrument.pips_between(entry_price, stop_price)
        if stop_pips <= 0:
            return SizingResult(rejected="invalid_stop")

        pip_value = pip_value_per_lot(instrument, rates, entry_price)
        if pip_value <= 0:
            return SizingResult(rejected="no_pip_value")

        risk_amount = equity * (self.limits.risk_per_trade_pct / 100.0) * risk_scaler
        risk_per_lot = stop_pips * pip_value
        min_lot_risk_pct = risk_per_lot * instrument.min_lot / equity * 100.0

        result = SizingResult(
            risk_amount=risk_amount,
            stop_pips=stop_pips,
            pip_value_per_lot=pip_value,
            min_lot_risk_pct=min_lot_risk_pct,
        )

        capped_risk = self._cap_by_currency_risk(
            risk_amount, instrument, is_long, equity, current_currency_risk or {}
        )
        if capped_risk <= 0:
            result.rejected = "currency_risk_cap"
            return result
        if capped_risk < risk_amount:
            result.capped_by = "currency_risk"
        risk_amount = capped_risk
        result.risk_amount = risk_amount

        lots = min(instrument.floor_lots(risk_amount / risk_per_lot), instrument.max_lot)
        result.lots = lots

        if lots < instrument.min_lot:
            result.lots = 0.0
            result.rejected = "below_min_lot"
            return result

        notional_per_lot = rates.convert(
            instrument.contract_size * entry_price, instrument.quote
        )
        if notional_per_lot <= 0:
            result.lots = 0.0
            result.rejected = "no_notional"
            return result

        lots = self._cap_by_margin(lots, notional_per_lot, free_margin, leverage)
        lots = self._cap_by_currency_exposure(
            lots,
            instrument,
            entry_price,
            equity,
            rates,
            is_long,
            current_exposure or {},
        )
        lots = self._cap_by_gross_exposure(lots, notional_per_lot, equity, gross_exposure)

        lots = instrument.floor_lots(lots)
        if lots < instrument.min_lot:
            result.lots = 0.0
            result.rejected = "capped_below_min_lot"
            return result

        result.lots = lots
        result.risk_amount = lots * risk_per_lot
        return result

    def _cap_by_currency_risk(
        self,
        risk_amount: float,
        instrument: FxInstrument,
        is_long: bool,
        equity: float,
        current_risk: dict[str, float],
    ) -> float:
        """Limit how much stop-risk may pile onto a single currency.

        Buying EURUSD is long EUR and short USD, so the new risk adds to one
        currency's tally and subtracts from the other's. Existing opposite-side
        risk therefore creates headroom rather than consuming it.
        """
        cap = equity * (self.limits.max_currency_risk_pct / 100.0)
        if cap <= 0:
            return 0.0

        sign = 1.0 if is_long else -1.0
        for currency, direction in ((instrument.base, sign), (instrument.quote, -sign)):
            existing = current_risk.get(currency, 0.0)
            headroom = cap - existing if direction > 0 else cap + existing
            risk_amount = min(risk_amount, max(headroom, 0.0))
        return risk_amount

    def _cap_by_margin(
        self, lots: float, notional_per_lot: float, free_margin: float, leverage: float
    ) -> float:
        margin_per_lot = notional_per_lot / leverage
        if margin_per_lot <= 0:
            return lots
        usable = free_margin * (1.0 - self.limits.min_free_margin_pct / 100.0)
        return min(lots, max(usable, 0.0) / margin_per_lot)

    def _cap_by_currency_exposure(
        self,
        lots: float,
        instrument: FxInstrument,
        price: float,
        equity: float,
        rates: RateBook,
        is_long: bool,
        current_exposure: dict[str, float],
    ) -> float:
        cap = equity * (self.limits.max_currency_exposure_pct / 100.0)
        if cap <= 0:
            return 0.0

        signed_lots = lots if is_long else -lots
        projected = projected_currency_exposure(
            current_exposure, instrument, signed_lots, price, rates
        )
        worst = max(
            abs(projected.get(instrument.base, 0.0)),
            abs(projected.get(instrument.quote, 0.0)),
        )
        if worst <= cap:
            return lots
        return lots * (cap / worst)

    def _cap_by_gross_exposure(
        self,
        lots: float,
        notional_per_lot: float,
        equity: float,
        gross_exposure: float,
    ) -> float:
        cap = equity * (self.limits.max_gross_exposure_pct / 100.0)
        remaining = cap - gross_exposure
        if remaining <= 0:
            return 0.0
        return min(lots, remaining / notional_per_lot)


def minimum_viable_equity(
    instrument: FxInstrument,
    stop_pips: float,
    rates: RateBook,
    risk_per_trade_pct: float,
    price: Optional[float] = None,
) -> float:
    """Smallest account that can take this trade at the intended risk.

    Answers the question a small account has to face: is my deposit large
    enough for the broker's minimum lot to still respect my risk budget?
    """
    pip_value = pip_value_per_lot(instrument, rates, price)
    risk_at_min_lot = stop_pips * pip_value * instrument.min_lot
    return risk_at_min_lot / (risk_per_trade_pct / 100.0)
