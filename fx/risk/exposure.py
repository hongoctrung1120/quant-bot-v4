"""Currency-level exposure netting.

Trading ten pairs is not ten independent bets. Long EURUSD, long GBPUSD and
short USDJPY are three expressions of one short-USD position, and a single USD
move hits all of them together. Netting exposure down to individual currencies
is what makes the position count honest.
"""

from __future__ import annotations

from dataclasses import dataclass

from fx.account import FxPosition
from fx.instruments import FxInstrument, RateBook


def position_notional(
    instrument: FxInstrument,
    lots: float,
    price: float,
    rates: RateBook,
) -> float:
    """Absolute position value in account currency."""
    quote_notional = instrument.notional_base(lots) * price
    return abs(rates.convert(quote_notional, instrument.quote))


def currency_exposure(
    positions: dict[str, FxPosition],
    prices: dict[str, float],
    instruments: dict[str, FxInstrument],
    rates: RateBook,
) -> dict[str, float]:
    """Signed exposure per currency, in account currency.

    A long EURUSD position is long EUR and short USD by the same value.
    """
    exposure: dict[str, float] = {}
    for symbol, position in positions.items():
        inst = instruments[symbol]
        price = prices.get(symbol, position.entry_price)
        notional = position_notional(inst, position.lots, price, rates)
        signed = notional if position.lots > 0 else -notional
        exposure[inst.base] = exposure.get(inst.base, 0.0) + signed
        exposure[inst.quote] = exposure.get(inst.quote, 0.0) - signed
    return exposure


def projected_currency_exposure(
    current: dict[str, float],
    instrument: FxInstrument,
    lots: float,
    price: float,
    rates: RateBook,
) -> dict[str, float]:
    """Exposure map as it would look after adding a candidate position."""
    notional = position_notional(instrument, lots, price, rates)
    signed = notional if lots > 0 else -notional
    projected = dict(current)
    projected[instrument.base] = projected.get(instrument.base, 0.0) + signed
    projected[instrument.quote] = projected.get(instrument.quote, 0.0) - signed
    return projected


def currency_risk(
    positions: dict[str, FxPosition],
    instruments: dict[str, FxInstrument],
) -> dict[str, float]:
    """Signed stop-loss risk attributable to each currency.

    This, not notional, is the meaningful correlation control: three positions
    that are all short USD lose together when USD rallies, so their risk adds
    up. A long EURUSD and a long USDJPY hold opposite USD views and net off.
    """
    risk: dict[str, float] = {}
    for symbol, position in positions.items():
        inst = instruments[symbol]
        signed = position.risk_amount if position.lots > 0 else -position.risk_amount
        risk[inst.base] = risk.get(inst.base, 0.0) + signed
        risk[inst.quote] = risk.get(inst.quote, 0.0) - signed
    return risk


@dataclass
class ExposureReport:
    by_currency: dict[str, float]
    gross: float
    largest_currency: str
    largest_abs: float

    @property
    def concentration(self) -> float:
        """Share of gross exposure sitting in the single largest currency."""
        return self.largest_abs / self.gross if self.gross > 0 else 0.0


def summarize(exposure: dict[str, float]) -> ExposureReport:
    if not exposure:
        return ExposureReport({}, 0.0, "", 0.0)
    gross = sum(abs(v) for v in exposure.values())
    currency, value = max(exposure.items(), key=lambda kv: abs(kv[1]))
    return ExposureReport(
        by_currency=dict(exposure),
        gross=gross,
        largest_currency=currency,
        largest_abs=abs(value),
    )
