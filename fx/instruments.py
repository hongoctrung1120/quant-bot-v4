"""FX instrument specifications and account-currency conversion.

Spread and swap values here are conservative retail placeholders. Replace them
with the actual figures from your broker's contract specification before
trusting any backtest number produced with them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

CENT_ACCOUNT_SCALE = 0.01


@dataclass(frozen=True)
class FxInstrument:
    """Contract specification for one currency pair."""

    symbol: str
    base: str
    quote: str
    pip_size: float
    contract_size: float = 100_000.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0
    typical_spread_pips: float = 1.0
    swap_long_pips_per_day: float = -0.5
    swap_short_pips_per_day: float = -0.3
    commission_per_lot_per_side: float = 0.0

    def pips_between(self, price_a: float, price_b: float) -> float:
        return abs(price_a - price_b) / self.pip_size

    def offset_price(self, price: float, pips: float) -> float:
        return price + pips * self.pip_size

    def notional_base(self, lots: float) -> float:
        """Position size expressed in units of the base currency."""
        return abs(lots) * self.contract_size

    def floor_lots(self, lots: float) -> float:
        """Round a desired size down to a tradable lot increment."""
        if lots <= 0:
            return 0.0
        steps = math.floor(round(lots / self.lot_step, 9))
        return round(steps * self.lot_step, 10)

    def as_cent_account(self) -> "FxInstrument":
        """Cent-account variant: same lot notation, 1/100 of the real exposure.

        A cent account denominates the balance in cents, so one lot controls
        1/100 of the real currency units a standard account would. This is what
        makes correct fractional risk sizing possible on a very small deposit.
        """
        return replace(self, contract_size=self.contract_size * CENT_ACCOUNT_SCALE)


def _pair(
    symbol: str,
    spread: float,
    swap_long: float = -0.5,
    swap_short: float = -0.3,
) -> FxInstrument:
    base, quote = symbol[:3], symbol[3:]
    pip = 0.01 if quote == "JPY" else 0.0001
    return FxInstrument(
        symbol=symbol,
        base=base,
        quote=quote,
        pip_size=pip,
        typical_spread_pips=spread,
        swap_long_pips_per_day=swap_long,
        swap_short_pips_per_day=swap_short,
    )


MAJORS: dict[str, FxInstrument] = {
    i.symbol: i
    for i in [
        _pair("EURUSD", 1.0, -0.60, 0.10),
        _pair("GBPUSD", 1.3, -0.70, 0.05),
        _pair("USDJPY", 1.0, 0.15, -0.85),
        _pair("AUDUSD", 1.2, -0.55, 0.00),
        _pair("USDCHF", 1.5, 0.20, -0.90),
        _pair("USDCAD", 1.5, 0.05, -0.70),
        _pair("NZDUSD", 1.8, -0.60, 0.00),
        _pair("EURGBP", 1.5, -0.45, -0.15),
        _pair("EURJPY", 1.5, -0.40, -0.45),
        _pair("GBPJPY", 2.2, -0.50, -0.55),
    ]
}


def get_instrument(symbol: str, cent_account: bool = False) -> FxInstrument:
    key = symbol.upper().replace("/", "")
    if key not in MAJORS:
        raise KeyError(f"Unknown instrument: {symbol}")
    inst = MAJORS[key]
    return inst.as_cent_account() if cent_account else inst


def build_universe(
    symbols: list[str], cent_account: bool = False
) -> dict[str, FxInstrument]:
    return {s: get_instrument(s, cent_account) for s in symbols}


class RateBook:
    """Latest mid prices, used to convert P&L and margin into account currency."""

    def __init__(self, account_currency: str = "USD") -> None:
        self.account_currency = account_currency.upper()
        self._prices: dict[str, float] = {}

    def update(self, symbol: str, price: float) -> None:
        if price > 0:
            self._prices[symbol.upper()] = price

    def price(self, symbol: str) -> Optional[float]:
        return self._prices.get(symbol.upper())

    def rate_to_account(self, currency: str) -> Optional[float]:
        """How many units of account currency one unit of `currency` buys."""
        ccy = currency.upper()
        acct = self.account_currency
        if ccy == acct:
            return 1.0

        direct = self._prices.get(f"{ccy}{acct}")
        if direct:
            return direct
        inverse = self._prices.get(f"{acct}{ccy}")
        if inverse:
            return 1.0 / inverse

        # One triangulation hop: find any pair containing `ccy` whose other leg
        # is itself convertible (e.g. EURGBP + EURUSD gives GBP->USD).
        for symbol, price in self._prices.items():
            base, quote = symbol[:3], symbol[3:]
            if base == ccy and quote not in (ccy, acct):
                leg = self.rate_to_account(quote)
                if leg:
                    return price * leg
            if quote == ccy and base not in (ccy, acct):
                leg = self.rate_to_account(base)
                if leg and price > 0:
                    return leg / price
        return None

    def convert(self, amount: float, currency: str) -> float:
        rate = self.rate_to_account(currency)
        if rate is None:
            raise ValueError(
                f"No conversion path from {currency} to {self.account_currency}"
            )
        return amount * rate


def pip_value_per_lot(
    instrument: FxInstrument, rates: RateBook, price: Optional[float] = None
) -> float:
    """Account-currency value of one pip for one lot.

    The raw pip value lands in the quote currency; `price` is only needed when
    the quote currency has to be converted and the rate book lacks a direct pair.
    """
    quote_value = instrument.pip_size * instrument.contract_size
    if price is not None:
        rates.update(instrument.symbol, price)
    return rates.convert(quote_value, instrument.quote)


def margin_required(
    instrument: FxInstrument,
    lots: float,
    price: float,
    leverage: float,
    rates: RateBook,
) -> float:
    """Account-currency margin locked by a position of `lots` at `price`."""
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    notional_quote = instrument.notional_base(lots) * price
    return rates.convert(notional_quote, instrument.quote) / leverage
