"""Margin-based FX account.

Unlike spot accounting (cash minus full notional), a leveraged FX position only
locks margin; profit and loss accrue on the price difference. On a $100 deposit
a 0.01-lot position controls far more notional than the balance, so spot
accounting would drive cash deeply negative and make equity meaningless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from fx.instruments import FxInstrument, RateBook, margin_required

logger = logging.getLogger(__name__)


@dataclass
class FxPosition:
    symbol: str
    lots: float  # signed: positive long, negative short
    entry_price: float
    opened_at: datetime
    strategy_id: str
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    initial_stop_price: Optional[float] = None  # fixed at entry; defines 1R
    risk_amount: float = 0.0  # account currency put at risk at entry (1R)
    swap_accrued: float = 0.0
    entry_costs: float = 0.0
    entry_session: str = ""
    entry_spread_cost: float = 0.0
    bars_held: int = 0
    mae_price: float = 0.0  # worst price reached against the position
    mfe_price: float = 0.0  # best price reached in favour

    def __post_init__(self) -> None:
        if self.mae_price == 0.0:
            self.mae_price = self.entry_price
        if self.mfe_price == 0.0:
            self.mfe_price = self.entry_price

    @property
    def is_long(self) -> bool:
        return self.lots > 0

    @property
    def direction(self) -> str:
        return "LONG" if self.lots > 0 else "SHORT"

    def track_excursion(self, high: float, low: float) -> None:
        if self.is_long:
            self.mae_price = min(self.mae_price, low)
            self.mfe_price = max(self.mfe_price, high)
        else:
            self.mae_price = max(self.mae_price, high)
            self.mfe_price = min(self.mfe_price, low)


@dataclass
class ClosedTrade:
    symbol: str
    strategy_id: str
    direction: str
    lots: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    exit_reason: str
    entry_session: str
    bars_held: int
    mae_pips: float
    mfe_pips: float
    spread_cost: float = 0.0  # informational: already inside the fill prices
    risk_amount: float = 0.0  # value of 1R for this trade


class FxAccount:
    """Leveraged account with one netted position per symbol."""

    def __init__(
        self,
        balance: float,
        leverage: float,
        rates: RateBook,
        instruments: dict[str, FxInstrument],
        stop_out_level: float = 0.5,
        margin_call_level: float = 1.0,
    ) -> None:
        self.initial_balance = balance
        self.balance = balance
        self.leverage = leverage
        self.rates = rates
        self.instruments = instruments
        self.stop_out_level = stop_out_level
        self.margin_call_level = margin_call_level

        self.positions: dict[str, FxPosition] = {}
        self.closed_trades: list[ClosedTrade] = []
        self.total_costs = 0.0
        self.total_swap = 0.0
        self.total_spread_cost = 0.0

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------
    def position_pnl(self, position: FxPosition, price: float) -> float:
        """Unrealized P&L of one position in account currency."""
        inst = self.instruments[position.symbol]
        move_quote = (price - position.entry_price) * position.lots * inst.contract_size
        return self.rates.convert(move_quote, inst.quote)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            total += self.position_pnl(pos, price)
        return total

    def equity(self, prices: dict[str, float]) -> float:
        return self.balance + self.unrealized_pnl(prices)

    def used_margin(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            inst = self.instruments[symbol]
            price = prices.get(symbol, pos.entry_price)
            total += margin_required(inst, pos.lots, price, self.leverage, self.rates)
        return total

    def free_margin(self, prices: dict[str, float]) -> float:
        return self.equity(prices) - self.used_margin(prices)

    def margin_level(self, prices: dict[str, float]) -> float:
        """Equity / used margin. Infinite when flat."""
        used = self.used_margin(prices)
        if used <= 0:
            return float("inf")
        return self.equity(prices) / used

    def is_stopped_out(self, prices: dict[str, float]) -> bool:
        return self.margin_level(prices) < self.stop_out_level

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------
    def can_open(self, symbol: str, lots: float, price: float, prices: dict[str, float]) -> bool:
        inst = self.instruments[symbol]
        needed = margin_required(inst, lots, price, self.leverage, self.rates)
        return self.free_margin(prices) >= needed

    def open_position(
        self,
        symbol: str,
        lots: float,
        fill_price: float,
        timestamp: datetime,
        strategy_id: str,
        cost: float,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        risk_amount: float = 0.0,
        session: str = "",
        spread_cost: float = 0.0,
    ) -> FxPosition:
        """Open a new position. Costs are charged to the balance immediately."""
        if symbol in self.positions:
            raise ValueError(f"Position already open for {symbol}")

        self.balance -= cost
        self.total_costs += cost
        self.total_spread_cost += spread_cost

        position = FxPosition(
            symbol=symbol,
            lots=lots,
            entry_price=fill_price,
            opened_at=timestamp,
            strategy_id=strategy_id,
            stop_price=stop_price,
            target_price=target_price,
            initial_stop_price=stop_price,
            risk_amount=risk_amount,
            entry_costs=cost,
            entry_session=session,
            entry_spread_cost=spread_cost,
        )
        self.positions[symbol] = position
        return position

    def close_position(
        self,
        symbol: str,
        fill_price: float,
        timestamp: datetime,
        cost: float,
        reason: str,
        spread_cost: float = 0.0,
    ) -> ClosedTrade:
        position = self.positions.pop(symbol)
        inst = self.instruments[symbol]

        gross = self.position_pnl(position, fill_price)
        # Swap already moved the balance as it accrued, so it is attributed to
        # the trade's cost total but must not be charged to the balance again.
        cost_attribution = position.entry_costs + cost - position.swap_accrued
        net = gross - cost - position.entry_costs + position.swap_accrued

        self.balance += gross - cost
        self.total_costs += cost
        self.total_spread_cost += spread_cost

        r_multiple = net / position.risk_amount if position.risk_amount > 0 else 0.0
        trade = ClosedTrade(
            symbol=symbol,
            strategy_id=position.strategy_id,
            direction=position.direction,
            lots=abs(position.lots),
            entry_time=position.opened_at,
            entry_price=position.entry_price,
            exit_time=timestamp,
            exit_price=fill_price,
            gross_pnl=gross,
            costs=cost_attribution,
            net_pnl=net,
            r_multiple=r_multiple,
            exit_reason=reason,
            entry_session=position.entry_session,
            bars_held=position.bars_held,
            mae_pips=inst.pips_between(position.entry_price, position.mae_price),
            mfe_pips=inst.pips_between(position.entry_price, position.mfe_price),
            spread_cost=position.entry_spread_cost + spread_cost,
            risk_amount=position.risk_amount,
        )
        self.closed_trades.append(trade)
        return trade

    def apply_swap(self, symbol: str, amount: float) -> None:
        """Credit or debit overnight financing for an open position."""
        position = self.positions.get(symbol)
        if position is None:
            return
        position.swap_accrued += amount
        self.balance += amount
        self.total_swap += amount
