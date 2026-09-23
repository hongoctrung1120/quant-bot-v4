"""Shared bar builder state and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.models.bar import Bar, BarType
from core.models.trade import Trade, TradeSide


@dataclass
class BarBuilderState:
    """Mutable state for a bar currently being formed."""

    symbol: str
    bar_type: BarType
    bar_threshold: float | str
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    dollar_volume: float = 0.0
    trade_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_dollar_volume: float = 0.0
    sell_dollar_volume: float = 0.0
    cumulative_dollar: float = 0.0
    carry_forward_dollar: float = 0.0

    def is_empty(self) -> bool:
        return self.trade_count == 0

    def add_trade(self, trade: Trade, quantity: float | None = None) -> None:
        """Add trade (or partial quantity) to the forming bar."""
        qty = quantity if quantity is not None else trade.quantity
        dollar = trade.price * qty

        if self.is_empty():
            self.start_timestamp = trade.timestamp
            self.open = trade.price
            self.high = trade.price
            self.low = trade.price
        else:
            self.high = max(self.high, trade.price)
            self.low = min(self.low, trade.price)

        self.close = trade.price
        self.end_timestamp = trade.timestamp
        self.volume += qty
        self.dollar_volume += dollar
        self.trade_count += 1
        self.cumulative_dollar += dollar

        if trade.side == TradeSide.BUY:
            self.buy_volume += qty
            self.buy_dollar_volume += dollar
        else:
            self.sell_volume += qty
            self.sell_dollar_volume += dollar

    def to_bar(self) -> Bar:
        """Materialize immutable Bar from current state."""
        if self.start_timestamp is None or self.end_timestamp is None:
            raise ValueError("Cannot create bar without timestamps")

        vwap = (
            self.dollar_volume / self.volume if self.volume > 0 else None
        )
        total_vol = self.buy_volume + self.sell_volume
        ofi = (
            (self.buy_volume - self.sell_volume) / total_vol
            if total_vol > 0
            else None
        )

        return Bar(
            timestamp=self.end_timestamp,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            dollar_volume=self.dollar_volume,
            trade_count=self.trade_count,
            bar_type=self.bar_type,
            bar_threshold=self.bar_threshold,
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            buy_dollar_volume=self.buy_dollar_volume,
            sell_dollar_volume=self.sell_dollar_volume,
            vwap=vwap,
            order_flow_imbalance=ofi,
        )

    def reset(self, carry_dollar: float = 0.0) -> None:
        """Reset state for next bar, optionally carrying dollar volume."""
        self.start_timestamp = None
        self.end_timestamp = None
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0
        self.dollar_volume = 0.0
        self.trade_count = 0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.buy_dollar_volume = 0.0
        self.sell_dollar_volume = 0.0
        self.cumulative_dollar = carry_dollar
        self.carry_forward_dollar = carry_dollar
