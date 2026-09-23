"""Event-driven backtest event loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.events.bus import SimpleEventBus
from core.events.types import BarClosedEvent, EventType
from core.models.bar import Bar
from core.models.trade import Trade

logger = logging.getLogger(__name__)


@dataclass
class BacktestContext:
    """Shared state during backtest run."""

    current_prices: dict[str, float] = field(default_factory=dict)
    bar_count: int = 0
    trade_count: int = 0


class EventLoop:
    """Process trades through the pipeline in timestamp order."""

    def __init__(self, event_bus: Optional[SimpleEventBus] = None) -> None:
        self.event_bus = event_bus or SimpleEventBus()
        self.context = BacktestContext()
        self._bar_handlers: list = []
        self._trade_handlers: list = []

    def on_bar_closed(self, handler) -> None:
        self._bar_handlers.append(handler)
        self.event_bus.subscribe(EventType.BAR_CLOSED.value, handler)

    def on_trade(self, handler) -> None:
        self._trade_handlers.append(handler)

    def emit_bar_closed(self, bar: Bar) -> None:
        self.context.current_prices[bar.symbol] = bar.close
        self.context.bar_count += 1
        event = BarClosedEvent(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            payload=bar,
            source="backtest",
        )
        self.event_bus.publish(event)

    def process_trade(self, trade: Trade) -> None:
        self.context.trade_count += 1
        self.context.current_prices[trade.symbol] = trade.price
        for handler in self._trade_handlers:
            handler(trade)
