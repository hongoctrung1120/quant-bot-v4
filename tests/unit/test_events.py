"""Unit tests for event bus."""

from datetime import datetime, timezone

import pytest

from core.events import EventType, TradeEvent, BarClosedEvent, RiskBreachedEvent
from core.events.bus import SimpleEventBus
from core.models.trade import Trade, TradeSide


TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


class TestSimpleEventBus:
    def setup_method(self):
        self.bus = SimpleEventBus()

    def test_publish_and_subscribe(self):
        received = []

        def handler(event):
            received.append(event)

        self.bus.subscribe(EventType.TRADE.value, handler)

        trade = Trade(
            timestamp=TS,
            symbol="BTC/USDT",
            price=50000.0,
            quantity=0.1,
            side=TradeSide.BUY,
            trade_id="t001",
            exchange="binance",
        )
        event = TradeEvent(
            timestamp=TS,
            symbol="BTC/USDT",
            payload=trade,
        )
        self.bus.publish(event)

        assert len(received) == 1
        assert received[0].symbol == "BTC/USDT"

    def test_multiple_handlers(self):
        count = {"a": 0, "b": 0}

        def handler_a(event):
            count["a"] += 1

        def handler_b(event):
            count["b"] += 1

        self.bus.subscribe(EventType.BAR_CLOSED.value, handler_a)
        self.bus.subscribe(EventType.BAR_CLOSED.value, handler_b)

        event = BarClosedEvent(
            timestamp=TS,
            symbol="ETH/USDT",
            payload=None,
        )
        self.bus.publish(event)

        assert count["a"] == 1
        assert count["b"] == 1

    def test_unsubscribe(self):
        received = []

        def handler(event):
            received.append(event)

        self.bus.subscribe(EventType.TRADE.value, handler)
        self.bus.unsubscribe(EventType.TRADE.value, handler)

        event = TradeEvent(timestamp=TS, symbol="BTC/USDT", payload=None)
        self.bus.publish(event)

        assert len(received) == 0

    def test_handler_exception_propagates(self):
        def bad_handler(event):
            raise RuntimeError("Handler failed")

        self.bus.subscribe(EventType.TRADE.value, bad_handler)

        event = TradeEvent(timestamp=TS, symbol="BTC/USDT", payload=None)
        with pytest.raises(RuntimeError, match="Handler failed"):
            self.bus.publish(event)

    def test_no_handlers_no_error(self):
        event = RiskBreachedEvent(timestamp=TS, payload=None)
        self.bus.publish(event)  # should not raise

    def test_clear(self):
        def handler(event):
            pass

        self.bus.subscribe(EventType.TRADE.value, handler)
        self.bus.clear()
        assert self.bus.handler_count == {}

    def test_event_has_unique_id(self):
        e1 = TradeEvent(timestamp=TS, symbol="BTC/USDT", payload=None)
        e2 = TradeEvent(timestamp=TS, symbol="BTC/USDT", payload=None)
        assert e1.event_id != e2.event_id
