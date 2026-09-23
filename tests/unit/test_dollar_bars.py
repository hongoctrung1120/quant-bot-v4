"""Unit tests for dollar bar engine."""

from datetime import datetime, timezone

import pytest

from core.bars.dollar_bars import (
    DollarBarEngine,
    OVERSHOOT_CARRY_FORWARD,
    OVERSHOOT_SPLIT_TRADE,
)
from core.models.trade import Trade, TradeSide


def _trade(price: float, qty: float, tid: str) -> Trade:
    return Trade(
        timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        symbol="BTC/USDT",
        price=price,
        quantity=qty,
        side=TradeSide.BUY,
        trade_id=tid,
        exchange="binance",
    )


class TestDollarBarThreshold:
    def test_bar_closes_at_threshold(self):
        engine = DollarBarEngine(threshold=1000.0)
        engine.on_trade(_trade(100.0, 5.0, "t1"))  # $500
        bar = engine.on_trade(_trade(100.0, 5.0, "t2"))  # $500 -> $1000
        assert bar is not None
        assert bar.dollar_volume >= 1000.0
        assert bar.trade_count == 2

    def test_overshoot_carry_forward(self):
        engine = DollarBarEngine(
            threshold=1000.0,
            overshoot_policy=OVERSHOOT_CARRY_FORWARD,
        )
        engine.on_trade(_trade(100.0, 9.0, "t1"))  # $900
        bar = engine.on_trade(_trade(100.0, 2.0, "t2"))  # $200 crosses
        assert bar is not None
        assert bar.dollar_volume == 1100.0  # full trade included

        # $100 carry + $900 trade reaches threshold; bar contains new trades only
        bar2 = engine.on_trade(_trade(100.0, 9.0, "t3"))
        assert bar2 is not None
        assert bar2.dollar_volume == pytest.approx(900.0)
        assert bar2.trade_count == 1

    def test_overshoot_split_trade(self):
        engine = DollarBarEngine(
            threshold=1000.0,
            overshoot_policy=OVERSHOOT_SPLIT_TRADE,
        )
        engine.on_trade(_trade(100.0, 9.0, "t1"))  # $900
        bar = engine.on_trade(_trade(100.0, 2.0, "t2"))  # crosses at $1000
        assert bar is not None
        assert bar.dollar_volume == pytest.approx(1000.0, rel=1e-6)

        # Remaining $100 from split + more trades
        bar2 = engine.on_trade(_trade(100.0, 9.0, "t3"))
        assert bar2 is not None


    def test_large_trade_can_close_multiple_bars(self):
        engine = DollarBarEngine(threshold=1000.0, overshoot_policy=OVERSHOOT_SPLIT_TRADE)
        bars = engine.on_trade_all(_trade(100.0, 25.0, "large"))
        assert len(bars) == 2
        assert all(b.dollar_volume == pytest.approx(1000.0) for b in bars)

    def test_timestamp_ordering(self):
        engine = DollarBarEngine(threshold=500.0)
        t1 = Trade(
            timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            symbol="BTC/USDT", price=100.0, quantity=3.0,
            side=TradeSide.BUY, trade_id="t1", exchange="binance",
        )
        t2 = Trade(
            timestamp=datetime(2024, 1, 15, 10, 1, tzinfo=timezone.utc),
            symbol="BTC/USDT", price=100.0, quantity=3.0,
            side=TradeSide.BUY, trade_id="t2", exchange="binance",
        )
        assert engine.on_trade(t1) is None
        bar = engine.on_trade(t2)
        assert bar is not None
        assert bar.start_timestamp <= bar.end_timestamp

    def test_flush_incomplete_bar(self):
        engine = DollarBarEngine(threshold=10000.0)
        engine.on_trade(_trade(100.0, 1.0, "t1"))
        bar = engine.flush("BTC/USDT")
        assert bar is not None
        assert bar.dollar_volume == 100.0
