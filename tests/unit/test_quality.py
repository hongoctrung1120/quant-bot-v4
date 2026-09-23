"""Unit tests for data quality engine."""

from datetime import datetime, timedelta, timezone

import pytest

from core.config import DataConfig
from core.data.quality import DataQualityEngine
from core.models.quality import DataQualityStatus
from core.models.trade import Trade, TradeSide


TS = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def _trade(
    trade_id: str = "t001",
    timestamp: datetime = TS,
    price: float = 50000.0,
    quantity: float = 0.1,
) -> Trade:
    return Trade(
        timestamp=timestamp,
        symbol="BTC/USDT",
        price=price,
        quantity=quantity,
        side=TradeSide.BUY,
        trade_id=trade_id,
        exchange="binance",
    )


class TestDataQualityEngine:
    def setup_method(self):
        self.config = DataConfig(
            duplicate_detection=True,
            out_of_order_tolerance_ms=100,
            gap_detection_enabled=True,
            max_price_change_pct=50.0,
            gap_threshold_seconds=300.0,
        )
        self.engine = DataQualityEngine(self.config)

    def test_valid_trade_accepted(self):
        result = self.engine.validate(_trade())
        assert result.accepted is True
        assert result.trade is not None

    def test_duplicate_detection(self):
        self.engine.validate(_trade(trade_id="dup-001"))
        result = self.engine.validate(_trade(trade_id="dup-001"))
        assert result.accepted is False
        assert any(e.event_type == "DUPLICATE_TRADE" for e in result.events)

    def test_out_of_order_within_tolerance(self):
        self.engine.validate(_trade(trade_id="t1", timestamp=TS))
        result = self.engine.validate(_trade(
            trade_id="t2",
            timestamp=TS + timedelta(milliseconds=50),
        ))
        assert result.accepted is True
        assert not any(e.event_type == "OUT_OF_ORDER" for e in result.events)

    def test_out_of_order_beyond_tolerance(self):
        self.engine.validate(_trade(trade_id="t1", timestamp=TS))
        result = self.engine.validate(_trade(
            trade_id="t2",
            timestamp=TS - timedelta(seconds=5),
        ))
        assert result.accepted is True  # warning, not rejection
        assert any(e.event_type == "OUT_OF_ORDER" for e in result.events)
        assert any(e.status == DataQualityStatus.WARNING for e in result.events)

    def test_timestamp_ordering_tracked(self):
        t1 = TS
        t2 = TS + timedelta(seconds=1)
        t3 = TS + timedelta(seconds=2)
        self.engine.validate(_trade(trade_id="a", timestamp=t1))
        self.engine.validate(_trade(trade_id="b", timestamp=t2))
        self.engine.validate(_trade(trade_id="c", timestamp=t3))
        state = self.engine._states["BTC/USDT"]
        assert state.last_timestamp == t3

    def test_price_spike_warning(self):
        self.engine.validate(_trade(trade_id="t1", price=50000.0))
        result = self.engine.validate(_trade(
            trade_id="t2", price=90000.0,
        ))
        assert any(e.event_type == "PRICE_SPIKE" for e in result.events)

    def test_data_gap_warning(self):
        self.engine.validate(_trade(trade_id="t1", timestamp=TS))
        result = self.engine.validate(_trade(
            trade_id="t2",
            timestamp=TS + timedelta(seconds=600),
        ))
        assert any(e.event_type == "DATA_GAP" for e in result.events)

    def test_reset_symbol(self):
        self.engine.validate(_trade(trade_id="t1"))
        self.engine.reset("BTC/USDT")
        state = self.engine._states.get("BTC/USDT")
        assert state is None

    def test_duplicate_after_reset_allowed(self):
        self.engine.validate(_trade(trade_id="t1"))
        self.engine.reset("BTC/USDT")
        result = self.engine.validate(_trade(trade_id="t1"))
        assert result.accepted is True

    def test_events_are_auditable(self):
        self.engine.validate(_trade(trade_id="t1"))
        self.engine.validate(_trade(trade_id="t1"))
        assert len(self.engine.events) >= 2
        dup_events = [
            e for e in self.engine.events if e.event_type == "DUPLICATE_TRADE"
        ]
        assert len(dup_events) == 1
        assert dup_events[0].trade_id == "t1"
