"""Integration tests for trade ingestion pipeline."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.config import DataConfig
from core.data.ingestion import TradeIngestionEngine
from core.data.loaders import CSVTradeLoader, MockTradeLoader
from core.events import EventType, SimpleEventBus
from core.models.trade import Trade


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestTradeIngestionEngine:
    def setup_method(self):
        self.config = DataConfig(
            duplicate_detection=True,
            out_of_order_tolerance_ms=100,
        )
        self.bus = SimpleEventBus()
        self.trades_received: list[Trade] = []
        self.bus.subscribe(
            EventType.TRADE.value,
            lambda e: self.trades_received.append(e.payload),
        )
        self.engine = TradeIngestionEngine(
            config=self.config,
            event_bus=self.bus,
        )

    def test_ingest_mock_trades(self):
        records = [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "quantity": 0.1,
                "side": "buy",
                "trade_id": "t001",
                "exchange": "binance",
            },
            {
                "timestamp": "2024-01-15T10:00:01Z",
                "symbol": "BTC/USDT",
                "price": 50010.0,
                "quantity": 0.05,
                "side": "sell",
                "trade_id": "t002",
                "exchange": "binance",
            },
        ]
        result = self.engine.ingest(MockTradeLoader(records))
        assert result.stats.accepted == 2
        assert result.stats.rejected == 0
        assert len(result.trades) == 2
        assert len(self.trades_received) == 2

    def test_ingest_csv_with_duplicate(self):
        loader = CSVTradeLoader(FIXTURES / "sample_trades.csv")
        result = self.engine.ingest(loader)
        assert result.stats.total_records == 5
        assert result.stats.accepted == 4
        assert result.stats.rejected == 1
        assert result.stats.duplicates == 1

    def test_ingest_rejects_invalid_record(self):
        records = [
            {
                "timestamp": None,
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "quantity": 0.1,
                "side": "buy",
                "trade_id": "bad-001",
                "exchange": "binance",
            },
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "quantity": 0.1,
                "side": "buy",
                "trade_id": "good-001",
                "exchange": "binance",
            },
        ]
        result = self.engine.ingest(MockTradeLoader(records))
        assert result.stats.accepted == 1
        assert result.stats.rejected == 1
        assert result.stats.normalization_errors == 1

    def test_ingest_one_streaming(self):
        raw = {
            "timestamp": "2024-01-15T10:00:00Z",
            "symbol": "ETH/USDT",
            "price": 3000.0,
            "quantity": 1.0,
            "side": "buy",
            "trade_id": "stream-001",
            "exchange": "binance",
        }
        trade = self.engine.ingest_one(raw)
        assert trade is not None
        assert trade.symbol == "ETH/USDT"

    def test_trades_are_immutable(self):
        records = [{
            "timestamp": "2024-01-15T10:00:00Z",
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "quantity": 0.1,
            "side": "buy",
            "trade_id": "imm-001",
            "exchange": "binance",
        }]
        result = self.engine.ingest(MockTradeLoader(records))
        trade = result.trades[0]
        with pytest.raises(AttributeError):
            trade.price = 99999.0  # type: ignore[misc]

    def test_quality_events_logged(self):
        records = [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "quantity": 0.1,
                "side": "buy",
                "trade_id": "dup-test",
                "exchange": "binance",
            },
            {
                "timestamp": "2024-01-15T10:00:01Z",
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "quantity": 0.1,
                "side": "buy",
                "trade_id": "dup-test",
                "exchange": "binance",
            },
        ]
        result = self.engine.ingest(MockTradeLoader(records))
        invalid_events = [
            e for e in result.quality_events
            if e.event_type == "DUPLICATE_TRADE"
        ]
        assert len(invalid_events) == 1
