"""Unit tests for trade loaders."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.data.loaders import CSVTradeLoader, MockTradeLoader, ParquetTradeWriter
from core.models.trade import Trade, TradeSide


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestMockTradeLoader:
    def test_load_records(self):
        records = [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "quantity": 0.1,
                "side": "buy",
                "trade_id": "t001",
                "exchange": "binance",
            },
        ]
        loader = MockTradeLoader(records)
        result = list(loader.load())
        assert len(result) == 1
        assert result[0].symbol == "BTC/USDT"


class TestCSVTradeLoader:
    def test_load_fixture(self):
        filepath = FIXTURES / "sample_trades.csv"
        loader = CSVTradeLoader(filepath)
        records = list(loader.load())
        assert len(records) == 5
        assert records[0].symbol == "BTC/USDT"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            CSVTradeLoader("/nonexistent/trades.csv")

    def test_missing_columns_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("timestamp,symbol\n2024-01-01,BTC\n")
        with pytest.raises(ValueError, match="missing required columns"):
            list(CSVTradeLoader(bad_csv).load())


class TestParquetRoundTrip:
    def test_write_and_read(self, tmp_path):
        pytest.importorskip("pyarrow")
        from core.data.loaders import ParquetTradeLoader

        filepath = tmp_path / "trades.parquet"
        writer = ParquetTradeWriter(filepath)

        trade = Trade(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="BTC/USDT",
            price=50000.0,
            quantity=0.1,
            side=TradeSide.BUY,
            trade_id="t001",
            exchange="binance",
        )
        writer.write(trade)
        writer.flush()

        loader = ParquetTradeLoader(filepath)
        records = list(loader.load())
        assert len(records) == 1
        assert records[0].symbol == "BTC/USDT"
        assert float(records[0].price) == 50000.0
