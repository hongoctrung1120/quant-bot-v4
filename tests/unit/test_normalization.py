"""Unit tests for trade normalization."""

from datetime import datetime, timezone

import pytest

from core.data.normalization import RawTradeRecord, TradeNormalizer, NormalizationError
from core.models.trade import TradeSide


TS = "2024-01-15T10:30:00Z"
TS_MS = 1705314600000


class TestRawTradeRecord:
    def test_from_dict(self):
        record = RawTradeRecord.from_dict({
            "timestamp": TS,
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "quantity": 0.1,
            "side": "buy",
            "trade_id": "t001",
            "exchange": "binance",
        })
        assert record.symbol == "BTC/USDT"
        assert record.trade_id == "t001"


class TestTradeNormalizer:
    def setup_method(self):
        self.normalizer = TradeNormalizer()

    def _record(self, **kwargs) -> RawTradeRecord:
        defaults = {
            "timestamp": TS,
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "quantity": 0.1,
            "side": "buy",
            "trade_id": "t001",
            "exchange": "binance",
        }
        defaults.update(kwargs)
        return RawTradeRecord.from_dict(defaults)

    def test_normalize_iso_string(self):
        trade = self.normalizer.normalize(self._record())
        assert trade.symbol == "BTC/USDT"
        assert trade.price == 50000.0
        assert trade.side == TradeSide.BUY
        assert trade.timestamp.tzinfo is not None

    def test_normalize_unix_millis(self):
        trade = self.normalizer.normalize(self._record(timestamp=TS_MS))
        assert trade.timestamp.year == 2024

    def test_normalize_side_variants(self):
        for side in ("buy", "BUY", "b", "sell", "SELL", "s"):
            trade = self.normalizer.normalize(
                self._record(side=side, trade_id=f"t-{side}")
            )
            expected = TradeSide.BUY if side.lower() in ("buy", "b") else TradeSide.SELL
            assert trade.side == expected

    def test_missing_timestamp_raises(self):
        with pytest.raises(NormalizationError, match="Missing timestamp"):
            self.normalizer.normalize(self._record(timestamp=None))

    def test_invalid_side_raises(self):
        with pytest.raises(NormalizationError, match="Unknown side"):
            self.normalizer.normalize(self._record(side="invalid"))

    def test_invalid_price_raises(self):
        with pytest.raises(ValueError, match="Invalid price"):
            self.normalizer.normalize(self._record(price=-1.0))

    def test_invalid_quantity_raises(self):
        with pytest.raises(ValueError, match="Invalid quantity"):
            self.normalizer.normalize(self._record(quantity=0.0))

    def test_optional_timestamps(self):
        trade = self.normalizer.normalize(self._record(
            exchange_timestamp=TS,
            local_timestamp=TS_MS,
        ))
        assert trade.exchange_timestamp is not None
        assert trade.local_timestamp is not None
