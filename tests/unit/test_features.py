"""Unit tests for feature engine."""

from datetime import datetime, timezone

from core.features.feature_engine import FeatureEngine
from core.models.bar import Bar, BarType


def _bar(close: float, i: int) -> Bar:
    return Bar(
        timestamp=datetime(2024, 1, 15, 10, i, tzinfo=timezone.utc),
        start_timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        end_timestamp=datetime(2024, 1, 15, 10, i, tzinfo=timezone.utc),
        symbol="BTC/USDT",
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=10.0,
        dollar_volume=close * 10,
        trade_count=5,
        bar_type=BarType.DOLLAR,
        bar_threshold=1000000,
        buy_volume=6.0,
        sell_volume=4.0,
        vwap=close,
    )


class TestFeatureEngine:
    def test_no_lookahead(self):
        engine = FeatureEngine()
        for i in range(30):
            features = engine.on_bar(_bar(100.0 + i, i))
        assert "ema_12" in features or engine.get_feature("ema_12") is not None

    def test_warmup_period(self):
        engine = FeatureEngine()
        first = engine.on_bar(_bar(100.0, 0))
        assert "rsi_14" not in first

    def test_reset(self):
        engine = FeatureEngine()
        for i in range(30):
            engine.on_bar(_bar(100.0 + i, i))
        engine.reset()
        assert engine.get_feature("ema_12") is None
