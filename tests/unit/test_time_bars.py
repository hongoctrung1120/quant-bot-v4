"""Unit tests for time bar engine."""

from datetime import datetime, timezone

from core.bars.time_bars import TimeBarEngine, bar_period_start, parse_interval
from core.models.trade import Trade, TradeSide


def _trade(ts_minute: int, price: float = 100.0) -> Trade:
    return Trade(
        timestamp=datetime(2024, 1, 15, 10, ts_minute, tzinfo=timezone.utc),
        symbol="BTC/USDT",
        price=price,
        quantity=1.0,
        side=TradeSide.BUY,
        trade_id=f"t{ts_minute}",
        exchange="binance",
    )


class TestTimeBarEngine:
    def test_parse_interval(self):
        assert parse_interval("15m").total_seconds() == 900
        assert parse_interval("1h").total_seconds() == 3600

    def test_bar_closes_on_period_change(self):
        engine = TimeBarEngine("15m")
        engine.on_trade(_trade(0, 100.0))
        engine.on_trade(_trade(5, 101.0))
        bar = engine.on_trade(_trade(16, 102.0))  # new 15m period
        assert bar is not None
        assert bar.open == 100.0
        assert bar.close == 101.0
        assert bar.trade_count == 2

    def test_period_alignment(self):
        ts = datetime(2024, 1, 15, 10, 17, tzinfo=timezone.utc)
        start = bar_period_start(ts, parse_interval("15m"))
        assert start.minute == 15
