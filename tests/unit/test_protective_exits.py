from datetime import datetime, timezone

from core.models.bar import Bar, BarType
from core.models.order import OrderSide, OrderType, OrderIntent
from core.models.signal import Signal, SignalDirection
from core.strategy.trend import TrendFollowingStrategy
from core.models.regime import RegimeOutput, MarketRegime


def test_trend_strategy_emits_three_r_target():
    bar = Bar(
        timestamp=datetime.now(timezone.utc),
        start_timestamp=datetime.now(timezone.utc),
        end_timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT", open=100, high=110, low=99, close=100,
        volume=10, dollar_volume=1000, trade_count=1,
        bar_type=BarType.DOLLAR, bar_threshold=1000,
    )
    regime = RegimeOutput(symbol="BTC/USDT", regime=MarketRegime.TREND_BULL, confidence=0.9, timestamp=bar.timestamp)
    features={"ema_12":101.0,"ema_26":100.0,"atr_14":2.0}
    signal=TrendFollowingStrategy(min_confidence=0.5).on_bar(bar,features,regime)
    assert signal is not None
    assert signal.stop_reference == 96
    assert signal.target_reference == 112
