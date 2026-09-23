"""Unit tests for core data models."""

from datetime import datetime, timezone

import pytest

from core.models.trade import Trade, TradeSide
from core.models.bar import Bar, BarType
from core.models.signal import Signal, SignalDirection
from core.models.order import OrderIntent, OrderSide, OrderType
from core.models.regime import RegimeOutput, MarketRegime
from core.models.risk import RiskState, RiskEvent
from core.models.allocation import AllocationResult
from core.models.quality import DataQualityStatus, DataQualityEvent


TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


class TestTrade:
    def test_valid_trade(self):
        trade = Trade(
            timestamp=TS,
            symbol="BTC/USDT",
            price=50000.0,
            quantity=0.1,
            side=TradeSide.BUY,
            trade_id="t001",
            exchange="binance",
        )
        assert trade.dollar_value == 5000.0
        assert trade.side == TradeSide.BUY

    def test_invalid_price_raises(self):
        with pytest.raises(ValueError, match="Invalid price"):
            Trade(
                timestamp=TS,
                symbol="BTC/USDT",
                price=-1.0,
                quantity=0.1,
                side=TradeSide.BUY,
                trade_id="t001",
                exchange="binance",
            )

    def test_invalid_quantity_raises(self):
        with pytest.raises(ValueError, match="Invalid quantity"):
            Trade(
                timestamp=TS,
                symbol="BTC/USDT",
                price=50000.0,
                quantity=0.0,
                side=TradeSide.BUY,
                trade_id="t001",
                exchange="binance",
            )

    def test_trade_is_immutable(self):
        trade = Trade(
            timestamp=TS,
            symbol="BTC/USDT",
            price=50000.0,
            quantity=0.1,
            side=TradeSide.BUY,
            trade_id="t001",
            exchange="binance",
        )
        with pytest.raises(AttributeError):
            trade.price = 60000.0  # type: ignore[misc]


class TestBar:
    def _make_bar(self, **kwargs) -> Bar:
        defaults = dict(
            timestamp=TS,
            start_timestamp=TS,
            end_timestamp=TS,
            symbol="BTC/USDT",
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=10.0,
            dollar_volume=500000.0,
            trade_count=100,
            bar_type=BarType.DOLLAR,
            bar_threshold=1000000,
        )
        defaults.update(kwargs)
        return Bar(**defaults)

    def test_valid_dollar_bar(self):
        bar = self._make_bar()
        assert bar.bar_type == BarType.DOLLAR
        assert bar.is_closed

    def test_valid_time_bar(self):
        bar = self._make_bar(
            bar_type=BarType.TIME,
            bar_threshold="15m",
        )
        assert bar.bar_threshold == "15m"

    def test_high_less_than_low_raises(self):
        with pytest.raises(ValueError, match="high"):
            self._make_bar(high=49000.0, low=50000.0)

    def test_open_outside_range_raises(self):
        with pytest.raises(ValueError, match="open"):
            self._make_bar(open=51000.0)


class TestSignal:
    def test_valid_signal(self):
        signal = Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.75,
            timestamp=TS,
            strategy_id="trend_following",
            regime="TREND_BULL",
            entry_reference=50000.0,
            stop_reference=49000.0,
        )
        assert signal.direction == SignalDirection.BUY
        assert signal.signal_id  # auto-generated

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            Signal(
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                confidence=1.5,
                timestamp=TS,
                strategy_id="test",
                regime="TREND_BULL",
            )


class TestOrderIntent:
    def test_valid_order_intent(self):
        intent = OrderIntent(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            entry_type=OrderType.LIMIT,
            timestamp=TS,
            strategy_id="trend_following",
            allocation_id="alloc-001",
            risk_id="risk-001",
            limit_price=50000.0,
            stop_loss=49000.0,
        )
        assert intent.quantity == 0.1

    def test_zero_quantity_raises(self):
        with pytest.raises(ValueError, match="quantity"):
            OrderIntent(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.0,
                entry_type=OrderType.MARKET,
                timestamp=TS,
                strategy_id="test",
                allocation_id="a",
                risk_id="r",
            )


class TestRegimeOutput:
    def test_valid_regime(self):
        regime = RegimeOutput(
            regime=MarketRegime.TREND_BULL,
            confidence=0.78,
            timestamp=TS,
            symbol="BTC/USDT",
            features_used=("adx", "ema_slope"),
        )
        assert regime.regime == MarketRegime.TREND_BULL


class TestRiskEvent:
    def test_risk_event_creation(self):
        event = RiskEvent(
            event_type="DAILY_LOSS_BREACH",
            timestamp=TS,
            risk_state=RiskState.LOCKED,
            message="Daily loss limit breached",
            start_equity=100000.0,
            current_equity=97000.0,
            daily_loss_pct=3.0,
            action_taken="LOCK_TRADING",
        )
        assert event.risk_state == RiskState.LOCKED


class TestAllocationResult:
    def test_valid_allocation(self):
        result = AllocationResult(
            timestamp=TS,
            total_equity=100000.0,
            reserve_amount=20000.0,
            trading_capital=80000.0,
            asset_allocations={"BTC/USDT": 40.0, "ETH/USDT": 25.0},
            strategy_allocations={"trend_following": 40.0},
            regime_adjustments={},
        )
        assert result.trading_capital == 80000.0

    def test_negative_allocation_raises(self):
        with pytest.raises(ValueError, match="Negative"):
            AllocationResult(
                timestamp=TS,
                total_equity=100000.0,
                reserve_amount=20000.0,
                trading_capital=80000.0,
                asset_allocations={"BTC/USDT": -10.0},
                strategy_allocations={},
                regime_adjustments={},
            )


class TestDataQualityEvent:
    def test_quality_event(self):
        event = DataQualityEvent(
            status=DataQualityStatus.INVALID,
            event_type="DUPLICATE_TRADE",
            timestamp=TS,
            message="Duplicate trade detected",
            symbol="BTC/USDT",
            trade_id="t001",
        )
        assert event.status == DataQualityStatus.INVALID
