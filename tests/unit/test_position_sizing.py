"""Unit tests for position sizing."""

import pytest

from datetime import datetime, timezone

from core.config import RiskConfig
from core.models.allocation import AllocationResult
from core.models.signal import Signal, SignalDirection
from core.risk.position_sizing import RiskBasedPositionSizer


class TestPositionSizing:
    def test_basic_sizing(self):
        sizer = RiskBasedPositionSizer(RiskConfig(risk_per_trade_pct=0.25))
        signal = Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            strategy_id="trend",
            regime="TREND_BULL",
            entry_reference=100000.0,
            stop_reference=98000.0,
        )
        allocation = AllocationResult(
            timestamp=datetime.now(timezone.utc),
            total_equity=100000.0,
            reserve_amount=20000.0,
            trading_capital=80000.0,
            asset_allocations={"BTC/USDT": 40.0},
            strategy_allocations={"trend": 40.0},
            regime_adjustments={},
        )
        intent = sizer.size(signal, allocation, 100000.0, 100000.0)
        assert intent is not None
        # Risk = 250, stop distance = 2000 -> 0.125 BTC
        assert intent.quantity == pytest.approx(0.125, rel=0.01)

    def test_zero_stop_returns_none(self):
        sizer = RiskBasedPositionSizer(RiskConfig())
        signal = Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            strategy_id="trend",
            regime="TREND_BULL",
            entry_reference=100.0,
            stop_reference=100.0,
        )
        allocation = AllocationResult(
            timestamp=datetime.now(timezone.utc),
            total_equity=100000.0,
            reserve_amount=0.0,
            trading_capital=100000.0,
            asset_allocations={"BTC/USDT": 100.0},
            strategy_allocations={"trend": 100.0},
            regime_adjustments={},
        )
        assert sizer.size(signal, allocation, 100000.0, 100.0) is None
