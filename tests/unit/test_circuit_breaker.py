"""Unit tests for circuit breaker and risk."""

from datetime import datetime, timezone

import pytest

from core.config import RiskConfig
from core.models.risk import RiskState
from core.risk.circuit_breaker import CircuitBreaker
from core.risk.portfolio_risk import PortfolioRiskMonitor


class TestCircuitBreaker:
    def setup_method(self):
        self.config = RiskConfig(
            daily_loss_limit_pct=3.0,
            warning_threshold_pct=2.0,
            circuit_breaker_enabled=True,
        )
        self.cb = CircuitBreaker(self.config)
        self.cb.reset_session(100000.0)

    def test_active_state_initially(self):
        assert self.cb.state == RiskState.ACTIVE
        assert self.cb.new_orders_allowed

    def test_warning_at_2pct(self):
        ts = datetime.now(timezone.utc)
        self.cb.update(98000.0, ts)
        assert self.cb.state == RiskState.WARNING

    def test_breach_at_3pct_locks_trading(self):
        ts = datetime.now(timezone.utc)
        state = self.cb.update(97000.0, ts)
        assert state == RiskState.LOCKED
        assert not self.cb.new_orders_allowed

    def test_breach_records_event(self):
        ts = datetime.now(timezone.utc)
        self.cb.update(97000.0, ts)
        events = [e for e in self.cb.events if e.event_type == "RISK_BREACH"]
        assert len(events) == 1
        assert events[0].daily_loss_pct == pytest.approx(3.0)

    def test_no_bypass_after_lock(self):
        ts = datetime.now(timezone.utc)
        self.cb.update(97000.0, ts)
        self.cb.update(96000.0, ts)
        assert self.cb.state == RiskState.LOCKED


class TestPortfolioRiskMonitor:
    def test_signal_blocked_when_locked(self):
        from core.models.signal import Signal, SignalDirection

        config = RiskConfig(daily_loss_limit_pct=3.0)
        capital_config = __import__(
            "core.config", fromlist=["CapitalConfig"]
        ).CapitalConfig()
        monitor = PortfolioRiskMonitor(config, capital_config)
        monitor.reset_session(100000.0)
        ts = datetime.now(timezone.utc)
        monitor.update_equity(97000.0, ts)

        signal = Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=ts,
            strategy_id="trend",
            regime="TREND_BULL",
        )
        assert not monitor.check_signal(signal)
