"""Daily loss circuit breaker — HARD 3% limit."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from core.config import RiskConfig
from core.models.risk import RiskEvent, RiskState

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Portfolio-level daily loss circuit breaker.

    When daily loss >= daily_loss_limit_pct (default 3%):
    - Enter BREACHED then LOCKED
    - No new orders allowed
    - Risk has final authority over all modules
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._state = RiskState.ACTIVE
        self._start_equity: Optional[float] = None
        self._events: list[RiskEvent] = []

    @property
    def state(self) -> RiskState:
        return self._state

    @property
    def start_equity(self) -> Optional[float]:
        return self._start_equity

    @property
    def events(self) -> list[RiskEvent]:
        return list(self._events)

    @property
    def new_orders_allowed(self) -> bool:
        return self._state in (RiskState.ACTIVE, RiskState.WARNING, RiskState.RECOVERY)

    def reset_session(self, start_equity: float) -> None:
        """Reset at start of trading session."""
        self._start_equity = start_equity
        self._state = RiskState.ACTIVE
        logger.info(
            "Circuit breaker session reset: start_equity=%.2f",
            start_equity,
            extra={"event": "CB_SESSION_RESET"},
        )

    def update(self, current_equity: float, timestamp: datetime) -> RiskState:
        """Update risk state based on current equity."""
        if not self._config.circuit_breaker_enabled:
            return self._state

        if self._start_equity is None or self._start_equity <= 0:
            self._start_equity = current_equity
            return self._state

        if self._state == RiskState.LOCKED:
            return self._state

        daily_loss_pct = (
            (self._start_equity - current_equity) / self._start_equity * 100.0
        )

        if daily_loss_pct >= self._config.daily_loss_limit_pct:
            self._state = RiskState.BREACHED
            event = RiskEvent(
                event_type="RISK_BREACH",
                timestamp=timestamp,
                risk_state=RiskState.BREACHED,
                message=(
                    f"Daily loss {daily_loss_pct:.2f}% >= "
                    f"limit {self._config.daily_loss_limit_pct}%"
                ),
                start_equity=self._start_equity,
                current_equity=current_equity,
                daily_loss_pct=daily_loss_pct,
                action_taken="LOCK_TRADING",
            )
            self._events.append(event)
            self._state = RiskState.LOCKED
            logger.error(
                "CIRCUIT BREAKER BREACHED: daily_loss=%.2f%%",
                daily_loss_pct,
                extra={
                    "event": "RISK_BREACH",
                    "start_equity": self._start_equity,
                    "current_equity": current_equity,
                    "action": "LOCK_TRADING",
                },
            )
        elif daily_loss_pct >= self._config.warning_threshold_pct:
            if self._state != RiskState.WARNING:
                self._state = RiskState.WARNING
                self._events.append(
                    RiskEvent(
                        event_type="RISK_WARNING",
                        timestamp=timestamp,
                        risk_state=RiskState.WARNING,
                        message=f"Daily loss approaching limit: {daily_loss_pct:.2f}%",
                        start_equity=self._start_equity,
                        current_equity=current_equity,
                        daily_loss_pct=daily_loss_pct,
                    )
                )

        return self._state

    def lock(self, timestamp: datetime, reason: str) -> None:
        """Manually lock trading."""
        self._state = RiskState.LOCKED
        self._events.append(
            RiskEvent(
                event_type="MANUAL_LOCK",
                timestamp=timestamp,
                risk_state=RiskState.LOCKED,
                message=reason,
                action_taken="LOCK_TRADING",
            )
        )
