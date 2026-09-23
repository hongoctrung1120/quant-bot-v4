"""Signal aggregation engine."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from core.interfaces import SignalEngine as SignalEngineABC
from core.models.signal import Signal, SignalDirection

logger = logging.getLogger(__name__)


class SignalEngine(SignalEngineABC):
    """Combine strategy signals without position sizing."""

    def __init__(
        self,
        signal_expiry_bars: int = 3,
        bar_duration_estimate: timedelta = timedelta(minutes=15),
    ) -> None:
        self._signals: list[Signal] = []
        self._consumed: set[str] = set()
        self._expiry_bars = signal_expiry_bars
        self._bar_duration = bar_duration_estimate

    def add_signal(self, signal: Signal) -> None:
        if signal.direction == SignalDirection.HOLD:
            return
        self._signals.append(signal)
        logger.debug(
            "Signal added: %s %s %s conf=%.2f",
            signal.strategy_id,
            signal.symbol,
            signal.direction.value,
            signal.confidence,
            extra={"event": "SIGNAL_ADDED"},
        )

    def get_actionable_signals(
        self,
        as_of: Optional[datetime] = None,
    ) -> list[Signal]:
        """Return non-expired signals, deduplicated by symbol+strategy."""
        if as_of is None:
            as_of = max((s.timestamp for s in self._signals), default=None)
        if as_of is None:
            return []

        expiry = self._bar_duration * self._expiry_bars
        active = [
            s for s in self._signals
            if s.signal_id not in self._consumed
            and (as_of - s.timestamp) <= expiry
            and s.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.EXIT)
        ]

        # Keep highest confidence per symbol+strategy
        best: dict[tuple[str, str], Signal] = {}
        for signal in active:
            key = (signal.symbol, signal.strategy_id)
            if key not in best or signal.confidence > best[key].confidence:
                best[key] = signal

        return list(best.values())

    def consume(self, signal_id: str) -> None:
        self._consumed.add(signal_id)

    def clear(self) -> None:
        self._signals.clear()
        self._consumed.clear()

    def combine_by_symbol(self) -> dict[str, list[Signal]]:
        """Group actionable signals by symbol."""
        result: dict[str, list[Signal]] = {}
        for signal in self.get_actionable_signals():
            result.setdefault(signal.symbol, []).append(signal)
        return result
