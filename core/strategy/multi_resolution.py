"""Multi-resolution strategy hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.models.bar import Bar
from core.models.regime import RegimeOutput
from core.models.signal import Signal, SignalDirection


@dataclass
class MultiResolutionState:
    """Tracks direction/setup/entry from different bar resolutions."""

    direction: Optional[SignalDirection] = None
    setup: Optional[str] = None
    entry: Optional[SignalDirection] = None
    last_large_bar_ts: Optional[object] = None
    last_medium_bar_ts: Optional[object] = None
    last_small_bar_ts: Optional[object] = None


class MultiResolutionCoordinator:
    """Combine large/medium/small bar signals into final signal.

    Large bars → market direction
    Medium bars → setup detection
    Small bars → entry trigger
    """

    def __init__(
        self,
        large_role: str = "direction",
        medium_role: str = "setup",
        small_role: str = "entry",
    ) -> None:
        self._states: dict[str, MultiResolutionState] = {}
        self._large_role = large_role
        self._medium_role = medium_role
        self._small_role = small_role

    def update_direction(
        self,
        symbol: str,
        bar: Bar,
        features: dict[str, float],
    ) -> None:
        state = self._states.setdefault(symbol, MultiResolutionState())
        ema12 = features.get("ema_12")
        ema26 = features.get("ema_26")
        if ema12 is None or ema26 is None:
            return
        if ema12 > ema26:
            state.direction = SignalDirection.BUY
        elif ema12 < ema26:
            state.direction = SignalDirection.SELL
        state.last_large_bar_ts = bar.timestamp

    def update_setup(
        self,
        symbol: str,
        bar: Bar,
        features: dict[str, float],
    ) -> None:
        state = self._states.setdefault(symbol, MultiResolutionState())
        rsi = features.get("rsi_14")
        if rsi is None:
            return
        if 35 <= rsi <= 55:
            state.setup = "pullback"
        elif rsi > 55:
            state.setup = "momentum"
        elif rsi < 35:
            state.setup = "oversold_bounce"
        state.last_medium_bar_ts = bar.timestamp

    def check_entry(
        self,
        symbol: str,
        bar: Bar,
        features: dict[str, float],
        regime: RegimeOutput,
        strategy_id: str = "multi_resolution",
        min_confidence: float = 0.6,
    ) -> Optional[Signal]:
        state = self._states.get(symbol)
        if state is None or state.direction is None:
            return None

        ofi = features.get("order_flow_imbalance")
        if ofi is None:
            return None

        entry: Optional[SignalDirection] = None
        if state.direction == SignalDirection.BUY and ofi > 0.1:
            entry = SignalDirection.BUY
        elif state.direction == SignalDirection.SELL and ofi < -0.1:
            entry = SignalDirection.SELL

        if entry is None:
            return None

        confidence = min(0.95, regime.confidence * 0.9)
        if confidence < min_confidence:
            return None

        return Signal(
            symbol=symbol,
            direction=entry,
            confidence=confidence,
            timestamp=bar.timestamp,
            strategy_id=strategy_id,
            regime=regime.regime.value,
            entry_reference=bar.close,
        )
