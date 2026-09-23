"""Breakout strategy."""

from __future__ import annotations

from collections import deque
from typing import Optional

from core.interfaces import Strategy
from core.models.bar import Bar
from core.models.regime import MarketRegime, RegimeOutput
from core.models.signal import Signal, SignalDirection


class BreakoutStrategy(Strategy):
    """N-bar high/low breakout with volume confirmation."""

    def __init__(
        self,
        strategy_id: str = "breakout",
        lookback: int = 20,
        volume_multiplier: float = 1.5,
        min_confidence: float = 0.5,
    ) -> None:
        self._strategy_id = strategy_id
        self._lookback = lookback
        self._volume_multiplier = volume_multiplier
        self._min_confidence = min_confidence
        self._highs: dict[str, deque[float]] = {}
        self._lows: dict[str, deque[float]] = {}
        self._volumes: dict[str, deque[float]] = {}

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def _update_history(self, bar: Bar) -> None:
        highs = self._highs.setdefault(bar.symbol, deque(maxlen=self._lookback))
        lows = self._lows.setdefault(bar.symbol, deque(maxlen=self._lookback))
        vols = self._volumes.setdefault(bar.symbol, deque(maxlen=self._lookback))
        highs.append(bar.high)
        lows.append(bar.low)
        vols.append(bar.volume)

    def on_bar(
        self,
        bar: Bar,
        features: dict[str, float],
        regime: RegimeOutput,
    ) -> Optional[Signal]:
        self._update_history(bar)
        highs = self._highs.get(bar.symbol, deque())
        lows = self._lows.get(bar.symbol, deque())
        vols = self._volumes.get(bar.symbol, deque())

        if len(highs) < self._lookback:
            return None

        if regime.regime == MarketRegime.HIGH_VOLATILITY:
            return None

        prev_high = max(list(highs)[:-1]) if len(highs) > 1 else None
        prev_low = min(list(lows)[:-1]) if len(lows) > 1 else None
        avg_vol = sum(list(vols)[:-1]) / max(len(vols) - 1, 1)

        if prev_high is None or prev_low is None:
            return None

        rel_vol = features.get("relative_volume_20", 1.0)
        direction: Optional[SignalDirection] = None
        confidence = 0.6

        if bar.close > prev_high and rel_vol >= self._volume_multiplier:
            direction = SignalDirection.BUY
            confidence = min(0.9, 0.6 + rel_vol * 0.1)
        elif bar.close < prev_low and rel_vol >= self._volume_multiplier:
            direction = SignalDirection.SELL
            confidence = min(0.9, 0.6 + rel_vol * 0.1)

        if direction is None or confidence < self._min_confidence:
            return None

        return Signal(
            symbol=bar.symbol,
            direction=direction,
            confidence=confidence,
            timestamp=bar.timestamp,
            strategy_id=self._strategy_id,
            regime=regime.regime.value,
            entry_reference=bar.close,
        )
