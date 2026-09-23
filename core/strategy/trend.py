"""Trend following strategy."""

from __future__ import annotations

from typing import Optional

from core.interfaces import Strategy
from core.models.bar import Bar
from core.models.regime import MarketRegime, RegimeOutput
from core.models.signal import Signal, SignalDirection


class TrendFollowingStrategy(Strategy):
    """EMA crossover trend strategy."""

    def __init__(
        self,
        strategy_id: str = "trend_following",
        min_confidence: float = 0.5,
    ) -> None:
        self._strategy_id = strategy_id
        self._min_confidence = min_confidence

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def on_bar(
        self,
        bar: Bar,
        features: dict[str, float],
        regime: RegimeOutput,
    ) -> Optional[Signal]:
        ema_fast = features.get("ema_12")
        ema_slow = features.get("ema_26")
        if ema_fast is None or ema_slow is None:
            return None

        if regime.regime not in (
            MarketRegime.TREND_BULL,
            MarketRegime.TREND_BEAR,
            MarketRegime.LOW_VOLATILITY,
        ):
            return None

        direction: Optional[SignalDirection] = None
        confidence = regime.confidence

        if ema_fast > ema_slow:
            direction = SignalDirection.BUY
        elif ema_fast < ema_slow:
            direction = SignalDirection.SELL

        if direction is None or confidence < self._min_confidence:
            return None

        atr = features.get("atr_14")
        stop = bar.close - 2 * atr if atr and direction == SignalDirection.BUY else None
        if direction == SignalDirection.SELL and atr:
            stop = bar.close + 2 * atr

        target = None
        if stop is not None:
            risk_distance = abs(bar.close - stop)
            target = (
                bar.close + 3 * risk_distance
                if direction == SignalDirection.BUY
                else bar.close - 3 * risk_distance
            )

        return Signal(
            symbol=bar.symbol,
            direction=direction,
            confidence=confidence,
            timestamp=bar.timestamp,
            strategy_id=self._strategy_id,
            regime=regime.regime.value,
            entry_reference=bar.close,
            stop_reference=stop,
            target_reference=target,
        )
