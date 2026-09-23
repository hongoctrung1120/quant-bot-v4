"""Momentum strategy."""

from __future__ import annotations

from typing import Optional

from core.interfaces import Strategy
from core.models.bar import Bar
from core.models.regime import MarketRegime, RegimeOutput
from core.models.signal import Signal, SignalDirection


class MomentumStrategy(Strategy):
    """RSI + ROC momentum strategy."""

    def __init__(
        self,
        strategy_id: str = "momentum",
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        min_confidence: float = 0.5,
    ) -> None:
        self._strategy_id = strategy_id
        self._rsi_overbought = rsi_overbought
        self._rsi_oversold = rsi_oversold
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
        rsi = features.get("rsi_14")
        roc = features.get("roc_10")
        if rsi is None or roc is None:
            return None

        if regime.regime == MarketRegime.MEAN_REVERTING:
            return None

        direction: Optional[SignalDirection] = None
        confidence = 0.5

        if rsi > 55 and roc > 0:
            direction = SignalDirection.BUY
            confidence = min(0.95, 0.5 + (rsi - 50) / 100)
        elif rsi < 45 and roc < 0:
            direction = SignalDirection.SELL
            confidence = min(0.95, 0.5 + (50 - rsi) / 100)

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
