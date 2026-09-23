"""Mean reversion strategy."""

from __future__ import annotations

from typing import Optional

from core.interfaces import Strategy
from core.models.bar import Bar
from core.models.regime import MarketRegime, RegimeOutput
from core.models.signal import Signal, SignalDirection


class MeanReversionStrategy(Strategy):
    """VWAP deviation mean reversion."""

    def __init__(
        self,
        strategy_id: str = "mean_reversion",
        zscore_entry: float = 0.002,
        min_confidence: float = 0.5,
    ) -> None:
        self._strategy_id = strategy_id
        self._zscore_entry = zscore_entry
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
        vwap_dev = features.get("vwap_deviation")
        if vwap_dev is None:
            return None

        if regime.regime not in (
            MarketRegime.MEAN_REVERTING,
            MarketRegime.LOW_VOLATILITY,
        ):
            return None

        direction: Optional[SignalDirection] = None
        confidence = regime.confidence

        if vwap_dev < -self._zscore_entry:
            direction = SignalDirection.BUY
        elif vwap_dev > self._zscore_entry:
            direction = SignalDirection.SELL

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
