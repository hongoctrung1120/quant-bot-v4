"""Rule-based market regime detection."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.interfaces import RegimeEngine
from core.models.regime import MarketRegime, RegimeOutput


class RuleBasedRegimeEngine(RegimeEngine):
    """Deterministic regime classifier using trend and volatility features.

    Uses only features available at bar close timestamp.
    """

    def __init__(
        self,
        adx_threshold: float = 25.0,
        vol_high_percentile: float = 0.7,
        vol_low_percentile: float = 0.3,
    ) -> None:
        self._adx_threshold = adx_threshold
        self._vol_history: dict[str, list[float]] = {}
        self._vol_high_pct = vol_high_percentile
        self._vol_low_pct = vol_low_percentile

    def on_features(
        self,
        symbol: str,
        features: dict[str, float],
        timestamp: datetime,
    ) -> RegimeOutput:
        used: list[str] = []
        ema12 = features.get("ema_12")
        ema26 = features.get("ema_26")
        rsi = features.get("rsi_14")
        realized_vol = features.get("realized_vol_20")
        bb_width = features.get("bb_width_20")

        if ema12 is not None:
            used.append("ema_12")
        if ema26 is not None:
            used.append("ema_26")
        if rsi is not None:
            used.append("rsi_14")
        if realized_vol is not None:
            used.append("realized_vol_20")
        if bb_width is not None:
            used.append("bb_width_20")

        regime = MarketRegime.TRANSITION
        confidence = 0.5

        vol_rank = self._vol_rank(symbol, realized_vol)

        if vol_rank is not None and vol_rank >= self._vol_high_pct:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = 0.6 + 0.3 * vol_rank
        elif vol_rank is not None and vol_rank <= self._vol_low_pct:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 0.6 + 0.3 * (1 - vol_rank)
        elif ema12 is not None and ema26 is not None:
            spread = (ema12 - ema26) / ema26 if ema26 != 0 else 0.0
            if spread > 0.005:
                regime = MarketRegime.TREND_BULL
                confidence = min(0.95, 0.6 + abs(spread) * 10)
            elif spread < -0.005:
                regime = MarketRegime.TREND_BEAR
                confidence = min(0.95, 0.6 + abs(spread) * 10)
            elif rsi is not None and 40 <= rsi <= 60:
                regime = MarketRegime.MEAN_REVERTING
                confidence = 0.65

        return RegimeOutput(
            regime=regime,
            confidence=min(confidence, 1.0),
            timestamp=timestamp,
            symbol=symbol,
            features_used=tuple(used),
        )

    def _vol_rank(self, symbol: str, vol: Optional[float]) -> Optional[float]:
        if vol is None:
            return None
        history = self._vol_history.setdefault(symbol, [])
        history.append(vol)
        if len(history) < 20:
            return None
        sorted_hist = sorted(history[-100:])
        rank = sorted_hist.index(vol) / max(len(sorted_hist) - 1, 1)
        return rank
