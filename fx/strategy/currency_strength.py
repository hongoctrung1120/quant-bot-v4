"""Cross-sectional currency-strength momentum.

Single-pair trend following asks "is EURUSD going up". This asks the better
question: "which currency is strongest and which is weakest, right now". Every
pair is decomposed into its two currencies, the currencies are ranked, and the
trade expresses the widest strength gap available in the universe.

Cross-sectional momentum — ranking assets and trading the spread between the
extremes — has far broader evidence behind it across asset classes than
time-series indicator rules on one instrument.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from fx.account import FxPosition
from fx.strategy.base import (
    FxSignal,
    FxStrategy,
    ManageAction,
    MarketContext,
    atr_trailing_stop,
)

MOMENTUM_LOOKBACK = 20


class CurrencyStrengthStrategy(FxStrategy):
    def __init__(
        self,
        evaluate_hour: int = 8,
        min_score_spread: float = 0.8,
        stop_atr_multiple: float = 2.0,
        target_r: float = 2.0,
        trail_activate_r: float = 1.0,
        trail_atr_multiple: float = 2.5,
        max_new_positions: int = 1,
    ) -> None:
        self.evaluate_hour = evaluate_hour
        self.min_score_spread = min_score_spread
        self.stop_atr_multiple = stop_atr_multiple
        self.target_r = target_r
        self.trail_activate_r = trail_activate_r
        self.trail_atr_multiple = trail_atr_multiple
        self.max_new_positions = max_new_positions
        self._last_evaluated: Optional[object] = None

    @property
    def strategy_id(self) -> str:
        return "currency_strength"

    def reset(self) -> None:
        self._last_evaluated = None

    @staticmethod
    def _normalized_momentum(ctx: MarketContext) -> Optional[float]:
        """Momentum in volatility units, so pairs are comparable."""
        ret = ctx.features.get(f"rolling_return_{MOMENTUM_LOOKBACK}")
        vol = ctx.features.get("realized_vol_20")
        if ret is None or vol is None or vol <= 0:
            return None
        return ret / (vol * math.sqrt(MOMENTUM_LOOKBACK))

    def currency_scores(
        self, contexts: dict[str, MarketContext]
    ) -> dict[str, float]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for ctx in contexts.values():
            momentum = self._normalized_momentum(ctx)
            if momentum is None:
                continue
            base, quote = ctx.instrument.base, ctx.instrument.quote
            totals[base] = totals.get(base, 0.0) + momentum
            totals[quote] = totals.get(quote, 0.0) - momentum
            counts[base] = counts.get(base, 0) + 1
            counts[quote] = counts.get(quote, 0) + 1
        return {c: totals[c] / counts[c] for c in totals if counts.get(c)}

    def on_slice(
        self, timestamp: datetime, contexts: dict[str, MarketContext]
    ) -> list[FxSignal]:
        if timestamp.hour != self.evaluate_hour:
            return []
        day = timestamp.date()
        if self._last_evaluated == day:
            return []
        self._last_evaluated = day

        scores = self.currency_scores(contexts)
        if len(scores) < 2:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        signals: list[FxSignal] = []

        for strong, strong_score in ranked:
            for weak, weak_score in reversed(ranked):
                if strong == weak:
                    continue
                spread = strong_score - weak_score
                if spread < self.min_score_spread:
                    continue
                signal = self._signal_for(contexts, strong, weak, spread)
                if signal is not None:
                    signals.append(signal)
                    if len(signals) >= self.max_new_positions:
                        return signals
        return signals

    def _signal_for(
        self,
        contexts: dict[str, MarketContext],
        strong: str,
        weak: str,
        spread: float,
    ) -> Optional[FxSignal]:
        direct = f"{strong}{weak}"
        inverse = f"{weak}{strong}"

        if direct in contexts:
            ctx, direction = contexts[direct], "LONG"
        elif inverse in contexts:
            ctx, direction = contexts[inverse], "SHORT"
        else:
            return None

        atr = ctx.features.get("atr_14")
        if not atr or atr <= 0:
            return None

        close = ctx.bar.close
        offset = self.stop_atr_multiple * atr
        stop = close - offset if direction == "LONG" else close + offset
        target = (
            close + self.target_r * offset
            if direction == "LONG"
            else close - self.target_r * offset
        )
        return FxSignal(
            symbol=ctx.symbol,
            direction=direction,
            stop_price=stop,
            target_price=target,
            strategy_id=self.strategy_id,
            confidence=min(0.4 + spread / 5.0, 0.9),
            reason=f"strength {strong}>{weak} spread={spread:.2f}",
        )

    def manage(self, position: FxPosition, ctx: MarketContext) -> Optional[ManageAction]:
        return atr_trailing_stop(
            position, ctx, self.trail_activate_r, self.trail_atr_multiple
        )
