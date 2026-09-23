"""Higher-timeframe trend, lower-timeframe pullback entry.

Direction comes from the H4 moving-average structure; the entry waits for a
counter-move to exhaust on H1 rather than buying strength. Entering on the
pullback is what keeps the stop close enough that the reward-to-risk is worth
taking, which matters far more than win rate for a trend system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fx.account import FxPosition
from fx.strategy.base import (
    FxSignal,
    FxStrategy,
    ManageAction,
    MarketContext,
    atr_trailing_stop,
)


@dataclass
class _PullbackState:
    armed_long: bool = False
    armed_short: bool = False
    bars_since_arm: int = 0


class TrendPullbackStrategy(FxStrategy):
    def __init__(
        self,
        pullback_rsi_long: float = 45.0,
        pullback_rsi_short: float = 55.0,
        arm_expiry_bars: int = 12,
        stop_atr_multiple: float = 1.8,
        target_r: float = 2.5,
        trail_activate_r: float = 1.0,
        trail_atr_multiple: float = 2.5,
        min_trend_separation: float = 0.0005,
    ) -> None:
        self.pullback_rsi_long = pullback_rsi_long
        self.pullback_rsi_short = pullback_rsi_short
        self.arm_expiry_bars = arm_expiry_bars
        self.stop_atr_multiple = stop_atr_multiple
        self.target_r = target_r
        self.trail_activate_r = trail_activate_r
        self.trail_atr_multiple = trail_atr_multiple
        self.min_trend_separation = min_trend_separation
        self._state: dict[str, _PullbackState] = {}

    @property
    def strategy_id(self) -> str:
        return "trend_pullback"

    def reset(self) -> None:
        self._state.clear()

    def _trend_direction(self, ctx: MarketContext) -> Optional[str]:
        fast = ctx.htf_features.get("ema_50")
        slow = ctx.htf_features.get("ema_200")
        if fast is None or slow is None or slow <= 0:
            return None
        separation = (fast - slow) / slow
        if separation > self.min_trend_separation:
            return "LONG"
        if separation < -self.min_trend_separation:
            return "SHORT"
        return None

    def on_bar(self, ctx: MarketContext) -> Optional[FxSignal]:
        state = self._state.setdefault(ctx.symbol, _PullbackState())
        rsi = ctx.features.get("rsi_14")
        atr = ctx.features.get("atr_14")
        if rsi is None or atr is None or atr <= 0:
            return None

        trend = self._trend_direction(ctx)
        if trend is None:
            state.armed_long = state.armed_short = False
            return None

        if state.armed_long or state.armed_short:
            state.bars_since_arm += 1
            if state.bars_since_arm > self.arm_expiry_bars:
                state.armed_long = state.armed_short = False

        signal: Optional[FxSignal] = None

        if trend == "LONG":
            state.armed_short = False
            if rsi < self.pullback_rsi_long:
                if not state.armed_long:
                    state.armed_long = True
                    state.bars_since_arm = 0
            elif state.armed_long:
                state.armed_long = False
                stop = ctx.bar.close - self.stop_atr_multiple * atr
                signal = self._build(ctx, "LONG", stop, rsi)
        else:
            state.armed_long = False
            if rsi > self.pullback_rsi_short:
                if not state.armed_short:
                    state.armed_short = True
                    state.bars_since_arm = 0
            elif state.armed_short:
                state.armed_short = False
                stop = ctx.bar.close + self.stop_atr_multiple * atr
                signal = self._build(ctx, "SHORT", stop, rsi)

        return signal

    def _build(
        self, ctx: MarketContext, direction: str, stop: float, rsi: float
    ) -> FxSignal:
        risk = abs(ctx.bar.close - stop)
        target = (
            ctx.bar.close + self.target_r * risk
            if direction == "LONG"
            else ctx.bar.close - self.target_r * risk
        )
        return FxSignal(
            symbol=ctx.symbol,
            direction=direction,
            stop_price=stop,
            target_price=target,
            strategy_id=self.strategy_id,
            confidence=0.6,
            reason=f"htf_trend_pullback rsi={rsi:.0f}",
        )

    def manage(self, position: FxPosition, ctx: MarketContext) -> Optional[ManageAction]:
        return atr_trailing_stop(
            position, ctx, self.trail_activate_r, self.trail_atr_multiple
        )
