"""Asian-range breakout at the London open.

The rationale is structural rather than a chart pattern: the Tokyo session is
thin and tends to range, then London arrives with an order of magnitude more
liquidity and repositions the market. The setup trades that liquidity handoff,
which is a property of how the FX day is organised, not a curve-fitted shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from fx import sessions
from fx.account import FxPosition
from fx.strategy.base import (
    FxSignal,
    FxStrategy,
    ManageAction,
    MarketContext,
    atr_trailing_stop,
)


@dataclass
class _DayState:
    day: Optional[date] = None
    high: float = float("-inf")
    low: float = float("inf")
    bars: int = 0
    traded: bool = False

    def reset(self, day: date) -> None:
        self.day = day
        self.high = float("-inf")
        self.low = float("inf")
        self.bars = 0
        self.traded = False

    @property
    def established(self) -> bool:
        return self.bars >= 3 and self.high > self.low


class SessionBreakoutStrategy(FxStrategy):
    def __init__(
        self,
        entry_deadline_hour: int = 12,
        min_range_atr: float = 1.5,
        max_range_atr: float = 6.0,
        target_r: float = 1.5,
        trail_activate_r: float = 1.0,
        trail_atr_multiple: float = 2.0,
        max_stop_atr: float = 4.0,
    ) -> None:
        self.entry_deadline_hour = entry_deadline_hour
        self.min_range_atr = min_range_atr
        self.max_range_atr = max_range_atr
        self.target_r = target_r
        self.trail_activate_r = trail_activate_r
        self.trail_atr_multiple = trail_atr_multiple
        self.max_stop_atr = max_stop_atr
        self._state: dict[str, _DayState] = {}

    @property
    def strategy_id(self) -> str:
        return "session_breakout"

    def reset(self) -> None:
        self._state.clear()

    def on_bar(self, ctx: MarketContext) -> Optional[FxSignal]:
        state = self._state.setdefault(ctx.symbol, _DayState())
        day = ctx.timestamp.date()
        if state.day != day:
            state.reset(day)

        if sessions.in_asian_range_window(ctx.timestamp):
            state.high = max(state.high, ctx.bar.high)
            state.low = min(state.low, ctx.bar.low)
            state.bars += 1
            return None

        if state.traded or not state.established:
            return None
        if not (sessions.LONDON_OPEN_HOUR <= ctx.timestamp.hour < self.entry_deadline_hour):
            return None

        atr = ctx.features.get("atr_14")
        if not atr or atr <= 0:
            return None

        range_size = state.high - state.low
        range_in_atr = range_size / atr
        if not (self.min_range_atr <= range_in_atr <= self.max_range_atr):
            return None

        close = ctx.bar.close
        if close > state.high:
            direction, stop = "LONG", state.low
        elif close < state.low:
            direction, stop = "SHORT", state.high
        else:
            return None

        # A range far wider than current volatility gives an unusable stop.
        if abs(close - stop) > self.max_stop_atr * atr:
            return None

        state.traded = True
        risk = abs(close - stop)
        target = close + self.target_r * risk if direction == "LONG" else close - self.target_r * risk

        return FxSignal(
            symbol=ctx.symbol,
            direction=direction,
            stop_price=stop,
            target_price=target,
            strategy_id=self.strategy_id,
            confidence=min(0.4 + 0.1 * range_in_atr, 0.9),
            reason=f"asian_range_break r={range_in_atr:.1f}atr",
        )

    def manage(self, position: FxPosition, ctx: MarketContext) -> Optional[ManageAction]:
        return atr_trailing_stop(
            position, ctx, self.trail_activate_r, self.trail_atr_multiple
        )
