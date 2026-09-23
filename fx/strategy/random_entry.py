"""Random-entry benchmark.

The honest comparison for any strategy is not zero — it is the same exits, the
same stops, the same position sizing and the same risk governor, but with the
entry timing replaced by a coin flip. Whatever this benchmark earns comes from
trade management, not from the signal. A strategy that cannot beat it has no
entry edge, however good its equity curve looks.
"""

from __future__ import annotations

import random
from typing import Optional

from fx.account import FxPosition
from fx.strategy.base import (
    FxSignal,
    FxStrategy,
    ManageAction,
    MarketContext,
    atr_trailing_stop,
)


class RandomEntryStrategy(FxStrategy):
    def __init__(
        self,
        entry_probability: float = 0.01,
        stop_atr_multiple: float = 1.8,
        target_r: float = 2.5,
        trail_activate_r: float = 1.0,
        trail_atr_multiple: float = 2.5,
        seed: int = 0,
        strategy_id: str = "random_entry",
    ) -> None:
        self.entry_probability = entry_probability
        self.stop_atr_multiple = stop_atr_multiple
        self.target_r = target_r
        self.trail_activate_r = trail_activate_r
        self.trail_atr_multiple = trail_atr_multiple
        self._rng = random.Random(seed)
        self._seed = seed
        self._strategy_id = strategy_id

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def on_bar(self, ctx: MarketContext) -> Optional[FxSignal]:
        atr = ctx.features.get("atr_14")
        if not atr or atr <= 0:
            return None
        if self._rng.random() >= self.entry_probability:
            return None

        is_long = self._rng.random() < 0.5
        offset = self.stop_atr_multiple * atr
        close = ctx.bar.close
        stop = close - offset if is_long else close + offset
        target = (
            close + self.target_r * offset
            if is_long
            else close - self.target_r * offset
        )
        return FxSignal(
            symbol=ctx.symbol,
            direction="LONG" if is_long else "SHORT",
            stop_price=stop,
            target_price=target,
            strategy_id=self.strategy_id,
            confidence=0.5,
            reason="random_entry_benchmark",
        )

    def manage(self, position: FxPosition, ctx: MarketContext) -> Optional[ManageAction]:
        return atr_trailing_stop(
            position, ctx, self.trail_activate_r, self.trail_atr_multiple
        )
