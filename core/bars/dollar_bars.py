"""Dollar bar construction from trade-level data."""

from __future__ import annotations

import logging
from typing import Optional

from core.bars.base import BarBuilderState
from core.config import BarsConfig
from core.interfaces import BarEngine
from core.models.bar import Bar, BarType
from core.models.trade import Trade

logger = logging.getLogger(__name__)

OVERSHOOT_CARRY_FORWARD = "CARRY_FORWARD"
OVERSHOOT_SPLIT_TRADE = "SPLIT_TRADE"
_QTY_EPS = 1e-12


class DollarBarEngine(BarEngine):
    """Construct dollar bars from immutable trade stream.

    Overshoot policies (configured via ``bars.yaml``):

    CARRY_FORWARD
        The crossing trade is fully included in the closing bar.
        Dollar volume above the threshold is carried to the next bar's
        cumulative counter without discarding excess.

    SPLIT_TRADE
        The crossing trade is split proportionally: the portion needed to
        reach the threshold closes the current bar; the remainder opens
        the next bar.
    """

    def __init__(
        self,
        threshold: float,
        overshoot_policy: str = OVERSHOOT_CARRY_FORWARD,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive: {threshold}")
        if overshoot_policy not in (
            OVERSHOOT_CARRY_FORWARD,
            OVERSHOOT_SPLIT_TRADE,
        ):
            raise ValueError(f"Unknown overshoot policy: {overshoot_policy}")

        self._threshold = threshold
        self._overshoot_policy = overshoot_policy
        self._states: dict[str, BarBuilderState] = {}

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def overshoot_policy(self) -> str:
        return self._overshoot_policy

    def _get_state(self, symbol: str) -> BarBuilderState:
        return self._states.setdefault(
            symbol,
            BarBuilderState(
                symbol=symbol,
                bar_type=BarType.DOLLAR,
                bar_threshold=self._threshold,
            ),
        )

    def on_trade_all(self, trade: Trade) -> list[Bar]:
        """Process a trade and return every bar closed by that trade.

        A single large trade can cross several thresholds. Returning all closed
        bars prevents silent loss of bars in the event-driven backtester.
        """
        state = self._get_state(trade.symbol)
        closed_bars: list[Bar] = []
        remaining_qty = trade.quantity

        while remaining_qty > _QTY_EPS:
            if state.cumulative_dollar >= self._threshold and not state.is_empty():
                closed_bars.append(state.to_bar())
                overshoot = state.cumulative_dollar - self._threshold
                state.reset(carry_dollar=max(overshoot, 0.0))
                continue

            needed_dollar = self._threshold - state.cumulative_dollar
            trade_dollar = trade.price * remaining_qty

            if trade_dollar <= needed_dollar + _QTY_EPS:
                state.add_trade(trade, remaining_qty)
                remaining_qty = 0.0
                if state.cumulative_dollar >= self._threshold - _QTY_EPS:
                    closed_bars.append(state.to_bar())
                    overshoot = state.cumulative_dollar - self._threshold
                    state.reset(carry_dollar=max(overshoot, 0.0))
            elif self._overshoot_policy == OVERSHOOT_SPLIT_TRADE:
                split_qty = needed_dollar / trade.price
                if split_qty <= _QTY_EPS:
                    closed_bars.append(state.to_bar())
                    state.reset()
                    continue
                state.add_trade(trade, split_qty)
                closed_bars.append(state.to_bar())
                state.reset(carry_dollar=0.0)
                remaining_qty -= split_qty
            else:
                # Compatibility mode: the crossing trade remains intact in the
                # closing bar. This mode is intentionally not the default because
                # carrying the overshoot into the next bar can double-count value.
                state.add_trade(trade, remaining_qty)
                closed_bars.append(state.to_bar())
                overshoot = state.cumulative_dollar - self._threshold
                state.reset(carry_dollar=max(overshoot, 0.0))
                remaining_qty = 0.0

        return closed_bars

    def on_trade(self, trade: Trade) -> Optional[Bar]:
        """Compatibility API: return the last bar closed by the trade."""
        bars = self.on_trade_all(trade)
        return bars[-1] if bars else None

    def get_current_bar(self, symbol: str) -> Optional[Bar]:
        state = self._states.get(symbol)
        if state is None or state.is_empty():
            return None
        return state.to_bar()

    def flush(self, symbol: str) -> Optional[Bar]:
        state = self._states.get(symbol)
        if state is None or state.is_empty():
            return None
        bar = state.to_bar()
        state.reset()
        return bar


class MultiResolutionDollarBarEngine:
    """Manage multiple dollar bar engines at different thresholds."""

    def __init__(self, config: BarsConfig) -> None:
        self._engines: dict[str, DollarBarEngine] = {}
        for name, threshold in config.dollar_thresholds.items():
            self._engines[name] = DollarBarEngine(
                threshold=threshold,
                overshoot_policy=config.overshoot_policy,
            )

    @property
    def resolution_names(self) -> list[str]:
        return list(self._engines.keys())

    def get_engine(self, name: str) -> DollarBarEngine:
        if name not in self._engines:
            raise KeyError(f"Unknown dollar bar resolution: {name}")
        return self._engines[name]

    def on_trade_all(self, trade: Trade) -> dict[str, list[Bar]]:
        closed: dict[str, list[Bar]] = {}
        for name, engine in self._engines.items():
            bars = engine.on_trade_all(trade)
            if bars:
                closed[name] = bars
        return closed

    def on_trade(self, trade: Trade) -> dict[str, Bar]:
        """Compatibility API returning the last closed bar per resolution."""
        return {name: bars[-1] for name, bars in self.on_trade_all(trade).items()}

    def get_current_bars(self, symbol: str) -> dict[str, Optional[Bar]]:
        return {
            name: engine.get_current_bar(symbol)
            for name, engine in self._engines.items()
        }

    def flush_all(self, symbol: str) -> dict[str, Bar]:
        flushed: dict[str, Bar] = {}
        for name, engine in self._engines.items():
            bar = engine.flush(symbol)
            if bar is not None:
                flushed[name] = bar
        return flushed
