"""Strategy interface and shared trade-management helpers.

Strategies decide direction and where the idea is wrong. They never decide
size, never place orders, and cannot see anything beyond the bar that just
closed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from core.features.momentum import RSI, ROC
from core.features.price import EMA, LogReturn, RollingReturn, SMA
from core.features.volatility import ATR, RealizedVolatility
from core.features.volume import RelativeVolume
from core.models.bar import Bar
from fx.account import FxPosition
from fx.instruments import FxInstrument
from fx.sessions import Session


def fx_feature_set() -> list:
    """Feature calculators used across the FX strategies."""
    return [
        LogReturn(),
        RollingReturn(20),
        EMA(20),
        EMA(50),
        EMA(200),
        SMA(20),
        ATR(14),
        RealizedVolatility(20),
        RSI(14),
        ROC(10),
        RelativeVolume(20),
    ]


@dataclass
class MarketContext:
    timestamp: datetime
    symbol: str
    bar: Bar
    features: dict[str, float]
    session: Session
    instrument: FxInstrument
    htf_features: dict[str, float] = field(default_factory=dict)
    history: Sequence[Bar] = field(default_factory=tuple)

    def feature(self, name: str, default: Optional[float] = None) -> Optional[float]:
        value = self.features.get(name)
        return default if value is None else value

    def htf(self, name: str, default: Optional[float] = None) -> Optional[float]:
        value = self.htf_features.get(name)
        return default if value is None else value


@dataclass
class FxSignal:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    stop_price: float
    strategy_id: str
    confidence: float = 0.5
    target_price: Optional[float] = None
    reason: str = ""

    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"


@dataclass
class ManageAction:
    new_stop: Optional[float] = None
    exit_now: bool = False
    reason: str = ""


class FxStrategy(ABC):
    """Base strategy. Override `on_bar` for per-symbol logic."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        ...

    def on_bar(self, ctx: MarketContext) -> Optional[FxSignal]:
        return None

    def on_slice(
        self, timestamp: datetime, contexts: dict[str, MarketContext]
    ) -> list[FxSignal]:
        """Called once per synchronized bar close across the whole universe."""
        signals: list[FxSignal] = []
        for ctx in contexts.values():
            signal = self.on_bar(ctx)
            if signal is not None:
                signals.append(signal)
        return signals

    def manage(
        self, position: FxPosition, ctx: MarketContext
    ) -> Optional[ManageAction]:
        return None

    def reset(self) -> None:
        return None


def r_progress(position: FxPosition, price: float) -> Optional[float]:
    """How far price has travelled in favour, measured in initial-risk units."""
    if position.initial_stop_price is None:
        return None
    risk_distance = abs(position.entry_price - position.initial_stop_price)
    if risk_distance <= 0:
        return None
    move = (
        price - position.entry_price
        if position.is_long
        else position.entry_price - price
    )
    return move / risk_distance


def atr_trailing_stop(
    position: FxPosition,
    ctx: MarketContext,
    activate_at_r: float = 1.0,
    atr_multiple: float = 2.0,
) -> Optional[ManageAction]:
    """Trail the stop once the trade is up `activate_at_r`, never loosening it.

    Letting winners run is what pays for a low win rate; the trail only ever
    moves in the profitable direction.
    """
    atr = ctx.features.get("atr_14")
    if not atr or atr <= 0:
        return None

    progress = r_progress(position, ctx.bar.close)
    if progress is None or progress < activate_at_r:
        return None

    if position.is_long:
        candidate = ctx.bar.close - atr_multiple * atr
        if position.stop_price is None or candidate > position.stop_price:
            return ManageAction(new_stop=candidate, reason="atr_trail")
    else:
        candidate = ctx.bar.close + atr_multiple * atr
        if position.stop_price is None or candidate < position.stop_price:
            return ManageAction(new_stop=candidate, reason="atr_trail")
    return None
