"""Momentum features."""

from __future__ import annotations

from collections import deque
from typing import Optional

from core.features.base import Feature, FeatureSpec
from core.features.price import EMA
from core.models.bar import Bar


class RSI(Feature):
    """Relative Strength Index."""

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._prev_close: Optional[float] = None
        self._gains: deque[float] = deque(maxlen=period)
        self._losses: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"rsi_{self._period}",
            formula=f"RSI({self._period})",
            input_data="close",
            lookback=self._period + 1,
            warmup_period=self._period + 1,
        )

    def update(self, bar: Bar) -> Optional[float]:
        if self._prev_close is not None:
            change = bar.close - self._prev_close
            self._gains.append(max(change, 0.0))
            self._losses.append(max(-change, 0.0))
        self._prev_close = bar.close
        if len(self._gains) < self._period:
            return None
        avg_gain = sum(self._gains) / len(self._gains)
        avg_loss = sum(self._losses) / len(self._losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def reset(self) -> None:
        self._prev_close = None
        self._gains.clear()
        self._losses.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._gains) >= self._period


class ROC(Feature):
    """Rate of change."""

    def __init__(self, period: int = 10) -> None:
        self._period = period
        self._closes: deque[float] = deque(maxlen=period + 1)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"roc_{self._period}",
            formula=f"(close - close_n) / close_n",
            input_data="close",
            lookback=self._period,
            warmup_period=self._period,
        )

    def update(self, bar: Bar) -> Optional[float]:
        self._closes.append(bar.close)
        if len(self._closes) <= self._period:
            return None
        old = self._closes[0]
        if old == 0:
            return None
        return ((bar.close - old) / old) * 100.0

    def reset(self) -> None:
        self._closes.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._closes) > self._period


class MACD(Feature):
    """MACD line (fast EMA - slow EMA)."""

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        self._fast_ema = EMA(fast)
        self._slow_ema = EMA(slow)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="macd",
            formula="EMA(12) - EMA(26)",
            input_data="close",
            lookback=26,
            warmup_period=26,
        )

    def update(self, bar: Bar) -> Optional[float]:
        fast = self._fast_ema.update(bar)
        slow = self._slow_ema.update(bar)
        if fast is None or slow is None:
            return None
        return fast - slow

    def reset(self) -> None:
        self._fast_ema.reset()
        self._slow_ema.reset()

    @property
    def is_ready(self) -> bool:
        return self._fast_ema.is_ready and self._slow_ema.is_ready
