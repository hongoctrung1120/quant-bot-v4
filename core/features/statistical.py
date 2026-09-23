"""Statistical features."""

from __future__ import annotations

from collections import deque
from typing import Optional

from core.features.base import Feature, FeatureSpec
from core.models.bar import Bar


class RollingStd(Feature):
    """Rolling standard deviation of close."""

    def __init__(self, period: int = 20) -> None:
        self._period = period
        self._closes: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"rolling_std_{self._period}",
            formula=f"std(close, {self._period})",
            input_data="close",
            lookback=self._period,
            warmup_period=self._period,
        )

    def update(self, bar: Bar) -> Optional[float]:
        import math

        self._closes.append(bar.close)
        if len(self._closes) < self._period:
            return None
        mean = sum(self._closes) / len(self._closes)
        var = sum((c - mean) ** 2 for c in self._closes) / (len(self._closes) - 1)
        return math.sqrt(var) if var > 0 else 0.0

    def reset(self) -> None:
        self._closes.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._closes) >= self._period


class RollingSkewness(Feature):
    """Rolling skewness of returns."""

    def __init__(self, period: int = 20) -> None:
        self._period = period
        self._prev_close: Optional[float] = None
        self._returns: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"rolling_skew_{self._period}",
            formula=f"skew(returns, {self._period})",
            input_data="close",
            lookback=self._period + 1,
            warmup_period=self._period + 1,
        )

    def update(self, bar: Bar) -> Optional[float]:
        if self._prev_close is not None and self._prev_close > 0:
            self._returns.append((bar.close - self._prev_close) / self._prev_close)
        self._prev_close = bar.close
        if len(self._returns) < self._period:
            return None
        n = len(self._returns)
        mean = sum(self._returns) / n
        m2 = sum((r - mean) ** 2 for r in self._returns) / n
        m3 = sum((r - mean) ** 3 for r in self._returns) / n
        if m2 == 0:
            return 0.0
        return m3 / (m2 ** 1.5)

    def reset(self) -> None:
        self._prev_close = None
        self._returns.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._returns) >= self._period
