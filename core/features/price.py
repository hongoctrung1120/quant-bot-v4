"""Price-based features."""

from __future__ import annotations

from collections import deque
from typing import Optional

from core.features.base import Feature, FeatureSpec
from core.models.bar import Bar


class LogReturn(Feature):
    """Log return: ln(close_t / close_{t-1})."""

    def __init__(self) -> None:
        self._prev_close: Optional[float] = None

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="log_return",
            formula="ln(close_t / close_{t-1})",
            input_data="close",
            lookback=1,
            warmup_period=1,
        )

    def update(self, bar: Bar) -> Optional[float]:
        if self._prev_close is None:
            self._prev_close = bar.close
            return None
        import math

        value = math.log(bar.close / self._prev_close)
        self._prev_close = bar.close
        return value

    def reset(self) -> None:
        self._prev_close = None

    @property
    def is_ready(self) -> bool:
        return self._prev_close is not None


class RollingReturn(Feature):
    """Rolling return over N bars."""

    def __init__(self, period: int = 10) -> None:
        self._period = period
        self._closes: deque[float] = deque(maxlen=period + 1)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"rolling_return_{self._period}",
            formula=f"(close_t / close_{{t-{self._period}}}) - 1",
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
        return (bar.close / old) - 1.0

    def reset(self) -> None:
        self._closes.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._closes) > self._period


class EMA(Feature):
    """Exponential moving average."""

    def __init__(self, period: int = 12) -> None:
        self._period = period
        self._alpha = 2.0 / (period + 1)
        self._value: Optional[float] = None
        self._count = 0

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"ema_{self._period}",
            formula=f"EMA({self._period})",
            input_data="close",
            lookback=self._period,
            warmup_period=self._period,
        )

    def update(self, bar: Bar) -> Optional[float]:
        self._count += 1
        if self._value is None:
            self._value = bar.close
        else:
            self._value = self._alpha * bar.close + (1 - self._alpha) * self._value
        if self._count < self._period:
            return None
        return self._value

    def reset(self) -> None:
        self._value = None
        self._count = 0

    @property
    def is_ready(self) -> bool:
        return self._count >= self._period


class SMA(Feature):
    """Simple moving average."""

    def __init__(self, period: int = 20) -> None:
        self._period = period
        self._values: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"sma_{self._period}",
            formula=f"SMA({self._period})",
            input_data="close",
            lookback=self._period,
            warmup_period=self._period,
        )

    def update(self, bar: Bar) -> Optional[float]:
        self._values.append(bar.close)
        if len(self._values) < self._period:
            return None
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        self._values.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._values) >= self._period
