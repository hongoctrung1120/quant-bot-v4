"""Volatility features."""

from __future__ import annotations

from collections import deque
from typing import Optional

from core.features.base import Feature, FeatureSpec
from core.models.bar import Bar


class ATR(Feature):
    """Average True Range."""

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._prev_close: Optional[float] = None
        self._trs: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"atr_{self._period}",
            formula=f"ATR({self._period})",
            input_data="high,low,close",
            lookback=self._period,
            warmup_period=self._period,
        )

    def update(self, bar: Bar) -> Optional[float]:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        self._prev_close = bar.close
        self._trs.append(tr)
        if len(self._trs) < self._period:
            return None
        return sum(self._trs) / len(self._trs)

    def reset(self) -> None:
        self._prev_close = None
        self._trs.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._trs) >= self._period


class RealizedVolatility(Feature):
    """Rolling std of log returns."""

    def __init__(self, period: int = 20) -> None:
        self._period = period
        self._prev_close: Optional[float] = None
        self._returns: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"realized_vol_{self._period}",
            formula=f"std(log_returns, {self._period})",
            input_data="close",
            lookback=self._period + 1,
            warmup_period=self._period + 1,
        )

    def update(self, bar: Bar) -> Optional[float]:
        import math

        if self._prev_close is not None and self._prev_close > 0:
            self._returns.append(math.log(bar.close / self._prev_close))
        self._prev_close = bar.close
        if len(self._returns) < self._period:
            return None
        mean = sum(self._returns) / len(self._returns)
        var = sum((r - mean) ** 2 for r in self._returns) / (len(self._returns) - 1)
        return math.sqrt(var) if var > 0 else 0.0

    def reset(self) -> None:
        self._prev_close = None
        self._returns.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._returns) >= self._period


class BollingerBandWidth(Feature):
    """(upper - lower) / middle using SMA and std."""

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self._period = period
        self._num_std = num_std
        self._closes: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"bb_width_{self._period}",
            formula=f"BB_width({self._period}, {self._num_std})",
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
        std = math.sqrt(var) if var > 0 else 0.0
        if mean == 0:
            return None
        upper = mean + self._num_std * std
        lower = mean - self._num_std * std
        return (upper - lower) / mean

    def reset(self) -> None:
        self._closes.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._closes) >= self._period
