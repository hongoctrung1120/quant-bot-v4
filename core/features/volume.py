"""Volume features."""

from __future__ import annotations

from collections import deque
from typing import Optional

from core.features.base import Feature, FeatureSpec
from core.models.bar import Bar


class RelativeVolume(Feature):
    """Current volume / SMA(volume)."""

    def __init__(self, period: int = 20) -> None:
        self._period = period
        self._volumes: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"relative_volume_{self._period}",
            formula=f"volume / SMA(volume, {self._period})",
            input_data="volume",
            lookback=self._period,
            warmup_period=self._period,
        )

    def update(self, bar: Bar) -> Optional[float]:
        self._volumes.append(bar.volume)
        if len(self._volumes) < self._period:
            return None
        avg = sum(self._volumes) / len(self._volumes)
        if avg == 0:
            return None
        return bar.volume / avg

    def reset(self) -> None:
        self._volumes.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._volumes) >= self._period


class DollarVolumeSMA(Feature):
    """SMA of dollar volume."""

    def __init__(self, period: int = 20) -> None:
        self._period = period
        self._values: deque[float] = deque(maxlen=period)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=f"dollar_volume_sma_{self._period}",
            formula=f"SMA(dollar_volume, {self._period})",
            input_data="dollar_volume",
            lookback=self._period,
            warmup_period=self._period,
        )

    def update(self, bar: Bar) -> Optional[float]:
        self._values.append(bar.dollar_volume)
        if len(self._values) < self._period:
            return None
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        self._values.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._values) >= self._period
