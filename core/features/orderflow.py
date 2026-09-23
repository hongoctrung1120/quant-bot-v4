"""Order flow features."""

from __future__ import annotations

from typing import Optional

from core.features.base import Feature, FeatureSpec
from core.models.bar import Bar


class VWAPDeviation(Feature):
    """Close deviation from bar VWAP."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="vwap_deviation",
            formula="(close - vwap) / vwap",
            input_data="close,vwap",
            lookback=1,
            warmup_period=1,
        )

    def update(self, bar: Bar) -> Optional[float]:
        if bar.vwap is None or bar.vwap == 0:
            return None
        return (bar.close - bar.vwap) / bar.vwap

    def reset(self) -> None:
        pass


class OrderFlowImbalance(Feature):
    """Buy/sell volume imbalance from bar."""

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            name="order_flow_imbalance",
            formula="(buy_vol - sell_vol) / total_vol",
            input_data="buy_volume,sell_volume",
            lookback=1,
            warmup_period=1,
        )

    def update(self, bar: Bar) -> Optional[float]:
        if bar.order_flow_imbalance is not None:
            return bar.order_flow_imbalance
        if bar.buy_volume is None or bar.sell_volume is None:
            return None
        total = bar.buy_volume + bar.sell_volume
        if total == 0:
            return None
        return (bar.buy_volume - bar.sell_volume) / total

    def reset(self) -> None:
        pass
