"""Feature computation orchestrator."""

from __future__ import annotations

import copy
import logging
from typing import Optional

from core.features.base import Feature
from core.features.momentum import MACD, ROC, RSI
from core.features.orderflow import OrderFlowImbalance, VWAPDeviation
from core.features.price import EMA, LogReturn, RollingReturn, SMA
from core.features.statistical import RollingSkewness, RollingStd
from core.features.volatility import ATR, BollingerBandWidth, RealizedVolatility
from core.features.volume import DollarVolumeSMA, RelativeVolume
from core.interfaces import FeatureEngine as FeatureEngineABC
from core.models.bar import Bar

logger = logging.getLogger(__name__)


def default_features() -> list[Feature]:
    """Standard feature set for baseline strategies."""
    return [
        LogReturn(),
        RollingReturn(10),
        EMA(12),
        EMA(26),
        SMA(20),
        ATR(14),
        RealizedVolatility(20),
        BollingerBandWidth(20),
        RSI(14),
        ROC(10),
        MACD(),
        RelativeVolume(20),
        DollarVolumeSMA(20),
        VWAPDeviation(),
        OrderFlowImbalance(),
        RollingStd(20),
        RollingSkewness(20),
    ]


class FeatureEngine(FeatureEngineABC):
    """Compute features from closed bars only (no look-ahead)."""

    def __init__(self, features: Optional[list[Feature]] = None) -> None:
        # Feature state must be isolated per symbol. Sharing state across
        # BTC/ETH/etc. would contaminate rolling windows and create invalid
        # cross-asset features.
        self._feature_templates = features or default_features()
        self._features_by_symbol: dict[str, list[Feature]] = {}
        self._latest_by_symbol: dict[str, dict[str, float]] = {}
        self._history_by_symbol: dict[str, dict[str, list[float]]] = {}

    @property
    def feature_names(self) -> list[str]:
        return [f.spec.name for f in self._feature_templates]

    def on_bar(self, bar: Bar) -> dict[str, float]:
        features = self._features_by_symbol.setdefault(
            bar.symbol, copy.deepcopy(self._feature_templates)
        )
        latest = self._latest_by_symbol.setdefault(bar.symbol, {})
        history = self._history_by_symbol.setdefault(bar.symbol, {})

        result: dict[str, float] = {}
        for feature in features:
            value = feature.update(bar)
            if value is not None:
                name = feature.spec.name
                result[name] = value
                latest[name] = value
                history.setdefault(name, []).append(value)
        return result

    def get_feature(self, name: str) -> Optional[float]:
        # Backward-compatible accessor for single-symbol use. For multi-symbol
        # research, use get_feature_for_symbol().
        for latest in self._latest_by_symbol.values():
            if name in latest:
                return latest[name]
        return None

    def get_history(self, name: str) -> list[float]:
        values: list[float] = []
        for history in self._history_by_symbol.values():
            values.extend(history.get(name, []))
        return list(values)

    def get_feature_for_symbol(self, symbol: str, name: str) -> Optional[float]:
        return self._latest_by_symbol.get(symbol, {}).get(name)

    def get_history_for_symbol(self, symbol: str, name: str) -> list[float]:
        return list(self._history_by_symbol.get(symbol, {}).get(name, []))

    def reset(self) -> None:
        self._features_by_symbol.clear()
        self._latest_by_symbol.clear()
        self._history_by_symbol.clear()

    def reset_symbol(self, symbol: str) -> None:
        self._features_by_symbol.pop(symbol, None)
        self._latest_by_symbol.pop(symbol, None)
        self._history_by_symbol.pop(symbol, None)
