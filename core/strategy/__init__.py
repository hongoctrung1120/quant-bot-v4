"""Baseline trading strategies."""

from core.strategy.breakout import BreakoutStrategy
from core.strategy.mean_reversion import MeanReversionStrategy
from core.strategy.momentum import MomentumStrategy
from core.strategy.multi_resolution import MultiResolutionCoordinator
from core.strategy.trend import TrendFollowingStrategy

__all__ = [
    "TrendFollowingStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "MultiResolutionCoordinator",
]
