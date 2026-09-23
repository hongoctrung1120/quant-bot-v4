"""Math utilities for quantitative computations."""

from __future__ import annotations

import math
from typing import Sequence


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(value, max_val))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0. Returns empty dict if sum is zero."""
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


def validate_allocation_sum(
    allocations: dict[str, float],
    max_total: float = 100.0,
    tolerance: float = 0.01,
) -> bool:
    """Validate allocation weights do not exceed max_total."""
    total = sum(allocations.values())
    return total <= max_total + tolerance


def annualized_sharpe(
    returns: Sequence[float],
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio from a return series."""
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_ret = sum(returns) / n
    variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    excess = mean_ret - risk_free_rate / periods_per_year
    return (excess / std) * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Compute maximum drawdown from an equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd
