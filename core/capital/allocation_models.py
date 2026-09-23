"""Capital allocation models with explicit constraint handling."""
from __future__ import annotations
from utils.math import clamp, normalize_weights, safe_divide


def fixed_weight_allocation(weights: dict[str, float], budget_pct: float = 100.0, **kwargs) -> dict[str, float]:
    # ``max_pct`` is retained as a backwards-compatible alias for tests/configs.
    if "max_pct" in kwargs:
        budget_pct = kwargs["max_pct"]
    positive = {k: max(float(v), 0.0) for k, v in weights.items()}
    total = sum(positive.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: v / total * budget_pct for k, v in positive.items()}


def equal_weight_allocation(symbols: list[str], budget_pct: float = 100.0) -> dict[str, float]:
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return {}
    return {s: budget_pct / len(symbols) for s in symbols}


def inverse_volatility_allocation(volatilities: dict[str, float], budget_pct: float = 100.0, **kwargs) -> dict[str, float]:
    if "max_pct" in kwargs:
        budget_pct = kwargs["max_pct"]
    inv = {k: safe_divide(1.0, float(v), default=0.0) for k, v in volatilities.items() if v > 0}
    normalized = normalize_weights(inv)
    return {k: v * budget_pct for k, v in normalized.items()}


def apply_regime_adjustments(weights: dict[str, float], regime: str, adjustments: dict[str, dict[str, float]]) -> dict[str, float]:
    regime_adj = adjustments.get(regime, {})
    adjusted = {k: max(v * regime_adj.get(k, regime_adj.get("all", 1.0)), 0.0) for k, v in weights.items()}
    return fixed_weight_allocation(adjusted, sum(weights.values()))


def apply_confidence_weights(weights: dict[str, float], confidences: dict[str, float] | None) -> dict[str, float]:
    if not confidences:
        return dict(weights)
    adjusted = {k: max(v, 0.0) * clamp(confidences.get(k, 1.0), 0.0, 1.0) for k, v in weights.items()}
    return fixed_weight_allocation(adjusted, sum(weights.values()))


def enforce_limits(allocations: dict[str, float], max_single: float, total_budget_pct: float | None = None) -> dict[str, float]:
    if not allocations:
        return {}
    target = sum(allocations.values()) if total_budget_pct is None else total_budget_pct
    capped = {k: clamp(float(v), 0.0, max_single) for k, v in allocations.items()}
    for _ in range(len(capped) + 2):
        total = sum(capped.values())
        if total >= target - 1e-9:
            break
        room = {k: max_single - v for k, v in capped.items() if v < max_single - 1e-12}
        room_total = sum(room.values())
        if room_total <= 0:
            break
        deficit = min(target - total, room_total)
        for k, r in room.items():
            capped[k] += deficit * r / room_total
    return capped
