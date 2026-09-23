"""Bootstrap significance and ruin simulation.

A backtest produces one path. These tools ask the two questions that one path
cannot answer: could this expectancy have come from luck, and how often does
this edge blow the account up before it pays off?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from fx.account import ClosedTrade


@dataclass
class BootstrapResult:
    observed_expectancy_r: float
    mean: float
    ci_low: float
    ci_high: float
    prob_positive: float
    num_trades: int

    @property
    def significant(self) -> bool:
        """Positive expectancy whose 95% interval excludes zero."""
        return self.ci_low > 0.0

    @property
    def significantly_negative(self) -> bool:
        """Reliably loses money — a different finding from "no edge found"."""
        return self.ci_high < 0.0

    @property
    def verdict(self) -> str:
        if self.significant:
            return "edge is statistically supported"
        if self.significantly_negative:
            return "reliably LOSES money (interval entirely below zero)"
        return "indistinguishable from zero: cannot reject luck"


@dataclass
class RuinResult:
    ruin_probability: float
    median_final_multiple: float
    p05_final_multiple: float
    p95_final_multiple: float
    median_max_drawdown_pct: float
    worst_max_drawdown_pct: float
    prob_target_reached: float
    paths: int
    horizon_trades: int


def _r_multiples(trades: Sequence[ClosedTrade]) -> np.ndarray:
    return np.array([t.r_multiple for t in trades], dtype=float)


def bootstrap_expectancy(
    trades: Sequence[ClosedTrade],
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Optional[BootstrapResult]:
    """Confidence interval for expectancy by resampling trades with replacement."""
    values = _r_multiples(trades)
    if values.size < 2:
        return None

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, values.size), replace=True)
    means = samples.mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    return BootstrapResult(
        observed_expectancy_r=float(values.mean()),
        mean=float(means.mean()),
        ci_low=float(np.quantile(means, tail)),
        ci_high=float(np.quantile(means, 1.0 - tail)),
        prob_positive=float((means > 0).mean()),
        num_trades=int(values.size),
    )


def simulate_ruin(
    trades: Sequence[ClosedTrade],
    risk_per_trade_pct: float,
    horizon_trades: Optional[int] = None,
    paths: int = 5_000,
    ruin_drawdown_pct: float = 50.0,
    target_multiple: float = 2.0,
    seed: int = 0,
) -> Optional[RuinResult]:
    """Resample the trade sequence to see how often fixed-fractional betting ruins.

    Each path replays the same edge in a different order, compounding at the
    configured risk. Order matters enormously at small account sizes: the same
    trades in an unlucky sequence can end the account before the edge arrives.
    """
    values = _r_multiples(trades)
    if values.size < 2:
        return None

    horizon = horizon_trades or values.size
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(paths, horizon), replace=True)

    fraction = risk_per_trade_pct / 100.0
    growth = 1.0 + draws * fraction
    # A single trade cannot lose more than the whole account.
    growth = np.maximum(growth, 1e-9)

    equity = np.cumprod(growth, axis=1)
    equity = np.concatenate([np.ones((paths, 1)), equity], axis=1)

    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdown = (running_peak - equity) / running_peak * 100.0
    max_dd = drawdown.max(axis=1)

    final = equity[:, -1]
    return RuinResult(
        ruin_probability=float((max_dd >= ruin_drawdown_pct).mean()),
        median_final_multiple=float(np.median(final)),
        p05_final_multiple=float(np.quantile(final, 0.05)),
        p95_final_multiple=float(np.quantile(final, 0.95)),
        median_max_drawdown_pct=float(np.median(max_dd)),
        worst_max_drawdown_pct=float(max_dd.max()),
        prob_target_reached=float((equity.max(axis=1) >= target_multiple).mean()),
        paths=paths,
        horizon_trades=horizon,
    )


def format_bootstrap(result: Optional[BootstrapResult]) -> str:
    if result is None:
        return "Bootstrap: not enough trades"
    return (
        f"Bootstrap ({result.num_trades} trades): "
        f"expectancy {result.observed_expectancy_r:+.4f}R  "
        f"95% CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}]  "
        f"P(>0)={result.prob_positive:.1%}  -> {result.verdict}"
    )


def format_ruin(result: Optional[RuinResult], ruin_drawdown_pct: float = 50.0) -> str:
    if result is None:
        return "Ruin simulation: not enough trades"
    return (
        f"Ruin simulation ({result.paths} paths x {result.horizon_trades} trades):\n"
        f"  P(drawdown >= {ruin_drawdown_pct:.0f}%) = {result.ruin_probability:.1%}\n"
        f"  Final equity multiple: p05 {result.p05_final_multiple:.2f}x  "
        f"median {result.median_final_multiple:.2f}x  "
        f"p95 {result.p95_final_multiple:.2f}x\n"
        f"  Max drawdown: median {result.median_max_drawdown_pct:.1f}%  "
        f"worst {result.worst_max_drawdown_pct:.1f}%\n"
        f"  P(equity ever doubles) = {result.prob_target_reached:.1%}"
    )
