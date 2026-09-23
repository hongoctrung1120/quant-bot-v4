"""The regression test that protects every future performance claim.

On a driftless random walk there is nothing to predict. With costs switched
off, any strategy must therefore earn exactly zero, and with costs on it must
lose exactly the cost of trading. If this test ever starts showing a profit,
something has begun to see the future — a look-ahead bug, an accounting leak,
or an unrealistic fill — and every backtest number produced after that point
is worthless until it is fixed.
"""

from __future__ import annotations

import statistics

import pytest

from fx.costs import FxCostConfig
from fx.data.synthetic import generate_market
from fx.engine import FxBacktestConfig, FxBacktestEngine
from fx.report import analyze
from fx.risk.governor import RiskLimits
from fx.strategy.currency_strength import CurrencyStrengthStrategy
from fx.strategy.session_breakout import SessionBreakoutStrategy
from fx.strategy.trend_pullback import TrendPullbackStrategy

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
SEEDS = (101, 202)
DAYS = 365

FREE = FxCostConfig(spread_stress_multiplier=0.0, slippage_pips=0.0, swap_enabled=False)

# The governor exists to suppress trading; for a statistical test of raw edge
# it must be opened up so the sample is not truncated by protective locks.
UNRESTRICTED = RiskLimits(
    risk_per_trade_pct=0.5,
    max_consecutive_losses=10_000,
    daily_loss_limit_pct=100.0,
    weekly_loss_limit_pct=100.0,
    monthly_loss_limit_pct=100.0,
    max_drawdown_pct=100.0,
    warning_drawdown_pct=100.0,
    kelly_min_trades=10_000,
)


def run(seed: int, costs: FxCostConfig):
    market = generate_market(SYMBOLS, days=DAYS, mode="random_walk", seed=seed)
    engine = FxBacktestEngine(
        symbols=SYMBOLS,
        strategies=[
            SessionBreakoutStrategy(),
            TrendPullbackStrategy(),
            CurrencyStrengthStrategy(),
        ],
        config=FxBacktestConfig(initial_balance=100.0, cent_account=True),
        limits=UNRESTRICTED,
        cost_config=costs,
    )
    return analyze(engine.run(market))


@pytest.fixture(scope="module")
def zero_cost_reports():
    return [run(seed, FREE) for seed in SEEDS]


@pytest.fixture(scope="module")
def real_cost_reports():
    return [run(seed, FxCostConfig()) for seed in SEEDS]


def _pooled_expectancy(reports) -> float:
    total_trades = sum(r.num_trades for r in reports)
    weighted = sum(r.expectancy_r * r.num_trades for r in reports)
    return weighted / total_trades if total_trades else 0.0


def test_random_walk_generates_a_usable_sample(zero_cost_reports):
    assert sum(r.num_trades for r in zero_cost_reports) > 500


def test_no_edge_on_a_driftless_walk_without_costs(zero_cost_reports):
    expectancy = _pooled_expectancy(zero_cost_reports)
    assert abs(expectancy) < 0.08, (
        f"expectancy {expectancy:+.4f}R on driftless data with zero costs — "
        "this indicates look-ahead bias or an accounting error, not alpha"
    )


def test_losses_match_transaction_costs_when_costs_are_on(real_cost_reports):
    expectancy = _pooled_expectancy(real_cost_reports)
    cost_drag = statistics.mean(r.cost_drag_r for r in real_cost_reports)
    assert expectancy < 0, "trading random data with costs must lose money"
    assert abs(expectancy + cost_drag) < 0.08, (
        f"expectancy {expectancy:+.4f}R does not reconcile with measured "
        f"cost drag {cost_drag:.4f}R — cost accounting is inconsistent"
    )


def test_costs_are_reported_rather_than_hidden_in_fills(real_cost_reports):
    for report in real_cost_reports:
        assert report.total_spread_cost > 0
        assert report.cost_drag_r > 0
