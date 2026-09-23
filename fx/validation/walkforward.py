"""Walk-forward analysis.

Optimizing parameters over a whole history and reporting the result is not a
backtest, it is curve fitting with extra steps. Walk-forward fits on a training
window, trades the next unseen window with those settings, rolls forward, and
reports only the out-of-sample trades. The gap between in-sample and
out-of-sample performance is the honest measure of how much of the edge was
real and how much was fitted to noise.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional, Sequence

from core.models.bar import Bar
from fx.account import ClosedTrade
from fx.engine import FxBacktestEngine, FxBacktestResult
from fx.report import PerformanceReport, analyze
from fx.strategy.base import FxStrategy

logger = logging.getLogger(__name__)

StrategyBuilder = Callable[[dict], Sequence[FxStrategy]]
EngineBuilder = Callable[[Sequence[FxStrategy]], FxBacktestEngine]
ScoreFn = Callable[[PerformanceReport], float]


@dataclass
class WalkForwardConfig:
    train_days: int = 180
    test_days: int = 60
    warmup_days: int = 30
    anchored: bool = False
    min_trades_in_sample: int = 15


@dataclass
class FoldResult:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict = field(default_factory=dict)
    in_sample_score: float = float("-inf")
    out_of_sample_score: float = 0.0
    in_sample_trades: int = 0
    out_of_sample_trades: int = 0
    oos_report: Optional[PerformanceReport] = None
    trades: list[ClosedTrade] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)
    oos_trades: list[ClosedTrade] = field(default_factory=list)

    @property
    def oos_expectancy_r(self) -> float:
        if not self.oos_trades:
            return 0.0
        return sum(t.r_multiple for t in self.oos_trades) / len(self.oos_trades)

    @property
    def mean_in_sample_score(self) -> float:
        scores = [f.in_sample_score for f in self.folds if f.in_sample_score > float("-inf")]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_out_of_sample_score(self) -> float:
        scores = [f.out_of_sample_score for f in self.folds]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def degradation(self) -> Optional[float]:
        """Out-of-sample score as a fraction of in-sample. Below ~0.5 is fitted.

        Undefined when the in-sample score is not positive: a ratio of two
        negative numbers looks reassuring while describing a system that lost
        money in both windows.
        """
        in_sample = self.mean_in_sample_score
        if in_sample <= 0:
            return None
        return self.mean_out_of_sample_score / in_sample

    @property
    def parameter_stability(self) -> dict[str, int]:
        """How many distinct values each parameter took across folds.

        A parameter that jumps around every fold is being fitted to noise.
        """
        counts: dict[str, set] = {}
        for fold in self.folds:
            for key, value in fold.best_params.items():
                counts.setdefault(key, set()).add(value)
        return {key: len(values) for key, values in counts.items()}


def slice_series(
    series: dict[str, list[Bar]], start: datetime, end: datetime
) -> dict[str, list[Bar]]:
    return {
        symbol: [b for b in bars if start <= b.timestamp < end]
        for symbol, bars in series.items()
    }


def _bounds(series: dict[str, list[Bar]]) -> tuple[datetime, datetime]:
    starts = [bars[0].timestamp for bars in series.values() if bars]
    ends = [bars[-1].timestamp for bars in series.values() if bars]
    if not starts:
        raise ValueError("Empty series")
    return min(starts), max(ends)


def build_folds(
    series: dict[str, list[Bar]], config: WalkForwardConfig
) -> list[FoldResult]:
    first, last = _bounds(series)
    folds: list[FoldResult] = []
    index = 0
    train_start = first
    train_end = first + timedelta(days=config.train_days)

    while train_end + timedelta(days=config.test_days) <= last:
        test_start = train_end
        test_end = test_start + timedelta(days=config.test_days)
        folds.append(
            FoldResult(
                index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        index += 1
        train_end = test_end
        if not config.anchored:
            train_start = train_end - timedelta(days=config.train_days)

    return folds


def expand_grid(grid: dict[str, Iterable]) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def default_score(report: PerformanceReport) -> float:
    """Edge per trade, penalised for thin samples.

    Multiplying by sqrt(n) turns expectancy into something t-statistic shaped,
    so a fold cannot win on three lucky trades.
    """
    if report.num_trades == 0:
        return float("-inf")
    return report.expectancy_r * (report.num_trades**0.5)


def _restrict(result: FxBacktestResult, start: datetime) -> FxBacktestResult:
    """Keep only the portion of a run at or after `start`."""
    curve = [(ts, eq) for ts, eq in result.equity_curve if ts >= start]
    trades = [t for t in result.trades if t.entry_time >= start]
    restricted = FxBacktestResult(
        equity_curve=curve,
        trades=trades,
        initial_balance=curve[0][1] if curve else result.initial_balance,
        final_equity=curve[-1][1] if curve else result.final_equity,
        total_costs=sum(t.costs for t in trades),
        total_swap=0.0,
        total_spread_cost=sum(t.spread_cost for t in trades),
        rejections=result.rejections,
        halted=result.halted,
        halt_reason=result.halt_reason,
    )
    return restricted


def run_walk_forward(
    series: dict[str, list[Bar]],
    build_strategies: StrategyBuilder,
    engine_builder: EngineBuilder,
    param_grid: Optional[dict[str, Iterable]] = None,
    config: Optional[WalkForwardConfig] = None,
    score_fn: Optional[ScoreFn] = None,
) -> WalkForwardResult:
    config = config or WalkForwardConfig()
    score_fn = score_fn or default_score
    combos = expand_grid(param_grid or {})

    result = WalkForwardResult()
    for fold in build_folds(series, config):
        train_data = slice_series(series, fold.train_start, fold.train_end)

        best_score = float("-inf")
        best_params: dict = combos[0]
        for params in combos:
            report = _evaluate(train_data, params, build_strategies, engine_builder)
            if report.num_trades < config.min_trades_in_sample:
                continue
            score = score_fn(report)
            if score > best_score:
                best_score, best_params = score, params
                fold.in_sample_trades = report.num_trades

        fold.best_params = best_params
        fold.in_sample_score = best_score

        # Feed warmup history so indicators are not cold at the test open,
        # then keep only trades entered inside the test window.
        warm_start = fold.test_start - timedelta(days=config.warmup_days)
        test_data = slice_series(series, warm_start, fold.test_end)
        raw = _run(test_data, best_params, build_strategies, engine_builder)
        oos = analyze(_restrict(raw, fold.test_start))

        fold.oos_report = oos
        fold.out_of_sample_score = score_fn(oos) if oos.num_trades else 0.0
        fold.out_of_sample_trades = oos.num_trades
        fold.trades = [t for t in raw.trades if t.entry_time >= fold.test_start]

        result.folds.append(fold)
        result.oos_trades.extend(fold.trades)

    return result


def _run(
    data: dict[str, list[Bar]],
    params: dict,
    build_strategies: StrategyBuilder,
    engine_builder: EngineBuilder,
) -> FxBacktestResult:
    strategies = build_strategies(params)
    engine = engine_builder(strategies)
    return engine.run(data)


def _evaluate(
    data: dict[str, list[Bar]],
    params: dict,
    build_strategies: StrategyBuilder,
    engine_builder: EngineBuilder,
) -> PerformanceReport:
    return analyze(_run(data, params, build_strategies, engine_builder))


def format_walk_forward(result: WalkForwardResult) -> str:
    lines = ["=== WALK-FORWARD ==="]
    for fold in result.folds:
        params = ", ".join(f"{k}={v}" for k, v in fold.best_params.items()) or "(none)"
        oos = fold.oos_report
        lines.append(
            f"Fold {fold.index}: test {fold.test_start.date()}..{fold.test_end.date()}  "
            f"IS score {fold.in_sample_score:+.2f}  OOS score {fold.out_of_sample_score:+.2f}  "
            f"OOS trades {fold.out_of_sample_trades}  "
            f"OOS exp {oos.expectancy_r if oos else 0.0:+.3f}R"
        )
        lines.append(f"    params: {params}")

    degradation = result.degradation
    degradation_line = (
        f"Degradation    : {degradation:.2f}  (1.0 = holds up, <0.5 = largely fitted)"
        if degradation is not None
        else "Degradation    : n/a — in-sample score was not positive, so there "
        "was no edge to degrade"
    )
    lines.append("")
    lines.append(
        f"Mean IS score  : {result.mean_in_sample_score:+.3f}\n"
        f"Mean OOS score : {result.mean_out_of_sample_score:+.3f}\n" + degradation_line
    )
    lines.append(
        f"Combined OOS   : {len(result.oos_trades)} trades  "
        f"expectancy {result.oos_expectancy_r:+.4f}R"
    )
    stability = result.parameter_stability
    if stability:
        lines.append(
            "Param stability: "
            + "  ".join(f"{k}={v} distinct" for k, v in sorted(stability.items()))
            + f"  (across {len(result.folds)} folds)"
        )
    return "\n".join(lines)
