"""Performance analysis for FX backtests."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

from fx.account import ClosedTrade
from fx.engine import FxBacktestResult

TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class PerformanceReport:
    initial_balance: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    annual_volatility_pct: float = 0.0

    trading_days: int = 0
    avg_daily_pnl: float = 0.0
    median_daily_pnl: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    pct_profitable_days: float = 0.0
    avg_monthly_return_pct: float = 0.0

    num_trades: int = 0
    trades_per_day: float = 0.0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    expectancy_cash: float = 0.0
    profit_factor: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    largest_loss_r: float = 0.0
    max_consecutive_losses: int = 0

    total_costs: float = 0.0
    total_swap: float = 0.0
    total_spread_cost: float = 0.0
    all_in_cost: float = 0.0
    cost_per_trade: float = 0.0
    cost_drag_r: float = 0.0
    cost_to_gross_profit: float = 0.0

    by_strategy: dict[str, dict[str, float]] = field(default_factory=dict)
    by_session: dict[str, dict[str, float]] = field(default_factory=dict)
    by_exit_reason: dict[str, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""


def _daily_equity(curve: Sequence[tuple]) -> list[tuple[date, float]]:
    by_day: dict[date, float] = {}
    for timestamp, equity in curve:
        by_day[timestamp.date()] = equity
    return sorted(by_day.items())


def _returns(values: Sequence[float]) -> list[float]:
    return [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(1, len(values))
        if values[i - 1] > 0
    ]


def _max_drawdown(values: Sequence[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst * 100.0


def _consecutive_losses(trades: Sequence[ClosedTrade]) -> int:
    streak = worst = 0
    for trade in trades:
        if trade.net_pnl < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def _group_stats(trades: Sequence[ClosedTrade], key) -> dict[str, dict[str, float]]:
    groups: dict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        groups[key(trade)].append(trade)

    out: dict[str, dict[str, float]] = {}
    for name, group in groups.items():
        wins = [t for t in group if t.net_pnl > 0]
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in group if t.net_pnl < 0))
        out[name] = {
            "trades": float(len(group)),
            "win_rate": len(wins) / len(group) * 100.0,
            "net_pnl": sum(t.net_pnl for t in group),
            "expectancy_r": sum(t.r_multiple for t in group) / len(group),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
        }
    return out


def analyze(result: FxBacktestResult) -> PerformanceReport:
    report = PerformanceReport(
        initial_balance=result.initial_balance,
        final_equity=result.final_equity,
        total_costs=result.total_costs,
        total_swap=result.total_swap,
        total_spread_cost=result.total_spread_cost,
        rejections=dict(result.rejections),
        halted=result.halted,
        halt_reason=result.halt_reason,
    )
    if result.initial_balance > 0:
        report.total_return_pct = (
            (result.final_equity - result.initial_balance) / result.initial_balance * 100.0
        )

    daily = _daily_equity(result.equity_curve)
    report.trading_days = len(daily)
    if len(daily) >= 2:
        values = [equity for _, equity in daily]
        daily_returns = _returns(values)
        daily_pnl = [values[i] - values[i - 1] for i in range(1, len(values))]

        report.avg_daily_pnl = sum(daily_pnl) / len(daily_pnl)
        report.median_daily_pnl = sorted(daily_pnl)[len(daily_pnl) // 2]
        report.best_day = max(daily_pnl)
        report.worst_day = min(daily_pnl)
        report.pct_profitable_days = (
            sum(1 for p in daily_pnl if p > 0) / len(daily_pnl) * 100.0
        )
        report.max_drawdown_pct = _max_drawdown(values)

        years = (daily[-1][0] - daily[0][0]).days / 365.25
        if years > 0 and values[0] > 0 and values[-1] > 0:
            report.cagr_pct = ((values[-1] / values[0]) ** (1 / years) - 1) * 100.0
            report.avg_monthly_return_pct = (
                (1 + report.cagr_pct / 100.0) ** (1 / 12) - 1
            ) * 100.0

        if len(daily_returns) > 1:
            mean = sum(daily_returns) / len(daily_returns)
            var = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            std = math.sqrt(var)
            report.annual_volatility_pct = std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
            if std > 0:
                report.sharpe = mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)
            downside = [r for r in daily_returns if r < 0]
            if downside:
                dstd = math.sqrt(sum(r**2 for r in downside) / len(downside))
                if dstd > 0:
                    report.sortino = mean / dstd * math.sqrt(TRADING_DAYS_PER_YEAR)
        if report.max_drawdown_pct > 0:
            report.calmar = report.cagr_pct / report.max_drawdown_pct

    trades = result.trades
    report.num_trades = len(trades)
    if trades:
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl < 0]
        report.win_rate = len(wins) / len(trades) * 100.0
        report.expectancy_r = sum(t.r_multiple for t in trades) / len(trades)
        report.expectancy_cash = sum(t.net_pnl for t in trades) / len(trades)
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        report.avg_win_r = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
        report.avg_loss_r = (
            sum(t.r_multiple for t in losses) / len(losses) if losses else 0.0
        )
        report.largest_loss_r = min((t.r_multiple for t in trades), default=0.0)
        report.max_consecutive_losses = _consecutive_losses(trades)

        # Swap is negative when it is a debit, so subtracting it adds the cost.
        report.all_in_cost = (
            result.total_spread_cost + result.total_costs - result.total_swap
        )
        report.cost_per_trade = report.all_in_cost / len(trades)
        risks = [t.risk_amount for t in trades if t.risk_amount > 0]
        if risks:
            report.cost_drag_r = report.cost_per_trade / (sum(risks) / len(risks))
        report.cost_to_gross_profit = (
            report.all_in_cost / gross_profit if gross_profit > 0 else float("inf")
        )
        if report.trading_days:
            report.trades_per_day = len(trades) / report.trading_days

        report.by_strategy = _group_stats(trades, lambda t: t.strategy_id)
        report.by_session = _group_stats(trades, lambda t: t.entry_session)
        counts: dict[str, int] = defaultdict(int)
        for trade in trades:
            counts[trade.exit_reason] += 1
        report.by_exit_reason = dict(counts)

    return report


def format_report(report: PerformanceReport, title: str = "FX BACKTEST") -> str:
    lines = [
        f"=== {title} ===",
        f"Balance      : {report.initial_balance:,.2f} -> {report.final_equity:,.2f} "
        f"({report.total_return_pct:+.2f}%)",
        f"CAGR         : {report.cagr_pct:+.2f}%   "
        f"avg month {report.avg_monthly_return_pct:+.2f}%",
        f"Max drawdown : {report.max_drawdown_pct:.2f}%",
        f"Sharpe       : {report.sharpe:.2f}   Sortino {report.sortino:.2f}   "
        f"Calmar {report.calmar:.2f}",
        f"Ann. vol     : {report.annual_volatility_pct:.2f}%",
        "",
        f"Trading days : {report.trading_days}",
        f"Daily P&L    : avg {report.avg_daily_pnl:+.4f}   "
        f"median {report.median_daily_pnl:+.4f}   "
        f"best {report.best_day:+.2f}   worst {report.worst_day:+.2f}",
        f"Winning days : {report.pct_profitable_days:.1f}%",
        "",
        f"Trades       : {report.num_trades}  ({report.trades_per_day:.2f}/day)",
        f"Win rate     : {report.win_rate:.1f}%",
        f"Expectancy   : {report.expectancy_r:+.3f}R  "
        f"({report.expectancy_cash:+.4f} per trade)",
        f"Profit factor: {report.profit_factor:.2f}",
        f"Avg win/loss : {report.avg_win_r:+.2f}R / {report.avg_loss_r:+.2f}R",
        f"Worst trade  : {report.largest_loss_r:+.2f}R   "
        f"max losing streak {report.max_consecutive_losses}",
        "",
        f"Costs all-in : {report.all_in_cost:,.2f}  "
        f"(spread {report.total_spread_cost:,.2f}  "
        f"commission {report.total_costs:,.2f}  swap {report.total_swap:+,.2f})",
        f"Cost/trade   : {report.cost_per_trade:.4f}  "
        f"= {report.cost_drag_r:.3f}R of edge needed just to break even",
        f"Cost/gross   : {report.cost_to_gross_profit:.2%}"
        if report.cost_to_gross_profit != float("inf")
        else "Cost/gross   : n/a (no gross profit)",
    ]

    if report.by_strategy:
        lines.append("")
        lines.append("By strategy:")
        for name, stats in sorted(report.by_strategy.items()):
            lines.append(
                f"  {name:<20} {int(stats['trades']):>4} trades  "
                f"win {stats['win_rate']:>5.1f}%  "
                f"exp {stats['expectancy_r']:+.3f}R  "
                f"pnl {stats['net_pnl']:+.2f}"
            )

    if report.by_session:
        lines.append("")
        lines.append("By entry session:")
        for name, stats in sorted(report.by_session.items()):
            lines.append(
                f"  {name:<20} {int(stats['trades']):>4} trades  "
                f"win {stats['win_rate']:>5.1f}%  "
                f"exp {stats['expectancy_r']:+.3f}R  "
                f"pnl {stats['net_pnl']:+.2f}"
            )

    if report.by_exit_reason:
        lines.append("")
        lines.append(
            "Exits: "
            + "  ".join(f"{k}={v}" for k, v in sorted(report.by_exit_reason.items()))
        )

    if report.rejections:
        lines.append("")
        top = sorted(report.rejections.items(), key=lambda kv: -kv[1])[:8]
        lines.append("Blocked entries: " + "  ".join(f"{k}={v}" for k, v in top))

    if report.halted:
        lines.append("")
        lines.append(f"*** SYSTEM HALTED: {report.halt_reason} ***")

    return "\n".join(lines)
