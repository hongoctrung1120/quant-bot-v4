"""Backtest performance metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Optional
from datetime import datetime

from utils.math import annualized_sharpe, max_drawdown, safe_divide


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    num_trades: int
    total_fees: float
    calmar_ratio: float
    expectancy: float


def compute_returns(equity_curve: Sequence[float]) -> list[float]:
    if len(equity_curve) < 2:
        return []
    return [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] > 0
    ]


def compute_metrics(
    equity_curve: Sequence[float],
    trade_pnls: Sequence[float],
    total_fees: float,
    periods_per_year: float = 252.0,
    years: Optional[float] = None,
    timestamps: Optional[Sequence[datetime]] = None,
) -> BacktestMetrics:
    if not equity_curve:
        return BacktestMetrics(
            total_return=0, cagr=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            profit_factor=0, win_rate=0, num_trades=0, total_fees=0,
            calmar_ratio=0, expectancy=0,
        )

    initial = equity_curve[0]
    final = equity_curve[-1]
    total_return = (final - initial) / initial if initial > 0 else 0.0
    if years is None and timestamps and len(timestamps) >= 2:
        elapsed = (timestamps[-1] - timestamps[0]).total_seconds() / (365.25 * 86400.0)
        years = max(elapsed, 1 / 365.25)
    years = 1.0 if years is None else years
    cagr = (final / initial) ** (1 / max(years, 0.01)) - 1 if initial > 0 else 0.0

    returns = compute_returns(equity_curve)
    vol = 0.0
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        vol = math.sqrt(var) * math.sqrt(periods_per_year)

    sharpe = annualized_sharpe(returns, periods_per_year)
    downside = [r for r in returns if r < 0]
    sortino = 0.0
    if downside:
        down_std = math.sqrt(sum(r ** 2 for r in downside) / len(downside))
        if down_std > 0:
            mean_r = sum(returns) / len(returns) if returns else 0
            sortino = (mean_r / down_std) * math.sqrt(periods_per_year)

    mdd = max_drawdown(equity_curve)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = safe_divide(gross_profit, gross_loss, default=0.0)
    expectancy = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0.0
    calmar = safe_divide(cagr, mdd, default=0.0)

    return BacktestMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=mdd,
        profit_factor=profit_factor,
        win_rate=win_rate,
        num_trades=len(trade_pnls),
        total_fees=total_fees,
        calmar_ratio=calmar,
        expectancy=expectancy,
    )
