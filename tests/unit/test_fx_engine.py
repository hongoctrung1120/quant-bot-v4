"""Engine behaviour: execution timing, gaps, forced exits, cost accounting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from fx.costs import FxCostConfig
from fx.data.ohlc import make_bar
from fx.engine import FxBacktestConfig, FxBacktestEngine
from fx.risk.governor import RiskLimits
from fx.strategy.base import FxSignal, FxStrategy, MarketContext

START = datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc)  # Wednesday
NO_COST = FxCostConfig(
    spread_stress_multiplier=0.0, slippage_pips=0.0, swap_enabled=False
)
PERMISSIVE = RiskLimits(
    risk_per_trade_pct=0.5,
    max_trades_per_day=100,
    max_consecutive_losses=1000,
    daily_loss_limit_pct=100.0,
    weekly_loss_limit_pct=100.0,
    monthly_loss_limit_pct=100.0,
    max_drawdown_pct=100.0,
    warning_drawdown_pct=100.0,
    kelly_min_trades=10_000,
)


def bars_from_closes(
    closes: list[float],
    symbol: str = "EURUSD",
    start: datetime = START,
    spread_hl: float = 0.0005,
    opens: Optional[list[float]] = None,
    highs: Optional[list[float]] = None,
    lows: Optional[list[float]] = None,
):
    out = []
    for i, close in enumerate(closes):
        open_ = opens[i] if opens else close
        high = highs[i] if highs else max(open_, close) + spread_hl
        low = lows[i] if lows else min(open_, close) - spread_hl
        out.append(
            make_bar(
                symbol=symbol,
                timestamp=start + timedelta(hours=i),
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=100.0,
                timeframe="H1",
            )
        )
    return out


class ScriptedStrategy(FxStrategy):
    """Fires one signal at a chosen bar index, with explicit levels."""

    def __init__(self, fire_at: int, direction: str, stop: float, target: float):
        self.fire_at = fire_at
        self.direction = direction
        self.stop = stop
        self.target = target
        self._seen = 0

    @property
    def strategy_id(self) -> str:
        return "scripted"

    def on_bar(self, ctx: MarketContext) -> Optional[FxSignal]:
        index = self._seen
        self._seen += 1
        if index != self.fire_at:
            return None
        return FxSignal(
            symbol=ctx.symbol,
            direction=self.direction,
            stop_price=self.stop,
            target_price=self.target,
            strategy_id=self.strategy_id,
        )


def build_engine(strategies, cost=NO_COST, limits=PERMISSIVE, **cfg):
    defaults = dict(initial_balance=1000.0, cent_account=True, leverage=100.0)
    defaults.update(cfg)
    return FxBacktestEngine(
        symbols=["EURUSD"],
        strategies=strategies,
        config=FxBacktestConfig(**defaults),
        limits=limits,
        cost_config=cost,
    )


class TestExecutionTiming:
    def test_entry_fills_at_next_bar_open_not_signal_bar_close(self):
        closes = [1.1000] * 5 + [1.1050] + [1.1100] * 5
        opens = [1.1000] * 5 + [1.1050] + [1.1080] + [1.1100] * 4
        bars = bars_from_closes(closes, opens=opens)

        engine = build_engine([ScriptedStrategy(5, "LONG", 1.0950, 1.2000)])
        engine.run({"EURUSD": bars})

        trades = engine.account.closed_trades
        assert len(trades) == 1
        # Bar 5 closed at 1.1050; bar 6 opened at 1.1080. Filling at 1.1050
        # would mean acting on a price we could not have traded at.
        assert trades[0].entry_price == pytest.approx(1.1080)

    def test_signal_expires_if_not_executable_next_slice(self):
        bars = bars_from_closes([1.1000] * 8)
        engine = build_engine([ScriptedStrategy(7, "LONG", 1.0950, 1.2000)])
        engine.run({"EURUSD": bars})
        # Last bar fires the signal but no bar follows it.
        assert engine.account.closed_trades == []


class TestGapHandling:
    def test_gap_through_stop_fills_at_open_not_at_stop(self):
        closes = [1.1000, 1.1000, 1.1000, 1.0800, 1.0800]
        opens = [1.1000, 1.1000, 1.1000, 1.0800, 1.0800]
        lows = [1.0995, 1.0995, 1.0995, 1.0795, 1.0795]
        bars = bars_from_closes(closes, opens=opens, lows=lows)

        engine = build_engine([ScriptedStrategy(1, "LONG", 1.0950, 1.2000)])
        engine.run({"EURUSD": bars})

        trade = engine.account.closed_trades[0]
        assert trade.exit_reason == "STOP_GAP"
        # A 1.0950 stop cannot fill at 1.0950 when the market opens at 1.0800.
        assert trade.exit_price == pytest.approx(1.0800)
        assert trade.r_multiple < -1.0, "gap losses must exceed one R"

    def test_entry_skipped_when_open_gaps_past_the_stop(self):
        closes = [1.1000, 1.1000, 1.0900, 1.0900]
        opens = [1.1000, 1.1000, 1.0900, 1.0900]
        lows = [1.0895] * 4
        bars = bars_from_closes(closes, opens=opens, lows=lows)

        engine = build_engine([ScriptedStrategy(1, "LONG", 1.0950, 1.2000)])
        result = engine.run({"EURUSD": bars})

        assert engine.account.closed_trades == []
        assert result.rejections["gapped_through_stop"] == 1


class TestProtectiveExits:
    def test_stop_takes_priority_when_both_levels_sit_inside_one_bar(self):
        bars = bars_from_closes(
            [1.1000, 1.1000, 1.1000],
            opens=[1.1000, 1.1000, 1.1000],
            highs=[1.1005, 1.1005, 1.2100],
            lows=[1.0995, 1.0995, 1.0940],
        )
        engine = build_engine([ScriptedStrategy(0, "LONG", 1.0950, 1.2000)])
        engine.run({"EURUSD": bars})

        trade = engine.account.closed_trades[0]
        assert trade.exit_reason == "STOP"
        assert trade.r_multiple == pytest.approx(-1.0, abs=0.01)

    def test_target_exit_pays_the_designed_multiple(self):
        bars = bars_from_closes(
            [1.1000, 1.1000, 1.1000],
            opens=[1.1000, 1.1000, 1.1000],
            highs=[1.1005, 1.1005, 1.1160],
            lows=[1.0995, 1.0995, 1.0995],
        )
        # Stop 50 pips away, target 100 pips away -> exactly +2R.
        engine = build_engine([ScriptedStrategy(0, "LONG", 1.0950, 1.1100)])
        engine.run({"EURUSD": bars})

        trade = engine.account.closed_trades[0]
        assert trade.exit_reason == "TARGET"
        assert trade.r_multiple == pytest.approx(2.0, abs=0.01)


class TestForcedExits:
    def test_positions_are_flat_before_the_weekend(self):
        friday = datetime(2024, 1, 5, 14, 0, tzinfo=timezone.utc)
        bars = bars_from_closes([1.1000] * 8, start=friday)
        engine = build_engine([ScriptedStrategy(0, "LONG", 1.0950, 1.3000)])
        engine.run({"EURUSD": bars})

        trades = engine.account.closed_trades
        assert trades and trades[0].exit_reason == "WEEKEND_FLAT"
        assert not engine.account.positions

    def test_open_position_is_liquidated_at_end_of_backtest(self):
        bars = bars_from_closes([1.1000] * 6)
        engine = build_engine([ScriptedStrategy(0, "LONG", 1.0950, 1.3000)])
        engine.run({"EURUSD": bars})
        assert engine.account.closed_trades[0].exit_reason == "BACKTEST_END"
        assert not engine.account.positions


class TestCostAccounting:
    def test_spread_is_charged_through_the_fill_price(self):
        bars = bars_from_closes([1.1000] * 6)
        engine = build_engine(
            [ScriptedStrategy(0, "LONG", 1.0950, 1.3000)],
            cost=FxCostConfig(slippage_pips=0.0, swap_enabled=False),
        )
        engine.run({"EURUSD": bars})

        trade = engine.account.closed_trades[0]
        assert trade.entry_price > 1.1000, "a buy must fill at the ask"
        assert trade.exit_price < 1.1000, "a sell must fill at the bid"
        assert trade.spread_cost > 0
        assert trade.net_pnl < 0, "a flat market still costs the round trip"

    def test_zero_cost_round_trip_in_a_flat_market_is_free(self):
        bars = bars_from_closes([1.1000] * 6)
        engine = build_engine([ScriptedStrategy(0, "LONG", 1.0950, 1.3000)])
        engine.run({"EURUSD": bars})
        assert engine.account.closed_trades[0].net_pnl == pytest.approx(0.0, abs=1e-9)


class TestSizingRefusal:
    def test_signal_is_dropped_when_min_lot_would_breach_the_risk_budget(self):
        bars = bars_from_closes([1.1000] * 6)
        engine = FxBacktestEngine(
            symbols=["EURUSD"],
            strategies=[ScriptedStrategy(0, "LONG", 1.0950, 1.3000)],
            # Standard lots on a $100 account: the smallest trade is too big.
            config=FxBacktestConfig(initial_balance=100.0, cent_account=False),
            limits=PERMISSIVE,
            cost_config=NO_COST,
        )
        result = engine.run({"EURUSD": bars})

        assert engine.account.closed_trades == []
        assert result.rejections["below_min_lot"] == 1
