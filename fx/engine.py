"""Event-driven FX portfolio backtest engine.

Three properties decide whether the numbers this produces mean anything:

1. A signal computed on a closed bar is executed at the NEXT bar's open. Filling
   at the close of the bar that produced the signal quietly assumes you knew the
   close before it happened.
2. Gaps are honoured. If price opens beyond the stop, the fill is the open, not
   the stop level. Weekend gaps are where "guaranteed" risk limits actually break.
3. Every fill crosses the spread and pays slippage, and financing is charged at
   the daily rollover.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from core.features.feature_engine import FeatureEngine
from core.models.bar import Bar
from fx import sessions
from fx.account import ClosedTrade, FxAccount, FxPosition
from fx.costs import FxCostConfig, FxCostModel
from fx.data.ohlc import align_multi_symbol, resample
from fx.instruments import FxInstrument, RateBook, build_universe
from fx.risk.exposure import currency_exposure, currency_risk
from fx.risk.governor import RiskGovernor, RiskLimits
from fx.risk.sizing import FxPositionSizer
from fx.strategy.base import FxSignal, FxStrategy, MarketContext, fx_feature_set

logger = logging.getLogger(__name__)


@dataclass
class FxBacktestConfig:
    initial_balance: float = 100.0
    leverage: float = 100.0
    account_currency: str = "USD"
    cent_account: bool = True
    trading_timeframe: str = "H1"
    higher_timeframe: str = "H4"
    stop_out_level: float = 0.5
    intrabar_priority: str = "STOP_FIRST"
    close_before_weekend: bool = True
    history_length: int = 250


@dataclass
class FxBacktestResult:
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)
    initial_balance: float = 0.0
    final_equity: float = 0.0
    total_costs: float = 0.0
    total_swap: float = 0.0
    total_spread_cost: float = 0.0
    bars_processed: int = 0
    signals_generated: int = 0
    rejections: Counter = field(default_factory=Counter)
    risk_events: list[str] = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""


class _HtfFeatureIndex:
    """Higher-timeframe features indexed by the time they became known.

    An H4 bar stamped 08:00 only exists once 12:00 has passed, so it may not be
    consumed by an H1 bar that closed earlier.
    """

    def __init__(self, series: dict[str, list[Bar]], timeframe: str) -> None:
        self._known_at: dict[str, list[datetime]] = {}
        self._features: dict[str, list[dict[str, float]]] = {}

        for symbol, bars in series.items():
            htf_bars = resample(bars, timeframe)
            engine = FeatureEngine(fx_feature_set())
            times: list[datetime] = []
            snapshots: list[dict[str, float]] = []
            running: dict[str, float] = {}
            for bar in htf_bars:
                running.update(engine.on_bar(bar))
                times.append(bar.end_timestamp)
                snapshots.append(dict(running))
            self._known_at[symbol] = times
            self._features[symbol] = snapshots

    def at(self, symbol: str, close_time: datetime) -> dict[str, float]:
        times = self._known_at.get(symbol)
        if not times:
            return {}
        index = bisect_right(times, close_time) - 1
        if index < 0:
            return {}
        return self._features[symbol][index]


class FxBacktestEngine:
    def __init__(
        self,
        symbols: Sequence[str],
        strategies: Sequence[FxStrategy],
        config: Optional[FxBacktestConfig] = None,
        limits: Optional[RiskLimits] = None,
        cost_config: Optional[FxCostConfig] = None,
        news_windows: Optional[Sequence[tuple[datetime, datetime]]] = None,
    ) -> None:
        self.config = config or FxBacktestConfig()
        self.symbols = list(symbols)
        self.strategies = list(strategies)
        self.limits = limits or RiskLimits()

        self.instruments: dict[str, FxInstrument] = build_universe(
            self.symbols, cent_account=self.config.cent_account
        )
        self.rates = RateBook(self.config.account_currency)
        self.costs = FxCostModel(cost_config)
        self.governor = RiskGovernor(self.limits, news_windows)
        self.sizer = FxPositionSizer(self.limits)
        self.account = FxAccount(
            balance=self.config.initial_balance,
            leverage=self.config.leverage,
            rates=self.rates,
            instruments=self.instruments,
            stop_out_level=self.config.stop_out_level,
        )

        self._features = FeatureEngine(fx_feature_set())
        self._history: dict[str, deque] = {
            symbol: deque(maxlen=self.config.history_length) for symbol in self.symbols
        }
        self._pending: dict[str, FxSignal] = {}
        self._strategy_by_id = {s.strategy_id: s for s in self.strategies}
        self._last_timestamp: Optional[datetime] = None
        self._result = FxBacktestResult(initial_balance=self.config.initial_balance)

    # ------------------------------------------------------------------
    def run(self, series: dict[str, list[Bar]]) -> FxBacktestResult:
        series = {s: series[s] for s in self.symbols if s in series}
        htf = _HtfFeatureIndex(series, self.config.higher_timeframe)
        slices = align_multi_symbol(series)

        for timestamp, bars in slices:
            self._process_slice(timestamp, bars, htf)

        self._liquidate_open_positions()

        self._result.trades = list(self.account.closed_trades)
        self._result.final_equity = self.account.equity(self._marks())
        self._result.total_costs = self.account.total_costs
        self._result.total_swap = self.account.total_swap
        self._result.total_spread_cost = self.account.total_spread_cost
        self._result.risk_events = list(self.governor.events)
        self._result.halted = self.governor.halted
        self._result.halt_reason = self.governor.halt_reason
        return self._result

    # ------------------------------------------------------------------
    def _process_slice(
        self, timestamp: datetime, bars: dict[str, Bar], htf: _HtfFeatureIndex
    ) -> None:
        self._result.bars_processed += len(bars)

        for symbol, bar in bars.items():
            self.rates.update(symbol, bar.open)
        self._execute_pending(timestamp, bars)

        for symbol, bar in bars.items():
            self.rates.update(symbol, bar.close)
            position = self.account.positions.get(symbol)
            if position is not None:
                position.track_excursion(bar.high, bar.low)
                position.bars_held += 1

        self._apply_rollover(timestamp)
        self._check_protective_exits(timestamp, bars)

        marks = self._marks()
        equity = self.account.equity(marks)
        self.governor.on_bar(timestamp, equity)
        self._result.equity_curve.append((timestamp, equity))

        if self.account.is_stopped_out(marks):
            self._close_all(timestamp, bars, "MARGIN_STOP_OUT")
        elif self.governor.halted:
            self._close_all(timestamp, bars, "RISK_HALT")
        elif self.config.close_before_weekend and self.governor.should_flatten_for_weekend(
            timestamp
        ):
            self._close_all(timestamp, bars, "WEEKEND_FLAT")

        contexts = self._build_contexts(timestamp, bars, htf)
        self._manage_positions(contexts)
        self._collect_signals(timestamp, contexts)
        self._last_timestamp = timestamp

    def _build_contexts(
        self, timestamp: datetime, bars: dict[str, Bar], htf: _HtfFeatureIndex
    ) -> dict[str, MarketContext]:
        session = sessions.classify(timestamp)
        contexts: dict[str, MarketContext] = {}
        for symbol, bar in bars.items():
            features = self._features.on_bar(bar)
            self._history[symbol].append(bar)
            contexts[symbol] = MarketContext(
                timestamp=timestamp,
                symbol=symbol,
                bar=bar,
                features=features,
                session=session,
                instrument=self.instruments[symbol],
                htf_features=htf.at(symbol, bar.end_timestamp),
                history=tuple(self._history[symbol]),
            )
        return contexts

    def _manage_positions(self, contexts: dict[str, MarketContext]) -> None:
        for symbol, position in list(self.account.positions.items()):
            ctx = contexts.get(symbol)
            strategy = self._strategy_by_id.get(position.strategy_id)
            if ctx is None or strategy is None:
                continue
            action = strategy.manage(position, ctx)
            if action is None:
                continue
            if action.new_stop is not None:
                position.stop_price = action.new_stop
            if action.exit_now:
                self._close_position(symbol, ctx.bar.close, ctx.timestamp, "STRATEGY_EXIT")

    def _collect_signals(
        self, timestamp: datetime, contexts: dict[str, MarketContext]
    ) -> None:
        self._pending.clear()
        for strategy in self.strategies:
            for signal in strategy.on_slice(timestamp, contexts):
                self._result.signals_generated += 1
                # One pending order per symbol; first strategy to claim it wins.
                self._pending.setdefault(signal.symbol, signal)

    # ------------------------------------------------------------------
    def _execute_pending(self, timestamp: datetime, bars: dict[str, Bar]) -> None:
        pending, self._pending = self._pending, {}
        for symbol, signal in pending.items():
            bar = bars.get(symbol)
            if bar is None:
                self._result.rejections["no_bar_next_slice"] += 1
                continue
            self._try_open(timestamp, signal, bar)

    def _try_open(self, timestamp: datetime, signal: FxSignal, bar: Bar) -> None:
        instrument = self.instruments[signal.symbol]
        marks = self._marks()
        equity = self.account.equity(marks)

        decision = self.governor.assess_entry(
            timestamp=timestamp,
            symbol=signal.symbol,
            open_symbols=list(self.account.positions),
            currency_counts=self._currency_counts(),
            base=instrument.base,
            quote=instrument.quote,
            margin_level=self.account.margin_level(marks),
            free_margin_pct=self._free_margin_pct(marks, equity),
        )
        if not decision.allowed:
            self._result.rejections[decision.reason] += 1
            return

        fill_price = self.costs.fill_price(
            instrument, bar.open, signal.is_long, timestamp
        )

        # The open already ran past the stop: the idea was invalidated before
        # we could take it.
        if signal.is_long and fill_price <= signal.stop_price:
            self._result.rejections["gapped_through_stop"] += 1
            return
        if not signal.is_long and fill_price >= signal.stop_price:
            self._result.rejections["gapped_through_stop"] += 1
            return

        sizing = self.sizer.size(
            instrument=instrument,
            entry_price=fill_price,
            stop_price=signal.stop_price,
            equity=equity,
            rates=self.rates,
            is_long=signal.is_long,
            risk_scaler=decision.risk_scaler,
            free_margin=self.account.free_margin(marks),
            leverage=self.config.leverage,
            current_exposure=currency_exposure(
                self.account.positions, marks, self.instruments, self.rates
            ),
            current_currency_risk=currency_risk(
                self.account.positions, self.instruments
            ),
            gross_exposure=self._gross_exposure(marks),
        )
        if not sizing.accepted:
            self._result.rejections[sizing.rejected or "unsized"] += 1
            return

        lots = sizing.lots if signal.is_long else -sizing.lots
        if not self.account.can_open(signal.symbol, lots, fill_price, marks):
            self._result.rejections["insufficient_margin"] += 1
            return

        self.account.open_position(
            symbol=signal.symbol,
            lots=lots,
            fill_price=fill_price,
            timestamp=timestamp,
            strategy_id=signal.strategy_id,
            cost=self.costs.commission(instrument, lots),
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            risk_amount=sizing.risk_amount,
            session=sessions.classify(timestamp).value,
            spread_cost=self.costs.implicit_cost(
                instrument, bar.open, fill_price, lots, self.rates
            ),
        )
        self.governor.on_position_opened()

    # ------------------------------------------------------------------
    def _check_protective_exits(self, timestamp: datetime, bars: dict[str, Bar]) -> None:
        for symbol in list(self.account.positions):
            bar = bars.get(symbol)
            if bar is None:
                continue
            position = self.account.positions[symbol]
            hit = self._exit_price(position, bar)
            if hit is None:
                continue
            price, reason = hit
            self._close_position(symbol, price, timestamp, reason)

    def _exit_price(
        self, position: FxPosition, bar: Bar
    ) -> Optional[tuple[float, str]]:
        stop, target = position.stop_price, position.target_price
        stop_first = self.config.intrabar_priority == "STOP_FIRST"

        if position.is_long:
            if stop is not None and bar.open <= stop:
                return bar.open, "STOP_GAP"
            if target is not None and bar.open >= target:
                return bar.open, "TARGET_GAP"
            stop_hit = stop is not None and bar.low <= stop
            target_hit = target is not None and bar.high >= target
        else:
            if stop is not None and bar.open >= stop:
                return bar.open, "STOP_GAP"
            if target is not None and bar.open <= target:
                return bar.open, "TARGET_GAP"
            stop_hit = stop is not None and bar.high >= stop
            target_hit = target is not None and bar.low <= target

        if stop_hit and target_hit:
            # Both levels sit inside one bar and the path is unknowable.
            # Assume the adverse one.
            return (stop, "STOP") if stop_first else (target, "TARGET")
        if stop_hit:
            return stop, "STOP"
        if target_hit:
            return target, "TARGET"
        return None

    def _close_position(
        self, symbol: str, exit_mid: float, timestamp: datetime, reason: str
    ) -> None:
        position = self.account.positions.get(symbol)
        if position is None:
            return
        instrument = self.instruments[symbol]
        fill = self.costs.fill_price(
            instrument, exit_mid, is_buy=not position.is_long, timestamp=timestamp
        )
        trade = self.account.close_position(
            symbol=symbol,
            fill_price=fill,
            timestamp=timestamp,
            cost=self.costs.commission(instrument, position.lots),
            reason=reason,
            spread_cost=self.costs.implicit_cost(
                instrument, exit_mid, fill, position.lots, self.rates
            ),
        )
        self.governor.on_trade_closed(trade)

    def _close_all(self, timestamp: datetime, bars: dict[str, Bar], reason: str) -> None:
        for symbol in list(self.account.positions):
            bar = bars.get(symbol)
            price = bar.close if bar else self.account.positions[symbol].entry_price
            self._close_position(symbol, price, timestamp, reason)

    def _liquidate_open_positions(self) -> None:
        if not self.account.positions or self._last_timestamp is None:
            return
        marks = self._marks()
        for symbol in list(self.account.positions):
            price = marks.get(symbol, self.account.positions[symbol].entry_price)
            self._close_position(symbol, price, self._last_timestamp, "BACKTEST_END")

    # ------------------------------------------------------------------
    def _apply_rollover(self, timestamp: datetime) -> None:
        if self._last_timestamp is None:
            return
        if not sessions.is_rollover(self._last_timestamp, timestamp):
            return
        for symbol, position in list(self.account.positions.items()):
            amount = self.costs.swap_amount(
                self.instruments[symbol], position.lots, timestamp, self.rates
            )
            self.account.apply_swap(symbol, amount)

    def _marks(self) -> dict[str, float]:
        marks = {}
        for symbol in self.symbols:
            price = self.rates.price(symbol)
            if price is not None:
                marks[symbol] = price
        return marks

    def _currency_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for symbol in self.account.positions:
            inst = self.instruments[symbol]
            counts[inst.base] = counts.get(inst.base, 0) + 1
            counts[inst.quote] = counts.get(inst.quote, 0) + 1
        return counts

    def _gross_exposure(self, marks: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.account.positions.items():
            inst = self.instruments[symbol]
            price = marks.get(symbol, position.entry_price)
            total += abs(
                self.rates.convert(inst.notional_base(position.lots) * price, inst.quote)
            )
        return total

    def _free_margin_pct(self, marks: dict[str, float], equity: float) -> float:
        if equity <= 0:
            return 0.0
        return self.account.free_margin(marks) / equity * 100.0
