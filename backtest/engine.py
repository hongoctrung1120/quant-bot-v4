"""Event-driven backtesting engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backtest.costs import CostConfig, TransactionCostModel
from backtest.funding import FundingConfig, FundingModel
from backtest.fills import FillSimulator
from backtest.execution import simulate_order_fills
from backtest.event_loop import EventLoop
from backtest.metrics import BacktestMetrics, compute_metrics
from core.bars.dollar_bars import DollarBarEngine, MultiResolutionDollarBarEngine
from core.bars.time_bars import MultiIntervalTimeBarEngine, TimeBarEngine
from core.capital.allocator import CapitalAllocationEngine
from core.config import Config
from core.data.ingestion.engine import TradeIngestionEngine
from core.data.loaders.base import TradeLoader
from core.events.bus import SimpleEventBus
from core.execution.execution_engine import SimulatedExecutionEngine
from core.features.feature_engine import FeatureEngine
from core.interfaces import Strategy
from core.models.bar import Bar, BarType
from core.models.order import OrderIntent, OrderSide, OrderType
from core.models.signal import SignalDirection
from core.models.risk import RiskState
from core.portfolio.accounting import PortfolioAccount
from core.regime.rules import RuleBasedRegimeEngine
from core.risk.portfolio_risk import PortfolioRiskMonitor
from core.risk.position_sizing import RiskBasedPositionSizer
from core.signal.signal_engine import SignalEngine
from core.strategy.trend import TrendFollowingStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_equity: float = 100000.0
    bar_type: str = "DOLLAR"  # DOLLAR | TIME
    dollar_threshold_name: str = "medium"
    time_interval: str = "15m"
    seed: int = 42
    sort_trades_by_timestamp: bool = True


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    equity_curve: list[float]
    bar_count: int
    trade_count: int
    fill_count: int
    risk_state: str
    experiment: dict = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtest with no look-ahead bias.

    Pipeline per closed bar:
        Features → Regime → Strategy → Signal → Allocation → Risk → Size → Execute
    """

    def __init__(
        self,
        config: Config,
        backtest_config: Optional[BacktestConfig] = None,
        strategies: Optional[list[Strategy]] = None,
    ) -> None:
        self._config = config
        self._bt_config = backtest_config or BacktestConfig()
        self._strategies = strategies or [TrendFollowingStrategy()]

        self._event_bus = SimpleEventBus()
        self._event_loop = EventLoop(self._event_bus)
        self._feature_engine = FeatureEngine()
        self._regime_engine = RuleBasedRegimeEngine()
        self._signal_engine = SignalEngine()
        self._risk_monitor = PortfolioRiskMonitor(config.risk, config.capital)
        self._position_sizer = RiskBasedPositionSizer(config.risk, config.capital)
        self._account = PortfolioAccount(
            cash=self._bt_config.initial_equity,
            initial_equity=self._bt_config.initial_equity,
        )
        self._cost_model = TransactionCostModel(CostConfig(
            maker_fee_pct=self._get_exec_cost("maker_fee_pct", 0.02),
            taker_fee_pct=self._get_exec_cost("taker_fee_pct", 0.05),
            slippage_pct=self._get_exec_cost("slippage_pct", 0.01),
            spread_pct=self._get_exec_cost("spread_pct", 0.005),
        ))
        self._funding = FundingModel(FundingConfig(
            rate_pct=self._get_exec_cost("funding_rate_pct", 0.01),
            interval_hours=int(self._get_exec_sim("funding_interval_hours", 8)),
        ))
        self._funding_last_timestamp = None
        self._protective_levels: dict[str, tuple[float | None, float | None]] = {}
        self._closed_trade_pnls: list[float] = []
        self._execution = SimulatedExecutionEngine(
            cost_config=CostConfig(
                maker_fee_pct=self._get_exec_cost("maker_fee_pct", 0.02),
                taker_fee_pct=self._get_exec_cost("taker_fee_pct", 0.05),
                slippage_pct=self._get_exec_cost("slippage_pct", 0.01),
                spread_pct=self._get_exec_cost("spread_pct", 0.005),
            ),
            partial_fills=self._get_exec_sim("partial_fills_enabled", True),
            fill_probability=self._get_exec_sim("fill_probability", 0.95),
            seed=self._bt_config.seed,
        )

        raw_capital = config.raw.get("capital", {})
        self._allocator = CapitalAllocationEngine(
            config=config.capital,
            asset_weights=raw_capital.get("asset_weights", {}),
            strategy_weights=raw_capital.get("strategy_weights", {}),
            regime_adjustments=raw_capital.get("regime_adjustments", {}),
            asset_volatilities=raw_capital.get("asset_volatilities", config.capital.asset_volatilities),
        )

        self._dollar_engine: Optional[MultiResolutionDollarBarEngine] = None
        self._time_engine: Optional[MultiIntervalTimeBarEngine] = None
        self._single_dollar: Optional[DollarBarEngine] = None
        self._single_time: Optional[TimeBarEngine] = None

        if self._bt_config.bar_type == "DOLLAR":
            threshold = config.bars.dollar_thresholds.get(
                self._bt_config.dollar_threshold_name
            )
            if threshold:
                self._single_dollar = DollarBarEngine(
                    threshold=threshold,
                    overshoot_policy=config.bars.overshoot_policy,
                )
            self._dollar_engine = MultiResolutionDollarBarEngine(config.bars)
        else:
            self._single_time = TimeBarEngine(self._bt_config.time_interval)
            self._time_engine = MultiIntervalTimeBarEngine(
                [self._bt_config.time_interval]
            )

        self._trade_pnls: list[float] = []
        self._current_prices: dict[str, float] = {}
        self._current_session = None
        self._event_loop.on_trade(self._on_trade)

    def _get_exec_cost(self, key: str, default: float) -> float:
        return self._config.raw.get("execution", {}).get("costs", {}).get(key, default)

    def _get_exec_sim(self, key: str, default):
        return (
            self._config.raw.get("execution", {})
            .get("simulation", {})
            .get(key, default)
        )

    def _on_trade(self, trade) -> None:
        self._current_prices[trade.symbol] = trade.price
        if self._single_dollar:
            for bar in self._single_dollar.on_trade_all(trade):
                self._on_bar_closed(bar)
        elif self._single_time:
            bar = self._single_time.on_trade(trade)
            if bar:
                self._on_bar_closed(bar)

    def _apply_funding(self, timestamp, prices: dict[str, float]) -> None:
        for event_time in self._funding.due_events(self._funding_last_timestamp, timestamp):
            for symbol, pos in list(self._account.position_book.positions.items()):
                mark = prices.get(symbol, pos.avg_entry_price)
                cost = self._funding.cost(pos.quantity, mark)
                self._account.cash -= cost
                self._account.total_fees += max(cost, 0.0)
            self._funding_last_timestamp = event_time

    def _force_close_all(self, bar: Bar) -> None:
        for symbol, pos in list(self._account.position_book.positions.items()):
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            intent = OrderIntent(
                symbol=symbol, side=side, quantity=abs(pos.quantity),
                entry_type=OrderType.MARKET, timestamp=bar.timestamp,
                strategy_id=pos.strategy_id, allocation_id="risk_close",
                risk_id="circuit_breaker", limit_price=bar.close,
            )
            fill = self._execution.process_intent(intent, bar.close)
            if fill:
                realized = self._account.apply_fill(fill.symbol, fill.side.value, fill.quantity, fill.price, fill.fee, pos.strategy_id, bar.timestamp)
                self._trade_pnls.append(realized - fill.fee)
                self._protective_levels.pop(symbol, None)

    def _check_protective_exits(self, bar: Bar) -> bool:
        pos = self._account.position_book.get(bar.symbol)
        levels = self._protective_levels.get(bar.symbol)
        if pos is None or levels is None:
            return False
        stop, target = levels
        exit_price = None
        reason = None
        priority = self._config.raw.get("risk", {}).get("protective_exits", {}).get("intrabar_priority", "STOP_FIRST")
        if pos.quantity > 0:
            stop_hit = stop is not None and bar.low <= stop
            target_hit = target is not None and bar.high >= target
            if stop_hit and target_hit:
                exit_price, reason = (stop, "STOP") if priority == "STOP_FIRST" else (target, "TARGET")
            elif stop_hit:
                exit_price, reason = stop, "STOP"
            elif target_hit:
                exit_price, reason = target, "TARGET"
            side = OrderSide.SELL
        else:
            stop_hit = stop is not None and bar.high >= stop
            target_hit = target is not None and bar.low <= target
            if stop_hit and target_hit:
                exit_price, reason = (stop, "STOP") if priority == "STOP_FIRST" else (target, "TARGET")
            elif stop_hit:
                exit_price, reason = stop, "STOP"
            elif target_hit:
                exit_price, reason = target, "TARGET"
            side = OrderSide.BUY
        if exit_price is None:
            return False
        intent = OrderIntent(
            symbol=bar.symbol, side=side, quantity=abs(pos.quantity),
            entry_type=OrderType.MARKET, timestamp=bar.timestamp,
            strategy_id=pos.strategy_id, allocation_id="protective_exit",
            risk_id=f"{reason.lower()}_exit", limit_price=exit_price,
        )
        fill = self._execution.process_intent(intent, exit_price)
        if not fill:
            return False
        realized = self._account.apply_fill(fill.symbol, fill.side.value, fill.quantity, fill.price, fill.fee, pos.strategy_id, bar.timestamp)
        self._trade_pnls.append(realized - fill.fee)
        self._closed_trade_pnls.append(realized - fill.fee)
        if self._account.position_book.get(bar.symbol) is None:
            self._protective_levels.pop(bar.symbol, None)
        return True

    def _on_bar_closed(self, bar: Bar) -> None:
        session = bar.timestamp.astimezone(timezone.utc).date()
        if self._current_session != session:
            session_equity = self._account.compute_equity(self._current_prices)
            self._risk_monitor.reset_session(session_equity if session_equity > 0 else self._bt_config.initial_equity)
            self._current_session = session
            self._funding_last_timestamp = bar.timestamp

        self._current_prices[bar.symbol] = bar.close
        self._apply_funding(bar.timestamp, self._current_prices)

        # Protective exits are evaluated before new signals and never depend on a future bar.
        self._check_protective_exits(bar)
        equity = self._account.snapshot_equity(bar.timestamp, self._current_prices)
        risk_state = self._risk_monitor.update_equity(equity, bar.timestamp)
        if risk_state == RiskState.LOCKED:
            self._execution.cancel_all_pending()
            if self._config.risk.close_positions_on_breach:
                self._force_close_all(bar)
            return

        features = self._feature_engine.on_bar(bar)
        regime = self._regime_engine.on_features(bar.symbol, features, bar.timestamp)
        signal_confidences: dict[str, float] = {}
        for strategy in self._strategies:
            signal = strategy.on_bar(bar, features, regime)
            if signal and self._risk_monitor.check_signal(signal):
                existing = self._account.position_book.get(signal.symbol)
                if existing is not None:
                    same_direction = (existing.quantity > 0 and signal.direction == SignalDirection.BUY) or (existing.quantity < 0 and signal.direction == SignalDirection.SELL)
                    if same_direction:
                        continue
                self._signal_engine.add_signal(signal)
                signal_confidences[signal.strategy_id] = max(signal_confidences.get(signal.strategy_id, 0.0), signal.confidence)

        equity = self._account.snapshot_equity(bar.timestamp, self._current_prices)
        if self._risk_monitor.update_equity(equity, bar.timestamp) == RiskState.LOCKED:
            self._execution.cancel_all_pending()
            if self._config.risk.close_positions_on_breach:
                self._force_close_all(bar)
            return

        allocation = self._allocator.allocate(equity, regime, signal_confidences)
        for signal in self._signal_engine.get_actionable_signals(as_of=bar.timestamp):
            if not self._risk_monitor.check_signal(signal):
                continue
            intent = self._position_sizer.size(signal, allocation, equity, bar.close)
            if intent is None or not self._risk_monitor.check_order_intent(intent):
                continue
            fill = self._execution.process_intent(intent, bar.close)
            if fill:
                realized = self._account.apply_fill(fill.symbol, fill.side.value, fill.quantity, fill.price, fill.fee, intent.strategy_id, bar.timestamp)
                if realized != 0.0:
                    self._trade_pnls.append(realized - fill.fee)
                self._protective_levels[fill.symbol] = (intent.stop_loss, intent.take_profit)
                self._signal_engine.consume(signal.signal_id)
                prices = dict(self._current_prices); prices[bar.symbol] = bar.close
                self._risk_monitor.update_exposure(self._account.position_book.gross_exposure(prices), self._account.position_book.net_exposure(prices), self._account.position_book.asset_exposure(prices), self._account.position_book.strategy_exposure(prices), self._account.position_book.net_asset_exposure(prices))
                post_fill_equity = self._account.compute_equity(prices)
                if self._risk_monitor.update_equity(post_fill_equity, bar.timestamp) == RiskState.LOCKED:
                    self._execution.cancel_all_pending()
                    if self._config.risk.close_positions_on_breach:
                        self._force_close_all(bar)
                    break
        self._event_loop.emit_bar_closed(bar)

    def run(self, loader: TradeLoader) -> BacktestResult:
        ingestion = TradeIngestionEngine(self._config.data)
        result = ingestion.ingest(loader)

        self._risk_monitor.reset_session(self._bt_config.initial_equity)

        trades = sorted(result.trades, key=lambda t: (t.timestamp, t.trade_id)) if self._bt_config.sort_trades_by_timestamp else result.trades
        for trade in trades:
            self._event_loop.process_trade(trade)

        # Flush remaining bars
        if self._single_dollar and trades:
            symbol = trades[-1].symbol
            bar = self._single_dollar.flush(symbol)
            if bar:
                self._on_bar_closed(bar)
        elif self._single_time and trades:
            symbol = trades[-1].symbol
            bar = self._single_time.flush(symbol)
            if bar:
                self._on_bar_closed(bar)

        equity_values = [e for _, e in self._account.equity_curve]
        if not equity_values:
            equity_values = [self._bt_config.initial_equity]

        equity_timestamps = [ts for ts, _ in self._account.equity_curve]
        metrics = compute_metrics(
            equity_values,
            self._trade_pnls,
            self._account.total_fees,
            timestamps=equity_timestamps,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_values,
            bar_count=self._event_loop.context.bar_count,
            trade_count=self._event_loop.context.trade_count,
            fill_count=len(self._execution.fill_history),
            risk_state=self._risk_monitor.get_risk_state(),
            experiment={
                "bar_type": self._bt_config.bar_type,
                "dollar_threshold": self._bt_config.dollar_threshold_name,
                "time_interval": self._bt_config.time_interval,
                "initial_equity": self._bt_config.initial_equity,
                "seed": self._bt_config.seed,
                "sort_trades_by_timestamp": self._bt_config.sort_trades_by_timestamp,
                "total_realized_pnl": self._account.total_realized_pnl,
                "total_fees": self._account.total_fees,
            },
        )

    def run_trades(self, trades: list) -> BacktestResult:
        from core.data.loaders.mock_loader import MockTradeLoader

        records = [
            {
                "timestamp": t.timestamp.isoformat() if hasattr(t.timestamp, "isoformat") else t.timestamp,
                "symbol": t.symbol,
                "price": t.price,
                "quantity": t.quantity,
                "side": t.side.value if hasattr(t.side, "value") else t.side,
                "trade_id": t.trade_id,
                "exchange": t.exchange,
            }
            for t in trades
        ]
        return self.run(MockTradeLoader(records))
