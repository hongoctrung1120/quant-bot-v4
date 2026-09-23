"""Unit tests for backtesting engine."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from backtest.engine import BacktestConfig, BacktestEngine
from core.config import ConfigEngine
from core.data.loaders import CSVTradeLoader


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestBacktestEngine:
    def setup_method(self):
        config_dir = ConfigEngine.get_default_config_dir()
        self.config = ConfigEngine(config_dir).load()

    def test_run_csv_backtest(self):
        engine = BacktestEngine(
            self.config,
            BacktestConfig(
                bar_type="DOLLAR",
                dollar_threshold_name="small",
                initial_equity=100000.0,
            ),
        )
        result = engine.run(CSVTradeLoader(FIXTURES / "sample_trades.csv"))
        assert result.bar_count >= 0
        assert len(result.equity_curve) >= 0
        assert result.experiment["bar_type"] == "DOLLAR"

    def test_metrics_computed(self):
        from backtest.metrics import compute_metrics

        curve = [100000, 101000, 100500, 102000]
        metrics = compute_metrics(curve, [1000, -500, 1500], total_fees=50.0)
        assert metrics.total_return > 0
        assert metrics.num_trades == 3

    def test_transaction_costs_applied(self):
        from backtest.costs import TransactionCostModel, CostConfig

        model = TransactionCostModel(CostConfig(slippage_pct=1.0, taker_fee_pct=0.05))
        price, fee = model.total_cost(100.0, 1.0, "BUY")
        assert price > 100.0
        assert fee > 0
