"""Unit tests for execution engine."""

from datetime import datetime, timezone

import pytest

from core.execution.execution_engine import SimulatedExecutionEngine
from core.execution.reconciliation import OrderReconciler
from core.models.order import OrderIntent, OrderSide, OrderType


class TestSimulatedExecution:
    @pytest.mark.asyncio
    async def test_submit_and_fill(self):
        engine = SimulatedExecutionEngine(
            fill_probability=1.0,
            partial_fills=False,
            seed=42,
        )
        intent = OrderIntent(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            entry_type=OrderType.MARKET,
            timestamp=datetime.now(timezone.utc),
            strategy_id="trend",
            allocation_id="a1",
            risk_id="r1",
        )
        order_id = await engine.submit(intent)
        fill = engine.process_intent(intent, 50000.0)
        assert fill is not None
        assert fill.quantity == pytest.approx(0.1, rel=0.01)
        assert fill.fee > 0

    def test_cancel_all_pending(self):
        engine = SimulatedExecutionEngine()
        count = engine.cancel_all_pending()
        assert count == 0


class TestReconciliation:
    def test_position_match(self):
        reconciler = OrderReconciler()
        result = reconciler.reconcile(
            {"BTC/USDT": 0.5},
            {"BTC/USDT": 0.5},
        )
        assert result.success

    def test_position_mismatch(self):
        reconciler = OrderReconciler()
        result = reconciler.reconcile(
            {"BTC/USDT": 0.5},
            {"BTC/USDT": 0.3},
        )
        assert not result.success
        assert len(result.mismatches) == 1
