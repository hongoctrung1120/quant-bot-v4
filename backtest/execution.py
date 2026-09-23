"""Helpers for realistic backtest execution semantics."""
from __future__ import annotations

from backtest.costs import TransactionCostModel
from backtest.fills import Fill, FillSimulator
from core.models.order import OrderIntent, OrderStatus


def simulate_order_fills(
    intent: OrderIntent,
    market_price: float,
    costs: TransactionCostModel,
    fills: FillSimulator,
) -> list[Fill]:
    """Simulate one intent until fully filled or rejected.

    Partial fills are represented as multiple fill events and the remaining
    quantity is retried deterministically within the same execution event.
    """
    remaining = intent.quantity
    out: list[Fill] = []
    while remaining > 1e-12:
        fill_price, fee = costs.total_cost(market_price, remaining, intent.side.value)
        partial_intent = OrderIntent(
            symbol=intent.symbol,
            side=intent.side,
            quantity=remaining,
            entry_type=intent.entry_type,
            timestamp=intent.timestamp,
            strategy_id=intent.strategy_id,
            allocation_id=intent.allocation_id,
            risk_id=intent.risk_id,
            order_id=intent.order_id,
            limit_price=intent.limit_price,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
        )
        fill = fills.simulate(partial_intent, market_price, fill_price, fee)
        if fill is None:
            break
        out.append(fill)
        remaining -= fill.quantity
        if fill.status != OrderStatus.PARTIALLY_FILLED:
            break
    return out
