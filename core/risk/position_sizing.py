"""Position sizing engine."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from core.config import CapitalConfig, RiskConfig
from core.interfaces import PositionSizer
from core.models.allocation import AllocationResult
from core.models.order import OrderIntent, OrderSide, OrderType
from core.models.signal import Signal, SignalDirection
from utils.math import clamp, safe_divide

logger = logging.getLogger(__name__)


class RiskBasedPositionSizer(PositionSizer):
    """Size positions based on risk per trade and stop distance."""

    def __init__(self, config: RiskConfig, capital_config: Optional[CapitalConfig] = None) -> None:
        self._config = config
        self._capital_config = capital_config or CapitalConfig()

    def size(
        self,
        signal: Signal,
        allocation: AllocationResult,
        equity: float,
        current_price: float,
    ) -> Optional[OrderIntent]:
        if signal.direction not in (SignalDirection.BUY, SignalDirection.SELL):
            return None

        if current_price <= 0 or equity <= 0:
            return None

        risk_amount = equity * (self._config.risk_per_trade_pct / 100.0)

        asset_pct = allocation.asset_allocations.get(signal.symbol, 0.0)
        strategy_pct = allocation.strategy_allocations.get(
            signal.strategy_id, 0.0
        )
        budget = allocation.trading_capital * (asset_pct / 100.0) * (strategy_pct / 100.0)

        if signal.stop_reference is not None:
            stop_distance = abs(current_price - signal.stop_reference)
        else:
            stop_distance = current_price * 0.02  # 2% default stop

        if stop_distance <= 0:
            return None

        quantity = safe_divide(risk_amount, stop_distance, default=0.0)
        max_qty = safe_divide(budget, current_price, default=0.0)
        quantity = min(quantity, max_qty)

        max_position_value = equity * (self._capital_config.max_single_position_pct / 100.0)
        max_qty_by_position = safe_divide(max_position_value, current_price, default=0.0)
        quantity = min(quantity, max_qty_by_position)

        if quantity <= 0:
            return None

        side = (
            OrderSide.BUY if signal.direction == SignalDirection.BUY
            else OrderSide.SELL
        )

        return OrderIntent(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            entry_type=OrderType.MARKET,
            timestamp=signal.timestamp,
            strategy_id=signal.strategy_id,
            allocation_id=allocation.allocation_id,
            risk_id=str(uuid.uuid4()),
            limit_price=current_price,
            stop_loss=signal.stop_reference,
            take_profit=signal.target_reference,
        )
