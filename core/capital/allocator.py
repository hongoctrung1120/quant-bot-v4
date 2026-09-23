"""Capital allocation engine."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from core.capital.allocation_models import apply_confidence_weights, apply_regime_adjustments, enforce_limits, equal_weight_allocation, fixed_weight_allocation, inverse_volatility_allocation
from core.capital.budget import CapitalBudget
from core.config import CapitalConfig
from core.interfaces import CapitalAllocator
from core.models.allocation import AllocationResult
from core.models.regime import RegimeOutput

class CapitalAllocationEngine(CapitalAllocator):
    """Allocate trading capital independently from position sizing."""
    def __init__(self, config: CapitalConfig, asset_weights: Optional[dict[str, float]] = None, strategy_weights: Optional[dict[str, float]] = None, regime_adjustments: Optional[dict[str, dict[str, float]]] = None, asset_volatilities: Optional[dict[str, float]] = None) -> None:
        self._config=config; self._asset_weights=asset_weights or {}; self._strategy_weights=strategy_weights or {}; self._regime_adjustments=regime_adjustments or {}; self._asset_volatilities=asset_volatilities or {}
    def allocate(self, total_equity: float, regime: Optional[RegimeOutput] = None, signal_confidences: Optional[dict[str,float]] = None) -> AllocationResult:
        if total_equity <= 0: raise ValueError("total_equity must be positive")
        budget=CapitalBudget.from_equity(total_equity,self._config.reserve_pct,self._config.max_trading_capital_pct)
        tradable_pct=100.0*budget.trading_capital/total_equity
        method=self._config.allocation_method.lower()
        if method=="equal" and self._asset_weights:
            assets=equal_weight_allocation(list(self._asset_weights),tradable_pct)
        elif method=="inverse_volatility" and self._asset_volatilities:
            assets=inverse_volatility_allocation(self._asset_volatilities,tradable_pct)
        else:
            assets=fixed_weight_allocation(self._asset_weights,tradable_pct)
        assets=enforce_limits(assets,self._config.max_asset_allocation_pct,tradable_pct)
        strategies=fixed_weight_allocation(self._strategy_weights,100.0)
        if regime is not None: strategies=apply_regime_adjustments(strategies,regime.regime.value,self._regime_adjustments)
        strategies=apply_confidence_weights(strategies,signal_confidences)
        strategies=enforce_limits(strategies,self._config.max_strategy_allocation_pct,sum(strategies.values()))
        return AllocationResult(timestamp=datetime.now(timezone.utc),total_equity=total_equity,reserve_amount=budget.reserve_amount,trading_capital=budget.trading_capital,asset_allocations=assets,strategy_allocations=strategies,regime_adjustments={regime.regime.value:1.0} if regime else {},method=method,notes="Asset allocations are percentages of trading capital; strategy allocations are relative weights.")
