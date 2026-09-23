"""Unit tests for capital allocation."""

import pytest

from datetime import datetime, timezone

from core.capital.allocation_models import (
    equal_weight_allocation,
    inverse_volatility_allocation,
    apply_regime_adjustments,
    enforce_limits,
)
from core.capital.allocator import CapitalAllocationEngine
from core.config import CapitalConfig
from core.models.regime import MarketRegime, RegimeOutput
from datetime import datetime, timezone


class TestAllocationModels:
    def test_equal_weight(self):
        result = equal_weight_allocation(["BTC", "ETH", "SOL"])
        assert sum(result.values()) == pytest.approx(100.0)
        assert len(result) == 3

    def test_inverse_volatility(self):
        vols = {"BTC": 2.0, "ETH": 4.0, "SOL": 8.0}
        result = inverse_volatility_allocation(vols, max_pct=80.0)
        assert result["BTC"] > result["ETH"] > result["SOL"]

    def test_regime_adjustment(self):
        weights = {"trend": 40.0, "mean_reversion": 20.0}
        adj = {"TREND_BULL": {"trend": 1.2, "mean_reversion": 0.7}}
        result = apply_regime_adjustments(weights, "TREND_BULL", adj)
        assert result["trend"] > weights["trend"]

    def test_enforce_limits(self):
        alloc = {"BTC": 50.0, "ETH": 30.0}
        capped = enforce_limits(alloc, max_single=40.0)
        assert capped["BTC"] == 40.0


class TestCapitalAllocationEngine:
    def test_allocate_respects_reserve(self):
        config = CapitalConfig(reserve_pct=20.0, max_trading_capital_pct=80.0)
        engine = CapitalAllocationEngine(
            config=config,
            asset_weights={"BTC/USDT": 40.0, "ETH/USDT": 25.0},
            strategy_weights={"trend": 40.0},
        )
        result = engine.allocate(100000.0)
        assert result.reserve_amount == 20000.0
        assert result.trading_capital == 80000.0

    def test_regime_adaptive(self):
        config = CapitalConfig()
        engine = CapitalAllocationEngine(
            config=config,
            strategy_weights={"trend_following": 40.0, "mean_reversion": 20.0},
            regime_adjustments={
                "TREND_BULL": {"trend_following": 1.2, "mean_reversion": 0.7}
            },
        )
        regime = RegimeOutput(
            regime=MarketRegime.TREND_BULL,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
        )
        result = engine.allocate(100000.0, regime=regime)
        assert "trend_following" in result.strategy_allocations
