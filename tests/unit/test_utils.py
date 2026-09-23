"""Unit tests for utility modules."""

from datetime import datetime, timezone

import pytest

from utils.time import ensure_utc, parse_timestamp, is_monotonic
from utils.math import (
    safe_divide,
    clamp,
    normalize_weights,
    validate_allocation_sum,
    annualized_sharpe,
    max_drawdown,
)


class TestTimeUtils:
    def test_ensure_utc_naive(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = ensure_utc(dt)
        assert result.tzinfo == timezone.utc

    def test_ensure_utc_aware(self):
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = ensure_utc(dt)
        assert result.tzinfo == timezone.utc

    def test_parse_timestamp_from_datetime(self):
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = parse_timestamp(dt)
        assert result == dt

    def test_parse_timestamp_from_unix(self):
        result = parse_timestamp(1705314600.0)
        assert result.year == 2024

    def test_parse_timestamp_from_millis(self):
        result = parse_timestamp(1705314600000)
        assert result.year == 2024

    def test_parse_timestamp_from_iso_string(self):
        result = parse_timestamp("2024-01-15T10:30:00Z")
        assert result.month == 1
        assert result.day == 15

    def test_is_monotonic_true(self):
        ts = [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        ]
        assert is_monotonic(ts) is True

    def test_is_monotonic_false(self):
        ts = [
            datetime(2024, 1, 3, tzinfo=timezone.utc),
            datetime(2024, 1, 1, tzinfo=timezone.utc),
        ]
        assert is_monotonic(ts) is False


class TestMathUtils:
    def test_safe_divide_normal(self):
        assert safe_divide(10.0, 2.0) == 5.0

    def test_safe_divide_zero_denominator(self):
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, 0.0, default=-1.0) == -1.0

    def test_clamp(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(-1.0, 0.0, 10.0) == 0.0
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_normalize_weights(self):
        weights = {"A": 30.0, "B": 70.0}
        result = normalize_weights(weights)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert abs(result["A"] - 0.3) < 1e-9

    def test_normalize_weights_zero_sum(self):
        weights = {"A": 0.0, "B": 0.0}
        result = normalize_weights(weights)
        assert all(v == 0.0 for v in result.values())

    def test_validate_allocation_sum_valid(self):
        assert validate_allocation_sum({"A": 40.0, "B": 30.0}) is True

    def test_validate_allocation_sum_exceeds(self):
        assert validate_allocation_sum({"A": 60.0, "B": 50.0}) is False

    def test_annualized_sharpe(self):
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        sharpe = annualized_sharpe(returns)
        assert isinstance(sharpe, float)

    def test_annualized_sharpe_insufficient_data(self):
        assert annualized_sharpe([0.01]) == 0.0

    def test_max_drawdown(self):
        equity = [100, 110, 105, 95, 100, 120]
        dd = max_drawdown(equity)
        assert dd > 0
        assert dd <= 1.0

    def test_max_drawdown_no_drawdown(self):
        equity = [100, 110, 120, 130]
        assert max_drawdown(equity) == 0.0
