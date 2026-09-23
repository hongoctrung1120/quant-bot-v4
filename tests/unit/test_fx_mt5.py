"""MT5 conversions, exercised without a running terminal."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from fx.broker.mt5 import (
    Mt5Client,
    Mt5Unavailable,
    bars_from_rates,
    instrument_from_symbol_info,
    median_spread_pips,
    pip_size_from_digits,
)

RATES_DTYPE = np.dtype(
    [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
        ("spread", "i4"),
        ("real_volume", "i8"),
    ]
)


def make_rates(rows):
    return np.array(rows, dtype=RATES_DTYPE)


def symbol_info(**overrides):
    defaults = dict(
        digits=5,
        point=0.00001,
        trade_contract_size=100_000.0,
        volume_min=0.01,
        volume_step=0.01,
        volume_max=200.0,
        spread=12,
        swap_long=-7.5,
        swap_short=1.2,
        swap_mode=1,
        currency_base="EUR",
        currency_profit="USD",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestPipSize:
    def test_five_digit_quote_has_ten_point_pip(self):
        assert pip_size_from_digits(5, 0.00001) == pytest.approx(0.0001)

    def test_three_digit_jpy_quote_has_ten_point_pip(self):
        assert pip_size_from_digits(3, 0.001) == pytest.approx(0.01)

    def test_four_digit_quote_pip_equals_point(self):
        assert pip_size_from_digits(4, 0.0001) == pytest.approx(0.0001)


class TestInstrumentFromSymbolInfo:
    def test_reads_contract_terms_from_the_broker(self):
        inst = instrument_from_symbol_info(symbol_info(), "EURUSD")
        assert inst.pip_size == pytest.approx(0.0001)
        assert inst.contract_size == 100_000.0
        assert inst.min_lot == 0.01
        assert inst.base == "EUR" and inst.quote == "USD"

    def test_converts_swap_points_into_pips(self):
        inst = instrument_from_symbol_info(symbol_info(), "EURUSD")
        # -7.5 points on a 5-digit quote is -0.75 pips per day.
        assert inst.swap_long_pips_per_day == pytest.approx(-0.75)
        assert inst.swap_short_pips_per_day == pytest.approx(0.12)

    def test_spread_points_become_pips(self):
        inst = instrument_from_symbol_info(symbol_info(spread=12), "EURUSD")
        assert inst.typical_spread_pips == pytest.approx(1.2)

    def test_explicit_spread_overrides_the_snapshot(self):
        inst = instrument_from_symbol_info(symbol_info(), "EURUSD", spread_pips=0.4)
        assert inst.typical_spread_pips == pytest.approx(0.4)

    def test_non_point_swap_mode_is_zeroed_rather_than_misread(self):
        inst = instrument_from_symbol_info(symbol_info(swap_mode=3), "EURUSD")
        assert inst.swap_long_pips_per_day == 0.0
        assert inst.swap_short_pips_per_day == 0.0

    def test_cent_account_contract_size_is_taken_as_reported(self):
        inst = instrument_from_symbol_info(
            symbol_info(trade_contract_size=1_000.0), "EURUSD"
        )
        assert inst.contract_size == 1_000.0

    def test_jpy_pair_specification(self):
        inst = instrument_from_symbol_info(
            symbol_info(digits=3, point=0.001, currency_base="USD",
                        currency_profit="JPY", spread=15, swap_long=-20.0),
            "USDJPY",
        )
        assert inst.pip_size == pytest.approx(0.01)
        assert inst.typical_spread_pips == pytest.approx(1.5)
        assert inst.swap_long_pips_per_day == pytest.approx(-2.0)


class TestBarsFromRates:
    def test_converts_rates_into_utc_bars(self):
        epoch = int(datetime(2024, 3, 5, 8, 0, tzinfo=timezone.utc).timestamp())
        rates = make_rates(
            [
                (epoch, 1.0850, 1.0870, 1.0840, 1.0865, 500, 10, 0),
                (epoch + 3600, 1.0865, 1.0880, 1.0860, 1.0875, 600, 11, 0),
            ]
        )
        bars = bars_from_rates(rates, "EURUSD", "H1")

        assert len(bars) == 2
        assert bars[0].timestamp == datetime(2024, 3, 5, 8, 0, tzinfo=timezone.utc)
        assert bars[0].close == pytest.approx(1.0865)
        assert bars[0].volume == 500
        assert bars[1].timestamp == datetime(2024, 3, 5, 9, 0, tzinfo=timezone.utc)

    def test_bar_duration_follows_the_timeframe(self):
        epoch = int(datetime(2024, 3, 5, 8, 0, tzinfo=timezone.utc).timestamp())
        rates = make_rates([(epoch, 1.08, 1.09, 1.07, 1.085, 100, 10, 0)])
        bar = bars_from_rates(rates, "EURUSD", "H4")[0]
        assert (bar.end_timestamp - bar.start_timestamp).total_seconds() == 4 * 3600


class TestMedianSpread:
    def test_median_of_historical_spread_in_pips(self):
        epoch = 1_700_000_000
        rates = make_rates(
            [
                (epoch + i * 3600, 1.08, 1.09, 1.07, 1.08, 100, spread, 0)
                for i, spread in enumerate([8, 10, 12, 40, 9])
            ]
        )
        # Points sorted: 8, 9, 10, 12, 40 -> median 10 points -> 1.0 pip.
        assert median_spread_pips(rates, 0.00001, 5) == pytest.approx(1.0)

    def test_returns_none_without_spread_data(self):
        dtype = np.dtype([("time", "i8"), ("open", "f8"), ("high", "f8"),
                          ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")])
        rates = np.array([(1, 1.0, 1.1, 0.9, 1.05, 10)], dtype=dtype)
        assert median_spread_pips(rates, 0.00001, 5) is None


class TestClientGuards:
    def test_using_the_client_before_connecting_is_an_error(self):
        with pytest.raises(Mt5Unavailable):
            Mt5Client().mt5

    def test_unknown_timeframe_is_rejected(self):
        client = Mt5Client()
        client._mt5 = SimpleNamespace(TIMEFRAME_H1=16385)
        with pytest.raises(ValueError):
            client._timeframe("H3")
