"""Instrument maths, margin accounting, sizing granularity and risk gates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fx.account import FxAccount
from fx.instruments import (
    RateBook,
    build_universe,
    get_instrument,
    margin_required,
    pip_value_per_lot,
)
from fx.risk.exposure import currency_exposure, currency_risk, summarize
from fx.risk.governor import RiskGovernor, RiskLimits
from fx.risk.sizing import FxPositionSizer, minimum_viable_equity

WEDNESDAY = datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc)


def make_rates() -> RateBook:
    rates = RateBook("USD")
    rates.update("EURUSD", 1.0850)
    rates.update("GBPUSD", 1.2700)
    rates.update("USDJPY", 148.0)
    rates.update("AUDUSD", 0.6550)
    return rates


class TestPipValue:
    def test_standard_account_eurusd_pip_is_ten_dollars(self):
        inst = get_instrument("EURUSD")
        assert pip_value_per_lot(inst, make_rates()) == pytest.approx(10.0)

    def test_cent_account_scales_pip_value_by_one_hundredth(self):
        inst = get_instrument("EURUSD", cent_account=True)
        assert pip_value_per_lot(inst, make_rates()) == pytest.approx(0.10)

    def test_jpy_quote_converts_through_usdjpy(self):
        inst = get_instrument("USDJPY")
        # 0.01 * 100_000 = 1000 JPY per pip per lot, converted at 148.
        assert pip_value_per_lot(inst, make_rates()) == pytest.approx(1000.0 / 148.0)

    def test_cross_pair_triangulates_to_account_currency(self):
        rates = make_rates()
        inst = get_instrument("EURGBP")
        # 10 GBP per pip per lot, GBP->USD via GBPUSD = 1.27
        assert pip_value_per_lot(inst, rates) == pytest.approx(10.0 * 1.27)

    def test_margin_uses_leverage(self):
        inst = get_instrument("EURUSD")
        margin = margin_required(inst, 0.1, 1.0850, 100.0, make_rates())
        assert margin == pytest.approx(0.1 * 100_000 * 1.0850 / 100.0)


class TestMarginAccount:
    def build(self, balance: float = 10_000.0) -> FxAccount:
        return FxAccount(
            balance=balance,
            leverage=100.0,
            rates=make_rates(),
            instruments=build_universe(["EURUSD", "USDJPY"]),
        )

    def test_long_profit_is_price_move_times_notional(self):
        account = self.build()
        account.open_position("EURUSD", 0.1, 1.0850, WEDNESDAY, "test", cost=0.0)
        assert account.unrealized_pnl({"EURUSD": 1.0950}) == pytest.approx(100.0)

    def test_short_profits_when_price_falls(self):
        account = self.build()
        account.open_position("EURUSD", -0.1, 1.0850, WEDNESDAY, "test", cost=0.0)
        assert account.unrealized_pnl({"EURUSD": 1.0750}) == pytest.approx(100.0)

    def test_equity_is_balance_plus_unrealized_not_cash_minus_notional(self):
        account = self.build(100.0)
        account.open_position("EURUSD", 0.01, 1.0850, WEDNESDAY, "test", cost=0.0)
        # Spot accounting would show 100 - 1085 = -985 here.
        assert account.equity({"EURUSD": 1.0850}) == pytest.approx(100.0)

    def test_closing_realizes_pnl_and_records_r_multiple(self):
        account = self.build()
        account.open_position(
            "EURUSD", 0.1, 1.0850, WEDNESDAY, "test", cost=1.0, risk_amount=50.0
        )
        trade = account.close_position(
            "EURUSD", 1.0950, WEDNESDAY + timedelta(hours=4), cost=1.0, reason="TARGET"
        )
        assert trade.gross_pnl == pytest.approx(100.0)
        assert trade.net_pnl == pytest.approx(98.0)
        assert trade.r_multiple == pytest.approx(98.0 / 50.0)
        assert account.balance == pytest.approx(10_000.0 + 98.0)

    def test_margin_level_reflects_leverage_usage(self):
        account = self.build(1_000.0)
        account.open_position("EURUSD", 0.1, 1.0850, WEDNESDAY, "test", cost=0.0)
        prices = {"EURUSD": 1.0850}
        assert account.used_margin(prices) == pytest.approx(108.50)
        assert account.margin_level(prices) == pytest.approx(1000.0 / 108.50)

    def test_swap_debits_balance_and_attaches_to_trade(self):
        account = self.build()
        account.open_position("EURUSD", 0.1, 1.0850, WEDNESDAY, "test", cost=0.0)
        account.apply_swap("EURUSD", -0.60)
        assert account.balance == pytest.approx(10_000.0 - 0.60)
        trade = account.close_position(
            "EURUSD", 1.0850, WEDNESDAY + timedelta(days=1), cost=0.0, reason="EXIT"
        )
        assert trade.net_pnl == pytest.approx(-0.60)


class TestSizingGranularity:
    """The $100-account granularity problem, encoded as tests."""

    def setup_method(self):
        self.rates = make_rates()
        self.limits = RiskLimits(risk_per_trade_pct=0.5)
        self.sizer = FxPositionSizer(self.limits)

    def test_micro_lot_account_cannot_honour_half_percent_risk_on_100_usd(self):
        inst = get_instrument("EURUSD")
        result = self.sizer.size(
            instrument=inst,
            entry_price=1.0850,
            stop_price=1.0830,  # 20 pips
            equity=100.0,
            rates=self.rates,
            is_long=True,
            free_margin=100.0,
        )
        assert not result.accepted
        assert result.rejected == "below_min_lot"
        # The smallest tradable lot would risk 2% — four times the budget.
        assert result.min_lot_risk_pct == pytest.approx(2.0)

    def test_cent_account_makes_the_same_trade_sizeable(self):
        inst = get_instrument("EURUSD", cent_account=True)
        result = self.sizer.size(
            instrument=inst,
            entry_price=1.0850,
            stop_price=1.0830,
            equity=100.0,
            rates=self.rates,
            is_long=True,
            free_margin=100.0,
        )
        assert result.accepted
        assert result.lots == pytest.approx(0.25)
        assert result.risk_amount == pytest.approx(0.50)

    def test_risk_scaler_shrinks_position(self):
        inst = get_instrument("EURUSD", cent_account=True)
        result = self.sizer.size(
            instrument=inst,
            entry_price=1.0850,
            stop_price=1.0830,
            equity=100.0,
            rates=self.rates,
            is_long=True,
            risk_scaler=0.5,
            free_margin=100.0,
        )
        assert result.lots == pytest.approx(0.12)  # 0.125 floored to lot step

    def test_wider_stop_produces_smaller_position(self):
        inst = get_instrument("EURUSD", cent_account=True)
        narrow = self.sizer.size(
            instrument=inst, entry_price=1.0850, stop_price=1.0830,
            equity=1000.0, rates=self.rates, is_long=True, free_margin=1000.0,
        )
        wide = self.sizer.size(
            instrument=inst, entry_price=1.0850, stop_price=1.0750,
            equity=1000.0, rates=self.rates, is_long=True, free_margin=1000.0,
        )
        assert wide.lots < narrow.lots
        assert narrow.risk_amount == pytest.approx(wide.risk_amount, rel=0.05)

    def test_minimum_viable_equity_reports_the_real_threshold(self):
        inst = get_instrument("EURUSD")
        needed = minimum_viable_equity(inst, 20.0, self.rates, 0.5)
        assert needed == pytest.approx(400.0)

    def test_margin_cap_reduces_size_when_free_margin_is_thin(self):
        inst = get_instrument("EURUSD", cent_account=True)
        result = self.sizer.size(
            instrument=inst, entry_price=1.0850, stop_price=1.0830,
            equity=100.0, rates=self.rates, is_long=True,
            free_margin=1.0, leverage=100.0,
        )
        assert result.lots < 0.25


class TestCurrencyExposure:
    def test_two_long_usd_quoted_pairs_double_the_short_usd_exposure(self):
        rates = make_rates()
        instruments = build_universe(["EURUSD", "GBPUSD"])
        account = FxAccount(10_000.0, 100.0, rates, instruments)
        account.open_position("EURUSD", 0.1, 1.0850, WEDNESDAY, "s", cost=0.0)
        account.open_position("GBPUSD", 0.1, 1.2700, WEDNESDAY, "s", cost=0.0)

        exposure = currency_exposure(
            account.positions, {"EURUSD": 1.0850, "GBPUSD": 1.2700}, instruments, rates
        )
        assert exposure["EUR"] == pytest.approx(10_850.0)
        assert exposure["GBP"] == pytest.approx(12_700.0)
        assert exposure["USD"] == pytest.approx(-23_550.0)

        report = summarize(exposure)
        assert report.largest_currency == "USD"

    def test_opposing_pairs_net_out(self):
        rates = make_rates()
        instruments = build_universe(["EURUSD", "GBPUSD"])
        account = FxAccount(10_000.0, 100.0, rates, instruments)
        account.open_position("EURUSD", 0.1, 1.0850, WEDNESDAY, "s", cost=0.0)
        account.open_position("GBPUSD", -0.1, 1.2700, WEDNESDAY, "s", cost=0.0)
        exposure = currency_exposure(
            account.positions, {"EURUSD": 1.0850, "GBPUSD": 1.2700}, instruments, rates
        )
        assert abs(exposure["USD"]) < 2_000.0

    def test_currency_risk_is_signed_so_opposite_views_net_off(self):
        rates = make_rates()
        instruments = build_universe(["EURUSD", "USDJPY"])
        account = FxAccount(10_000.0, 100.0, rates, instruments)
        account.open_position(
            "EURUSD", 0.1, 1.0850, WEDNESDAY, "s", cost=0.0, risk_amount=50.0
        )
        account.open_position(
            "USDJPY", 0.1, 148.0, WEDNESDAY, "s", cost=0.0, risk_amount=50.0
        )
        risk = currency_risk(account.positions, instruments)
        # Long EURUSD is short USD, long USDJPY is long USD: the views cancel.
        assert risk["USD"] == pytest.approx(0.0)
        assert risk["EUR"] == pytest.approx(50.0)
        assert risk["JPY"] == pytest.approx(-50.0)

    def test_third_same_side_position_is_refused_by_currency_risk_cap(self):
        rates = make_rates()
        limits = RiskLimits(risk_per_trade_pct=0.5, max_currency_risk_pct=1.5)
        sizer = FxPositionSizer(limits)
        inst = get_instrument("AUDUSD", cent_account=True)
        # Two existing short-USD positions already used the full 1.5% budget.
        existing_risk = {"USD": -15.0, "EUR": 5.0, "GBP": 5.0}
        result = sizer.size(
            instrument=inst, entry_price=0.6550, stop_price=0.6530,
            equity=1000.0, rates=rates, is_long=True, free_margin=1000.0,
            current_currency_risk=existing_risk,
        )
        assert not result.accepted
        assert result.rejected == "currency_risk_cap"

    def test_opposite_side_position_still_allowed_under_risk_cap(self):
        rates = make_rates()
        limits = RiskLimits(risk_per_trade_pct=0.5, max_currency_risk_pct=1.5)
        sizer = FxPositionSizer(limits)
        inst = get_instrument("AUDUSD", cent_account=True)
        result = sizer.size(
            instrument=inst, entry_price=0.6550, stop_price=0.6530,
            equity=1000.0, rates=rates, is_long=False, free_margin=1000.0,
            current_currency_risk={"USD": -15.0},
        )
        assert result.accepted, "selling AUDUSD reduces net short-USD risk"


class TestRiskGovernor:
    def test_daily_loss_breach_locks_new_entries(self):
        gov = RiskGovernor(RiskLimits(daily_loss_limit_pct=3.0))
        gov.on_bar(WEDNESDAY, 100.0)
        gov.on_bar(WEDNESDAY + timedelta(hours=1), 96.5)
        decision = self.entry(gov, WEDNESDAY + timedelta(hours=2))
        assert not decision.allowed
        assert decision.reason == "loss_limit_lock"

    def test_lock_expires_next_day(self):
        gov = RiskGovernor(RiskLimits(daily_loss_limit_pct=3.0))
        gov.on_bar(WEDNESDAY, 100.0)
        gov.on_bar(WEDNESDAY + timedelta(hours=1), 96.5)
        gov.on_bar(WEDNESDAY + timedelta(days=1, hours=1), 96.5)
        assert self.entry(gov, WEDNESDAY + timedelta(days=1, hours=2)).allowed

    def test_max_drawdown_halts_permanently(self):
        gov = RiskGovernor(RiskLimits(max_drawdown_pct=20.0, daily_loss_limit_pct=50.0))
        gov.on_bar(WEDNESDAY, 100.0)
        gov.on_bar(WEDNESDAY + timedelta(days=1), 79.0)
        assert gov.halted
        gov.on_bar(WEDNESDAY + timedelta(days=30), 100.0)
        assert gov.halted, "halt must survive equity recovery until reviewed"

    def test_drawdown_scaler_reduces_risk_progressively(self):
        gov = RiskGovernor(
            RiskLimits(warning_drawdown_pct=10.0, max_drawdown_pct=20.0,
                       daily_loss_limit_pct=50.0)
        )
        gov.on_bar(WEDNESDAY, 100.0)
        assert gov.drawdown_scaler() == pytest.approx(1.0)
        gov.on_bar(WEDNESDAY + timedelta(days=1), 92.0)
        assert gov.drawdown_scaler() == pytest.approx(1.0)
        gov.on_bar(WEDNESDAY + timedelta(days=2), 85.0)
        assert gov.drawdown_scaler() == pytest.approx(0.625)

    def test_scalers_never_exceed_one(self):
        gov = RiskGovernor()
        gov.on_bar(WEDNESDAY, 100.0)
        for day in range(1, 40):
            gov.on_bar(WEDNESDAY + timedelta(days=day), 100.0 + day * 0.01)
        assert gov.risk_scaler() <= 1.0

    def test_weekend_window_blocks_entries(self):
        gov = RiskGovernor(RiskLimits(flat_before_weekend_hours=2.0))
        friday_late = datetime(2024, 1, 5, 20, 0, tzinfo=timezone.utc)
        gov.on_bar(friday_late, 100.0)
        assert gov.should_flatten_for_weekend(friday_late)
        assert not self.entry(gov, friday_late).allowed

    def test_news_blackout_blocks_entries(self):
        event = WEDNESDAY + timedelta(hours=3)
        gov = RiskGovernor(news_windows=[(event, event)])
        gov.on_bar(WEDNESDAY, 100.0)
        assert not self.entry(gov, event).allowed
        assert self.entry(gov, event + timedelta(hours=2)).allowed

    def test_currency_position_cap(self):
        gov = RiskGovernor(RiskLimits(max_positions_per_currency=2))
        gov.on_bar(WEDNESDAY, 100.0)
        decision = gov.assess_entry(
            timestamp=WEDNESDAY, symbol="AUDUSD", open_symbols=["EURUSD", "GBPUSD"],
            currency_counts={"USD": 2, "EUR": 1, "GBP": 1}, base="AUD", quote="USD",
            margin_level=10.0, free_margin_pct=90.0,
        )
        assert not decision.allowed
        assert "max_positions_currency" in decision.reason

    @staticmethod
    def entry(gov: RiskGovernor, timestamp: datetime):
        return gov.assess_entry(
            timestamp=timestamp, symbol="EURUSD", open_symbols=[],
            currency_counts={}, base="EUR", quote="USD",
            margin_level=10.0, free_margin_pct=90.0,
        )
