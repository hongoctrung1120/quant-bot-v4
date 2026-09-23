from datetime import datetime, timedelta, timezone

from backtest.funding import FundingConfig, FundingModel


def test_positive_funding_long_pays_short_receives():
    model = FundingModel(FundingConfig(rate_pct=0.1, interval_hours=8))
    assert model.cost(2.0, 100.0) == 0.2
    assert model.cost(-2.0, 100.0) == -0.2


def test_funding_event_schedule():
    model = FundingModel(FundingConfig(rate_pct=0.1, interval_hours=8))
    t0 = datetime(2024, 1, 1, 10, tzinfo=timezone.utc)
    t1 = datetime(2024, 1, 2, 3, tzinfo=timezone.utc)
    events = model.due_events(t0, t1)
    assert events
    assert events[0] == datetime(2024, 1, 1, 18, tzinfo=timezone.utc)
