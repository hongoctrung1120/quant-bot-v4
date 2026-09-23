from datetime import datetime, timedelta, timezone

from research.experiments import WalkForwardPlanner


def test_walk_forward_windows_are_chronological():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    windows = WalkForwardPlanner(30, 10).windows(start, end)
    assert windows
    for w in windows:
        assert w.train_start < w.train_end <= w.test_start < w.test_end
