"""Time bar construction from trade-level data."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from core.bars.base import BarBuilderState
from core.interfaces import BarEngine
from core.models.bar import Bar, BarType
from core.models.trade import Trade
from utils.time import ensure_utc


def parse_interval(interval: str) -> timedelta:
    """Parse interval string like '15m', '1h', '1d' to timedelta."""
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return timedelta(days=int(interval[:-1]))
    raise ValueError(f"Unsupported interval: {interval}")


def bar_period_start(timestamp: datetime, interval: timedelta) -> datetime:
    """Align timestamp to interval boundary (UTC)."""
    ts = ensure_utc(timestamp)
    epoch = datetime(1970, 1, 1, tzinfo=ts.tzinfo)
    elapsed = (ts - epoch).total_seconds()
    period_seconds = interval.total_seconds()
    aligned = int(elapsed // period_seconds) * period_seconds
    return epoch + timedelta(seconds=aligned)


class TimeBarEngine(BarEngine):
    """Construct time bars from trade stream."""

    def __init__(self, interval: str) -> None:
        self._interval_str = interval
        self._interval = parse_interval(interval)
        self._states: dict[str, BarBuilderState] = {}
        self._period_starts: dict[str, datetime] = {}

    @property
    def interval(self) -> str:
        return self._interval_str

    def on_trade(self, trade: Trade) -> Optional[Bar]:
        period = bar_period_start(trade.timestamp, self._interval)
        state = self._states.setdefault(
            trade.symbol,
            BarBuilderState(
                symbol=trade.symbol,
                bar_type=BarType.TIME,
                bar_threshold=self._interval_str,
            ),
        )
        current_period = self._period_starts.get(trade.symbol)

        closed: Optional[Bar] = None
        if current_period is not None and period > current_period:
            if not state.is_empty():
                closed = state.to_bar()
            state.reset()
            self._period_starts[trade.symbol] = period
        elif current_period is None:
            self._period_starts[trade.symbol] = period

        state.add_trade(trade)
        return closed

    def get_current_bar(self, symbol: str) -> Optional[Bar]:
        state = self._states.get(symbol)
        if state is None or state.is_empty():
            return None
        return state.to_bar()

    def flush(self, symbol: str) -> Optional[Bar]:
        state = self._states.get(symbol)
        if state is None or state.is_empty():
            return None
        bar = state.to_bar()
        state.reset()
        self._period_starts.pop(symbol, None)
        return bar


class MultiIntervalTimeBarEngine:
    """Manage multiple time bar engines."""

    def __init__(self, intervals: list[str]) -> None:
        self._engines = {iv: TimeBarEngine(iv) for iv in intervals}

    def on_trade(self, trade: Trade) -> dict[str, Bar]:
        closed: dict[str, Bar] = {}
        for name, engine in self._engines.items():
            bar = engine.on_trade(trade)
            if bar is not None:
                closed[name] = bar
        return closed

    def flush_all(self, symbol: str) -> dict[str, Bar]:
        flushed: dict[str, Bar] = {}
        for name, engine in self._engines.items():
            bar = engine.flush(symbol)
            if bar is not None:
                flushed[name] = bar
        return flushed
