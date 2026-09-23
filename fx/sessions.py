"""Trading session classification.

Session boundaries are fixed UTC approximations. Real session times shift by an
hour when London and New York observe DST on different schedules; treat these
as regime labels, not exchange hours.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from enum import Enum


class Session(str, Enum):
    SYDNEY = "SYDNEY"
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    NEW_YORK = "NEW_YORK"
    CLOSED = "CLOSED"


# Liquidity multipliers relative to the London/NY overlap, used to model how
# spreads widen when the book thins out.
SPREAD_MULTIPLIER: dict[Session, float] = {
    Session.LONDON_NY_OVERLAP: 1.0,
    Session.LONDON: 1.15,
    Session.NEW_YORK: 1.25,
    Session.TOKYO: 1.6,
    Session.SYDNEY: 2.2,
    Session.CLOSED: 4.0,
}

ASIAN_RANGE_START_HOUR = 0
ASIAN_RANGE_END_HOUR = 7
LONDON_OPEN_HOUR = 7
NY_CLOSE_HOUR = 21


def classify(timestamp: datetime) -> Session:
    """Map a UTC timestamp to its dominant session."""
    weekday = timestamp.weekday()  # Monday == 0
    hour = timestamp.hour

    if is_market_closed(timestamp):
        return Session.CLOSED
    if 12 <= hour < 16:
        return Session.LONDON_NY_OVERLAP
    if 7 <= hour < 12:
        return Session.LONDON
    if 16 <= hour < 21:
        return Session.NEW_YORK
    if 0 <= hour < 7:
        return Session.TOKYO
    return Session.SYDNEY


def is_market_closed(timestamp: datetime) -> bool:
    """Forex closes Friday 21:00 UTC and reopens Sunday 21:00 UTC."""
    weekday = timestamp.weekday()
    hour = timestamp.hour
    if weekday == 5:  # Saturday
        return True
    if weekday == 4 and hour >= NY_CLOSE_HOUR:  # Friday after NY close
        return True
    if weekday == 6 and hour < NY_CLOSE_HOUR:  # Sunday before reopen
        return True
    return False


def spread_multiplier(timestamp: datetime) -> float:
    return SPREAD_MULTIPLIER[classify(timestamp)]


def is_rollover(previous: datetime, current: datetime) -> bool:
    """True when a 21:00 UTC swap boundary falls between two bars."""
    if previous >= current:
        return False
    boundary = previous.replace(hour=21, minute=0, second=0, microsecond=0)
    if boundary <= previous:
        boundary += timedelta(days=1)
    return boundary <= current


def swap_multiplier(timestamp: datetime) -> float:
    """Wednesday rollover carries three days of swap to cover the weekend."""
    return 3.0 if timestamp.weekday() == 2 else 1.0


def hours_to_weekend_close(timestamp: datetime) -> float:
    """Hours until Friday 21:00 UTC. Large value when the weekend is far off."""
    weekday = timestamp.weekday()
    if weekday > 4:
        return 0.0
    days_ahead = 4 - weekday
    close = (timestamp + timedelta(days=days_ahead)).replace(
        hour=NY_CLOSE_HOUR, minute=0, second=0, microsecond=0
    )
    return max((close - timestamp).total_seconds() / 3600.0, 0.0)


def in_asian_range_window(timestamp: datetime) -> bool:
    return ASIAN_RANGE_START_HOUR <= timestamp.hour < ASIAN_RANGE_END_HOUR
