"""Time utilities for timestamp normalization and alignment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Union


def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_timestamp(value: Union[str, int, float, datetime]) -> datetime:
    """Parse various timestamp formats to UTC datetime."""
    if isinstance(value, datetime):
        return ensure_utc(value)

    if isinstance(value, (int, float)):
        if value > 1e12:
            value = value / 1000.0
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(value, fmt)
                return ensure_utc(dt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse timestamp: {value}")

    raise TypeError(f"Unsupported timestamp type: {type(value)}")


def is_monotonic(
    timestamps: list[datetime], allow_equal: bool = True
) -> bool:
    """Check if timestamps are in non-decreasing order."""
    if len(timestamps) < 2:
        return True
    for i in range(1, len(timestamps)):
        if allow_equal:
            if timestamps[i] < timestamps[i - 1]:
                return False
        else:
            if timestamps[i] <= timestamps[i - 1]:
                return False
    return True
