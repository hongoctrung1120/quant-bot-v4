"""Event type definitions for the event-driven architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid


class EventType(str, Enum):
    TRADE = "TradeEvent"
    BAR_CLOSED = "BarClosedEvent"
    FEATURE_UPDATED = "FeatureUpdatedEvent"
    REGIME_CHANGED = "RegimeChangedEvent"
    SIGNAL_GENERATED = "SignalGeneratedEvent"
    ALLOCATION_UPDATED = "AllocationUpdatedEvent"
    RISK_BREACHED = "RiskBreachedEvent"
    ORDER_CREATED = "OrderCreatedEvent"
    ORDER_FILLED = "OrderFilledEvent"
    POSITION_UPDATED = "PositionUpdatedEvent"
    DATA_QUALITY = "DataQualityEvent"


@dataclass(frozen=True)
class Event:
    """Base event with traceable metadata."""

    event_type: EventType
    timestamp: datetime
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: Optional[str] = None
    source: Optional[str] = None
    payload: Any = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            raise ValueError("Event timestamp is required")


@dataclass(frozen=True)
class TradeEvent(Event):
    """Emitted when a raw trade is ingested."""

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        payload: Any,
        source: str = "ingestion",
        event_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "event_type", EventType.TRADE)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "event_id", event_id or str(uuid.uuid4())
        )


@dataclass(frozen=True)
class BarClosedEvent(Event):
    """Emitted when a bar (time or dollar) is closed."""

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        payload: Any,
        source: str = "bar_engine",
        event_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "event_type", EventType.BAR_CLOSED)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "event_id", event_id or str(uuid.uuid4())
        )


@dataclass(frozen=True)
class SignalGeneratedEvent(Event):
    """Emitted when a strategy generates a signal."""

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        payload: Any,
        source: str = "signal_engine",
        event_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "event_type", EventType.SIGNAL_GENERATED)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "event_id", event_id or str(uuid.uuid4())
        )


@dataclass(frozen=True)
class RiskBreachedEvent(Event):
    """Emitted when a risk limit is breached."""

    def __init__(
        self,
        timestamp: datetime,
        payload: Any,
        symbol: str | None = None,
        source: str = "risk_engine",
        event_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "event_type", EventType.RISK_BREACHED)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "event_id", event_id or str(uuid.uuid4())
        )


@dataclass(frozen=True)
class DataQualityEvent(Event):
    """Emitted when a data quality issue is detected."""

    def __init__(
        self,
        timestamp: datetime,
        payload: Any,
        symbol: str | None = None,
        source: str = "quality_engine",
        event_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "event_type", EventType.DATA_QUALITY)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "event_id", event_id or str(uuid.uuid4())
        )
