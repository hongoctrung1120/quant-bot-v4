"""Data quality event models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class DataQualityStatus(str, Enum):
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


@dataclass(frozen=True)
class DataQualityEvent:
    """Auditable data quality event."""

    status: DataQualityStatus
    event_type: str
    timestamp: datetime
    message: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    details: Optional[dict] = None
