"""Risk state and risk event models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class RiskState(str, Enum):
    ACTIVE = "ACTIVE"
    WARNING = "WARNING"
    BREACHED = "BREACHED"
    LOCKED = "LOCKED"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class RiskEvent:
    """Auditable risk event record."""

    event_type: str
    timestamp: datetime
    risk_state: RiskState
    message: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: Optional[str] = None
    start_equity: Optional[float] = None
    current_equity: Optional[float] = None
    daily_loss_pct: Optional[float] = None
    action_taken: Optional[str] = None
