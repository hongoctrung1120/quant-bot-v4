"""Structured logging and audit trail."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SENSITIVE_KEYS = frozenset({
    "api_key", "api_secret", "password", "token", "secret",
    "private_key", "access_key", "passphrase",
})


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for auditability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "event"):
            log_entry["event"] = record.event
        if hasattr(record, "symbol"):
            log_entry["symbol"] = record.symbol
        if hasattr(record, "strategy"):
            log_entry["strategy"] = record.strategy
        if hasattr(record, "risk_state"):
            log_entry["risk_state"] = record.risk_state
        if hasattr(record, "order_id"):
            log_entry["order_id"] = record.order_id
        if hasattr(record, "position"):
            log_entry["position"] = record.position
        if hasattr(record, "pnl"):
            log_entry["pnl"] = record.pnl

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        return json.dumps(log_entry, default=str)


def _redact_sensitive(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive keys from log data."""
    return {
        k: ("***REDACTED***" if k.lower() in SENSITIVE_KEYS else v)
        for k, v in data.items()
    }


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = True,
) -> logging.Logger:
    """Configure root logger with structured output."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if structured:
        formatter: logging.Formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)


def log_risk_event(
    logger: logging.Logger,
    event_type: str,
    risk_state: str,
    message: str,
    **kwargs: Any,
) -> None:
    """Log a high-priority risk event."""
    safe_kwargs = _redact_sensitive(kwargs)
    logger.critical(
        message,
        extra={
            "event": event_type,
            "risk_state": risk_state,
            **safe_kwargs,
        },
    )
