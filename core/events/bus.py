"""In-process event bus implementation."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

from core.events.types import Event, EventType
from core.interfaces import EventBus, EventHandler

logger = logging.getLogger(__name__)


class SimpleEventBus(EventBus):
    """Synchronous in-process event bus for backtesting and research."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def publish(self, event: Event) -> None:
        event_type = (
            event.event_type.value
            if isinstance(event.event_type, EventType)
            else str(event.event_type)
        )
        logger.debug(
            "Publishing event",
            extra={
                "event_type": event_type,
                "event_id": event.event_id,
                "symbol": event.symbol,
            },
        )
        for handler in self._handlers.get(event_type, []):
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Handler failed for event %s", event_type
                )
                raise

    def subscribe(
        self, event_type: str, handler: EventHandler
    ) -> None:
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(
        self, event_type: str, handler: EventHandler
    ) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def clear(self) -> None:
        """Remove all handlers. Useful for test isolation."""
        self._handlers.clear()

    @property
    def handler_count(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._handlers.items()}
