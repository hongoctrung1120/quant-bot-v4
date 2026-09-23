"""Order reconciliation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconciliationResult:
    success: bool
    message: str
    mismatches: list[str]


class OrderReconciler:
    """Compare local state vs exchange state."""

    def reconcile(
        self,
        local_positions: dict[str, float],
        exchange_positions: dict[str, float],
        tolerance: float = 1e-8,
    ) -> ReconciliationResult:
        mismatches: list[str] = []
        all_symbols = set(local_positions) | set(exchange_positions)

        for symbol in all_symbols:
            local = local_positions.get(symbol, 0.0)
            remote = exchange_positions.get(symbol, 0.0)
            if abs(local - remote) > tolerance:
                mismatches.append(
                    f"{symbol}: local={local}, exchange={remote}"
                )

        if mismatches:
            logger.error(
                "Reconciliation failed: %s",
                mismatches,
                extra={"event": "RECONCILIATION_FAILED"},
            )
            return ReconciliationResult(
                success=False,
                message="Position mismatch detected",
                mismatches=mismatches,
            )

        return ReconciliationResult(
            success=True,
            message="Reconciliation passed",
            mismatches=[],
        )
