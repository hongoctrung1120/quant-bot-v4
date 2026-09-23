"""Feature computation base classes and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from core.models.bar import Bar


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for a computed feature."""

    name: str
    formula: str
    input_data: str
    lookback: int
    warmup_period: int


class Feature(ABC):
    """Single feature calculator."""

    @property
    @abstractmethod
    def spec(self) -> FeatureSpec:
        """Feature specification."""

    @abstractmethod
    def update(self, bar: Bar) -> Optional[float]:
        """Update with a closed bar. Returns value when ready."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""

    @property
    def is_ready(self) -> bool:
        return True
