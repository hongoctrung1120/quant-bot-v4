"""Core abstract interfaces for engine components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

from core.models.bar import Bar
from core.models.signal import Signal
from core.models.order import OrderIntent
from core.models.regime import RegimeOutput
from core.models.allocation import AllocationResult
from core.models.trade import Trade

if TYPE_CHECKING:
    # Deferred: core.events.bus imports this module, so importing the events
    # package eagerly here makes the two mutually dependent at import time.
    from core.events.types import Event


class EventHandler(Protocol):
    """Protocol for event handlers in the event bus."""

    def __call__(self, event: Event) -> None: ...


class EventBus(ABC):
    """Abstract event bus for decoupled component communication."""

    @abstractmethod
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""

    @abstractmethod
    def subscribe(
        self, event_type: str, handler: EventHandler
    ) -> None:
        """Subscribe a handler to an event type."""

    @abstractmethod
    def unsubscribe(
        self, event_type: str, handler: EventHandler
    ) -> None:
        """Unsubscribe a handler from an event type."""


class BarEngine(ABC):
    """Abstract bar construction engine."""

    @abstractmethod
    def on_trade(self, trade: Trade) -> Optional[Bar]:
        """Process a trade. Returns closed bar if threshold met."""

    @abstractmethod
    def get_current_bar(self, symbol: str) -> Optional[Bar]:
        """Return the currently forming (unclosed) bar."""

    @abstractmethod
    def flush(self, symbol: str) -> Optional[Bar]:
        """Force-close the current bar for a symbol."""


class FeatureEngine(ABC):
    """Abstract feature computation engine."""

    @abstractmethod
    def on_bar(self, bar: Bar) -> dict[str, float]:
        """Compute features for a closed bar."""

    @abstractmethod
    def get_feature(self, name: str) -> Optional[float]:
        """Get the latest value of a named feature."""


class RegimeEngine(ABC):
    """Abstract market regime detection engine."""

    @abstractmethod
    def on_features(
        self, symbol: str, features: dict[str, float], timestamp: Any
    ) -> RegimeOutput:
        """Detect regime from current features."""


class Strategy(ABC):
    """Abstract strategy. Must NOT execute orders directly."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique strategy identifier."""

    @abstractmethod
    def on_bar(
        self,
        bar: Bar,
        features: dict[str, float],
        regime: RegimeOutput,
    ) -> Optional[Signal]:
        """Process a bar and optionally emit a signal."""


class SignalEngine(ABC):
    """Abstract signal aggregation engine."""

    @abstractmethod
    def add_signal(self, signal: Signal) -> None:
        """Register a strategy signal."""

    @abstractmethod
    def get_actionable_signals(self) -> list[Signal]:
        """Return combined actionable signals. Does NOT size positions."""


class CapitalAllocator(ABC):
    """Abstract capital allocation engine."""

    @abstractmethod
    def allocate(
        self,
        total_equity: float,
        regime: Optional[RegimeOutput] = None,
        signal_confidences: Optional[dict[str, float]] = None,
    ) -> AllocationResult:
        """Compute capital allocation across assets and strategies."""


class PortfolioRiskEngine(ABC):
    """Abstract portfolio risk engine with override authority."""

    @abstractmethod
    def check_signal(self, signal: Signal) -> bool:
        """Return True if signal is allowed under current risk state."""

    @abstractmethod
    def check_order_intent(self, intent: OrderIntent) -> bool:
        """Return True if order intent passes risk checks."""

    @abstractmethod
    def get_risk_state(self) -> str:
        """Return current risk state."""


class PositionSizer(ABC):
    """Abstract position sizing engine."""

    @abstractmethod
    def size(
        self,
        signal: Signal,
        allocation: AllocationResult,
        equity: float,
        current_price: float,
    ) -> Optional[OrderIntent]:
        """Convert signal + allocation into an order intent."""


class ExecutionEngine(ABC):
    """Abstract execution engine."""

    @abstractmethod
    async def submit(self, intent: OrderIntent) -> str:
        """Submit an order intent. Returns order ID."""

    @abstractmethod
    async def cancel(self, order_id: str) -> bool:
        """Cancel a pending order."""


class ExchangeAdapter(ABC):
    """Abstract exchange interface. No strategy-specific logic."""

    @abstractmethod
    async def get_balance(self) -> dict[str, float]:
        """Return account balances."""

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return open positions."""

    @abstractmethod
    async def get_orderbook(self, symbol: str) -> dict:
        """Return current orderbook."""

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> dict:
        """Submit order to exchange."""

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order on exchange."""

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> dict:
        """Get order status."""

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> list[dict]:
        """Get all open orders for a symbol."""
