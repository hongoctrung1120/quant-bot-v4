"""Portfolio accounting and equity tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from core.portfolio.positions import PositionBook


@dataclass
class PortfolioAccount:
    """Portfolio-level accounting."""

    cash: float
    initial_equity: float
    position_book: PositionBook = field(default_factory=PositionBook)
    total_fees: float = 0.0
    total_realized_pnl: float = 0.0
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)

    def compute_equity(self, prices: dict[str, float]) -> float:
        equity = self.cash
        for symbol, pos in self.position_book.positions.items():
            price = prices.get(symbol, pos.avg_entry_price)
            equity += pos.quantity * price
        return equity

    def snapshot_equity(self, timestamp: datetime, prices: dict[str, float]) -> float:
        equity = self.compute_equity(prices)
        self.equity_curve.append((timestamp, equity))
        return equity

    def apply_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        strategy_id: str,
        timestamp: datetime,
    ) -> float:
        cost = quantity * price
        is_buy = side.upper() == "BUY"
        if is_buy:
            self.cash -= cost + fee
        else:
            self.cash += cost - fee
        self.total_fees += fee
        realized = self.position_book.update_from_fill(symbol, side, quantity, price, strategy_id, timestamp)
        self.total_realized_pnl += realized
        return realized
