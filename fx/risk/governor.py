"""Risk governor: the layer with veto authority over every entry.

No strategy can override this. It enforces loss limits across four horizons
(day / week / month / peak-to-trough), scales risk down as drawdown deepens,
refuses to add correlated currency exposure, and halts the system outright
when the hard drawdown ceiling is hit.

Every scaler here can only reduce risk below the configured base, never raise
it above. Adding size after losses is how small accounts die.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence

from fx import sessions
from fx.account import ClosedTrade

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    risk_per_trade_pct: float = 0.5

    daily_loss_limit_pct: float = 3.0
    weekly_loss_limit_pct: float = 6.0
    monthly_loss_limit_pct: float = 10.0

    warning_drawdown_pct: float = 10.0
    max_drawdown_pct: float = 20.0

    max_concurrent_positions: int = 5
    max_positions_per_currency: int = 2
    # Primary correlation control: aggregate stop-risk allowed on one currency.
    # At 0.5% risk per trade this permits three concurrent same-side positions.
    max_currency_risk_pct: float = 1.5
    # Secondary leverage guard. Tight stops legitimately need several times
    # equity in notional, so this must stay loose or it silently undersizes.
    max_currency_exposure_pct: float = 800.0
    max_gross_exposure_pct: float = 2000.0
    max_trades_per_day: int = 12

    min_margin_level: float = 3.0
    min_free_margin_pct: float = 20.0

    max_consecutive_losses: int = 5
    cooldown_hours_after_streak: float = 24.0

    flat_before_weekend_hours: float = 2.0
    block_new_trades_on_friday_after_hour: int = 18

    news_blackout_minutes: int = 30

    target_annual_volatility_pct: float = 12.0
    vol_scaler_min: float = 0.4
    vol_scaler_max: float = 1.0

    kelly_fraction: float = 0.25
    kelly_min_trades: int = 30
    kelly_scaler_min: float = 0.3
    kelly_scaler_max: float = 1.0


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    risk_scaler: float = 1.0

    @staticmethod
    def ok(scaler: float = 1.0) -> "RiskDecision":
        return RiskDecision(True, "", scaler)

    @staticmethod
    def deny(reason: str) -> "RiskDecision":
        return RiskDecision(False, reason, 0.0)


@dataclass
class RiskSnapshot:
    timestamp: Optional[datetime] = None
    equity: float = 0.0
    high_water_mark: float = 0.0
    drawdown_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    monthly_pnl_pct: float = 0.0
    risk_scaler: float = 1.0
    halted: bool = False
    locked_until: Optional[datetime] = None
    consecutive_losses: int = 0


class RiskGovernor:
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        news_windows: Optional[Sequence[tuple[datetime, datetime]]] = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.news_windows = list(news_windows or [])

        self._initialized = False
        self._high_water_mark = 0.0
        self._day_anchor = 0.0
        self._week_anchor = 0.0
        self._month_anchor = 0.0
        self._current_day: Optional[object] = None
        self._current_week: Optional[object] = None
        self._current_month: Optional[object] = None

        self._equity = 0.0
        self._timestamp: Optional[datetime] = None
        self._halted = False
        self._halt_reason = ""
        self._locked_until: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None
        self._consecutive_losses = 0
        self._trades_today = 0

        self._daily_equity: list[tuple[object, float]] = []
        self._closed_trades: list[ClosedTrade] = []
        self.events: list[str] = []

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def on_bar(self, timestamp: datetime, equity: float) -> None:
        self._timestamp = timestamp
        self._equity = equity

        if not self._initialized:
            self._high_water_mark = equity
            self._day_anchor = self._week_anchor = self._month_anchor = equity
            self._current_day = timestamp.date()
            self._current_week = timestamp.isocalendar()[:2]
            self._current_month = (timestamp.year, timestamp.month)
            self._initialized = True

        self._roll_periods(timestamp, equity)
        self._high_water_mark = max(self._high_water_mark, equity)
        self._record_daily_equity(timestamp, equity)
        self._enforce_hard_limits(timestamp, equity)

    def _roll_periods(self, timestamp: datetime, equity: float) -> None:
        day = timestamp.date()
        week = timestamp.isocalendar()[:2]
        month = (timestamp.year, timestamp.month)

        if day != self._current_day:
            self._current_day = day
            self._day_anchor = equity
            self._trades_today = 0
        if week != self._current_week:
            self._current_week = week
            self._week_anchor = equity
        if month != self._current_month:
            self._current_month = month
            self._month_anchor = equity

        if self._locked_until is not None and timestamp >= self._locked_until:
            self._locked_until = None

    def _record_daily_equity(self, timestamp: datetime, equity: float) -> None:
        day = timestamp.date()
        if self._daily_equity and self._daily_equity[-1][0] == day:
            self._daily_equity[-1] = (day, equity)
        else:
            self._daily_equity.append((day, equity))

    def _enforce_hard_limits(self, timestamp: datetime, equity: float) -> None:
        if self._halted:
            return

        drawdown = self.drawdown_pct
        if drawdown >= self.limits.max_drawdown_pct:
            self._halted = True
            self._halt_reason = f"max_drawdown {drawdown:.2f}%"
            self._log(timestamp, f"HALT: drawdown {drawdown:.2f}% >= limit")
            return

        if self._pct_change(self._day_anchor, equity) <= -self.limits.daily_loss_limit_pct:
            self._lock_until(timestamp + timedelta(days=1), "daily_loss_limit", timestamp)
        elif self._pct_change(self._week_anchor, equity) <= -self.limits.weekly_loss_limit_pct:
            self._lock_until(self._next_week(timestamp), "weekly_loss_limit", timestamp)
        elif self._pct_change(self._month_anchor, equity) <= -self.limits.monthly_loss_limit_pct:
            self._lock_until(self._next_month(timestamp), "monthly_loss_limit", timestamp)

    def _lock_until(self, until: datetime, reason: str, now: datetime) -> None:
        if self._locked_until is not None and self._locked_until >= until:
            return
        self._locked_until = until
        self._log(now, f"LOCK until {until.isoformat()}: {reason}")

    @staticmethod
    def _next_week(timestamp: datetime) -> datetime:
        days_ahead = 7 - timestamp.weekday()
        return (timestamp + timedelta(days=days_ahead)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    @staticmethod
    def _next_month(timestamp: datetime) -> datetime:
        year = timestamp.year + (1 if timestamp.month == 12 else 0)
        month = 1 if timestamp.month == 12 else timestamp.month + 1
        return timestamp.replace(
            year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0
        )

    @staticmethod
    def _pct_change(anchor: float, equity: float) -> float:
        if anchor <= 0:
            return 0.0
        return (equity - anchor) / anchor * 100.0

    def _log(self, timestamp: datetime, message: str) -> None:
        entry = f"{timestamp.isoformat()} {message}"
        self.events.append(entry)
        logger.warning("RISK %s", message, extra={"event": "RISK_GOVERNOR"})

    def on_trade_closed(self, trade: ClosedTrade) -> None:
        self._closed_trades.append(trade)
        if trade.net_pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.limits.max_consecutive_losses:
                until = trade.exit_time + timedelta(
                    hours=self.limits.cooldown_hours_after_streak
                )
                self._cooldown_until = until
                self._log(
                    trade.exit_time,
                    f"COOLDOWN until {until.isoformat()}: "
                    f"{self._consecutive_losses} consecutive losses",
                )
                self._consecutive_losses = 0
        else:
            self._consecutive_losses = 0

    def on_position_opened(self) -> None:
        self._trades_today += 1

    # ------------------------------------------------------------------
    # Derived risk state
    # ------------------------------------------------------------------
    @property
    def drawdown_pct(self) -> float:
        if self._high_water_mark <= 0:
            return 0.0
        return max(
            (self._high_water_mark - self._equity) / self._high_water_mark * 100.0, 0.0
        )

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def locked(self) -> bool:
        return self._locked_until is not None

    def drawdown_scaler(self) -> float:
        """Cut risk as drawdown deepens; never increase it."""
        dd = self.drawdown_pct
        warn = self.limits.warning_drawdown_pct
        hard = self.limits.max_drawdown_pct
        if dd <= warn:
            return 1.0
        if dd >= hard or hard <= warn:
            return 0.25
        span = (dd - warn) / (hard - warn)
        return max(1.0 - 0.75 * span, 0.25)

    def volatility_scaler(self) -> float:
        """Throttle risk when the equity curve is running hotter than target.

        Applied at portfolio level: per-trade risk is already volatility-adjusted
        through the ATR stop, so scaling by instrument volatility again would
        double-count.
        """
        returns = self._daily_returns()
        if len(returns) < 20:
            return 1.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        realized = math.sqrt(var) * math.sqrt(252.0) * 100.0
        if realized <= 0:
            return self.limits.vol_scaler_max
        raw = self.limits.target_annual_volatility_pct / realized
        return min(max(raw, self.limits.vol_scaler_min), self.limits.vol_scaler_max)

    def kelly_scaler(self) -> float:
        """Shrink risk when realized edge is weaker than assumed.

        Uses a fraction of Kelly on measured win rate and payoff, capped so it
        can only reduce the configured base risk.
        """
        trades = self._closed_trades[-100:]
        if len(trades) < self.limits.kelly_min_trades:
            return 1.0

        wins = [t.r_multiple for t in trades if t.net_pnl > 0]
        losses = [abs(t.r_multiple) for t in trades if t.net_pnl < 0]
        if not wins or not losses:
            return 1.0

        win_rate = len(wins) / len(trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        if avg_loss <= 0:
            return 1.0

        payoff = avg_win / avg_loss
        kelly = win_rate - (1.0 - win_rate) / payoff
        if kelly <= 0:
            return self.limits.kelly_scaler_min

        target_fraction = self.limits.kelly_fraction * kelly * 100.0
        raw = target_fraction / self.limits.risk_per_trade_pct
        return min(max(raw, self.limits.kelly_scaler_min), self.limits.kelly_scaler_max)

    def _daily_returns(self) -> list[float]:
        values = [equity for _, equity in self._daily_equity]
        return [
            (values[i] - values[i - 1]) / values[i - 1]
            for i in range(1, len(values))
            if values[i - 1] > 0
        ]

    def risk_scaler(self) -> float:
        return self.drawdown_scaler() * self.volatility_scaler() * self.kelly_scaler()

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    def in_news_blackout(self, timestamp: datetime) -> bool:
        pad = timedelta(minutes=self.limits.news_blackout_minutes)
        return any(start - pad <= timestamp <= end + pad for start, end in self.news_windows)

    def should_flatten_for_weekend(self, timestamp: datetime) -> bool:
        remaining = sessions.hours_to_weekend_close(timestamp)
        return 0.0 <= remaining <= self.limits.flat_before_weekend_hours

    def assess_entry(
        self,
        timestamp: datetime,
        symbol: str,
        open_symbols: Sequence[str],
        currency_counts: dict[str, int],
        base: str,
        quote: str,
        margin_level: float,
        free_margin_pct: float,
    ) -> RiskDecision:
        """Gate a candidate entry. Only reduces or blocks; never enlarges."""
        if self._halted:
            return RiskDecision.deny(f"halted:{self._halt_reason}")
        if self._locked_until is not None:
            return RiskDecision.deny("loss_limit_lock")
        if self._cooldown_until is not None and timestamp < self._cooldown_until:
            return RiskDecision.deny("loss_streak_cooldown")
        if sessions.is_market_closed(timestamp):
            return RiskDecision.deny("market_closed")
        if self.in_news_blackout(timestamp):
            return RiskDecision.deny("news_blackout")
        if self.should_flatten_for_weekend(timestamp):
            return RiskDecision.deny("weekend_flat_window")
        if (
            timestamp.weekday() == 4
            and timestamp.hour >= self.limits.block_new_trades_on_friday_after_hour
        ):
            return RiskDecision.deny("friday_late")
        if symbol in open_symbols:
            return RiskDecision.deny("position_already_open")
        if len(open_symbols) >= self.limits.max_concurrent_positions:
            return RiskDecision.deny("max_concurrent_positions")
        if self._trades_today >= self.limits.max_trades_per_day:
            return RiskDecision.deny("max_trades_per_day")
        for currency in (base, quote):
            if currency_counts.get(currency, 0) >= self.limits.max_positions_per_currency:
                return RiskDecision.deny(f"max_positions_currency:{currency}")
        if margin_level < self.limits.min_margin_level:
            return RiskDecision.deny("margin_level_floor")
        if free_margin_pct < self.limits.min_free_margin_pct:
            return RiskDecision.deny("free_margin_floor")

        return RiskDecision.ok(self.risk_scaler())

    def snapshot(self) -> RiskSnapshot:
        return RiskSnapshot(
            timestamp=self._timestamp,
            equity=self._equity,
            high_water_mark=self._high_water_mark,
            drawdown_pct=self.drawdown_pct,
            daily_pnl_pct=self._pct_change(self._day_anchor, self._equity),
            weekly_pnl_pct=self._pct_change(self._week_anchor, self._equity),
            monthly_pnl_pct=self._pct_change(self._month_anchor, self._equity),
            risk_scaler=self.risk_scaler(),
            halted=self._halted,
            locked_until=self._locked_until,
            consecutive_losses=self._consecutive_losses,
        )
