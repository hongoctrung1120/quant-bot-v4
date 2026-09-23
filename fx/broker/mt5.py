"""MetaTrader 5 bridge: history download and real contract specifications.

The `MetaTrader5` package is only an IPC client. It talks to the MT5 desktop
terminal, which must be installed, logged in, and running on the same machine.

Two things come from here that matter more than convenience:

* Real contract specs. The spreads and swaps hardcoded in `fx.instruments` are
  conservative guesses; a broker's actual figures decide whether a strategy
  clears its cost hurdle, and they differ substantially between brokers.
* Real bar ranges. Synthetic highs and lows are drawn from a model. Broker bars
  are built from ticks, so stops that sit inside a bar genuinely traded.

Cent accounts: MT5 already denominates a cent account in cents (balance shows
10,000 USC for a $100 deposit) and reports the standard 100,000 contract size.
That is self-consistent, so run with `cent_account=False` and pass the balance
in cents. Setting the flag as well would scale the exposure twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from core.models.bar import Bar
from fx.data.ohlc import TIMEFRAME_MINUTES, make_bar, write_csv
from fx.instruments import FxInstrument

logger = logging.getLogger(__name__)

MT5_TIMEFRAME_NAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}

SWAP_MODE_POINTS = 1


class Mt5Unavailable(RuntimeError):
    """Raised when the MT5 terminal cannot be reached."""


# ----------------------------------------------------------------------
# Pure conversions — no terminal required, so they stay unit-testable.
# ----------------------------------------------------------------------
def pip_size_from_digits(digits: int, point: float) -> float:
    """A pip is ten points on 3- and 5-digit quotes, one point otherwise."""
    return point * 10 if digits in (3, 5) else point


def instrument_from_symbol_info(
    info: Any, symbol: str, spread_pips: Optional[float] = None
) -> FxInstrument:
    """Build an FxInstrument from an MT5 SymbolInfo structure."""
    pip = pip_size_from_digits(int(info.digits), float(info.point))
    base = getattr(info, "currency_base", symbol[:3])
    quote = getattr(info, "currency_profit", symbol[3:])

    points_per_pip = pip / float(info.point) if info.point else 1.0
    swap_mode = int(getattr(info, "swap_mode", SWAP_MODE_POINTS))
    if swap_mode == SWAP_MODE_POINTS and points_per_pip:
        swap_long = float(info.swap_long) / points_per_pip
        swap_short = float(info.swap_short) / points_per_pip
    else:
        # Other swap modes are quoted in interest or account currency; leaving
        # them at zero is safer than silently misreading the units.
        logger.warning(
            "%s uses swap_mode=%s, not points — swap set to 0, fill it in manually",
            symbol,
            swap_mode,
        )
        swap_long = swap_short = 0.0

    if spread_pips is None:
        spread_pips = float(getattr(info, "spread", 0)) / points_per_pip

    return FxInstrument(
        symbol=symbol,
        base=base,
        quote=quote,
        pip_size=pip,
        contract_size=float(info.trade_contract_size),
        min_lot=float(info.volume_min),
        lot_step=float(info.volume_step),
        max_lot=float(info.volume_max),
        typical_spread_pips=spread_pips,
        swap_long_pips_per_day=swap_long,
        swap_short_pips_per_day=swap_short,
    )


def bars_from_rates(rates: Iterable[Any], symbol: str, timeframe: str) -> list[Bar]:
    """Convert an MT5 rates array into engine bars (timestamps are UTC)."""
    minutes = TIMEFRAME_MINUTES[timeframe]
    bars: list[Bar] = []
    for row in rates:
        timestamp = datetime.fromtimestamp(int(row["time"]), tz=timezone.utc)
        volume = float(row["tick_volume"]) if "tick_volume" in row.dtype.names else 0.0
        bars.append(
            make_bar(
                symbol=symbol,
                timestamp=timestamp,
                open_=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=volume,
                timeframe=timeframe,
                duration_minutes=minutes,
            )
        )
    return bars


def median_spread_pips(rates: Iterable[Any], point: float, digits: int) -> Optional[float]:
    """Median historical spread from the rates array, in pips."""
    values = [
        float(row["spread"]) for row in rates if "spread" in row.dtype.names
    ]
    if not values:
        return None
    values.sort()
    median_points = values[len(values) // 2]
    points_per_pip = pip_size_from_digits(digits, point) / point if point else 1.0
    return median_points / points_per_pip


# ----------------------------------------------------------------------
@dataclass
class Mt5Credentials:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    terminal_path: Optional[str] = None


class Mt5Client:
    """Context manager around the MT5 terminal connection."""

    def __init__(self, credentials: Optional[Mt5Credentials] = None) -> None:
        self.credentials = credentials or Mt5Credentials()
        self._mt5 = None

    def __enter__(self) -> "Mt5Client":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()

    def connect(self) -> None:
        try:
            import MetaTrader5 as mt5  # noqa: N813
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise Mt5Unavailable(
                "The MetaTrader5 package is not installed: pip install MetaTrader5"
            ) from exc

        creds = self.credentials
        kwargs: dict[str, Any] = {}
        if creds.terminal_path:
            kwargs["path"] = creds.terminal_path
        if creds.login:
            kwargs.update(
                login=int(creds.login), password=creds.password, server=creds.server
            )

        if not mt5.initialize(**kwargs):
            code, message = mt5.last_error()
            raise Mt5Unavailable(
                f"Could not reach the MT5 terminal ({code}: {message}). "
                "Install the MT5 desktop application from your broker, log in, "
                "and leave it running."
            )
        self._mt5 = mt5

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    @property
    def mt5(self):
        if self._mt5 is None:
            raise Mt5Unavailable("Not connected — call connect() first")
        return self._mt5

    # ------------------------------------------------------------------
    def account_summary(self) -> dict[str, Any]:
        info = self.mt5.account_info()
        if info is None:
            raise Mt5Unavailable("No account information; is the terminal logged in?")
        return {
            "login": info.login,
            "server": info.server,
            "currency": info.currency,
            "balance": info.balance,
            "equity": info.equity,
            "leverage": info.leverage,
            "margin_free": info.margin_free,
            "is_cent_account": str(info.currency).upper().endswith("C"),
        }

    def _timeframe(self, timeframe: str):
        name = MT5_TIMEFRAME_NAMES.get(timeframe.upper())
        if name is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return getattr(self.mt5, name)

    def _ensure_symbol(self, symbol: str) -> None:
        if not self.mt5.symbol_select(symbol, True):
            raise Mt5Unavailable(
                f"Symbol {symbol} is not available on this account. "
                "Check the exact name in Market Watch — brokers add suffixes "
                "such as EURUSD.m or EURUSDmicro."
            )

    def symbol_spec(self, symbol: str, use_historical_spread: bool = True) -> FxInstrument:
        """Contract specification as the broker actually defines it."""
        self._ensure_symbol(symbol)
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise Mt5Unavailable(f"No symbol_info for {symbol}")

        spread_pips = None
        if use_historical_spread:
            rates = self.mt5.copy_rates_from_pos(symbol, self._timeframe("H1"), 0, 2000)
            if rates is not None and len(rates):
                spread_pips = median_spread_pips(rates, float(info.point), int(info.digits))

        return instrument_from_symbol_info(info, symbol, spread_pips)

    def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "H1",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        chunk_days: int = 180,
    ) -> list[Bar]:
        """Download history in chunks so terminal bar limits do not truncate it."""
        self._ensure_symbol(symbol)
        tf = self._timeframe(timeframe)
        end = end or datetime.now(timezone.utc)
        start = start or end - timedelta(days=365 * 3)

        collected: dict[datetime, Bar] = {}
        cursor = start
        while cursor < end:
            window_end = min(cursor + timedelta(days=chunk_days), end)
            rates = self.mt5.copy_rates_range(symbol, tf, cursor, window_end)
            if rates is not None and len(rates):
                for bar in bars_from_rates(rates, symbol, timeframe):
                    collected[bar.timestamp] = bar
            cursor = window_end

        bars = [collected[ts] for ts in sorted(collected)]
        logger.info("Fetched %d %s bars for %s", len(bars), timeframe, symbol)
        return bars

    def download_to_csv(
        self,
        symbols: Iterable[str],
        directory: Path | str,
        timeframe: str = "H1",
        years: float = 3.0,
    ) -> dict[str, int]:
        """Save history as SYMBOL.csv, ready for `--data-dir`."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(365 * years))

        counts: dict[str, int] = {}
        for symbol in symbols:
            bars = self.fetch_bars(symbol, timeframe, start, end)
            write_csv(out_dir / f"{symbol}.csv", bars)
            counts[symbol] = len(bars)
        return counts
