"""Command line entry point for the FX system.

    python -m fx.cli capital      # what account size does the risk model need
    python -m fx.cli backtest     # single run with full performance report
    python -m fx.cli validate     # null test, bootstrap, ruin, random benchmark
    python -m fx.cli walkforward  # out-of-sample parameter validation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from fx.costs import FxCostConfig
from fx.data import ohlc
from fx.data.synthetic import generate_market
from fx.engine import FxBacktestConfig, FxBacktestEngine
from fx.instruments import RateBook, get_instrument
from fx.report import analyze, format_report
from fx.risk.governor import RiskLimits
from fx.risk.sizing import minimum_viable_equity
from fx.strategy.base import FxStrategy
from fx.strategy.currency_strength import CurrencyStrengthStrategy
from fx.strategy.random_entry import RandomEntryStrategy
from fx.strategy.session_breakout import SessionBreakoutStrategy
from fx.strategy.trend_pullback import TrendPullbackStrategy
from fx.validation.monte_carlo import (
    bootstrap_expectancy,
    format_bootstrap,
    format_ruin,
    simulate_ruin,
)
from fx.validation.walkforward import (
    WalkForwardConfig,
    format_walk_forward,
    run_walk_forward,
)

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURGBP"]
REFERENCE_PRICES = {
    "EURUSD": 1.0850,
    "GBPUSD": 1.2700,
    "USDJPY": 148.0,
    "AUDUSD": 0.6550,
    "USDCHF": 0.8800,
    "USDCAD": 1.3600,
    "NZDUSD": 0.6050,
    "EURGBP": 0.8547,
    "EURJPY": 160.6,
    "GBPJPY": 188.0,
}


def default_strategies() -> list[FxStrategy]:
    return [
        SessionBreakoutStrategy(),
        TrendPullbackStrategy(),
        CurrencyStrengthStrategy(),
    ]


def load_market(args) -> dict:
    if args.data_dir:
        directory = Path(args.data_dir)
        series = {}
        for symbol in args.symbols:
            path = directory / f"{symbol}.csv"
            if not path.exists():
                raise SystemExit(f"Missing data file: {path}")
            series[symbol] = ohlc.load_csv(path, symbol, args.timeframe)
        return series
    return generate_market(
        args.symbols, days=args.days, mode=args.mode, seed=args.seed,
        timeframe=args.timeframe,
    )


def build_limits(args) -> RiskLimits:
    return RiskLimits(
        risk_per_trade_pct=args.risk,
        daily_loss_limit_pct=args.daily_limit,
        max_concurrent_positions=args.max_positions,
    )


def build_engine(args, strategies: Sequence[FxStrategy]) -> FxBacktestEngine:
    return FxBacktestEngine(
        symbols=args.symbols,
        strategies=strategies,
        config=FxBacktestConfig(
            initial_balance=args.balance,
            leverage=args.leverage,
            cent_account=not args.standard_account,
        ),
        limits=build_limits(args),
        cost_config=FxCostConfig(spread_stress_multiplier=args.spread_multiplier),
    )


# ----------------------------------------------------------------------
def cmd_capital(args) -> int:
    """How large must the account be for the risk model to be honourable."""
    rates = RateBook("USD")
    for symbol, price in REFERENCE_PRICES.items():
        rates.update(symbol, price)

    print(f"Minimum account size for {args.risk}% risk per trade")
    print(f"{'pair':<9}{'account':>10}{'stop':>7}{'pip value':>12}{'min-lot risk':>14}")
    for account_type, cent in (("standard", False), ("cent", True)):
        for symbol in args.symbols:
            inst = get_instrument(symbol, cent_account=cent)
            for stop_pips in (args.stop_pips,):
                needed = minimum_viable_equity(
                    inst, stop_pips, rates, args.risk, REFERENCE_PRICES.get(symbol)
                )
                from fx.instruments import pip_value_per_lot

                pip_value = pip_value_per_lot(inst, rates, REFERENCE_PRICES.get(symbol))
                risk_at_min = stop_pips * pip_value * inst.min_lot
                print(
                    f"{symbol:<9}{account_type:>10}{stop_pips:>7.0f}"
                    f"{pip_value:>12.4f}{risk_at_min:>10.4f} -> "
                    f"need {needed:,.0f} USD"
                )
        print()
    return 0


def cmd_backtest(args) -> int:
    market = load_market(args)
    engine = build_engine(args, default_strategies())
    result = engine.run(market)
    report = analyze(result)
    print(format_report(report, f"FX BACKTEST ({args.mode} data)"))

    if args.data_dir is None:
        print()
        print(
            "NOTE: this ran on SYNTHETIC data. It validates that the machinery "
            "works.\n      It says nothing about whether the strategies have a "
            "real edge.\n      Only out-of-sample real market data can answer that."
        )
    return 0


def cmd_validate(args) -> int:
    market = load_market(args)

    engine = build_engine(args, default_strategies())
    report = analyze(engine.run(market))
    print(format_report(report, "STRATEGY"))

    bench_engine = build_engine(
        args, [RandomEntryStrategy(entry_probability=args.random_entry_prob)]
    )
    bench = analyze(bench_engine.run(market))

    print()
    print("=== RANDOM-ENTRY BENCHMARK ===")
    print(
        f"Random entries, identical exits/sizing/risk: "
        f"{bench.num_trades} trades, expectancy {bench.expectancy_r:+.4f}R, "
        f"return {bench.total_return_pct:+.2f}%"
    )
    edge = report.expectancy_r - bench.expectancy_r
    print(
        f"Strategy minus random: {edge:+.4f}R  -> "
        + (
            "entries add value"
            if edge > 0
            else "entries add NOTHING over coin flips"
        )
    )

    print()
    boot = bootstrap_expectancy(engine.account.closed_trades, seed=args.seed)
    print(format_bootstrap(boot))

    print()
    ruin = simulate_ruin(
        engine.account.closed_trades,
        risk_per_trade_pct=args.risk,
        horizon_trades=args.ruin_horizon,
        ruin_drawdown_pct=args.ruin_drawdown,
        seed=args.seed,
    )
    print(format_ruin(ruin, args.ruin_drawdown))
    return 0


def cmd_walkforward(args) -> int:
    market = load_market(args)

    def build_strategies(params) -> list[FxStrategy]:
        return [
            SessionBreakoutStrategy(
                min_range_atr=params.get("min_range_atr", 1.5),
                target_r=params.get("breakout_target_r", 1.5),
            ),
            TrendPullbackStrategy(
                stop_atr_multiple=params.get("stop_atr", 1.8),
                target_r=params.get("pullback_target_r", 2.5),
            ),
            CurrencyStrengthStrategy(
                min_score_spread=params.get("min_score_spread", 0.8),
            ),
        ]

    result = run_walk_forward(
        series=market,
        build_strategies=build_strategies,
        engine_builder=lambda strategies: build_engine(args, strategies),
        param_grid={
            "min_range_atr": [1.2, 2.0],
            "pullback_target_r": [2.0, 3.0],
            "min_score_spread": [0.6, 1.0],
        },
        config=WalkForwardConfig(
            train_days=args.train_days, test_days=args.test_days
        ),
    )
    print(format_walk_forward(result))

    print()
    boot = bootstrap_expectancy(result.oos_trades, seed=args.seed)
    print(format_bootstrap(boot))
    return 0


def cmd_mt5_check(args) -> int:
    """Diagnose the terminal connection and print the broker's real specs."""
    from fx.broker.mt5 import Mt5Client, Mt5Credentials, Mt5Unavailable

    credentials = Mt5Credentials(
        login=args.login, password=args.password, server=args.server,
        terminal_path=args.terminal_path,
    )
    try:
        with Mt5Client(credentials) as client:
            account = client.account_summary()
            print("=== ACCOUNT ===")
            for key, value in account.items():
                print(f"  {key:<16} {value}")
            if account["is_cent_account"]:
                print(
                    f"\n  Cent account detected. Backtest it with "
                    f"--standard-account --balance {account['balance']:.0f}\n"
                    "  (MT5 already reports this account in cent units, so the "
                    "built-in cent scaling would apply it twice)."
                )

            print("\n=== BROKER CONTRACT SPECS (replaces built-in estimates) ===")
            for symbol in args.symbols:
                try:
                    spec = client.symbol_spec(symbol)
                except Mt5Unavailable as exc:
                    print(f"  {symbol:<10} unavailable: {exc}")
                    continue
                print(
                    f"  {spec.symbol:<10} pip={spec.pip_size:<8g} "
                    f"contract={spec.contract_size:<10g} "
                    f"min_lot={spec.min_lot:<6g} step={spec.lot_step:<6g} "
                    f"spread={spec.typical_spread_pips:.2f}p  "
                    f"swap L/S={spec.swap_long_pips_per_day:+.2f}/"
                    f"{spec.swap_short_pips_per_day:+.2f} pips/day"
                )
    except Mt5Unavailable as exc:
        print(f"MT5 not reachable: {exc}")
        return 1
    return 0


def cmd_fetch(args) -> int:
    from fx.broker.mt5 import Mt5Client, Mt5Credentials, Mt5Unavailable

    credentials = Mt5Credentials(
        login=args.login, password=args.password, server=args.server,
        terminal_path=args.terminal_path,
    )
    try:
        with Mt5Client(credentials) as client:
            counts = client.download_to_csv(
                args.symbols, args.out, args.timeframe, args.years
            )
    except Mt5Unavailable as exc:
        print(f"MT5 not reachable: {exc}")
        return 1

    for symbol, count in counts.items():
        print(f"  {symbol:<10} {count:>7} bars -> {Path(args.out) / (symbol + '.csv')}")
    print(f"\nNow run: python -m fx.cli validate --data-dir {args.out}")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FX quant system")
    parser.add_argument("--log-level", default="ERROR")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    common.add_argument("--balance", type=float, default=100.0)
    common.add_argument("--leverage", type=float, default=100.0)
    common.add_argument("--risk", type=float, default=0.5)
    common.add_argument("--daily-limit", type=float, default=3.0)
    common.add_argument("--max-positions", type=int, default=5)
    common.add_argument("--standard-account", action="store_true",
                        help="Use standard lots instead of a cent account")
    common.add_argument("--spread-multiplier", type=float, default=1.0,
                        help="Stress-test costs, e.g. 2.0 for double spread")
    common.add_argument("--data-dir", default=None,
                        help="Directory of SYMBOL.csv OHLC files (real data)")
    common.add_argument("--timeframe", default="H1")
    common.add_argument("--days", type=int, default=730)
    common.add_argument("--mode", default="regime_switching",
                        choices=["regime_switching", "random_walk"])
    common.add_argument("--seed", type=int, default=7)

    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capital", parents=[common])
    cap.add_argument("--stop-pips", type=float, default=20.0)
    cap.set_defaults(func=cmd_capital)

    bt = sub.add_parser("backtest", parents=[common])
    bt.set_defaults(func=cmd_backtest)

    val = sub.add_parser("validate", parents=[common])
    val.add_argument("--random-entry-prob", type=float, default=0.01)
    val.add_argument("--ruin-horizon", type=int, default=200)
    val.add_argument("--ruin-drawdown", type=float, default=50.0)
    val.set_defaults(func=cmd_validate)

    wf = sub.add_parser("walkforward", parents=[common])
    wf.add_argument("--train-days", type=int, default=180)
    wf.add_argument("--test-days", type=int, default=60)
    wf.set_defaults(func=cmd_walkforward)

    mt5_common = argparse.ArgumentParser(add_help=False)
    mt5_common.add_argument("--login", type=int, default=None)
    mt5_common.add_argument("--password", default=None)
    mt5_common.add_argument("--server", default=None)
    mt5_common.add_argument("--terminal-path", default=None,
                            help="Path to terminal64.exe if auto-detection fails")

    check = sub.add_parser("mt5-check", parents=[common, mt5_common])
    check.set_defaults(func=cmd_mt5_check)

    fetch = sub.add_parser("fetch", parents=[common, mt5_common])
    fetch.add_argument("--out", default="data", help="Output directory for CSV files")
    fetch.add_argument("--years", type=float, default=3.0)
    fetch.set_defaults(func=cmd_fetch)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
