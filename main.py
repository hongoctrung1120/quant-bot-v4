"""Application entry point for Multi-Resolution Adaptive Quant Bot v4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import ConfigEngine, ConfigurationError
from utils.logger import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Resolution Adaptive Quant Bot v4.0"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to configuration directory",
    )
    parser.add_argument(
        "--mode",
        choices=["BACKTEST", "PAPER", "LIVE"],
        default=None,
        help="Override trading mode",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override log level",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run sample backtest on fixture data",
    )
    parser.add_argument(
        "--ingest",
        type=Path,
        default=None,
        help="Ingest trades from CSV file",
    )
    return parser.parse_args()


def run_backtest(config_dir: Path, logger) -> int:
    from backtest.engine import BacktestConfig, BacktestEngine
    from core.data.loaders import CSVTradeLoader

    engine_cfg = ConfigEngine(config_dir)
    config = engine_cfg.load()

    fixtures = Path(__file__).parent / "tests" / "fixtures" / "sample_trades.csv"
    if not fixtures.exists():
        logger.error("Fixture file not found: %s", fixtures)
        return 1

    bt = BacktestEngine(
        config,
        BacktestConfig(
            bar_type="DOLLAR",
            dollar_threshold_name="small",
            initial_equity=100000.0,
        ),
    )
    result = bt.run(CSVTradeLoader(fixtures))
    m = result.metrics
    logger.info(
        "Backtest complete: bars=%d fills=%d return=%.2f%% sharpe=%.2f mdd=%.2f%%",
        result.bar_count,
        result.fill_count,
        m.total_return * 100,
        m.sharpe_ratio,
        m.max_drawdown * 100,
        extra={"event": "BACKTEST_COMPLETE"},
    )
    return 0


def run_ingest(csv_path: Path, config_dir: Path, logger) -> int:
    from core.config import DataConfig
    from core.data.ingestion import TradeIngestionEngine
    from core.data.loaders import CSVTradeLoader

    engine_cfg = ConfigEngine(config_dir)
    config = engine_cfg.load()
    ingestion = TradeIngestionEngine(config.data)
    result = ingestion.ingest(CSVTradeLoader(csv_path))
    logger.info(
        "Ingestion: %d accepted, %d rejected from %s",
        result.stats.accepted,
        result.stats.rejected,
        csv_path,
        extra={"event": "INGESTION_COMPLETE"},
    )
    return 0


def main() -> int:
    args = parse_args()

    config_dir = args.config_dir or ConfigEngine.get_default_config_dir()

    try:
        engine = ConfigEngine(config_dir)
        config = engine.load()
    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    log_level = args.log_level or config.system.log_level
    setup_logging(level=log_level)
    logger = get_logger("quant_bot")

    mode = args.mode or config.system.mode

    logger.info(
        "Starting %s v%s in %s mode",
        config.system.name,
        config.system.version,
        mode,
        extra={"event": "SYSTEM_START"},
    )

    if mode == "LIVE":
        if not config.system.live_trading_enabled:
            logger.error(
                "LIVE mode blocked: live_trading_enabled is false",
                extra={"event": "LIVE_BLOCKED"},
            )
            return 1
        logger.warning(
            "LIVE trading mode active",
            extra={"event": "LIVE_MODE"},
        )

    logger.info(
        "Configuration loaded successfully",
        extra={
            "event": "CONFIG_LOADED",
            "dollar_thresholds": list(config.bars.dollar_thresholds.keys()),
            "daily_loss_limit": config.risk.daily_loss_limit_pct,
        },
    )

    if args.ingest:
        return run_ingest(args.ingest, config_dir, logger)

    if args.backtest:
        return run_backtest(config_dir, logger)

    logger.info(
        "Phases 1-13 infrastructure ready. "
        "Use --backtest or --ingest <csv> to run pipelines.",
        extra={"event": "SYSTEM_READY"},
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
