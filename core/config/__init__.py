"""Configuration engine for YAML-based system configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""


@dataclass
class SystemConfig:
    name: str = "Multi-Resolution Adaptive Quant Bot"
    version: str = "4.0.0"
    mode: str = "BACKTEST"
    log_level: str = "INFO"
    timezone: str = "UTC"
    random_seed: int = 42
    live_trading_enabled: bool = False


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.25
    daily_loss_limit_pct: float = 3.0
    max_leverage: float = 3.0
    circuit_breaker_enabled: bool = True
    close_positions_on_breach: bool = True
    warning_threshold_pct: float = 2.0


@dataclass
class CapitalConfig:
    reserve_pct: float = 20.0
    max_trading_capital_pct: float = 80.0
    max_asset_allocation_pct: float = 40.0
    max_strategy_allocation_pct: float = 50.0
    max_single_position_pct: float = 15.0
    max_gross_exposure_pct: float = 100.0
    max_net_exposure_pct: float = 80.0
    allocation_method: str = "inverse_volatility"
    asset_volatilities: dict[str, float] = field(default_factory=dict)


@dataclass
class BarsConfig:
    dollar_thresholds: dict[str, float] = field(default_factory=dict)
    overshoot_policy: str = "CARRY_FORWARD"
    time_intervals: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    primary_source: str = "trades"
    symbols: list[str] = field(default_factory=list)
    duplicate_detection: bool = True
    out_of_order_tolerance_ms: int = 100
    reconnect_max_retries: int = 10
    reconnect_backoff_seconds: int = 5
    invalid_price_action: str = "reject"
    invalid_quantity_action: str = "reject"
    gap_detection_enabled: bool = True
    max_price_change_pct: float = 50.0
    gap_threshold_seconds: float = 300.0


@dataclass
class Config:
    """Root configuration container."""

    system: SystemConfig = field(default_factory=SystemConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    bars: BarsConfig = field(default_factory=BarsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    raw: dict[str, Any] = field(default_factory=dict)


class ConfigEngine:
    """Loads and validates YAML configuration files."""

    REQUIRED_FILES = [
        "system.yaml",
        "data.yaml",
        "bars.yaml",
        "strategy.yaml",
        "capital.yaml",
        "risk.yaml",
        "execution.yaml",
    ]

    VALID_MODES = {"BACKTEST", "PAPER", "LIVE"}

    def __init__(self, config_dir: Path | str) -> None:
        self.config_dir = Path(config_dir)

    def load(self) -> Config:
        """Load all configuration files and return validated Config."""
        if not self.config_dir.exists():
            raise ConfigurationError(
                f"Config directory not found: {self.config_dir}"
            )

        merged: dict[str, Any] = {}
        for filename in self.REQUIRED_FILES:
            filepath = self.config_dir / filename
            if not filepath.exists():
                raise ConfigurationError(
                    f"Required config file missing: {filepath}"
                )
            with open(filepath, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            merged.update(data)

        config = self._build_config(merged)
        self.validate(config)
        return config

    def _build_config(self, merged: dict[str, Any]) -> Config:
        system_data = merged.get("system", {})
        risk_data = merged.get("risk", {})
        capital_data = merged.get("capital", {})
        bars_data = merged.get("bars", {})
        data_section = merged.get("data", {})
        ingestion_cfg = data_section.get("ingestion", {})
        quality_cfg = data_section.get("quality", {})

        dollar_cfg = bars_data.get("dollar", {})
        time_cfg = bars_data.get("time", {})

        return Config(
            system=SystemConfig(
                name=system_data.get("name", SystemConfig.name),
                version=system_data.get("version", SystemConfig.version),
                mode=system_data.get("mode", "BACKTEST").upper(),
                log_level=system_data.get("log_level", "INFO"),
                timezone=system_data.get("timezone", "UTC"),
                random_seed=system_data.get("random_seed", 42),
                live_trading_enabled=system_data.get(
                    "live_trading_enabled", False
                ),
            ),
            risk=RiskConfig(
                risk_per_trade_pct=risk_data.get(
                    "risk_per_trade_pct", 0.25
                ),
                daily_loss_limit_pct=risk_data.get(
                    "daily_loss_limit_pct", 3.0
                ),
                max_leverage=risk_data.get("max_leverage", 3.0),
                circuit_breaker_enabled=risk_data.get(
                    "circuit_breaker_enabled", True
                ),
                close_positions_on_breach=risk_data.get(
                    "close_positions_on_breach", True
                ),
                warning_threshold_pct=risk_data.get(
                    "warning_threshold_pct", 2.0
                ),
            ),
            capital=CapitalConfig(
                reserve_pct=capital_data.get("reserve_pct", 20.0),
                max_trading_capital_pct=capital_data.get(
                    "max_trading_capital_pct", 80.0
                ),
                max_asset_allocation_pct=capital_data.get(
                    "max_asset_allocation_pct", 40.0
                ),
                max_strategy_allocation_pct=capital_data.get(
                    "max_strategy_allocation_pct", 50.0
                ),
                max_single_position_pct=capital_data.get(
                    "max_single_position_pct", 15.0
                ),
                max_gross_exposure_pct=capital_data.get(
                    "max_gross_exposure_pct", 100.0
                ),
                max_net_exposure_pct=capital_data.get(
                    "max_net_exposure_pct", 80.0
                ),
                allocation_method=capital_data.get("allocation_method", "inverse_volatility"),
                asset_volatilities=capital_data.get("asset_volatilities", {}),
            ),
            bars=BarsConfig(
                dollar_thresholds=dollar_cfg.get("thresholds", {}),
                overshoot_policy=dollar_cfg.get(
                    "overshoot_policy", "CARRY_FORWARD"
                ),
                time_intervals=time_cfg.get("intervals", []),
            ),
            data=DataConfig(
                primary_source=data_section.get("primary_source", "trades"),
                symbols=data_section.get("symbols", []),
                duplicate_detection=ingestion_cfg.get(
                    "duplicate_detection", True
                ),
                out_of_order_tolerance_ms=ingestion_cfg.get(
                    "out_of_order_tolerance_ms", 100
                ),
                reconnect_max_retries=ingestion_cfg.get(
                    "reconnect_max_retries", 10
                ),
                reconnect_backoff_seconds=ingestion_cfg.get(
                    "reconnect_backoff_seconds", 5
                ),
                invalid_price_action=quality_cfg.get(
                    "invalid_price_action", "reject"
                ),
                invalid_quantity_action=quality_cfg.get(
                    "invalid_quantity_action", "reject"
                ),
                gap_detection_enabled=quality_cfg.get(
                    "gap_detection_enabled", True
                ),
                max_price_change_pct=quality_cfg.get(
                    "max_price_change_pct", 50.0
                ),
                gap_threshold_seconds=quality_cfg.get(
                    "gap_threshold_seconds", 300.0
                ),
            ),
            raw=merged,
        )

    def validate(self, config: Config) -> None:
        """Validate configuration values. Fail closed on invalid config."""
        if config.system.mode not in self.VALID_MODES:
            raise ConfigurationError(
                f"Invalid mode: {config.system.mode}. "
                f"Must be one of {self.VALID_MODES}"
            )

        if config.system.mode == "LIVE" and not config.system.live_trading_enabled:
            raise ConfigurationError(
                "LIVE mode requires live_trading_enabled: true in system.yaml"
            )

        if config.risk.daily_loss_limit_pct <= 0:
            raise ConfigurationError(
                "daily_loss_limit_pct must be positive"
            )

        if not (0 < config.capital.reserve_pct < 100):
            raise ConfigurationError(
                "reserve_pct must be between 0 and 100"
            )

        if not config.bars.dollar_thresholds:
            raise ConfigurationError(
                "At least one dollar bar threshold must be configured"
            )

        for name, threshold in config.bars.dollar_thresholds.items():
            if threshold <= 0:
                raise ConfigurationError(
                    f"Dollar threshold '{name}' must be positive: {threshold}"
                )

    @staticmethod
    def get_default_config_dir() -> Path:
        """Return default config directory relative to project root."""
        return Path(__file__).resolve().parent.parent.parent / "config"
