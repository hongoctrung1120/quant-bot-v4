"""Unit tests for configuration engine."""

from pathlib import Path

import pytest
import yaml

from core.config import ConfigEngine, ConfigurationError


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


class TestConfigEngine:
    def test_load_default_config(self):
        engine = ConfigEngine(CONFIG_DIR)
        config = engine.load()

        assert config.system.mode == "BACKTEST"
        assert config.system.live_trading_enabled is False
        assert config.risk.daily_loss_limit_pct == 3.0
        assert config.risk.risk_per_trade_pct == 0.25
        assert config.capital.reserve_pct == 20.0

    def test_dollar_thresholds_loaded(self):
        engine = ConfigEngine(CONFIG_DIR)
        config = engine.load()

        thresholds = config.bars.dollar_thresholds
        assert "small" in thresholds
        assert thresholds["small"] == 250000
        assert thresholds["medium"] == 1000000
        assert thresholds["large"] == 5000000
        assert thresholds["very_large"] == 20000000

    def test_overshoot_policy_loaded(self):
        engine = ConfigEngine(CONFIG_DIR)
        config = engine.load()
        assert config.bars.overshoot_policy == "SPLIT_TRADE"

    def test_time_intervals_loaded(self):
        engine = ConfigEngine(CONFIG_DIR)
        config = engine.load()
        assert "15m" in config.bars.time_intervals
        assert "1h" in config.bars.time_intervals

    def test_missing_config_dir_raises(self):
        engine = ConfigEngine("/nonexistent/path")
        with pytest.raises(ConfigurationError, match="not found"):
            engine.load()

    def test_live_mode_without_enable_raises(self, tmp_path):
        config_data = {
            "system": {"mode": "LIVE", "live_trading_enabled": False},
            "bars": {"dollar": {"thresholds": {"small": 250000}}},
            "risk": {},
            "capital": {},
        }
        for name in ConfigEngine.REQUIRED_FILES:
            filepath = tmp_path / name
            key = name.replace(".yaml", "")
            if key == "system":
                data = {"system": config_data["system"]}
            elif key == "bars":
                data = {"bars": config_data["bars"]}
            elif key == "risk":
                data = {"risk": config_data["risk"]}
            elif key == "capital":
                data = {"capital": config_data["capital"]}
            else:
                data = {}
            with open(filepath, "w") as f:
                yaml.dump(data, f)

        engine = ConfigEngine(tmp_path)
        with pytest.raises(ConfigurationError, match="LIVE mode requires"):
            engine.load()

    def test_invalid_mode_raises(self, tmp_path):
        for name in ConfigEngine.REQUIRED_FILES:
            filepath = tmp_path / name
            if name == "system.yaml":
                data = {"system": {"mode": "INVALID"}}
            elif name == "bars.yaml":
                data = {"bars": {"dollar": {"thresholds": {"small": 250000}}}}
            else:
                data = {}
            with open(filepath, "w") as f:
                yaml.dump(data, f)

        engine = ConfigEngine(tmp_path)
        with pytest.raises(ConfigurationError, match="Invalid mode"):
            engine.load()

    def test_get_default_config_dir(self):
        default = ConfigEngine.get_default_config_dir()
        assert default.name == "config"
        assert default.exists()
