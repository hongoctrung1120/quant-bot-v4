"""Reproducible backtest experiments and walk-forward evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Callable, Sequence

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult

@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    name: str
    bar_type: str
    dollar_threshold_name: str = "medium"
    time_interval: str = "15m"
    seed: int = 42
    parameters: dict = field(default_factory=dict)

class ExperimentRunner:
    """Run and persist comparable experiments without changing risk assumptions."""
    def __init__(self, output_dir: str | Path = "research/results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, engine_factory: Callable[[BacktestConfig], BacktestEngine], loader_factory: Callable[[], object], spec: ExperimentSpec) -> BacktestResult:
        cfg = BacktestConfig(
            bar_type=spec.bar_type,
            dollar_threshold_name=spec.dollar_threshold_name,
            time_interval=spec.time_interval,
            seed=spec.seed,
        )
        result = engine_factory(cfg).run(loader_factory())
        payload = {
            "spec": asdict(spec),
            "result": {
                "metrics": asdict(result.metrics),
                "bar_count": result.bar_count,
                "trade_count": result.trade_count,
                "fill_count": result.fill_count,
                "risk_state": result.risk_state,
                "experiment": result.experiment,
            },
        }
        (self.output_dir / f"{spec.experiment_id}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return result

@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

class WalkForwardPlanner:
    """Create chronological train/test windows with no overlap into the future."""
    def __init__(self, train_days: int, test_days: int, step_days: int | None = None) -> None:
        if train_days <= 0 or test_days <= 0:
            raise ValueError("train_days and test_days must be positive")
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days or test_days

    def windows(self, start: datetime, end: datetime) -> list[WalkForwardWindow]:
        out=[]
        cursor=start + timedelta(days=self.train_days)
        while cursor + timedelta(days=self.test_days) <= end:
            out.append(WalkForwardWindow(
                train_start=cursor - timedelta(days=self.train_days),
                train_end=cursor,
                test_start=cursor,
                test_end=cursor + timedelta(days=self.test_days),
            ))
            cursor += timedelta(days=self.step_days)
        return out
