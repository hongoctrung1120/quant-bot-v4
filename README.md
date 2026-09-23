# Multi-Resolution Adaptive Quant Bot v4.0

Production-oriented, research-driven quantitative trading platform for backtesting, paper trading, and live trading.

## Design Philosophy

This is **not** a simple indicator bot. It is a modular quantitative research and execution framework with strict separation of concerns:

```
Raw Market Data → Data Quality → Bar Construction → Features → Regime
→ Strategy → Signal → Capital Allocation → Portfolio Risk → Position Sizing
→ Order Intent → Execution → Exchange
```

**Risk always has final authority.** No strategy, signal, or ML model can bypass the 3% daily loss circuit breaker.

## Quick Start

```powershell
cd "D:\Bot quant\quant_bot_v4_complete\qbwork"
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==22.3.1
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt

# Infrastructure check
.\.venv\Scripts\python.exe main.py

# Ingest CSV trades
.\.venv\Scripts\python.exe main.py --ingest tests/fixtures/sample_trades.csv

# Run sample backtest
.\.venv\Scripts\python.exe main.py --backtest

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

See `SETUP_PYTHON310.md` if PowerShell blocks virtual-environment activation.

## Development Phases

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Infrastructure | ✅ Complete |
| 2 | Trade Ingestion | ✅ Complete |
| 3 | Dollar Bar Engine | ✅ Complete |
| 4 | Time Bar Engine | ✅ Complete |
| 5 | Feature Engine | ✅ Complete |
| 6 | Regime Engine | ✅ Complete |
| 7 | Baseline Strategies | ✅ Complete |
| 8 | Backtesting Engine | ✅ Complete |
| 9 | Capital Allocation | ✅ Complete |
| 10 | Portfolio Risk | ✅ Complete |
| 11 | Circuit Breaker (3%) | ✅ Complete |
| 12 | Position Sizing | ✅ Complete |
| 13 | Execution Engine | ✅ Complete |
| 14 | Paper Trading | Pending |
| 15 | Monitoring | Pending |
| 16 | ML Research | Pending |
| 17 | Adaptive Allocation | Pending |
| 18 | Live Trading | Pending |

## Architecture

```
quant_bot_v4/
├── config/           # YAML configuration (no magic numbers)
├── core/
│   ├── data/         # Ingestion, loaders, quality, normalization
│   ├── bars/         # Dollar + time bar engines
│   ├── features/     # Modular feature calculators
│   ├── regime/       # Rule-based regime detection
│   ├── strategy/     # Trend, momentum, mean reversion, breakout
│   ├── signal/       # Signal aggregation
│   ├── capital/      # Allocation engine
│   ├── risk/         # Circuit breaker, portfolio risk, sizing
│   ├── execution/    # Simulated execution + reconciliation
│   └── portfolio/    # Accounting + positions
├── backtest/         # Event-driven backtest engine + metrics
└── tests/            # Unit + integration tests
```

## Key Design Decisions

### Dollar Bar Overshoot

Configured in `config/bars.yaml`:

- **CARRY_FORWARD**: Crossing trade fully included in closing bar; excess dollar volume carries to next bar
- **SPLIT_TRADE**: Crossing trade split proportionally across two bars

### Circuit Breaker

Hard 3% daily loss limit (`risk.yaml`). When breached:
1. Cancel pending orders
2. Block new orders
3. Lock until next session
4. Record auditable risk event

### No Look-Ahead

Features and signals use **closed bars only**. Unclosed bar OHLC is never exposed downstream.

## Configuration

| File | Purpose |
|------|---------|
| `system.yaml` | Mode, logging, session |
| `data.yaml` | Data sources, quality rules |
| `bars.yaml` | Dollar/time bar thresholds |
| `strategy.yaml` | Strategy parameters |
| `capital.yaml` | Allocation limits |
| `risk.yaml` | Risk limits, circuit breaker |
| `execution.yaml` | Fees, slippage, simulation |

## License

Private — internal research use.

## Backtest research capabilities (V4.0)

The backtest stack now includes:

- Trade-level Dollar Bars with `SPLIT_TRADE` overshoot handling.
- Time Bars for benchmark comparisons.
- ATR-based stop loss and 3R take-profit on the baseline trend strategy.
- Intrabar protective-exit simulation with configurable `STOP_FIRST` / `TARGET_FIRST` priority.
- Transaction costs: maker/taker fees, spread and slippage.
- Deterministic perpetual funding accrual at configurable intervals.
- Portfolio-level 3% daily-loss circuit breaker with optional forced liquidation.
- Symbol-aware gross/net exposure projections for long/short reduction and reversal.
- Capital allocation separated from position sizing.
- Reproducible experiment persistence and walk-forward window planning.

### Safety hierarchy

`Portfolio Risk > Capital Allocation > Strategy > Signal > Execution Preference`

The 3% daily loss circuit breaker cannot be overridden by a strategy or model.

### Research rule

Dollar Bars are not assumed to be superior to Time Bars. Run controlled experiments with identical data, fees, slippage, funding and risk configuration before making that conclusion.

## Forex system (`fx/`)

A second, independent system for leveraged FX trading on a small (cent-account)
deposit, built on top of the shared feature/indicator code above but with its
own margin accounting, risk governor, and execution engine — spot accounting
cannot represent a leveraged position correctly.

```powershell
python -m fx.cli capital                          # minimum viable account size per instrument
python -m fx.cli mt5-check --symbols EURUSD GBPUSD # verify MT5 terminal + real broker specs
python -m fx.cli fetch --symbols EURUSD GBPUSD --out data   # download history via MT5
python -m fx.cli backtest --data-dir data          # backtest on real data
python -m fx.cli validate --data-dir data          # + random-entry benchmark, bootstrap, ruin sim
python -m fx.cli walkforward --data-dir data        # out-of-sample parameter validation
```

Key modules: `fx/account.py` (margin accounting), `fx/risk/` (loss-limit
governor, pip-based sizing, currency exposure netting), `fx/strategy/`
(session breakout, trend pullback, currency strength), `fx/validation/`
(walk-forward, bootstrap, ruin simulation), `fx/broker/mt5.py` (MetaTrader 5
data + contract specs).

**Status**: infrastructure complete and tested
(`tests/unit/test_fx_*.py`, `tests/integration/test_fx_null_hypothesis.py`),
but not yet validated on real market data. Every result produced on synthetic
data proves the machinery is honest (see the null-hypothesis test), not that
any strategy has a real edge — only out-of-sample real data can show that.
