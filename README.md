# Trading Research Engine

A Python framework for testing trading strategies with statistical rigor. The core goal is to make the biases that invalidate backtests — look-ahead bias, in-sample overfitting, inconsistent fees — difficult to introduce by accident.

---

## Common Backtesting Failure Modes

| Common failure | How it happens | How this framework prevents it |
|---|---|---|
| **Look-ahead bias (calibration)** | Threshold calibrated on the full dataset, including the period being evaluated | `fit()` receives only training data. The framework controls the train/test split — the strategy cannot influence it. Look-ahead **inside** `generate_signals()` (e.g., `data.shift(-k)`) is the strategy author's responsibility — see [Signal generation contract](#framework-guarantees). |
| **In-sample overfitting** | Reported result uses the same data used to tune parameters | Walk-forward is the only entry point. There is no way to run an in-sample backtest as the main result. |
| **Inconsistent fees** | Fee applied to wins but not losses, or ignored entirely | `FeeModel.apply()` is called on every trade — wins and losses. Fee is structural, not optional. |
| **Unrealistic fills** | Entry at the same candle's close where the signal was generated | Entry at next candle's open. You cannot execute on a price you haven't seen yet. |

---

## Research context

This framework was born from three research sprints where a funding-rate-based strategy showed 57% win rate and +11.5% return in-sample — then collapsed to 32% win rate and -0.56% edge when walk-forward validation was applied.

Every design decision in this framework exists to prevent a specific failure mode discovered during those sprints. The full sprint reports — with data tables, root-cause analysis, and the failure-to-framework mapping — are in [`docs/RESEARCH.md`](docs/RESEARCH.md).

An actual walk-forward run of this framework against the same strategy and dataset is in [`reports/funding_rate_4h_2y.md`](reports/funding_rate_4h_2y.md) — verdict **REJECTED**. The framework caught the invalidation that two sprints of in-sample analysis missed.

---

## What this is NOT

- **Not a trading bot.** This framework does not connect to exchanges, place orders, or manage positions. It is a research tool.
- **Not a signal provider.** The bundled `FundingRateStrategy` is an invalidated example — included as a reference implementation of the `Strategy` interface, not as a profitable signal. The framework's purpose is to test whether *your* strategy has real edge.
- **Not an HFT system.** It prioritizes statistical correctness over execution speed.

---

## Architecture

The engine enforces a strict research workflow where each component has a single responsibility and data flows in one direction:

```
Strategy.fit(train)
        ↓
Strategy.generate_signals(test)
        ↓
Backtest Engine (execute trades, compute MAE)
        ↓
Walk-Forward (repeat across time windows, all OOS)
        ↓
Metrics (performance, risk, statistical significance)
        ↓
Report (automated markdown with conclusion)
```

Key design decisions:

- **Walk-forward is the entry point, not backtest.** There is no way to run an in-sample backtest as the main result.
- **Strategy never sees test data during `fit()`.** The train/test split is the framework's responsibility. `generate_signals()` receives the test window — strategies must not look at future candles inside it (see [Signal generation contract](#framework-guarantees)).
- **Fee is structural, not optional.** Applied on every trade via `FeeModel` — wins and losses.
- **Entry at next-open, not same-close.** Signals detected at candle close execute at the next candle's open. This models the realistic execution delay.

---

## Project structure

```
backtester/
├── data/
│   ├── loader.py          # Load local parquet files
│   ├── fetcher.py         # Fetch data from exchanges with retry
│   └── validator.py       # Detect gaps, NaN, and duplicates
│
├── core/
│   ├── bt_types.py        # Signal, Trade, BacktestResult, Direction
│   ├── backtest_engine.py # Trade execution loop (internal use)
│   ├── walk_forward.py    # Walk-forward validation — main entry point
│   └── fee_model.py       # Fee + slippage, always applied
│
├── strategy/
│   ├── base.py            # Abstract Strategy interface
│   └── examples/
│       └── funding_rate.py # H3 strategy as reference implementation
│
├── metrics/
│   ├── performance.py     # Win rate, Sharpe, Sortino, profit factor
│   ├── statistical.py     # One-tailed t-test, p-value, 95% CI
│   ├── risk.py            # Drawdown, MAE, consecutive loss streak
│   └── baselines.py       # Buy-and-hold and random entry baselines
│
├── report/
│   └── generator.py       # Generates markdown report automatically
│
├── config.py              # Global parameters with sensible defaults
└── run.py                 # CLI entry point

docs/
└── RESEARCH.md            # Research sprints, root causes, failure → framework
```

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Implement a strategy

Create a class that inherits from `Strategy` and implement the four required methods:

```python
from backtester.strategy.base import Strategy
from backtester.core.bt_types import Signal, Direction

class MyStrategy(Strategy):

    def name(self) -> str:
        return "my_strategy_v1"

    def warmup_periods(self) -> int:
        # candles discarded before the first signal
        return 50

    def holding_periods(self) -> int:
        # candles to hold after entry
        return 4

    def fit(self, train_data) -> None:
        # calibrate parameters using ONLY training data
        self.threshold = train_data['my_feature'].quantile(0.95)

    def generate_signals(self, data) -> list:
        signals = []
        for ts, row in data.iterrows():
            if row['my_feature'] > self.threshold:
                signals.append(Signal(
                    timestamp=ts,
                    direction=Direction.SHORT,
                    metadata={'feature': row['my_feature']}
                ))
        return signals
```

### Run via CLI

```bash
# List available strategies
python run.py --list

# Quick run — defaults from config.py (train=500, test=100, step=100, fee=0.001),
# prints a summary to stdout
python run.py --strategy funding_rate --data btc_4h_2y.parquet

# Full run — generates the markdown report (with real baselines) at the given path
python run.py --strategy funding_rate --data btc_4h_2y.parquet \
              --report --output reports/funding_rate.md

# Override walk-forward window sizes and fee
python run.py --strategy funding_rate --data btc_4h_2y.parquet \
              --train 600 --test 150 --step 150 --fee 0.0008
```

> When `--report` is passed, the report includes buy-and-hold and random-entry
> baselines computed via Monte Carlo (100 simulations by default). This adds a
> few tens of seconds to the run.

### Run via code

```python
from backtester.core.fee_model import FeeModel
from backtester.core.walk_forward import walk_forward, WalkForwardConfig
from backtester.config import FrameworkConfig
from backtester.data.loader import load_parquet
from backtester.report.generator import generate_report
from backtester.strategy.examples.funding_rate import FundingRateStrategy

config = FrameworkConfig()
data = load_parquet("data/btc_4h_2y.parquet")
fee_model = FeeModel(taker_fee=config.taker_fee, slippage_estimate=config.slippage_estimate)

results = walk_forward(
    data=data,
    strategy=FundingRateStrategy(percentile=0.95),
    fee_model=fee_model,
    config=WalkForwardConfig(
        train_size=config.train_size,
        test_size=config.test_size,
        step_size=config.step_size,
        min_trades_per_window=config.min_trades_per_window,
    ),
)

# Per-window metrics (already populated by walk_forward)
for r in results:
    perf = r.metrics["performance"]
    stat = r.metrics["statistical"]
    print(f"window {r.config['window']}: "
          f"trades={len(r.trades)}, win_rate={perf['win_rate']:.1%}, "
          f"p={stat['p_value']:.3f}")

# Generate the markdown report. Passing `data` and `fee_model` activates real
# baseline comparison (buy-and-hold + random entry) and adds the `beats random
# entry` criterion to the verdict.
report_md = generate_report(
    results,
    strategy_name="funding_rate",
    config=config,
    data=data,
    fee_model=fee_model,
)
print(report_md)
```

---

## Testing

![Tests](https://img.shields.io/badge/tests-179%20passing-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)

```bash
pytest --cov=backtester --cov-fail-under=90
```

CI runs on every push via GitHub Actions (`.github/workflows/tests.yml`).

---

## Framework guarantees

| Guarantee | Implementation |
|---|---|
| No look-ahead in calibration | `fit()` receives only training data. The train/test split is the framework's responsibility — the strategy has no control over it. |
| Signal generation contract | Inside `generate_signals(data)`, strategies must not access future information (e.g., `data.shift(-k)`, reverse iteration, future-windowed rolling). The framework cannot enforce this — it is the strategy author's responsibility. See the docstring of `Strategy.generate_signals`. |
| Results are always OOS | All `BacktestResult` objects have `is_oos=True`. In-sample results are diagnostic only, never the main output. |
| Fee always applied | `fee_model.apply()` is called on every trade — wins and losses, no exceptions. |
| Realistic execution | Entry at next candle's open, not at signal candle's close. |
| Warmup enforced | The framework discards `warmup_periods()` candles before the first signal. |
| One-tailed t-test | Hypothesis test is always H1: return > 0, never two-tailed. |

---

## Generated metrics

### Performance

- Win rate, mean/median return, total return
- Sharpe ratio, Sortino ratio
- Profit factor, win/loss ratio
- Trades per day

### Risk

- Max drawdown
- Longest consecutive loss streak
- Max adverse excursion (MAE), mean MAE, p90 MAE
- Minimum capital reached during simulation

### Statistics

- t-statistic, p-value (one-tailed)
- 95% confidence interval
- Per-window significance across walk-forward

### Baselines

- Buy-and-hold per window (with fee), compounded total return
- Random entry (Monte Carlo, 100 simulations by default) with matched trade count and holding
- One-tailed p-value of strategy total return vs random entry distribution

---

## Automatic report

Each run generates a markdown report with:

1. Run configuration
2. Results per walk-forward window
3. Global OOS statistical significance
4. Risk metrics
5. Comparison against all baselines
6. **Automatic conclusion** based on objective criteria:
   - OOS p-value < `p_value_threshold` (default 0.05)
   - OOS win rate > `min_win_rate` (default 52%)
   - OOS mean return > 0 (after fee)
   - Max drawdown within `max_acceptable_drawdown` (default -25%)
   - Significant windows ≥ 60% (per-window dispersion check)
   - Beats random entry (one-tailed p < `p_value_threshold`) — only when baseline data is provided

### Example output

[`reports/funding_rate_4h_2y.md`](reports/funding_rate_4h_2y.md) — walk-forward report on 2 years of BTC/USDT 4h data, generated by:

```bash
python run.py --strategy funding_rate --data btc_funding_2y.parquet \
              --report --output reports/funding_rate_4h_2y.md
```

**Verdict: REJECTED** — 5 of 6 criteria fail across 3 OOS windows and 42 trades (win rate 35.7%, p-value 0.94, mean return -0.40%, 0% significant windows, loses to random entry at p=0.81). Only `max_drawdown` (-17.69%) passes. Additionally, 35 of 38 walk-forward windows were skipped for insufficient trades — direct evidence of the regime-dependency described in [`docs/RESEARCH.md`](docs/RESEARCH.md). This is the framework correctly invalidating the H3 funding-rate strategy.

---

## Current features

- Walk-forward validation with strict train/test separation
- Realistic trade execution (next-open entry, fixed holding)
- Fee model with configurable slippage
- Maximum adverse excursion tracking
- Statistical significance testing (one-tailed t-test)
- Buy-and-hold and random entry baselines with Monte Carlo p-value
- Automatic report generation with objective conclusions

## Out of scope for v1.0

| Feature | Reason |
|---|---|
| Simultaneous positions | Unnecessary complexity for the MVP |
| Stop loss / take profit | Add after validating fixed holding |
| Leverage | Simple multiplier, not needed in the framework |
| Parameter optimization | Overfit risk — walk-forward is sufficient control |
| UI / Dashboard | Markdown report is sufficient |

## Known limitations — deferred to v1.1

| Limitation | Notes |
|---|---|
| Look-ahead enforcement in `generate_signals` | Documented as contract; not detected. Sampling-based auditor planned (`backtester.audit`) — see `CHANGELOG.md`. |
| Random-entry baseline direction | Currently bidirectional (50/50 LONG/SHORT). Penalizes unidirectional strategies. To be matched to the strategy's direction distribution. |
| `n_simulations` for random entry | Hardcoded at 100. Will become configurable via `FrameworkConfig`. |
| `--no-baselines` CLI flag | Baselines always run when `--report` is used. Slow for iteration. |
| "Too good to be true" sanity check | No automatic flag for win-rate > 80% / Sharpe > 5 — manual auditor judgment required. |
| Signal-rejection visibility | `rejection_rate` exists on `BacktestResult` but is not surfaced in CLI summary or report. |
| Slippage model | Symmetric and constant `(taker_fee + slippage) * 2`. Real slippage is directional and size-dependent. |
| Sharpe risk-free rate | Implicit RFR = 0. |
| Confidence interval semantics with n<2 | Returns `(0.0, 0.0)` instead of `None`/NaN. |
| `min_trades_per_window=10` justification | Default chosen pragmatically; not derived from a power-analysis. |
| CLI strategy parameterization | `FundingRateStrategy(percentile=...)` configurable in code only, not via CLI. |
| Strategy registry API | `_REGISTRY` is mutated by import side-effect in `run.py`. No public `register_strategy()` function. |

---

## Default configuration

Defined in `config.py`. All parameters have sensible defaults:

```python
taker_fee = 0.001            # 0.10% per side
slippage_estimate = 0.0      # calibrate with real execution data
train_size = 500             # training candles per window
test_size = 100              # test candles per window
step_size = 100              # advance between windows
min_trades_per_window = 10   # windows with fewer trades are discarded
p_value_threshold = 0.05
min_win_rate = 0.52
max_acceptable_drawdown = -0.25
```

---

## Success criteria

**Goal:** a new strategy can be tested end-to-end in under 1 hour of work — implement the `Strategy` interface, point the CLI at a parquet file, get a markdown report with an automatic verdict.

**Status:** v1.0 meets this on the bundled (invalidated) `FundingRateStrategy` example. See `python run.py --strategy funding_rate --data your.parquet --report`.
