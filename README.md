# Trading Research Engine

A Python framework for testing trading strategies with statistical rigor. The core goal is to make the biases that invalidate backtests — look-ahead bias, in-sample overfitting, inconsistent fees — impossible to introduce by accident.

---

## Why this framework exists

Every naive backtest lies. The most common problems:

- Threshold calibrated on the full dataset → look-ahead bias
- Reported result is in-sample → guaranteed overfit
- Fee applied only when convenient → unrealistic returns
- No benchmark → no way to know if the strategy has real edge

This framework was built to make these mistakes structurally impossible — not through discipline, but through architecture.

It was born from three research sprints where promising strategies were invalidated by walk-forward validation. The framework encodes every lesson from those failures into enforceable constraints.

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
- **Strategy never sees test data.** The train/test split is the framework's responsibility. The strategy only implements `fit()` and `generate_signals()`.
- **Fee is structural, not optional.** Applied on every trade via `FeeModel` — wins and losses, no exceptions.
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
│   └── risk.py            # Drawdown, MAE, consecutive loss streak
│
├── report/
│   ├── generator.py       # Generates markdown report automatically
│   └── templates/
│       └── sprint_report.md
│
├── config.py              # Global parameters with sensible defaults
└── run.py                 # CLI entry point
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
                    confidence=1.0,
                    metadata={'feature': row['my_feature']}
                ))
        return signals
```

### Run via CLI

```bash
# Walk-forward with report
python run.py --strategy funding_rate --data btc_4h_2y.parquet --report

# Custom parameters
python run.py --strategy funding_rate --train 600 --test 150 --fee 0.001

# List available strategies
python run.py --list
```

### Run via code

```python
from backtester.core.walk_forward import walk_forward, WalkForwardConfig
from backtester.core.fee_model import FeeModel
from backtester.data.loader import load_parquet
from backtester.strategy.examples.funding_rate import FundingRateStrategy

data = load_parquet("data/btc_4h_2y.parquet")

results = walk_forward(
    data=data,
    strategy=FundingRateStrategy(),
    fee_model=FeeModel(taker_fee=0.001),
    config=WalkForwardConfig(train_size=500, test_size=100, step_size=100),
)
```

---

## Testing

![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-99.59%25-brightgreen)

```bash
pytest --cov=backtester --cov-fail-under=90
```

CI runs on every push via GitHub Actions (`.github/workflows/tests.yml`).

---

## Implementation status

| Module | Status | Description |
|---|---|---|
| `core/bt_types.py` | ✅ | Signal, Trade, BacktestResult, Direction |
| `core/fee_model.py` | ✅ | Round-trip fee + slippage |
| `core/backtest_engine.py` | ✅ | Trade execution loop |
| `core/walk_forward.py` | ✅ | Walk-forward validation — main entry point |
| `strategy/base.py` | ✅ | Abstract Strategy interface |
| `metrics/performance.py` | ✅ | Win rate, Sharpe, Sortino, profit factor |
| `metrics/risk.py` | ✅ | Drawdown, MAE, consecutive loss |
| `metrics/statistical.py` | ✅ | One-tailed t-test, p-value, 95% CI |
| `data/loader.py` | ✅ | Load local parquet files |
| `data/validator.py` | ✅ | Detect NaN, duplicates, gaps |
| `config.py` | ✅ | Global parameters with defaults |
| `report/generator.py` | ✅ | Markdown report with auto conclusion |
| `cli.py` / `run.py` | ✅ | CLI entry point |
| `data/fetcher.py` | ⬜ | Fetch from exchanges with retry |
| `strategy/examples/funding_rate.py` | ⬜ | H3 reference implementation |

---

## Framework guarantees

| Guarantee | Implementation |
|---|---|
| No look-ahead bias | `fit()` receives only training data. The train/test split is the framework's responsibility — the strategy has no control over it. |
| Results are always OOS | All `BacktestResult` objects have `is_oos=True`. In-sample results are diagnostic only, never the main output. |
| Fee always applied | `fee_model.apply()` is called on every trade — wins and losses, no exceptions. |
| Realistic execution | Entry at next candle's open, not at signal candle's close. |
| Warmup enforced | The framework discards `warmup_periods()` candles before the first signal. |
| One-tailed t-test | Hypothesis test is always H1: return > 0, never two-tailed. |
| Mandatory benchmarks | Every result includes comparison against buy-and-hold, random entry, and the inverse strategy. |
| Reproducible | Fixed seed (`random_seed=42`), no hidden randomization. |

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

---

## Automatic report

Each run generates a markdown report with:

1. Run configuration
2. Results per walk-forward window
3. Global OOS statistical significance
4. Risk metrics
5. Comparison against all baselines
6. **Automatic conclusion** based on objective criteria:
   - OOS p-value < 0.05
   - OOS win rate > 52%
   - OOS edge > 0 after fee
   - Beats random entry with significance
   - Max drawdown < 25%

---

## Research context

This framework was built during a structured research process documented in three sprints:

- **Sprint 1** tested three hypotheses (liquidity grab reversal, volume confirmation, extreme funding rate) and found apparent edge in funding rate signals — later proven to be overfit.
- **Sprint 2** added regime detection via daily EMA50, improving in-sample results to 57% win rate and +11.5% return — but without out-of-sample validation.
- **Sprint 3** applied walk-forward validation and revealed that the in-sample edge was a statistical artifact. Win rate dropped from 57% to 32% out-of-sample. The strategy was invalidated.

Every design decision in this framework exists to prevent a specific failure mode discovered during those sprints. The full research reports are available in the `docs/` directory.

---

## Current features

- Walk-forward validation with strict train/test separation
- Realistic trade execution (next-open entry, fixed holding)
- Fee model with configurable slippage
- Maximum adverse excursion tracking
- Statistical significance testing (one-tailed t-test)
- Automatic report generation with objective conclusions
- Mandatory baseline comparison

## Out of scope for v1.0

| Feature | Reason |
|---|---|
| Simultaneous positions | Unnecessary complexity for the MVP |
| Stop loss / take profit | Add after validating fixed holding |
| Leverage | Simple multiplier, not needed in the framework |
| Parameter optimization | Overfit risk — walk-forward is sufficient control |
| UI / Dashboard | Markdown report is sufficient |

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
random_entry_simulations = 100
random_seed = 42
```

---

## Success criteria

The framework is ready when a new strategy can be tested in under 1 hour of work: implement the interface, run the walk-forward, and get a report with an automatic conclusion.
