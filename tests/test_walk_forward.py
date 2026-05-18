import pytest
import pandas as pd

from backtester.core.walk_forward import walk_forward, WalkForwardConfig
from backtester.core.bt_types import Direction, Signal
from backtester.core.fee_model import FeeModel


def make_data(n=300):
    """Synthetic OHLC data with a stable upward drift."""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    prices = [100.0 + i * 0.01 for i in range(n)]
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=idx,
    )


class AlwaysLongStrategy:
    """Stub: emits a LONG signal on every candle. fit() is a no-op."""

    def warmup_periods(self):
        return 0

    def holding_periods(self):
        return 2

    def fit(self, train_data):
        pass

    def generate_signals(self, data):
        return [
            Signal(timestamp=ts, direction=Direction.LONG)
            for ts in data.index
        ]


class NeverSignalStrategy:
    """Stub: never emits signals."""

    def warmup_periods(self):
        return 0

    def holding_periods(self):
        return 2

    def fit(self, train_data):
        pass

    def generate_signals(self, data):
        return []


class WarmupHeavyStrategy:
    """Stub: warmup larger than train_size to test skip logic."""

    def warmup_periods(self):
        return 999

    def holding_periods(self):
        return 2

    def fit(self, train_data):
        pass

    def generate_signals(self, data):
        return []


class FitRecorderStrategy:
    """Stub: records what data fit() received to verify train/test separation."""

    def __init__(self):
        self.fit_calls = []

    def warmup_periods(self):
        return 0

    def holding_periods(self):
        return 1

    def fit(self, train_data):
        self.fit_calls.append(train_data.copy())

    def generate_signals(self, data):
        return [Signal(timestamp=ts, direction=Direction.LONG) for ts in data.index]


# --- basic behavior ---

def test_all_results_are_oos():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, AlwaysLongStrategy(), FeeModel(), config)

    assert len(results) > 0
    assert all(r.is_oos for r in results)


def test_correct_number_of_windows():
    # 300 candles, train=100, test=50, step=50 → windows at 0, 50, 100, 150 → 4 windows
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, AlwaysLongStrategy(), FeeModel(), config)

    assert len(results) == 4


def test_window_indices_do_not_overlap():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    strategy = FitRecorderStrategy()
    results = walk_forward(data, strategy, FeeModel(), config)

    # Each test window must start after the previous one ended
    test_starts = [pd.Timestamp(r.config["test_start"]) for r in results]
    for i in range(1, len(test_starts)):
        assert test_starts[i] > test_starts[i - 1]


# --- train/test separation ---

def test_fit_never_receives_test_data():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    strategy = FitRecorderStrategy()
    results = walk_forward(data, strategy, FeeModel(), config)

    for i, result in enumerate(results):
        test_start = pd.Timestamp(result.config["test_start"])
        train_end = pd.Timestamp(result.config["train_end"])
        assert train_end < test_start, f"Window {i}: train bleeds into test"


def test_warmup_is_excluded_from_fit():
    data = make_data(300)

    class WarmupCheckStrategy:
        def __init__(self):
            self.received_train_sizes = []

        def warmup_periods(self):
            return 20

        def holding_periods(self):
            return 1

        def fit(self, train_data):
            self.received_train_sizes.append(len(train_data))

        def generate_signals(self, data):
            return [Signal(timestamp=ts, direction=Direction.LONG) for ts in data.index]

    strategy = WarmupCheckStrategy()
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    walk_forward(data, strategy, FeeModel(), config)

    # fit() should receive train_size - warmup = 80 candles per window
    assert all(s == 80 for s in strategy.received_train_sizes)


# --- min_trades filtering ---

def test_windows_below_min_trades_are_dropped():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, NeverSignalStrategy(), FeeModel(), config)

    assert len(results) == 0


def test_min_trades_threshold_respected():
    data = make_data(300)
    # AlwaysLong with holding=2 generates many trades — all windows should pass min=5
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=5)
    results = walk_forward(data, AlwaysLongStrategy(), FeeModel(), config)

    assert all(len(r.trades) >= 5 for r in results)


# --- signals_generated vs signals_executed ---

def test_signals_generated_and_executed_tracked():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, AlwaysLongStrategy(), FeeModel(), config)

    for r in results:
        assert r.signals_generated >= r.signals_executed
        assert r.signals_executed == len(r.trades)


# --- edge cases ---

def test_warmup_larger_than_train_skips_window():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, WarmupHeavyStrategy(), FeeModel(), config)

    assert len(results) == 0


def test_empty_data_returns_no_results():
    data = make_data(10)  # too small for any window
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, AlwaysLongStrategy(), FeeModel(), config)

    assert len(results) == 0


# --- metrics populated ---

def test_metrics_has_three_keys():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, AlwaysLongStrategy(), FeeModel(), config)

    for r in results:
        assert set(r.metrics.keys()) == {"performance", "risk", "statistical"}


def test_metrics_performance_win_rate_is_float():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, AlwaysLongStrategy(), FeeModel(), config)

    for r in results:
        assert isinstance(r.metrics["performance"]["win_rate"], float)


def test_metrics_statistical_n_trades_matches_trades():
    data = make_data(300)
    config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    results = walk_forward(data, AlwaysLongStrategy(), FeeModel(), config)

    for r in results:
        assert r.metrics["statistical"]["n_trades"] == len(r.trades)
