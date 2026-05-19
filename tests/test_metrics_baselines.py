import numpy as np
import pandas as pd
import pytest

from backtester.core.bt_types import Direction, Signal
from backtester.core.fee_model import FeeModel
from backtester.core.walk_forward import WalkForwardConfig, walk_forward
from backtester.metrics.baselines import buy_and_hold, random_entry


def _make_data(n: int = 300, drift: float = 0.05) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    prices = [100.0 + i * drift for i in range(n)]
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=idx,
    )


class AlwaysLongStrategy:
    def warmup_periods(self): return 0
    def holding_periods(self): return 2
    def fit(self, train_data): pass
    def generate_signals(self, data):
        return [Signal(timestamp=ts, direction=Direction.LONG) for ts in data.index]


def _make_results(drift: float = 0.05):
    data = _make_data(300, drift=drift)
    wf_config = WalkForwardConfig(train_size=100, test_size=50, step_size=50,
                                  min_trades_per_window=1)
    return walk_forward(data, AlwaysLongStrategy(), FeeModel(), wf_config), data


# --- buy_and_hold ---

def test_buy_and_hold_empty_results():
    bh = buy_and_hold([], pd.DataFrame(), FeeModel())
    assert bh["per_window_returns"] == []
    assert bh["total_return"] == 0.0
    assert bh["mean_return"] == 0.0


def test_buy_and_hold_returns_one_value_per_window():
    results, data = _make_results()
    bh = buy_and_hold(results, data, FeeModel())
    assert len(bh["per_window_returns"]) == len(results)


def test_buy_and_hold_applies_fee():
    results, data = _make_results()
    no_fee = buy_and_hold(results, data, FeeModel(taker_fee=0.0, slippage_estimate=0.0))
    with_fee = buy_and_hold(results, data, FeeModel(taker_fee=0.001, slippage_estimate=0.0))
    # With fee, each window's net return must be lower by exactly round_trip
    for nf, wf in zip(no_fee["per_window_returns"], with_fee["per_window_returns"]):
        assert wf == pytest.approx(nf - 0.002, abs=1e-12)


def test_buy_and_hold_total_is_compounded():
    results, data = _make_results()
    bh = buy_and_hold(results, data, FeeModel())
    expected = float(np.prod([1 + r for r in bh["per_window_returns"]]) - 1)
    assert bh["total_return"] == pytest.approx(expected)


# --- random_entry ---

def test_random_entry_empty_results():
    re = random_entry([], pd.DataFrame(), FeeModel(), n_simulations=5)
    assert re["n_simulations"] == 5
    assert re["mean_total_return"] == 0.0
    assert re["p_value_vs_strategy"] == 1.0


def test_random_entry_returns_required_keys():
    results, data = _make_results()
    re = random_entry(results, data, FeeModel(), n_simulations=10, seed=42)
    for key in ("n_simulations", "mean_total_return", "median_total_return",
                "p5_total_return", "p95_total_return", "p_value_vs_strategy"):
        assert key in re


def test_random_entry_reproducible_with_same_seed():
    results, data = _make_results()
    r1 = random_entry(results, data, FeeModel(), n_simulations=10, seed=42)
    r2 = random_entry(results, data, FeeModel(), n_simulations=10, seed=42)
    assert r1["mean_total_return"] == r2["mean_total_return"]
    assert r1["p_value_vs_strategy"] == r2["p_value_vs_strategy"]


def test_random_entry_p_value_in_unit_interval():
    results, data = _make_results()
    re = random_entry(results, data, FeeModel(), n_simulations=20, seed=42)
    assert 0.0 <= re["p_value_vs_strategy"] <= 1.0


def test_random_entry_p5_le_median_le_p95():
    results, data = _make_results()
    re = random_entry(results, data, FeeModel(), n_simulations=30, seed=42)
    assert re["p5_total_return"] <= re["median_total_return"] <= re["p95_total_return"]
