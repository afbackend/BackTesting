import pytest
import pandas as pd

from backtester.config import FrameworkConfig
from backtester.core.bt_types import Direction, Signal
from backtester.core.fee_model import FeeModel
from backtester.core.walk_forward import WalkForwardConfig, walk_forward
from backtester.report.generator import generate_report


# --- helpers ---

def make_data(n: int = 300, drift: float = 0.01) -> pd.DataFrame:
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


class AlwaysShortStrategy:
    def warmup_periods(self): return 0
    def holding_periods(self): return 2
    def fit(self, train_data): pass
    def generate_signals(self, data):
        return [Signal(timestamp=ts, direction=Direction.SHORT) for ts in data.index]


def make_results(strategy=None, drift=0.01):
    if strategy is None:
        strategy = AlwaysLongStrategy()
    data = make_data(300, drift=drift)
    wf_config = WalkForwardConfig(train_size=100, test_size=50, step_size=50, min_trades_per_window=1)
    return walk_forward(data, strategy, FeeModel(), wf_config)


# --- empty results ---

def test_empty_results_returns_string():
    report = generate_report([], "TestStrategy", FrameworkConfig())
    assert isinstance(report, str)
    assert len(report) > 0


def test_empty_results_contains_no_data_message():
    report = generate_report([], "TestStrategy", FrameworkConfig())
    assert "No valid walk-forward windows" in report


def test_empty_results_contains_strategy_name():
    report = generate_report([], "MyStrategy", FrameworkConfig())
    assert "MyStrategy" in report


# --- section headers ---

def test_report_contains_all_section_headers():
    results = make_results()
    report = generate_report(results, "TestStrategy", FrameworkConfig())
    assert "## 1. Configuration" in report
    assert "## 2. Walk-Forward Results" in report
    assert "## 3. Statistical Significance" in report
    assert "## 4. Risk Metrics" in report
    assert "## 5. Baseline Comparison" in report
    assert "## 6. Conclusion" in report


def test_report_contains_strategy_name():
    results = make_results()
    report = generate_report(results, "FundingRateH3", FrameworkConfig())
    assert "FundingRateH3" in report


# --- walk-forward table ---

def test_walk_forward_table_has_one_row_per_window():
    results = make_results()
    report = generate_report(results, "TestStrategy", FrameworkConfig())
    # Each result window produces one data row in the table (lines starting with "| <digit>")
    table_rows = [l for l in report.splitlines() if l.startswith("| ") and "Window" not in l and "---" not in l and "Criterion" not in l and "p-value" not in l]
    # At least len(results) rows across both the walk-forward and conclusion tables
    assert len(table_rows) >= len(results)


# --- data_info ---

def test_data_info_appears_in_section_1():
    results = make_results()
    data_info = {"symbol": "BTCUSDT", "timeframe": "4h", "total_candles": 300}
    report = generate_report(results, "TestStrategy", FrameworkConfig(), data_info=data_info)
    assert "BTCUSDT" in report
    assert "4h" in report
    assert "300" in report


def test_no_data_info_omits_data_block():
    results = make_results()
    report = generate_report(results, "TestStrategy", FrameworkConfig(), data_info=None)
    assert "symbol" not in report
    assert "timeframe" not in report


# --- conclusion ---

def test_conclusion_approved_when_all_criteria_pass():
    # drift=5.0 → gross ≈ 10/100 = 10% per 2-candle hold, well above 0.2% fee
    results = make_results(drift=5.0)
    report = generate_report(results, "TestStrategy", FrameworkConfig())
    assert "APPROVED" in report


def test_conclusion_rejected_when_criteria_fail():
    # AlwaysShort on upward drift → losses
    results = make_results(strategy=AlwaysShortStrategy(), drift=0.05)
    report = generate_report(results, "TestStrategy", FrameworkConfig())
    assert "REJECTED" in report


def test_conclusion_table_has_four_criteria():
    results = make_results()
    report = generate_report(results, "TestStrategy", FrameworkConfig())
    assert "p-value < threshold" in report
    assert "win_rate > min_win_rate" in report
    assert "mean_return > 0" in report
    assert "max_drawdown within limit" in report


# --- statistical section ---

def test_significant_windows_count_in_report():
    results = make_results(drift=0.05)
    report = generate_report(results, "TestStrategy", FrameworkConfig())
    assert "Significant windows:" in report


# --- baselines placeholder ---

def test_baselines_section_has_placeholder():
    results = make_results()
    report = generate_report(results, "TestStrategy", FrameworkConfig())
    assert "not implemented in v1.0" in report
