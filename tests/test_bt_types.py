import pytest
import pandas as pd

from backtester.core.bt_types import Direction, Signal, Trade, BacktestResult


def make_trade(**kwargs):
    defaults = dict(
        entry_time=pd.Timestamp("2024-01-01"),
        exit_time=pd.Timestamp("2024-01-02"),
        direction=Direction.LONG,
        entry_price=100.0,
        exit_price=102.0,
        gross_return=0.02,
        fee=0.002,
        net_return=0.018,
        holding_candles=4,
        max_adverse=0.005,
    )
    defaults.update(kwargs)
    return Trade(**defaults)


# --- Trade ---

def test_trade_consistent_net_return_passes():
    t = make_trade()
    assert t.net_return == pytest.approx(0.018)


def test_trade_inconsistent_net_return_raises():
    with pytest.raises(ValueError, match="net_return"):
        make_trade(net_return=0.999)


def test_trade_metadata_default_is_isolated():
    t1 = make_trade()
    t2 = make_trade()
    t1.metadata["key"] = "value"
    assert "key" not in t2.metadata


# --- Signal ---

def test_signal_metadata_default_is_isolated():
    s1 = Signal(timestamp=pd.Timestamp("2024-01-01"), direction=Direction.LONG)
    s2 = Signal(timestamp=pd.Timestamp("2024-01-01"), direction=Direction.SHORT)
    s1.metadata["key"] = "value"
    assert "key" not in s2.metadata


def test_signal_confidence_default():
    s = Signal(timestamp=pd.Timestamp("2024-01-01"), direction=Direction.LONG)
    assert s.confidence == 1.0


# --- BacktestResult ---

def test_rejection_rate_with_signals():
    result = BacktestResult(
        trades=[], metrics={}, config={}, is_oos=True,
        signals_generated=10, signals_executed=6,
    )
    assert result.rejection_rate == pytest.approx(0.4)


def test_rejection_rate_zero_signals_generated():
    result = BacktestResult(
        trades=[], metrics={}, config={}, is_oos=True,
        signals_generated=0, signals_executed=0,
    )
    assert result.rejection_rate == 0.0


def test_net_returns_series():
    trades = [make_trade(net_return=0.018), make_trade(net_return=-0.012, gross_return=-0.010, fee=0.002)]
    result = BacktestResult(
        trades=trades, metrics={}, config={}, is_oos=True,
        signals_generated=2, signals_executed=2,
    )
    series = result.net_returns
    assert isinstance(series, pd.Series)
    assert list(series) == pytest.approx([0.018, -0.012])
