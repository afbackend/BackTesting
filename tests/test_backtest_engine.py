import pytest
import pandas as pd
import numpy as np

from backtester.core.backtest_engine import run_backtest
from backtester.core.bt_types import Direction, Signal
from backtester.core.fee_model import FeeModel


def make_data(closes, highs=None, lows=None):
    """Build a minimal OHLC DataFrame with a DatetimeIndex."""
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else closes,
            "low": lows if lows is not None else closes,
            "close": closes,
        },
        index=idx,
    )


def signal(data, i, direction=Direction.LONG):
    return Signal(timestamp=data.index[i], direction=direction)


# --- basic execution ---

def test_long_trade_returns_correctly():
    data = make_data([100.0, 100.0, 100.0, 100.0, 110.0])
    fee = FeeModel(taker_fee=0.001)
    trades = run_backtest(data, [signal(data, 0)], holding=4, fee_model=fee)

    assert len(trades) == 1
    t = trades[0]
    assert t.direction == Direction.LONG
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(110.0)
    assert t.gross_return == pytest.approx(0.10)
    assert t.fee == pytest.approx(0.002)
    assert t.net_return == pytest.approx(0.098)
    assert t.holding_candles == 4


def test_short_trade_returns_correctly():
    data = make_data([100.0, 100.0, 100.0, 100.0, 90.0])
    fee = FeeModel(taker_fee=0.001)
    trades = run_backtest(data, [signal(data, 0, Direction.SHORT)], holding=4, fee_model=fee)

    assert len(trades) == 1
    t = trades[0]
    assert t.gross_return == pytest.approx(0.10)
    assert t.net_return == pytest.approx(0.098)


# --- overlap handling ---

def test_overlapping_signal_is_ignored():
    data = make_data([100.0] * 10)
    signals = [signal(data, 0), signal(data, 2)]  # second is inside holding=4
    trades = run_backtest(data, signals, holding=4, fee_model=FeeModel())

    assert len(trades) == 1
    assert trades[0].entry_time == data.index[0]


def test_signal_at_exit_candle_is_executed():
    # Trade 1: entry=0, exit=4. Signal at 4 should execute (active_until=4, 4 < 4 is False).
    data = make_data([100.0] * 10)
    signals = [signal(data, 0), signal(data, 4)]
    trades = run_backtest(data, signals, holding=4, fee_model=FeeModel())

    assert len(trades) == 2
    assert trades[1].entry_time == data.index[4]


# --- boundary ---

def test_signal_too_close_to_end_is_skipped():
    data = make_data([100.0] * 5)
    # holding=4, entry at index 2 would need exit at 6, which doesn't exist
    trades = run_backtest(data, [signal(data, 2)], holding=4, fee_model=FeeModel())

    assert len(trades) == 0


def test_direction_none_is_ignored():
    data = make_data([100.0] * 10)
    s = Signal(timestamp=data.index[0], direction=Direction.NONE)
    trades = run_backtest(data, [s], holding=4, fee_model=FeeModel())

    assert len(trades) == 0


# --- MAE ---

def test_mae_long_uses_lowest_low():
    # Entry at 100, holding window lows: [95, 98, 99, 100]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0]
    lows    = [100.0,  95.0,  98.0,  99.0, 100.0]
    highs   = [100.0, 100.0, 100.0, 100.0, 100.0]
    data = make_data(closes, highs=highs, lows=lows)

    trades = run_backtest(data, [signal(data, 0)], holding=4, fee_model=FeeModel())

    # Worst adverse: (100 - 95) / 100 = 0.05
    assert trades[0].max_adverse == pytest.approx(0.05)


def test_mae_short_uses_highest_high():
    # Entry at 100, holding window highs: [105, 102, 101, 100]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0]
    highs   = [100.0, 105.0, 102.0, 101.0, 100.0]
    lows    = [100.0, 100.0, 100.0, 100.0, 100.0]
    data = make_data(closes, highs=highs, lows=lows)

    trades = run_backtest(data, [signal(data, 0, Direction.SHORT)], holding=4, fee_model=FeeModel())

    # Worst adverse: (105 - 100) / 100 = 0.05
    assert trades[0].max_adverse == pytest.approx(0.05)


def test_mae_is_zero_when_no_adverse_move():
    # LONG: price only goes up — no adverse move
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    lows   = [100.0, 101.0, 102.0, 103.0, 104.0]  # lows above entry
    data = make_data(closes, lows=lows)

    trades = run_backtest(data, [signal(data, 0)], holding=4, fee_model=FeeModel())

    assert trades[0].max_adverse == pytest.approx(0.0)


# --- input validation ---

def test_raises_on_missing_columns():
    idx = pd.date_range("2024-01-01", periods=5, freq="4h")
    data = pd.DataFrame({"close": [100.0] * 5}, index=idx)
    with pytest.raises(ValueError, match="missing required columns"):
        run_backtest(data, [], holding=4, fee_model=FeeModel())


def test_raises_on_unsorted_index():
    idx = pd.date_range("2024-01-01", periods=5, freq="4h")[::-1]
    data = pd.DataFrame(
        {"open": [100.0] * 5, "high": [100.0] * 5, "low": [100.0] * 5, "close": [100.0] * 5},
        index=idx,
    )
    with pytest.raises(ValueError, match="sorted chronologically"):
        run_backtest(data, [], holding=4, fee_model=FeeModel())


# --- metadata propagation ---

def test_signal_metadata_is_copied_to_trade():
    data = make_data([100.0] * 10)
    s = Signal(timestamp=data.index[0], direction=Direction.LONG, metadata={"key": "value"})
    trades = run_backtest(data, [s], holding=4, fee_model=FeeModel())

    assert trades[0].metadata == {"key": "value"}


def test_signal_metadata_is_not_shared_with_trade():
    data = make_data([100.0] * 10)
    meta = {"key": "value"}
    s = Signal(timestamp=data.index[0], direction=Direction.LONG, metadata=meta)
    trades = run_backtest(data, [s], holding=4, fee_model=FeeModel())

    meta["key"] = "changed"
    assert trades[0].metadata["key"] == "value"
