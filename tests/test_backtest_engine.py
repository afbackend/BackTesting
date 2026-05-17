import pytest
import pandas as pd

from backtester.core.backtest_engine import run_backtest
from backtester.core.bt_types import Direction, Signal
from backtester.core.fee_model import FeeModel


def make_data(closes, highs=None, lows=None, opens=None):
    """Build a minimal OHLC DataFrame with a DatetimeIndex."""
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame(
        {
            "open":  opens  if opens  is not None else closes,
            "high":  highs  if highs  is not None else closes,
            "low":   lows   if lows   is not None else closes,
            "close": closes,
        },
        index=idx,
    )


def signal(data, i, direction=Direction.LONG):
    return Signal(timestamp=data.index[i], direction=direction)


# --- basic execution ---

def test_long_trade_entry_at_next_open():
    # Signal at candle 0, entry at open[1], exit at close[5]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 110.0]
    data = make_data(closes)
    fee = FeeModel(taker_fee=0.001)
    trades = run_backtest(data, [signal(data, 0)], holding=4, fee_model=fee)

    assert len(trades) == 1
    t = trades[0]
    assert t.entry_time == data.index[1]
    assert t.exit_time == data.index[5]
    assert t.direction == Direction.LONG
    assert t.entry_price == pytest.approx(100.0)  # open[1]
    assert t.exit_price == pytest.approx(110.0)   # close[5]
    assert t.gross_return == pytest.approx(0.10)
    assert t.fee == pytest.approx(0.002)
    assert t.net_return == pytest.approx(0.098)
    assert t.holding_candles == 4


def test_short_trade_entry_at_next_open():
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 90.0]
    data = make_data(closes)
    trades = run_backtest(data, [signal(data, 0, Direction.SHORT)], holding=4, fee_model=FeeModel())

    t = trades[0]
    assert t.entry_price == pytest.approx(100.0)  # open[1]
    assert t.exit_price == pytest.approx(90.0)    # close[5]
    assert t.gross_return == pytest.approx(0.10)
    assert t.net_return == pytest.approx(0.098)


# --- overlap handling ---

def test_overlapping_signal_is_ignored():
    # Signal at 0: active_until = exit_idx = 1+4=5. Signal at 2: 2<=5 → ignored.
    data = make_data([100.0] * 10)
    signals = [signal(data, 0), signal(data, 2)]
    trades = run_backtest(data, signals, holding=4, fee_model=FeeModel())

    assert len(trades) == 1
    assert trades[0].entry_time == data.index[1]


def test_signal_at_exit_candle_is_ignored():
    # Trade 1: signal at 0, active_until=5. Signal at 5: 5<=5 → ignored.
    data = make_data([100.0] * 12)
    signals = [signal(data, 0), signal(data, 5)]
    trades = run_backtest(data, signals, holding=4, fee_model=FeeModel())

    assert len(trades) == 1


def test_signal_after_exit_candle_is_executed():
    # Trade 1: signal at 0, active_until=5. Signal at 6: 6<=5 → False → executes.
    data = make_data([100.0] * 15)
    signals = [signal(data, 0), signal(data, 6)]
    trades = run_backtest(data, signals, holding=4, fee_model=FeeModel())

    assert len(trades) == 2
    assert trades[1].entry_time == data.index[7]


# --- boundary ---

def test_signal_too_close_to_end_is_skipped():
    # 5 candles. Signal at 2: entry_idx=3, exit_idx=7, 7>=5 → skip.
    data = make_data([100.0] * 5)
    trades = run_backtest(data, [signal(data, 2)], holding=4, fee_model=FeeModel())

    assert len(trades) == 0


def test_signal_on_last_candle_is_skipped():
    # Signal on last candle: entry_idx would be out of bounds.
    data = make_data([100.0] * 5)
    trades = run_backtest(data, [signal(data, 4)], holding=1, fee_model=FeeModel())

    assert len(trades) == 0


def test_direction_none_is_ignored():
    data = make_data([100.0] * 10)
    s = Signal(timestamp=data.index[0], direction=Direction.NONE)
    trades = run_backtest(data, [s], holding=4, fee_model=FeeModel())

    assert len(trades) == 0


# --- MAE ---

def test_mae_long_includes_entry_candle():
    # Entry candle (idx+1) has a low below entry_price → counts for MAE.
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    lows   = [100.0,  95.0, 100.0, 100.0, 100.0, 100.0]  # low at entry candle
    data = make_data(closes, lows=lows)

    trades = run_backtest(data, [signal(data, 0)], holding=4, fee_model=FeeModel())

    assert trades[0].max_adverse == pytest.approx(5.0 / 100.0)


def test_mae_short_includes_entry_candle():
    # Entry candle (idx+1) has a high above entry_price → counts for MAE.
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    highs  = [100.0, 105.0, 100.0, 100.0, 100.0, 100.0]  # high at entry candle
    data = make_data(closes, highs=highs)

    trades = run_backtest(data, [signal(data, 0, Direction.SHORT)], holding=4, fee_model=FeeModel())

    assert trades[0].max_adverse == pytest.approx(5.0 / 100.0)


def test_mae_is_zero_when_no_adverse_move():
    # LONG: all lows >= entry_price → no adverse move.
    closes = [100.0, 100.0, 101.0, 102.0, 103.0, 104.0]
    lows   = [100.0, 100.0, 101.0, 102.0, 103.0, 104.0]
    data = make_data(closes, lows=lows)

    trades = run_backtest(data, [signal(data, 0)], holding=4, fee_model=FeeModel())

    assert trades[0].max_adverse == pytest.approx(0.0)


# --- input validation ---

def test_raises_on_missing_columns():
    idx = pd.date_range("2024-01-01", periods=6, freq="4h")
    data = pd.DataFrame({"close": [100.0] * 6}, index=idx)
    with pytest.raises(ValueError, match="missing required columns"):
        run_backtest(data, [], holding=4, fee_model=FeeModel())


def test_raises_on_unsorted_index():
    idx = pd.date_range("2024-01-01", periods=6, freq="4h")[::-1]
    data = pd.DataFrame(
        {"open": [100.0]*6, "high": [100.0]*6, "low": [100.0]*6, "close": [100.0]*6},
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


def test_signal_with_unknown_timestamp_is_skipped(caplog):
    import logging
    data = make_data([100.0] * 10)
    unknown_ts = pd.Timestamp("2099-01-01")
    s = Signal(timestamp=unknown_ts, direction=Direction.LONG)

    with caplog.at_level(logging.WARNING, logger="backtester.core.backtest_engine"):
        trades = run_backtest(data, [s], holding=4, fee_model=FeeModel())

    assert len(trades) == 0
    assert any("not in data index" in msg for msg in caplog.messages)
