import logging
import pytest
import numpy as np
import pandas as pd

from backtester.data.validator import DataValidationError, validate


def make_df(n: int = 5, freq: str = "4h") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
        index=idx,
    )


# --- valid ---

def test_valid_df_does_not_raise():
    validate(make_df())


def test_valid_df_with_expected_freq_does_not_raise():
    validate(make_df(), expected_freq="4h")


# --- empty ---

def test_empty_raises():
    with pytest.raises(DataValidationError, match="empty"):
        validate(pd.DataFrame())


# --- index type ---

def test_non_datetime_index_raises():
    df = make_df()
    df.index = range(len(df))
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        validate(df)


# --- monotonic ---

def test_non_monotonic_index_raises():
    df = make_df()
    df = df.iloc[::-1]  # reverse order
    with pytest.raises(DataValidationError, match="monotonic"):
        validate(df)


# --- duplicates ---

def test_duplicate_timestamps_raises():
    idx = pd.date_range("2024-01-01", periods=4, freq="4h")
    idx = idx.append(idx[[0]])  # duplicate first timestamp
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
        index=idx,
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        validate(df)


# --- missing columns ---

def test_missing_column_raises():
    df = make_df().drop(columns=["close"])
    with pytest.raises(DataValidationError, match="close"):
        validate(df)


def test_multiple_missing_columns_raises():
    df = make_df().drop(columns=["open", "high"])
    with pytest.raises(DataValidationError, match="Missing"):
        validate(df)


# --- NaN ---

def test_nan_in_open_raises():
    df = make_df()
    df.iloc[2, df.columns.get_loc("open")] = np.nan
    with pytest.raises(DataValidationError, match="'open'"):
        validate(df)


def test_nan_in_close_raises():
    df = make_df()
    df.iloc[0, df.columns.get_loc("close")] = np.nan
    with pytest.raises(DataValidationError, match="'close'"):
        validate(df)


# --- gap detection (warning only) ---

def test_gap_logs_warning(caplog):
    idx = pd.date_range("2024-01-01", periods=3, freq="4h")
    # insert a 12h gap between candle 1 and 2
    extra = pd.date_range("2024-01-01 20:00", periods=2, freq="4h")
    combined = idx[:2].append(extra)
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
        index=combined,
    )
    with caplog.at_level(logging.WARNING, logger="backtester.data.validator"):
        validate(df, expected_freq="4h")
    assert "gap" in caplog.text.lower()


def test_no_gap_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="backtester.data.validator"):
        validate(make_df(10, freq="4h"), expected_freq="4h")
    assert caplog.text == ""
