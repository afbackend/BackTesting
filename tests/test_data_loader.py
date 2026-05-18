import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from backtester.data.loader import load_parquet


def make_ohlc(n: int = 10, freq: str = "4h") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
        index=idx,
    )


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


# --- basic loading ---

def test_load_parquet_returns_dataframe():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "data.parquet"
        save_parquet(make_ohlc(), p)
        df = load_parquet(p)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


def test_loaded_df_has_datetime_index():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "data.parquet"
        save_parquet(make_ohlc(), p)
        df = load_parquet(p)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_loaded_df_is_sorted():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "data.parquet"
        # save in reverse order
        save_parquet(make_ohlc().iloc[::-1], p)
        df = load_parquet(p)
    assert df.index.is_monotonic_increasing


# --- timestamp column as index ---

def test_load_parquet_with_timestamp_column():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "data.parquet"
        df = make_ohlc().reset_index().rename(columns={"index": "timestamp"})
        df.to_parquet(p, index=False)
        loaded = load_parquet(p)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert len(loaded) == 10


def test_load_parquet_with_time_column():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "data.parquet"
        df = make_ohlc().reset_index().rename(columns={"index": "time"})
        df.to_parquet(p, index=False)
        loaded = load_parquet(p)
    assert isinstance(loaded.index, pd.DatetimeIndex)


# --- errors ---

def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_parquet("/nonexistent/path/data.parquet")


def test_accepts_string_path():
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "data.parquet")
        save_parquet(make_ohlc(), Path(p))
        df = load_parquet(p)
    assert len(df) == 10


def test_load_parquet_with_no_named_timestamp_column_falls_back_to_index_conversion():
    # No 'timestamp'/'time'/'date' column; existing index is convertible to datetime.
    # Exercises the pd.to_datetime(df.index) fallback path in _set_datetime_index.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "data.parquet"
        df = make_ohlc().reset_index(drop=True)  # RangeIndex, no timestamp col
        df.to_parquet(p, index=False)
        loaded = load_parquet(p)
    assert isinstance(loaded.index, pd.DatetimeIndex)
