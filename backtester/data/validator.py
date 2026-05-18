import logging
from typing import Optional

import pandas as pd

_REQUIRED_COLUMNS = {"open", "high", "low", "close"}

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, expected_freq: Optional[str] = None) -> None:
    """
    Validate OHLC DataFrame integrity.

    Raises DataValidationError for: empty DataFrame, non-DatetimeIndex,
    non-monotonic index, duplicate timestamps, missing OHLC columns, NaN values.

    If expected_freq is provided, logs a warning for any gaps exceeding that
    frequency (does not raise — gaps are common in real market data).
    """
    if df.empty:
        raise DataValidationError("DataFrame is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Index must be DatetimeIndex, got {type(df.index).__name__}."
        )

    if df.index.duplicated().any():
        n = int(df.index.duplicated().sum())
        raise DataValidationError(f"Found {n} duplicate timestamp(s) in index.")

    if not df.index.is_monotonic_increasing:
        raise DataValidationError("Index is not monotonically increasing.")

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {sorted(missing)}."
        )

    for col in _REQUIRED_COLUMNS:
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            raise DataValidationError(
                f"Column '{col}' contains {n_nan} NaN value(s)."
            )

    if expected_freq is not None:
        _check_gaps(df.index, expected_freq)


def _check_gaps(index: pd.DatetimeIndex, expected_freq: str) -> None:
    expected = pd.Timedelta(expected_freq)
    diffs = pd.Series(index).diff().dropna()
    gaps = diffs[diffs > expected]
    if not gaps.empty:
        largest = gaps.max()
        logger.warning(
            "Found %d gap(s) in data; largest: %s (expected: %s).",
            len(gaps),
            largest,
            expected,
        )
