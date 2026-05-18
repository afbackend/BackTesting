from pathlib import Path
from typing import Union

import pandas as pd

_TIMESTAMP_COLUMN_NAMES = {"timestamp", "time", "date"}


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load an OHLC parquet file and return a DataFrame with a sorted DatetimeIndex.

    If the parquet file already has a DatetimeIndex it is used as-is (then sorted).
    Otherwise, looks for a column named 'timestamp', 'time', or 'date' to use as
    the index. Falls back to converting the existing index via pd.to_datetime.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if a DatetimeIndex cannot be constructed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        df = _set_datetime_index(df, path)

    return df.sort_index()


def _set_datetime_index(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    candidate = _TIMESTAMP_COLUMN_NAMES & set(df.columns)
    if candidate:
        col = next(iter(candidate))
        df = df.set_index(col)
        df.index = pd.to_datetime(df.index)
        return df

    try:
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:
        raise ValueError(
            f"Cannot build DatetimeIndex from '{path}': {exc}"
        ) from exc
