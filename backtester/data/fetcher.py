import time
from typing import Union

import ccxt
import pandas as pd

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    exchange_id: str = "binance",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from a ccxt-supported exchange.

    Paginates automatically until all candles in [start, end] are collected.
    Retries on NetworkError or ExchangeError with exponential backoff.

    Returns a DataFrame with DatetimeIndex (UTC), columns: open/high/low/close/volume.
    Index is sorted and deduplicated.

    Raises ValueError for unknown exchange_id.
    Re-raises ccxt errors after max_retries is exhausted.
    """
    if exchange_id not in ccxt.exchanges:
        raise ValueError(f"Unknown exchange: {exchange_id!r}. Available: ccxt.exchanges")

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    timeframe_ms = int(exchange.parse_timeframe(timeframe)) * 1000

    start_ms = int(pd.Timestamp(start).value // 1_000_000)
    end_ms = int(pd.Timestamp(end).value // 1_000_000)

    all_candles: list = []
    cursor = start_ms
    prev_last_ts = -1

    while True:
        batch = _fetch_with_retry(exchange, symbol, timeframe, cursor, max_retries, retry_delay)
        if not batch:
            break
        last_ts = batch[-1][0]
        if last_ts <= prev_last_ts:
            raise RuntimeError(
                f"Cursor did not advance (last_ts={last_ts}, prev={prev_last_ts}). "
                "Possible exchange bug or stale data."
            )
        all_candles.extend(batch)
        prev_last_ts = last_ts
        if last_ts >= end_ms:
            break
        cursor = last_ts + timeframe_ms

    if not all_candles:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    timestamps = [c[0] for c in all_candles]
    rows = [c[1:6] for c in all_candles]

    index = pd.to_datetime(timestamps, unit="ms", utc=True)
    df = pd.DataFrame(rows, index=index, columns=_OHLCV_COLUMNS)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df.loc[
        (df.index >= _to_utc(start))
        & (df.index <= _to_utc(end))
    ]


def _to_utc(ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t if t.tzinfo is not None else t.tz_localize("UTC")


def _fetch_with_retry(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int,
    max_retries: int,
    retry_delay: float,
) -> list:
    for attempt in range(max_retries + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except (ccxt.NetworkError, ccxt.ExchangeError):
            if attempt == max_retries:
                raise
            time.sleep(retry_delay * (2 ** attempt))
    return []
