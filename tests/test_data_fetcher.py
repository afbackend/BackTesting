from unittest.mock import MagicMock

import ccxt
import pandas as pd
import pytest

from backtester.data.fetcher import fetch_ohlcv

# 2024-01-01 00:00 UTC in ms
_T0 = 1704067200000
_4H_SEC = 4 * 3600
_4H = _4H_SEC * 1000


def _candle(ts: int, price: float = 100.0) -> list:
    return [ts, price, price + 5, price - 5, price + 1, 1000.0]


def _mock_exchange(candles_by_call: list, timeframe_seconds: int = _4H_SEC):
    """Mock exchange whose fetch_ohlcv side_effect follows candles_by_call."""
    ex = MagicMock(spec=ccxt.Exchange)
    ex.fetch_ohlcv.side_effect = candles_by_call
    ex.parse_timeframe.return_value = timeframe_seconds
    return ex


def _patch_binance(monkeypatch, ex):
    monkeypatch.setattr(ccxt, "binance", lambda *args, **kwargs: ex)


# --- basic shape ---

def test_fetch_returns_dataframe(monkeypatch):
    candles = [_candle(_T0), _candle(_T0 + _4H)]
    ex = _mock_exchange([candles, []])
    _patch_binance(monkeypatch, ex)

    df = fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", "2024-01-02", retry_delay=0)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert len(df) == 2


def test_fetch_returns_sorted_index(monkeypatch):
    candles = [_candle(_T0 + _4H), _candle(_T0)]
    ex = _mock_exchange([candles, []])
    _patch_binance(monkeypatch, ex)

    df = fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", "2024-01-02", retry_delay=0)

    assert df.index.is_monotonic_increasing


def test_fetch_deduplicates(monkeypatch):
    candles = [_candle(_T0), _candle(_T0)]
    ex = _mock_exchange([candles, []])
    _patch_binance(monkeypatch, ex)

    df = fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", "2024-01-02", retry_delay=0)

    assert df.index.is_unique


# --- pagination ---

def test_pagination_fetches_multiple_batches(monkeypatch):
    batch1 = [_candle(_T0 + i * _4H) for i in range(3)]
    batch2 = [_candle(_T0 + i * _4H) for i in range(3, 6)]
    ex = _mock_exchange([batch1, batch2])
    _patch_binance(monkeypatch, ex)

    end_dt = pd.Timestamp(_T0 + 5 * _4H, unit="ms", tz="UTC")
    fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", end_dt, retry_delay=0)

    assert ex.fetch_ohlcv.call_count == 2


def test_cursor_advances_by_timeframe(monkeypatch):
    # First batch ends at _T0 + 2*_4H; next `since` must be _T0 + 3*_4H
    batch1 = [_candle(_T0 + i * _4H) for i in range(3)]
    batch2 = [_candle(_T0 + i * _4H) for i in range(3, 6)]
    ex = _mock_exchange([batch1, batch2])
    _patch_binance(monkeypatch, ex)

    end_dt = pd.Timestamp(_T0 + 5 * _4H, unit="ms", tz="UTC")
    fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", end_dt, retry_delay=0)

    calls = ex.fetch_ohlcv.call_args_list
    assert calls[1].kwargs["since"] == _T0 + 3 * _4H


def test_empty_response_stops_pagination(monkeypatch):
    ex = _mock_exchange([[]])
    _patch_binance(monkeypatch, ex)

    df = fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", "2024-01-02", retry_delay=0)

    assert len(df) == 0
    assert ex.fetch_ohlcv.call_count == 1


def test_cursor_not_advancing_raises(monkeypatch):
    # Two batches with same last_ts → must raise RuntimeError
    same = [_candle(_T0)]
    ex = _mock_exchange([same, same])
    _patch_binance(monkeypatch, ex)

    end_dt = pd.Timestamp(_T0 + 10 * _4H, unit="ms", tz="UTC")
    with pytest.raises(RuntimeError, match="Cursor did not advance"):
        fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", end_dt, retry_delay=0)


# --- retry ---

def test_retry_exhausted_raises(monkeypatch):
    ex = MagicMock(spec=ccxt.Exchange)
    ex.fetch_ohlcv.side_effect = ccxt.NetworkError("timeout")
    ex.parse_timeframe.return_value = _4H_SEC
    _patch_binance(monkeypatch, ex)

    with pytest.raises(ccxt.NetworkError):
        fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", "2024-01-02",
                    max_retries=2, retry_delay=0)

    assert ex.fetch_ohlcv.call_count == 3


def test_retry_succeeds_on_second_attempt(monkeypatch):
    candles = [_candle(_T0 + 4 * _4H)]
    ex = MagicMock(spec=ccxt.Exchange)
    ex.fetch_ohlcv.side_effect = [ccxt.NetworkError("fail"), candles]
    ex.parse_timeframe.return_value = _4H_SEC
    _patch_binance(monkeypatch, ex)

    end_dt = pd.Timestamp(_T0 + 2 * _4H, unit="ms", tz="UTC")
    df = fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", end_dt,
                     max_retries=2, retry_delay=0)

    assert ex.fetch_ohlcv.call_count == 2
    assert isinstance(df, pd.DataFrame)


# --- rate limit ---

def test_rate_limit_enabled_on_construction(monkeypatch):
    captured = {}
    ex = _mock_exchange([[]])

    def factory(config=None):
        captured["config"] = config
        return ex

    monkeypatch.setattr(ccxt, "binance", factory)
    fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", "2024-01-02", retry_delay=0)

    assert captured["config"] == {"enableRateLimit": True}


# --- validation ---

def test_unknown_exchange_raises_value_error():
    with pytest.raises(ValueError, match="Unknown exchange"):
        fetch_ohlcv("BTC/USDT", "4h", "2024-01-01", "2024-01-02",
                    exchange_id="totally_fake_exchange_xyz")
