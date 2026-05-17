import logging
from typing import List

import pandas as pd

from backtester.core.bt_types import Direction, Signal, Trade
from backtester.core.fee_model import FeeModel

logger = logging.getLogger(__name__)


_REQUIRED_COLUMNS = {"open", "high", "low", "close"}


def run_backtest(
    data: pd.DataFrame,
    signals: List[Signal],
    holding: int,
    fee_model: FeeModel,
) -> List[Trade]:
    """
    Runs a simple fixed-holding backtest.

    - Entry at close of signal candle.
    - Exit at close of candle `holding` periods later.
    - Signals during an active holding window are ignored (no simultaneous positions).
    - Re-entry on the exit candle is allowed: active_until uses strict < so a signal
      at the exact exit candle index passes. The prior trade closed at that candle's
      close, and the new trade opens at the same price.
    - MAE window covers [entry+1, exit] inclusive — the full holding period including
      the exit candle, since price can move adversely before the closing print.

    Raises ValueError if data is missing required columns or has an unsorted index.
    """
    missing = _REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Data index must be sorted chronologically (monotonic increasing)")

    trades: List[Trade] = []
    active_until = -1  # candle index up to which a position is active

    for signal in sorted(signals, key=lambda s: s.timestamp):
        if signal.direction == Direction.NONE:
            continue

        idx = data.index.searchsorted(signal.timestamp)

        if idx >= len(data) or data.index[idx] != signal.timestamp:
            logger.warning("Signal timestamp %s not in data index, skipping", signal.timestamp)
            continue

        if idx < active_until:
            continue

        exit_idx = idx + holding
        if exit_idx >= len(data):
            continue

        entry_price = float(data.iloc[idx]["close"])
        exit_price = float(data.iloc[exit_idx]["close"])

        if signal.direction == Direction.LONG:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        # Adverse move: how far price moved against the position during holding.
        window = data.iloc[idx + 1 : exit_idx + 1]
        if signal.direction == Direction.LONG:
            adverse = ((entry_price - window["low"]) / entry_price).clip(lower=0)
        else:
            adverse = ((window["high"] - entry_price) / entry_price).clip(lower=0)

        max_adverse = float(adverse.max()) if not adverse.empty else 0.0

        fee = fee_model.round_trip
        net_return = fee_model.apply(gross_return)

        trades.append(Trade(
            entry_time=data.index[idx],
            exit_time=data.index[exit_idx],
            direction=signal.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_return=gross_return,
            fee=fee,
            net_return=net_return,
            holding_candles=holding,
            max_adverse=max_adverse,
            metadata=signal.metadata.copy(),
        ))

        active_until = exit_idx

    return trades
