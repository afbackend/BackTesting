import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

from backtester.core.backtest_engine import run_backtest
from backtester.core.bt_types import BacktestResult
from backtester.core.fee_model import FeeModel
from backtester.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    train_size: int
    test_size: int
    step_size: int
    min_trades_per_window: int = 10


def walk_forward(
    data: pd.DataFrame,
    strategy: Strategy,
    fee_model: FeeModel,
    config: WalkForwardConfig,
) -> List[BacktestResult]:
    """
    Walk-forward validation. All results are out-of-sample.

    For each window:
    1. Slice train and test without overlap.
    2. Discard warmup_periods from the start of train before fit().
       Strategy calibrates on clean data only.
    3. Call strategy.fit(train) — strategy never sees test data.
    4. Call strategy.generate_signals(test).
    5. Run backtest on test window.
    6. Discard window if trades < min_trades_per_window.

    Guarantees:
    - fit() never receives test data.
    - All BacktestResult objects are marked is_oos=True.
    - Windows with insufficient trades are logged and dropped.
    """
    results: List[BacktestResult] = []
    warmup = strategy.warmup_periods()
    holding = strategy.holding_periods()

    i = 0
    window_index = 0

    while i + config.train_size + config.test_size <= len(data):
        train = data.iloc[i : i + config.train_size]
        test  = data.iloc[i + config.train_size : i + config.train_size + config.test_size]

        effective_train = train.iloc[warmup:]

        if len(effective_train) == 0:
            logger.warning(
                "Window %d: warmup_periods (%d) >= train_size (%d), skipping",
                window_index, warmup, config.train_size,
            )
            i += config.step_size
            window_index += 1
            continue

        strategy.fit(effective_train)
        signals = strategy.generate_signals(test)
        trades = run_backtest(test, signals, holding, fee_model)

        if len(trades) < config.min_trades_per_window:
            logger.warning(
                "Window %d: only %d trade(s), minimum is %d, skipping",
                window_index, len(trades), config.min_trades_per_window,
            )
            i += config.step_size
            window_index += 1
            continue

        results.append(BacktestResult(
            trades=trades,
            metrics={},
            config={
                "window": window_index,
                "train_start": str(train.index[0]),
                "train_end": str(train.index[-1]),
                "test_start": str(test.index[0]),
                "test_end": str(test.index[-1]),
            },
            is_oos=True,
            signals_generated=len(signals),
            signals_executed=len(trades),
        ))

        i += config.step_size
        window_index += 1

    return results
