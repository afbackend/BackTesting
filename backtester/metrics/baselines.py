from typing import Dict, List

import numpy as np
import pandas as pd

from backtester.core.backtest_engine import run_backtest
from backtester.core.bt_types import BacktestResult, Direction, Signal
from backtester.core.fee_model import FeeModel


def buy_and_hold(
    results: List[BacktestResult],
    data: pd.DataFrame,
    fee_model: FeeModel,
) -> Dict:
    """
    For each test window in results, compute buy-and-hold net return:
    enter LONG at first candle's open, exit at last candle's close, apply round-trip fee.

    Returns per-window net returns, compounded total, and arithmetic mean.
    """
    if not results:
        return _empty_buy_and_hold()

    per_window: List[float] = []
    for r in results:
        window = _window_slice(r, data)
        if window is None or len(window) < 2:
            continue
        entry = float(window.iloc[0]["open"])
        exit_price = float(window.iloc[-1]["close"])
        gross = (exit_price - entry) / entry
        per_window.append(fee_model.apply(gross))

    if not per_window:
        return _empty_buy_and_hold()

    total = float(np.prod([1 + r for r in per_window]) - 1)
    return {
        "per_window_returns": per_window,
        "total_return": total,
        "mean_return": float(np.mean(per_window)),
    }


def random_entry(
    results: List[BacktestResult],
    data: pd.DataFrame,
    fee_model: FeeModel,
    n_simulations: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Run n_simulations of random-timing entries. Each window uses the same holding
    period as the strategy and oversamples entry candidates 2x (overlapping signals
    are dropped by the engine).

    Returns aggregate stats on simulation totals and a one-tailed Monte Carlo
    p-value: P(random_total >= strategy_total).
    """
    if not results or not any(r.trades for r in results):
        return _empty_random_entry(n_simulations)

    holding = next(t.holding_candles for r in results for t in r.trades)
    sim_totals: List[float] = []

    for sim in range(n_simulations):
        rng = np.random.default_rng(seed + sim)
        sim_window_returns: List[float] = []

        for r in results:
            window = _window_slice(r, data)
            if window is None:
                continue
            n_target = len(r.trades)
            available = len(window) - holding - 1
            if n_target == 0 or available <= 0:
                continue

            n_signals = min(n_target * 2, available)
            indices = sorted(rng.choice(available, size=n_signals, replace=False))
            directions = rng.choice(
                [Direction.LONG, Direction.SHORT], size=n_signals
            )
            signals = [
                Signal(timestamp=window.index[i], direction=d)
                for i, d in zip(indices, directions)
            ]
            trades = run_backtest(window, signals, holding, fee_model)
            if trades:
                wr = float(np.prod([1 + t.net_return for t in trades]) - 1)
                sim_window_returns.append(wr)

        if sim_window_returns:
            sim_totals.append(
                float(np.prod([1 + r for r in sim_window_returns]) - 1)
            )

    if not sim_totals:
        return _empty_random_entry(n_simulations)

    strat_total = _strategy_total_return(results)
    p_value = float(np.mean([t >= strat_total for t in sim_totals]))

    arr = np.array(sim_totals)
    return {
        "n_simulations": n_simulations,
        "mean_total_return": float(np.mean(arr)),
        "median_total_return": float(np.median(arr)),
        "p5_total_return": float(np.percentile(arr, 5)),
        "p95_total_return": float(np.percentile(arr, 95)),
        "p_value_vs_strategy": p_value,
    }


def _window_slice(r: BacktestResult, data: pd.DataFrame):
    try:
        test_start = pd.Timestamp(r.config["test_start"])
        test_end = pd.Timestamp(r.config["test_end"])
    except (KeyError, ValueError):
        return None
    return data.loc[test_start:test_end]


def _strategy_total_return(results: List[BacktestResult]) -> float:
    returns = [t.net_return for r in results for t in r.trades]
    if not returns:
        return 0.0
    return float(np.prod([1 + r for r in returns]) - 1)


def _empty_buy_and_hold() -> Dict:
    return {"per_window_returns": [], "total_return": 0.0, "mean_return": 0.0}


def _empty_random_entry(n_simulations: int) -> Dict:
    return {
        "n_simulations": n_simulations,
        "mean_total_return": 0.0,
        "median_total_return": 0.0,
        "p5_total_return": 0.0,
        "p95_total_return": 0.0,
        "p_value_vs_strategy": 1.0,
    }
