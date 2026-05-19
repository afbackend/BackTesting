from math import sqrt
from typing import List

import numpy as np

from backtester.core.bt_types import Trade

_SECONDS_PER_YEAR = 365.25 * 86400


def calculate_performance(trades: List[Trade]) -> dict:
    """
    Compute performance metrics from a list of trades.

    Sharpe and Sortino are annualized using calendar time:
        ann_factor = sqrt(n_trades / years_elapsed)
    where years_elapsed = (last_exit - first_entry).total_seconds() / SECONDS_PER_YEAR.
    This correctly handles strategies with irregular trade frequency and avoids
    the methodological error of treating every candle as a period.

    trades_per_day uses total_seconds() to handle sub-day spans correctly;
    returns float('inf') when span is zero.

    Returns zeros for empty trade lists. profit_factor is float('inf')
    when there are no losing trades — mathematically correct, not a sentinel.
    """
    if not trades:
        return _empty_metrics()

    returns = np.array([t.net_return for t in trades])
    total = len(trades)
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if total > 1 else 0.0

    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0

    span_seconds = (trades[-1].exit_time - trades[0].entry_time).total_seconds()
    span_days = span_seconds / 86400
    trades_per_day = total / span_days if span_days > 0 else float("inf")

    if span_seconds > 0:
        ann = sqrt(total * _SECONDS_PER_YEAR / span_seconds)
    else:
        ann = 0.0
    sharpe = (mean / std * ann) if std > 0 else 0.0
    sortino = (mean / downside_std * ann) if downside_std > 0 else 0.0

    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    sum_losses = float(np.sum(losses))

    profit_factor = (
        float(np.sum(wins)) / abs(sum_losses)
        if sum_losses < 0
        else float("inf")
    )

    win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf") if avg_win > 0 else 0.0

    return {
        "total_trades": total,
        "win_rate": len(wins) / total,
        "mean_return": mean,
        "median_return": float(np.median(returns)),
        "total_return": float(np.prod(1 + returns) - 1),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "trades_per_day": trades_per_day,
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0,
        "win_rate": 0.0,
        "mean_return": 0.0,
        "median_return": 0.0,
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "win_loss_ratio": 0.0,
        "trades_per_day": 0.0,
    }
