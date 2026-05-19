import math
import pytest
import numpy as np
import pandas as pd

from backtester.core.bt_types import Direction, Trade
from backtester.metrics.performance import _SECONDS_PER_YEAR, calculate_performance


def make_trade(net_return: float, days_offset: int = 0) -> Trade:
    base = pd.Timestamp("2024-01-01")
    gross = net_return + 0.002
    return Trade(
        entry_time=base + pd.Timedelta(days=days_offset),
        exit_time=base + pd.Timedelta(days=days_offset + 1),
        direction=Direction.LONG,
        entry_price=100.0,
        exit_price=100.0 * (1 + gross),
        gross_return=gross,
        fee=0.002,
        net_return=net_return,
        holding_candles=4,
        max_adverse=0.0,
    )


# --- empty ---

def test_empty_trades_returns_zeros():
    m = calculate_performance([])
    assert m["total_trades"] == 0
    assert m["win_rate"] == 0.0
    assert m["profit_factor"] == 0.0


# --- basic counts ---

def test_total_trades():
    trades = [make_trade(0.01, i) for i in range(5)]
    assert calculate_performance(trades)["total_trades"] == 5


def test_win_rate_all_wins():
    trades = [make_trade(0.01, i) for i in range(4)]
    assert calculate_performance(trades)["win_rate"] == pytest.approx(1.0)


def test_win_rate_mixed():
    trades = [make_trade(0.01, 0), make_trade(-0.01, 1),
              make_trade(0.01, 2), make_trade(-0.01, 3)]
    assert calculate_performance(trades)["win_rate"] == pytest.approx(0.5)


def test_win_rate_all_losses():
    trades = [make_trade(-0.01, i) for i in range(4)]
    assert calculate_performance(trades)["win_rate"] == pytest.approx(0.0)


# --- returns ---

def test_mean_return():
    trades = [make_trade(0.02, 0), make_trade(-0.01, 1), make_trade(0.03, 2)]
    assert calculate_performance(trades)["mean_return"] == pytest.approx(
        (0.02 - 0.01 + 0.03) / 3
    )


def test_median_return():
    trades = [make_trade(0.01, 0), make_trade(0.05, 1), make_trade(0.02, 2)]
    assert calculate_performance(trades)["median_return"] == pytest.approx(0.02)


def test_total_return_is_compound():
    trades = [make_trade(0.10, 0), make_trade(0.10, 1)]
    expected = (1.10 * 1.10) - 1  # 0.21
    assert calculate_performance(trades)["total_return"] == pytest.approx(expected)


def test_total_return_compound_differs_from_arithmetic():
    trades = [make_trade(0.10, 0), make_trade(-0.10, 1)]
    # Arithmetic: 0.10 + (-0.10) = 0.0
    # Compound: 1.10 * 0.90 - 1 = -0.01
    m = calculate_performance(trades)
    assert m["total_return"] == pytest.approx(-0.01)
    assert m["total_return"] != pytest.approx(0.0)


# --- edge cases ---

def test_profit_factor_no_losses_is_inf():
    trades = [make_trade(0.01, i) for i in range(3)]
    assert calculate_performance(trades)["profit_factor"] == float("inf")


def test_profit_factor_no_wins_is_zero():
    trades = [make_trade(-0.01, i) for i in range(3)]
    assert calculate_performance(trades)["profit_factor"] == pytest.approx(0.0)


def test_profit_factor_mixed():
    trades = [make_trade(0.02, 0), make_trade(-0.01, 1)]
    assert calculate_performance(trades)["profit_factor"] == pytest.approx(2.0)


def test_avg_win_no_wins_is_zero():
    trades = [make_trade(-0.01, i) for i in range(3)]
    assert calculate_performance(trades)["avg_win"] == 0.0


def test_avg_loss_no_losses_is_zero():
    trades = [make_trade(0.01, i) for i in range(3)]
    assert calculate_performance(trades)["avg_loss"] == 0.0


def test_win_loss_ratio_no_losses_is_inf():
    trades = [make_trade(0.01, i) for i in range(3)]
    assert calculate_performance(trades)["win_loss_ratio"] == float("inf")


def test_win_loss_ratio_no_wins_is_zero():
    trades = [make_trade(-0.01, i) for i in range(3)]
    assert calculate_performance(trades)["win_loss_ratio"] == 0.0


def test_sharpe_zero_std_is_zero():
    # All returns identical → std=0
    trades = [make_trade(0.01, i) for i in range(3)]
    # std of [0.01, 0.01, 0.01] = 0
    m = calculate_performance(trades)
    assert m["sharpe_ratio"] == 0.0


def test_sharpe_positive_edge():
    trades = [make_trade(0.01, i) for i in range(0, 20)]
    trades += [make_trade(-0.001, i) for i in range(20, 30)]
    m = calculate_performance(trades)
    assert m["sharpe_ratio"] > 0


def test_sortino_zero_downside_std_is_zero():
    # All positive returns → no downside → sortino = 0
    trades = [make_trade(0.01, 0), make_trade(0.02, 1), make_trade(0.03, 2)]
    assert calculate_performance(trades)["sortino_ratio"] == 0.0


# --- trades_per_day ---

def test_trades_per_day():
    # 2 trades over 10 days → 0.2 trades/day
    trades = [make_trade(0.01, 0), make_trade(0.01, 10)]
    m = calculate_performance(trades)
    # span = exit_time[-1] - entry_time[0] = day11 - day0 = 11 days
    assert m["trades_per_day"] == pytest.approx(2 / 11)


def test_trades_per_day_sub_day():
    # Two trades with total span of 8 hours.
    # Old bug: (exit[-1] - entry[0]).days == 0 → float(2) = 2.0 (wrong)
    # Fix: total_seconds() / 86400 = 0.333 days → 2 / 0.333 ≈ 6.0
    base = pd.Timestamp("2024-01-01")
    gross = 0.012
    def _trade(entry_h, exit_h):
        return Trade(
            entry_time=base + pd.Timedelta(hours=entry_h),
            exit_time=base + pd.Timedelta(hours=exit_h),
            direction=Direction.LONG,
            entry_price=100.0,
            exit_price=100.0 * (1 + gross),
            gross_return=gross,
            fee=0.002,
            net_return=0.01,
            holding_candles=2,
            max_adverse=0.0,
        )
    trades = [_trade(0, 4), _trade(4, 8)]
    m = calculate_performance(trades)
    span_days = 8 / 24
    assert m["trades_per_day"] == pytest.approx(2 / span_days)


def test_trades_per_day_zero_span_returns_inf():
    base = pd.Timestamp("2024-01-01")
    gross = 0.012
    t = Trade(
        entry_time=base,
        exit_time=base,
        direction=Direction.LONG,
        entry_price=100.0,
        exit_price=100.0 * (1 + gross),
        gross_return=gross,
        fee=0.002,
        net_return=0.01,
        holding_candles=0,
        max_adverse=0.0,
    )
    assert calculate_performance([t])["trades_per_day"] == float("inf")


# --- Sharpe calendar-time annualization ---

def test_sharpe_uses_calendar_time_annualization():
    trades = [make_trade(0.02, i) for i in range(5)]
    trades += [make_trade(-0.005, i) for i in range(5, 10)]
    m = calculate_performance(trades)

    returns = np.array([t.net_return for t in trades])
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    span_seconds = (trades[-1].exit_time - trades[0].entry_time).total_seconds()
    ann = math.sqrt(len(trades) * _SECONDS_PER_YEAR / span_seconds)
    expected = mean_r / std_r * ann

    assert m["sharpe_ratio"] == pytest.approx(expected, rel=1e-6)
