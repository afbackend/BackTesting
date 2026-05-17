import pytest
import pandas as pd

from backtester.strategy.base import Strategy
from backtester.core.bt_types import Direction, Signal


class ConcreteStrategy(Strategy):
    """Minimal valid implementation for testing the interface."""

    def name(self) -> str:
        return "concrete_strategy"

    def warmup_periods(self) -> int:
        return 10

    def holding_periods(self) -> int:
        return 4

    def fit(self, train_data: pd.DataFrame) -> None:
        self.threshold = train_data["close"].mean()

    def generate_signals(self, data: pd.DataFrame):
        return [
            Signal(timestamp=ts, direction=Direction.LONG)
            for ts, row in data.iterrows()
            if row["close"] > self.threshold
        ]


class IncompleteStrategy(Strategy):
    """Missing generate_signals — instantiation must fail."""

    def name(self) -> str:
        return "incomplete"

    def warmup_periods(self) -> int:
        return 0

    def holding_periods(self) -> int:
        return 1

    def fit(self, train_data: pd.DataFrame) -> None:
        pass


def test_concrete_strategy_instantiates():
    s = ConcreteStrategy()
    assert s.name() == "concrete_strategy"
    assert s.warmup_periods() == 10
    assert s.holding_periods() == 4


def test_incomplete_strategy_raises_on_instantiation():
    with pytest.raises(TypeError):
        IncompleteStrategy()


def test_generate_signals_returns_list_of_signals():
    s = ConcreteStrategy()
    idx = pd.date_range("2024-01-01", periods=10, freq="4h")
    data = pd.DataFrame({"close": [100.0] * 10}, index=idx)

    s.fit(data)
    signals = s.generate_signals(data)

    assert isinstance(signals, list)
    assert all(isinstance(sig, Signal) for sig in signals)


def test_signals_use_direction_enum():
    s = ConcreteStrategy()
    idx = pd.date_range("2024-01-01", periods=5, freq="4h")
    data = pd.DataFrame({"close": [90.0, 90.0, 110.0, 110.0, 110.0]}, index=idx)

    s.fit(data)
    signals = s.generate_signals(data)

    assert all(sig.direction in list(Direction) for sig in signals)
