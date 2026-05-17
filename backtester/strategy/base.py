from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from backtester.core.bt_types import Signal


class Strategy(ABC):
    """
    Interface every strategy must implement.

    The framework guarantees:
    - fit() receives ONLY training data.
    - generate_signals() receives test data with no future returns.
    - The strategy NEVER sees test data during fit().
    """

    @abstractmethod
    def name(self) -> str:
        """Unique strategy name used in reports."""

    @abstractmethod
    def warmup_periods(self) -> int:
        """
        Number of candles required before the first valid signal.
        The framework discards these candles from the start of train
        before calling fit(), ensuring calibration uses clean data only.
        """

    @abstractmethod
    def holding_periods(self) -> int:
        """Number of candles to hold after entry."""

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Calibrate strategy parameters using ONLY training data.
        Called once per walk-forward window before generate_signals().
        """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate entry signals for each candle in data.
        Parameters must come from the last fit() call.
        data contains no future returns.
        """
