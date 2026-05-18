from dataclasses import dataclass


@dataclass
class FrameworkConfig:
    taker_fee: float = 0.001
    slippage_estimate: float = 0.0
    train_size: int = 500
    test_size: int = 100
    step_size: int = 100
    min_trades_per_window: int = 10
    p_value_threshold: float = 0.05
    min_win_rate: float = 0.52
    max_acceptable_drawdown: float = -0.25
    random_entry_simulations: int = 100
    random_seed: int = 42
