from backtester.config import FrameworkConfig


def test_defaults():
    cfg = FrameworkConfig()
    assert cfg.taker_fee == 0.001
    assert cfg.slippage_estimate == 0.0
    assert cfg.train_size == 500
    assert cfg.test_size == 100
    assert cfg.step_size == 100
    assert cfg.min_trades_per_window == 10
    assert cfg.p_value_threshold == 0.05
    assert cfg.min_win_rate == 0.52
    assert cfg.max_acceptable_drawdown == -0.25
    assert cfg.random_entry_simulations == 100
    assert cfg.random_seed == 42


def test_override_fields():
    cfg = FrameworkConfig(taker_fee=0.002, train_size=1000, random_seed=0)
    assert cfg.taker_fee == 0.002
    assert cfg.train_size == 1000
    assert cfg.random_seed == 0
    # Non-overridden fields keep defaults
    assert cfg.test_size == 100
    assert cfg.min_win_rate == 0.52
