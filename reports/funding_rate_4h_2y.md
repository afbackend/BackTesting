# funding_rate — Validation Report

Generated: 2026-05-23  |  Windows: 3  |  Total trades: 42

## 1. Configuration

**Data**

- start: 2024-05-03
- end: 2026-05-03
- total_candles: 4380

**Fee model**

- taker_fee: 0.0010
- slippage_estimate: 0.0000
- round_trip: 0.0020

**Walk-forward**

- train_size: 500
- test_size: 100
- step_size: 100
- min_trades_per_window: 10

**Thresholds**

- p_value_threshold: 0.05
- min_win_rate: 52%
- max_acceptable_drawdown: -25%

## 2. Walk-Forward Results

| Window | Test Period | Trades | Win Rate | Mean Return | p-value | Sig |
|--------|------------|--------|----------|-------------|---------|-----|
| 5 | 2024-10-17 → 2024-11-02 | 16 | 31.2% | -0.365% | 0.859 | ✗ |
| 7 | 2024-11-19 → 2024-12-06 | 11 | 27.3% | -1.085% | 0.937 | ✗ |
| 21 | 2025-07-11 → 2025-07-27 | 15 | 46.7% | 0.059% | 0.437 | ✗ |

## 3. Statistical Significance

**Global (all OOS trades combined)**

- t-statistic: -1.5978
- p-value (one-tailed): 0.9411
- significant: No
- 95% CI: (-0.00911, 0.00106)
- n_trades: 42

**Significant windows: 0 / 3 (0%)**

## 4. Risk Metrics

- max_drawdown: -17.69%
- capital_min: 0.8231
- max_consecutive_loss: 4
- max_adverse_excursion: 0.0708
- mean_adverse_excursion: 0.0144
- p90_adverse_excursion: 0.0319

## 5. Baseline Comparison

**Buy-and-hold (one trade per window, with fee)**

- mean per-window return: 3.639%
- total compounded return: 11.274%

**Random entry (100 simulations, matched trade count and holding)**

- mean total return: -7.061%
- median total return: -8.652%
- p5–p95 range: -21.661% to 14.286%
- p-value (strategy vs random, one-tailed): 0.8100

## 6. Conclusion

| Criterion | Value | Threshold | Pass |
|-----------|-------|-----------|------|
| p-value < threshold | 0.9411 | 0.0500 | ✗ |
| win_rate > min_win_rate | 35.7% | 52.0% | ✗ |
| mean_return > 0 | -0.4023% | 0.0000% | ✗ |
| max_drawdown within limit | -17.69% | -25.00% | ✓ |
| significant_windows >= 60% | 0% | 60% | ✗ |
| beats random entry (p < 0.05) | 0.8100 | 0.0500 | ✗ |

### Verdict: **REJECTED** ✗
