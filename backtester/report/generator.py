from datetime import date
from typing import Dict, List, Optional

from backtester.config import FrameworkConfig
from backtester.core.bt_types import BacktestResult
from backtester.metrics.performance import calculate_performance
from backtester.metrics.risk import calculate_risk
from backtester.metrics.statistical import calculate_statistical


def generate_report(
    results: List[BacktestResult],
    strategy_name: str,
    config: FrameworkConfig,
    data_info: Optional[Dict] = None,
) -> str:
    """
    Generate a markdown validation report from walk-forward results.

    data_info optional keys: symbol, timeframe, start, end, total_candles.
    Returns a markdown string; the caller decides where to write it.
    """
    sections = [_header(strategy_name, results)]
    sections.append(_section_configuration(config, data_info))

    if not results:
        sections.append(
            "> No valid walk-forward windows found. "
            "Check min_trades_per_window and data length.\n"
        )
        return "\n".join(sections)

    sections.append(_section_walk_forward(results))
    sections.append(_section_statistical(results, config))
    sections.append(_section_risk(results))
    sections.append(_section_baselines())
    sections.append(_section_conclusion(results, config))

    return "\n".join(sections)


# --- sections ---

def _header(strategy_name: str, results: List[BacktestResult]) -> str:
    total_trades = sum(len(r.trades) for r in results)
    return (
        f"# {strategy_name} — Validation Report\n\n"
        f"Generated: {date.today()}  |  "
        f"Windows: {len(results)}  |  "
        f"Total trades: {total_trades}\n"
    )


def _section_configuration(config: FrameworkConfig, data_info: Optional[Dict]) -> str:
    lines = ["## 1. Configuration\n"]

    if data_info:
        lines.append("**Data**\n")
        for key in ("symbol", "timeframe", "start", "end", "total_candles"):
            if key in data_info:
                lines.append(f"- {key}: {data_info[key]}")
        lines.append("")

    lines.append("**Fee model**\n")
    round_trip = (config.taker_fee + config.slippage_estimate) * 2
    lines.append(f"- taker_fee: {config.taker_fee:.4f}")
    lines.append(f"- slippage_estimate: {config.slippage_estimate:.4f}")
    lines.append(f"- round_trip: {round_trip:.4f}")
    lines.append("")

    lines.append("**Walk-forward**\n")
    lines.append(f"- train_size: {config.train_size}")
    lines.append(f"- test_size: {config.test_size}")
    lines.append(f"- step_size: {config.step_size}")
    lines.append(f"- min_trades_per_window: {config.min_trades_per_window}")
    lines.append("")

    lines.append("**Thresholds**\n")
    lines.append(f"- p_value_threshold: {config.p_value_threshold}")
    lines.append(f"- min_win_rate: {config.min_win_rate:.0%}")
    lines.append(f"- max_acceptable_drawdown: {config.max_acceptable_drawdown:.0%}")
    lines.append("")

    return "\n".join(lines)


def _section_walk_forward(results: List[BacktestResult]) -> str:
    lines = ["## 2. Walk-Forward Results\n"]
    lines.append("| Window | Test Period | Trades | Win Rate | Mean Return | p-value | Sig |")
    lines.append("|--------|------------|--------|----------|-------------|---------|-----|")

    for r in results:
        w = r.config.get("window", "—")
        test_start = r.config.get("test_start", "")[:10]
        test_end = r.config.get("test_end", "")[:10]
        period = f"{test_start} → {test_end}"
        perf = r.metrics["performance"]
        stat = r.metrics["statistical"]
        n = len(r.trades)
        wr = f"{perf['win_rate']:.1%}"
        mr = f"{perf['mean_return']:.3%}"
        pv = f"{stat['p_value']:.3f}"
        sig = "✓" if stat["is_significant"] else "✗"
        lines.append(f"| {w} | {period} | {n} | {wr} | {mr} | {pv} | {sig} |")

    lines.append("")
    return "\n".join(lines)


def _section_statistical(results: List[BacktestResult], config: FrameworkConfig) -> str:
    all_trades = [t for r in results for t in r.trades]
    gs = calculate_statistical(all_trades, p_threshold=config.p_value_threshold)

    sig_windows = sum(1 for r in results if r.metrics["statistical"]["is_significant"])
    sig_pct = sig_windows / len(results) if results else 0.0

    ci = gs["confidence_interval_95"]
    lines = [
        "## 3. Statistical Significance\n",
        "**Global (all OOS trades combined)**\n",
        f"- t-statistic: {gs['t_statistic']:.4f}",
        f"- p-value (one-tailed): {gs['p_value']:.4f}",
        f"- significant: {'Yes' if gs['is_significant'] else 'No'}",
        f"- 95% CI: ({ci[0]:.5f}, {ci[1]:.5f})",
        f"- n_trades: {gs['n_trades']}",
        "",
        f"**Significant windows: {sig_windows} / {len(results)} "
        f"({sig_pct:.0%})**\n",
    ]
    return "\n".join(lines)


def _section_risk(results: List[BacktestResult]) -> str:
    all_trades = [t for r in results for t in r.trades]
    gr = calculate_risk(all_trades)

    lines = [
        "## 4. Risk Metrics\n",
        f"- max_drawdown: {gr['max_drawdown']:.2%}",
        f"- capital_min: {gr['capital_min']:.4f}",
        f"- max_consecutive_loss: {gr['max_consecutive_loss']}",
        f"- max_adverse_excursion: {gr['max_adverse_excursion']:.4f}",
        f"- mean_adverse_excursion: {gr['mean_adverse_excursion']:.4f}",
        f"- p90_adverse_excursion: {gr['p90_adverse_excursion']:.4f}",
        "",
    ]
    return "\n".join(lines)


def _section_baselines() -> str:
    return "## 5. Baseline Comparison\n\n> Baselines not implemented in v1.0.\n"


def _section_conclusion(results: List[BacktestResult], config: FrameworkConfig) -> str:
    all_trades = [t for r in results for t in r.trades]
    gp = calculate_performance(all_trades)
    gr = calculate_risk(all_trades)
    gs = calculate_statistical(all_trades, p_threshold=config.p_value_threshold)

    criteria = [
        (
            "p-value < threshold",
            f"{gs['p_value']:.4f}",
            f"{config.p_value_threshold:.4f}",
            gs["p_value"] < config.p_value_threshold,
        ),
        (
            "win_rate > min_win_rate",
            f"{gp['win_rate']:.1%}",
            f"{config.min_win_rate:.1%}",
            gp["win_rate"] > config.min_win_rate,
        ),
        (
            "mean_return > 0",
            f"{gp['mean_return']:.4%}",
            "0.0000%",
            gp["mean_return"] > 0,
        ),
        (
            "max_drawdown within limit",
            f"{gr['max_drawdown']:.2%}",
            f"{config.max_acceptable_drawdown:.2%}",
            gr["max_drawdown"] > config.max_acceptable_drawdown,
        ),
    ]

    all_pass = all(c[3] for c in criteria)
    verdict = "**APPROVED** ✓" if all_pass else "**REJECTED** ✗"

    lines = [
        "## 6. Conclusion\n",
        "| Criterion | Value | Threshold | Pass |",
        "|-----------|-------|-----------|------|",
    ]
    for name, value, threshold, passed in criteria:
        mark = "✓" if passed else "✗"
        lines.append(f"| {name} | {value} | {threshold} | {mark} |")

    lines.append("")
    lines.append(f"### Verdict: {verdict}\n")

    return "\n".join(lines)
