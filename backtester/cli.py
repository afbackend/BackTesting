import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from backtester.config import FrameworkConfig
from backtester.core.fee_model import FeeModel
from backtester.core.walk_forward import WalkForwardConfig, walk_forward
from backtester.data.loader import load_parquet
from backtester.data.validator import DataValidationError, validate
from backtester.metrics.performance import calculate_performance
from backtester.metrics.risk import calculate_risk
from backtester.metrics.statistical import calculate_statistical
from backtester.report.generator import generate_report

# Populated when strategy modules are imported.
# Example: _REGISTRY["funding_rate"] = FundingRateStrategy
_REGISTRY: Dict[str, Type] = {}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns exit code: 0 = success, 1 = error."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _print_strategies()
        return 0

    if not args.data:
        print("Error: --data is required.", file=sys.stderr)
        return 1

    if not args.strategy:
        print("Error: --strategy is required.", file=sys.stderr)
        return 1

    return _run(args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Walk-forward backtesting framework.",
    )
    p.add_argument("--strategy", help="Strategy name (see --list).")
    p.add_argument("--data", help="Path to OHLC parquet file.")
    p.add_argument("--report", action="store_true", help="Save markdown report.")
    p.add_argument("--output", default="report.md", help="Report output path (default: report.md).")
    p.add_argument("--train", type=int, help="Train window size (candles).")
    p.add_argument("--test", type=int, help="Test window size (candles).")
    p.add_argument("--step", type=int, help="Step size (candles).")
    p.add_argument("--fee", type=float, help="Taker fee override.")
    p.add_argument("--list", action="store_true", dest="list", help="List available strategies.")
    return p


def _print_strategies() -> None:
    if not _REGISTRY:
        print("No strategies registered.")
        return
    print("Available strategies:")
    for name in sorted(_REGISTRY):
        print(f"  {name}")


def _run(args: argparse.Namespace) -> int:
    defaults = FrameworkConfig()
    config = FrameworkConfig(
        taker_fee=args.fee if args.fee is not None else defaults.taker_fee,
        train_size=args.train if args.train is not None else defaults.train_size,
        test_size=args.test if args.test is not None else defaults.test_size,
        step_size=args.step if args.step is not None else defaults.step_size,
    )

    try:
        df = load_parquet(args.data)
        validate(df)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (DataValidationError, ValueError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1

    if args.strategy not in _REGISTRY:
        print(
            f"Error: unknown strategy '{args.strategy}'. "
            "Use --list to see available strategies.",
            file=sys.stderr,
        )
        return 1

    strategy = _REGISTRY[args.strategy]()
    fee_model = FeeModel(taker_fee=config.taker_fee, slippage_estimate=config.slippage_estimate)
    wf_config = WalkForwardConfig(
        train_size=config.train_size,
        test_size=config.test_size,
        step_size=config.step_size,
        min_trades_per_window=config.min_trades_per_window,
    )

    results = walk_forward(df, strategy, fee_model, wf_config)
    _print_summary(results, args.strategy)

    if args.report:
        data_info = {
            "start": str(df.index[0])[:10],
            "end": str(df.index[-1])[:10],
            "total_candles": len(df),
        }
        report_text = generate_report(results, args.strategy, config, data_info=data_info)
        Path(args.output).write_text(report_text)
        print(f"Report saved to {args.output}")

    return 0


def _print_summary(results: list, strategy_name: str) -> None:
    total_trades = sum(len(r.trades) for r in results)
    print(f"\nStrategy : {strategy_name}")
    print(f"Windows  : {len(results)}  |  Total trades: {total_trades}")

    if not results:
        print("No valid windows found.")
        return

    all_trades = [t for r in results for t in r.trades]
    perf = calculate_performance(all_trades)
    risk = calculate_risk(all_trades)
    stat = calculate_statistical(all_trades)

    print(
        f"Win rate : {perf['win_rate']:.1%}  "
        f"Mean return: {perf['mean_return']:.3%}  "
        f"Sharpe: {perf['sharpe_ratio']:.2f}"
    )
    print(
        f"Drawdown : {risk['max_drawdown']:.2%}  "
        f"p-value: {stat['p_value']:.4f}  "
        f"Significant: {'Yes' if stat['is_significant'] else 'No'}"
    )
