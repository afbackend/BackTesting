import tempfile
from pathlib import Path

import pandas as pd
import pytest

import backtester.cli as cli
from backtester.core.bt_types import Direction, Signal


# --- stub strategy ---

class StubStrategy:
    def warmup_periods(self): return 0
    def holding_periods(self): return 2
    def fit(self, train_data): pass
    def generate_signals(self, data):
        return [Signal(timestamp=ts, direction=Direction.LONG) for ts in data.index]


def make_parquet(tmp_path: Path, n: int = 300, drift: float = 5.0) -> Path:
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    prices = [100.0 + i * drift for i in range(n)]
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=idx,
    )
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    return p


# --- --list ---

def test_list_empty_registry(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {})
    rc = cli.main(["--list"])
    assert rc == 0
    assert "No strategies registered" in capsys.readouterr().out


def test_list_with_registered_strategy(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    rc = cli.main(["--list"])
    assert rc == 0
    assert "stub" in capsys.readouterr().out


# --- missing required args ---

def test_missing_data_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    rc = cli.main(["--strategy", "stub"])
    assert rc == 1


def test_missing_strategy_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    p = make_parquet(tmp_path)
    rc = cli.main(["--data", str(p)])
    assert rc == 1


# --- error cases ---

def test_unknown_strategy_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {})
    p = make_parquet(tmp_path)
    rc = cli.main(["--strategy", "nonexistent", "--data", str(p)])
    assert rc == 1
    assert "unknown strategy" in capsys.readouterr().err


def test_nonexistent_file_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    rc = cli.main(["--strategy", "stub", "--data", "/no/such/file.parquet"])
    assert rc == 1


# --- valid run ---

def test_valid_run_returns_0(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    p = make_parquet(tmp_path)
    rc = cli.main(["--strategy", "stub", "--data", str(p),
                   "--train", "100", "--test", "50", "--step", "50"])
    assert rc == 0


def test_valid_run_prints_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    p = make_parquet(tmp_path)
    cli.main(["--strategy", "stub", "--data", str(p),
              "--train", "100", "--test", "50", "--step", "50"])
    out = capsys.readouterr().out
    assert "Strategy" in out
    assert "Win rate" in out


# --- --report ---

def test_report_flag_saves_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    p = make_parquet(tmp_path)
    out_path = tmp_path / "out.md"
    rc = cli.main(["--strategy", "stub", "--data", str(p),
                   "--train", "100", "--test", "50", "--step", "50",
                   "--report", "--output", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    assert len(out_path.read_text()) > 0


# --- overrides ---

def test_fee_override(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    p = make_parquet(tmp_path)
    rc = cli.main(["--strategy", "stub", "--data", str(p),
                   "--train", "100", "--test", "50", "--step", "50",
                   "--fee", "0.005"])
    assert rc == 0


def test_train_test_overrides(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    p = make_parquet(tmp_path)
    rc = cli.main(["--strategy", "stub", "--data", str(p),
                   "--train", "150", "--test", "75", "--step", "75"])
    assert rc == 0


def test_invalid_data_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    # Parquet with NaN in close → DataValidationError
    idx = pd.date_range("2024-01-01", periods=10, freq="4h")
    import numpy as np
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": float("nan")},
        index=idx,
    )
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    rc = cli.main(["--strategy", "stub", "--data", str(p)])
    assert rc == 1
    assert "Data error" in capsys.readouterr().err


def test_no_valid_windows_prints_message(monkeypatch, tmp_path, capsys):
    # Data too small for any walk-forward window → 0 results
    monkeypatch.setattr(cli, "_REGISTRY", {"stub": StubStrategy})
    p = make_parquet(tmp_path, n=50)  # smaller than default train_size=500
    cli.main(["--strategy", "stub", "--data", str(p)])
    assert "No valid windows found" in capsys.readouterr().out
