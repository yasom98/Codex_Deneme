"""Tests for atomic write behavior and health gate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import PipelineConfig
from core.io_atomic import atomic_write_parquet
from data.standardize import standardize_file


def test_atomic_write_parquet_success(monkeypatch: object, tmp_path: Path) -> None:
    df = pd.DataFrame({"timestamp": ["2024-01-01"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
    dest = tmp_path / "sample.parquet"
    tmp_dest = dest.with_suffix(".parquet.tmp")

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    atomic_write_parquet(df, dest)

    assert dest.exists()
    assert not tmp_dest.exists()


def test_atomic_write_parquet_cleans_tmp_on_failure(monkeypatch: object, tmp_path: Path) -> None:
    df = pd.DataFrame({"timestamp": ["2024-01-01"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
    dest = tmp_path / "broken.parquet"
    tmp_dest = dest.with_suffix(".parquet.tmp")

    def fake_to_parquet_fail(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet_fail)

    try:
        atomic_write_parquet(df, dest)
    except RuntimeError:
        pass

    assert not dest.exists()
    assert not tmp_dest.exists()


def test_health_gate_blocks_output_write(tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    reports_root = tmp_path / "reports"
    input_root.mkdir(parents=True, exist_ok=True)

    bad_csv = input_root / "bad.csv"
    bad_csv.write_text(
        "timestamp,open,high,low,close\n2024-01-01 00:00:00,1,2,0.5,1.5\n",
        encoding="utf-8",
    )

    cfg = PipelineConfig(
        input_root=input_root,
        output_root=output_root,
        reports_root=reports_root,
        csv_glob="**/*.csv",
        timestamp_aliases=("timestamp", "ts", "date", "datetime", "time", "candle_time", "open_time", "close_time"),
        required_columns=("open", "high", "low", "close", "volume"),
        float_columns=("open", "high", "low", "close", "volume"),
        fail_on_critical=True,
        duplicate_policy="last",
        seed=42,
    )

    report = standardize_file(bad_csv, cfg=cfg, dry_run=False)
    expected_out = output_root / "bad.parquet"
    expected_report = reports_root / "bad.health.json"

    assert not report.success
    assert not expected_out.exists()
    assert expected_report.exists()
