"""Regression tests for timestamp strategy selection in standardization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import PipelineConfig
from data.standardize import standardize_file


def _build_cfg(input_root: Path, runs_root: Path) -> PipelineConfig:
    return PipelineConfig(
        input_root=input_root,
        runs_root=runs_root,
        csv_glob="**/*.csv",
        timestamp_aliases=("timestamp", "ts", "date", "datetime", "time", "candle_time", "open_time", "close_time"),
        required_columns=("open", "high", "low", "close", "volume"),
        float_columns=("open", "high", "low", "close", "volume"),
        duplicate_policy="last",
        seed=42,
    )


def _patch_parquet(monkeypatch: object) -> None:
    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del index
        self.to_json(path.with_suffix(".json"), orient="records")
        path.write_text("fake-parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)


def test_standardize_selects_epoch_milliseconds(monkeypatch: object, tmp_path: Path) -> None:
    _patch_parquet(monkeypatch)
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)

    src = input_root / "epoch_ms.csv"
    src.write_text(
        "\n".join(
            [
                "Time,Open,High,Low,Price-Close,Volume",
                "1704067200000,10,11,9,10.5,100",
                "1704153600000,11,12,10,11.5,110",
                "1704240000000,12,13,11,12.5,120",
            ]
        ),
        encoding="utf-8",
    )

    report = standardize_file(
        src_csv=src,
        cfg=_build_cfg(input_root, runs_root),
        parquet_root=runs_root / "test_run" / "data_standardized" / "parquet",
        per_file_reports_root=runs_root / "test_run" / "data_standardized" / "reports" / "per_file",
        dry_run=False,
    )

    assert report.status == "success"
    assert report.selected_timestamp_strategy == "epoch_milliseconds"
    assert report.parse_valid_ratio == 1.0
    assert report.unique_days == 3
    assert report.timestamp_min == "2024-01-01T00:00:00+00:00"
    assert report.timestamp_max == "2024-01-03T00:00:00+00:00"


def test_standardize_selects_epoch_seconds(monkeypatch: object, tmp_path: Path) -> None:
    _patch_parquet(monkeypatch)
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)

    src = input_root / "epoch_s.csv"
    src.write_text(
        "\n".join(
            [
                "Time,Open,High,Low,Price-Close,Volume",
                "1704067200,10,11,9,10.5,100",
                "1704153600,11,12,10,11.5,110",
                "1704240000,12,13,11,12.5,120",
            ]
        ),
        encoding="utf-8",
    )

    report = standardize_file(
        src_csv=src,
        cfg=_build_cfg(input_root, runs_root),
        parquet_root=runs_root / "test_run" / "data_standardized" / "parquet",
        per_file_reports_root=runs_root / "test_run" / "data_standardized" / "reports" / "per_file",
        dry_run=False,
    )

    assert report.status == "success"
    assert report.selected_timestamp_strategy == "epoch_seconds"
    assert report.parse_valid_ratio == 1.0
    assert report.unique_days == 3
    assert report.timestamp_min == "2024-01-01T00:00:00+00:00"
    assert report.timestamp_max == "2024-01-03T00:00:00+00:00"


def test_standardize_selects_excel_serial(monkeypatch: object, tmp_path: Path) -> None:
    _patch_parquet(monkeypatch)
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)

    src = input_root / "excel_serial.csv"
    src.write_text(
        "\n".join(
            [
                "Time,Open,High,Low,Price-Close,Volume",
                "45292,10,11,9,10.5,100",
                "45293,11,12,10,11.5,110",
                "45294,12,13,11,12.5,120",
            ]
        ),
        encoding="utf-8",
    )

    report = standardize_file(
        src_csv=src,
        cfg=_build_cfg(input_root, runs_root),
        parquet_root=runs_root / "test_run" / "data_standardized" / "parquet",
        per_file_reports_root=runs_root / "test_run" / "data_standardized" / "reports" / "per_file",
        dry_run=False,
    )

    assert report.status == "success"
    assert report.selected_timestamp_strategy == "excel_serial"
    assert report.parse_valid_ratio == 1.0
    assert report.unique_days == 3
    assert report.timestamp_min == "2024-01-01T00:00:00+00:00"
    assert report.timestamp_max == "2024-01-03T00:00:00+00:00"


def test_standardize_fails_on_malformed_timestamp(tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)

    src = input_root / "malformed_ts.csv"
    src.write_text(
        "\n".join(
            [
                "Time,Open,High,Low,Price-Close,Volume",
                "bad-ts-1,10,11,9,10.5,100",
                "bad-ts-2,11,12,10,11.5,110",
                "bad-ts-3,12,13,11,12.5,120",
            ]
        ),
        encoding="utf-8",
    )

    report = standardize_file(
        src_csv=src,
        cfg=_build_cfg(input_root, runs_root),
        parquet_root=runs_root / "test_run" / "data_standardized" / "parquet",
        per_file_reports_root=runs_root / "test_run" / "data_standardized" / "reports" / "per_file",
        dry_run=False,
    )

    assert report.status == "failed"
    assert report.rows_out == 0
    assert report.parse_valid_ratio == 0.0
    assert report.timestamp_min is None
    assert report.timestamp_max is None
    assert any(err.code == "TIMESTAMP_PARSE_INVALID" for err in report.errors)
