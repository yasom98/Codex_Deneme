"""Tests for end-to-end standardization engine behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.config import PipelineConfig
from data.standardize import standardize_file


def test_standardize_file_drops_invalid_ts_duplicates_and_invalid_numeric(monkeypatch: object, tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)

    src = input_root / "sample.csv"
    src.write_text(
        "\n".join(
            [
                "Time,Open,High,Low,Price-Close,Volume",
                "2024-01-01T00:00:00Z,10,11,9,10.5,100",
                "not-a-time,11,12,10,11.5,101",
                "2024-01-01T00:00:00Z,12,13,11,12.5,102",
                "2024-01-03T00:00:00Z,bad,14,12,13.5,103",
            ]
        ),
        encoding="utf-8",
    )

    cfg = PipelineConfig(
        input_root=input_root,
        runs_root=runs_root,
        csv_glob="**/*.csv",
        timestamp_aliases=("timestamp", "ts", "date", "datetime", "time", "candle_time", "open_time", "close_time"),
        required_columns=("open", "high", "low", "close", "volume"),
        float_columns=("open", "high", "low", "close", "volume"),
        duplicate_policy="last",
        seed=42,
    )
    parquet_root = runs_root / "test_run" / "data_standardized" / "parquet"
    per_file_reports_root = runs_root / "test_run" / "data_standardized" / "reports" / "per_file"

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del index
        self.to_json(path.with_suffix(".json"), orient="records")
        path.write_text("fake-parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    report = standardize_file(
        src_csv=src,
        cfg=cfg,
        parquet_root=parquet_root,
        per_file_reports_root=per_file_reports_root,
        dry_run=False,
    )

    out_path = parquet_root / "sample.parquet"
    assert report.status == "success"
    assert report.rows_in == 4
    assert report.rows_out == 1
    assert report.dropped_invalid_ts == 1
    assert report.dropped_duplicates == 1
    assert report.dropped_invalid_numeric == 1
    assert report.schema_ok is True
    assert report.monotonic_ok is True
    assert report.unique_ts_ok is True
    assert report.dtype_ok is True
    assert report.no_nan_ok is True
    assert out_path.exists()

    per_file_report = per_file_reports_root / "sample.json"
    payload = json.loads(per_file_report.read_text(encoding="utf-8"))
    assert payload["status"] == "success"


def test_standardize_file_fails_when_timestamp_alias_missing(tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)

    src = input_root / "missing_ts.csv"
    src.write_text("open,high,low,close,volume\n1,2,0.5,1.5,100\n", encoding="utf-8")

    cfg = PipelineConfig(
        input_root=input_root,
        runs_root=runs_root,
        csv_glob="**/*.csv",
        timestamp_aliases=("timestamp", "ts", "date", "datetime", "time", "candle_time", "open_time", "close_time"),
        required_columns=("open", "high", "low", "close", "volume"),
        float_columns=("open", "high", "low", "close", "volume"),
        duplicate_policy="last",
        seed=42,
    )
    parquet_root = runs_root / "test_run" / "data_standardized" / "parquet"
    per_file_reports_root = runs_root / "test_run" / "data_standardized" / "reports" / "per_file"

    report = standardize_file(
        src_csv=src,
        cfg=cfg,
        parquet_root=parquet_root,
        per_file_reports_root=per_file_reports_root,
        dry_run=False,
    )

    assert report.status == "failed"
    assert report.rows_out == 0
    assert len(report.errors) == 1
    assert report.errors[0].code == "TIMESTAMP_ALIAS_NOT_FOUND"


def test_standardize_file_parses_locale_scientific_numeric(monkeypatch: object, tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)

    src = input_root / "locale_numeric.csv"
    src.write_text(
        "\n".join(
            [
                "Time,Open,High,Low,Price-Close,Volume",
                "2024-01-01T00:00:00Z,10,11,9,10.5,\"1,01E+11\"",
                "2024-01-01T00:15:00Z,11,12,10,11.5,\"9,99E+10\"",
            ]
        ),
        encoding="utf-8",
    )

    cfg = PipelineConfig(
        input_root=input_root,
        runs_root=runs_root,
        csv_glob="**/*.csv",
        timestamp_aliases=("timestamp", "ts", "date", "datetime", "time", "candle_time", "open_time", "close_time"),
        required_columns=("open", "high", "low", "close", "volume"),
        float_columns=("open", "high", "low", "close", "volume"),
        duplicate_policy="last",
        seed=42,
    )
    parquet_root = runs_root / "test_run" / "data_standardized" / "parquet"
    per_file_reports_root = runs_root / "test_run" / "data_standardized" / "reports" / "per_file"

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del index
        self.to_json(path.with_suffix(".json"), orient="records")
        path.write_text("fake-parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    report = standardize_file(
        src_csv=src,
        cfg=cfg,
        parquet_root=parquet_root,
        per_file_reports_root=per_file_reports_root,
        dry_run=False,
    )

    assert report.status == "success"
    assert report.rows_in == 2
    assert report.rows_out == 2
    assert report.dropped_invalid_numeric == 0
