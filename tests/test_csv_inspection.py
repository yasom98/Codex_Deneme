"""Tests for CSV inspection workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from data.csv_inspection import inspect_all_csvs, inspect_csv_file, write_inspection_report


def test_inspect_csv_file_success_with_indicator(tmp_path: Path) -> None:
    src = tmp_path / "ok.csv"
    src.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume,rsi_14",
                "2024-01-01T00:00:00Z,10,11,9,10.5,100,55",
                "2024-01-02T00:00:00Z,11,12,10,11.5,120,56",
            ]
        ),
        encoding="utf-8",
    )

    result = inspect_csv_file(src, sample_size=100)

    assert result.passed is True
    assert result.timestamp_alias == "timestamp"
    assert result.timestamp_column == "timestamp"
    assert result.ohlcv_mapping == {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    assert result.indicator_columns == ["rsi_14"]
    assert result.parse_valid_ratio == 1.0
    assert result.min_ts is not None
    assert result.max_ts is not None
    assert result.unique_days_sample == 2


def test_inspect_csv_file_flags_implausible_timestamp_range(tmp_path: Path) -> None:
    src = tmp_path / "bad_ts.csv"
    src.write_text(
        "\n".join(
            [
                "ts,open,high,low,close,volume",
                "1,10,11,9,10.5,100",
                "2,11,12,10,11.5,120",
                "3,12,13,11,12.5,130",
            ]
        ),
        encoding="utf-8",
    )

    result = inspect_csv_file(src, sample_size=100)

    assert result.passed is False
    assert "TIMESTAMP_IMPLAUSIBLE_RANGE" in result.failure_reason_codes


def test_inspect_csv_file_detects_duplicate_headers(tmp_path: Path) -> None:
    src = tmp_path / "dup_header.csv"
    src.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume,volume",
                "2024-01-01T00:00:00Z,10,11,9,10.5,100,100",
                "2024-01-02T00:00:00Z,11,12,10,11.5,120,120",
            ]
        ),
        encoding="utf-8",
    )

    result = inspect_csv_file(src, sample_size=100)

    assert result.passed is False
    assert result.duplicate_header_names == ["volume"]
    assert "DUPLICATE_HEADER_NAMES" in result.failure_reason_codes


def test_inspect_all_and_write_report(tmp_path: Path) -> None:
    ok_file = tmp_path / "ok.csv"
    ok_file.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume",
                "2024-01-01T00:00:00Z,10,11,9,10.5,100",
                "2024-01-02T00:00:00Z,11,12,10,11.5,120",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_all_csvs(input_root=tmp_path, run_id="run_test", sample_size=100)
    assert report.total_files == 1
    assert report.failed_files == 0

    destination = tmp_path / "runs" / "run_test" / "csv_inspection" / "inspection_report.json"
    write_inspection_report(report, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_test"
    assert payload["total_files"] == 1
    assert payload["passed_files"] == 1


def test_cli_prints_exact_success_message(tmp_path: Path) -> None:
    src = tmp_path / "ok.csv"
    src.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume",
                "2024-01-01T00:00:00Z,10,11,9,10.5,100",
                "2024-01-02T00:00:00Z,11,12,10,11.5,120",
            ]
        ),
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "inspect_csvs.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input-root",
            str(tmp_path),
            "--run-id",
            "cli_success",
            "--log-level",
            "ERROR",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert proc.stdout == "çıktı beklediğim gibi,başarılı ✅\n"
