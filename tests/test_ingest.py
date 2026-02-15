"""Tests for CSV ingestion helpers."""

from __future__ import annotations

from pathlib import Path

from data.ingest import read_csv_ohlcv


def test_read_csv_ohlcv_reads_comma_delimited_file(tmp_path: Path) -> None:
    src = tmp_path / "comma.csv"
    src.write_text(
        "\n".join(
            [
                "ts,open,high,low,close,volume",
                "2024-01-01T00:00:00Z,1,2,0.5,1.5,100",
            ]
        ),
        encoding="utf-8",
    )

    frame = read_csv_ohlcv(src)
    assert list(frame.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert len(frame) == 1


def test_read_csv_ohlcv_reads_semicolon_delimited_file(tmp_path: Path) -> None:
    src = tmp_path / "semicolon.csv"
    src.write_text(
        "\n".join(
            [
                "ts;open;high;low;close;volume",
                "2024-01-01T00:00:00Z;1;2;0.5;1.5;100",
            ]
        ),
        encoding="utf-8",
    )

    frame = read_csv_ohlcv(src)
    assert list(frame.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert len(frame) == 1
