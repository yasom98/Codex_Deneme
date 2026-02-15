"""CSV ingestion helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def _detect_delimiter(path: Path) -> str:
    """Detect delimiter from CSV sample. Fallback to comma."""
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        sample = handle.read(8192)

    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return ","
    return dialect.delimiter


def read_csv_ohlcv(path: Path) -> pd.DataFrame:
    """Read OHLCV CSV file into a dataframe."""
    if not path.exists():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"CSV path is not a file: {path}")

    try:
        delimiter = _detect_delimiter(path)
        frame = pd.read_csv(path, sep=delimiter)
        if len(frame.columns) == 1 and delimiter != ";" and ";" in str(frame.columns[0]):
            frame = pd.read_csv(path, sep=";")
        return frame
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {path}") from exc


def parse_timestamp_utc(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Parse timestamp column as UTC-aware datetime."""
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column not found: {timestamp_col}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    return out
