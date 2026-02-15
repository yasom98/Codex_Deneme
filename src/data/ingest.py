"""CSV ingestion helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv_ohlcv(path: Path) -> pd.DataFrame:
    """Read OHLCV CSV file into a dataframe."""
    if not path.exists():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"CSV path is not a file: {path}")

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {path}") from exc


def parse_timestamp_utc(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Parse timestamp column as UTC-aware datetime."""
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column not found: {timestamp_col}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    return out

