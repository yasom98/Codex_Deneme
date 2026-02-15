"""Tests for schema mapping and enforcement."""

from __future__ import annotations

import pandas as pd
import pytest

from data.schema import (
    CANONICAL_COLUMNS,
    detect_timestamp_alias,
    enforce_schema,
    map_to_canonical,
    normalize_column_names,
)


def test_schema_enforcement_sorts_dedups_and_casts() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2024-01-02 00:00:00", "2024-01-01 00:00:00", "2024-01-02 00:00:00"],
            "Open": ["11", "10", "12"],
            "High": ["13", "11", "14"],
            "Low": ["9", "8", "10"],
            "Close": ["12", "10.5", "13"],
            "Volume": ["100", "90", "110"],
        }
    )
    df.columns = normalize_column_names(df.columns)
    ts_col = detect_timestamp_alias(df.columns, ["timestamp", "ts", "date"]).selected_column

    mapped = map_to_canonical(df, ts_col)
    out = enforce_schema(mapped)

    assert tuple(out.columns) == CANONICAL_COLUMNS
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].is_unique
    assert len(out) == 2
    for col in ("open", "high", "low", "close", "volume"):
        assert str(out[col].dtype) == "float32"


def test_schema_enforcement_raises_for_invalid_numeric_value() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00"],
            "open": ["bad-number"],
            "high": ["1"],
            "low": ["1"],
            "close": ["1"],
            "volume": ["1"],
        }
    )
    with pytest.raises(ValueError, match="Invalid numeric values in open"):
        enforce_schema(df)


def test_mixed_case_header_normalization_and_alias_mapping() -> None:
    df = pd.DataFrame(
        {
            " TimeStamp ": ["2024-01-01 00:00:00"],
            " Open ": ["1"],
            "HIGH": ["2"],
            " low ": ["0.5"],
            "Price-Close": ["1.5"],
            "BASE VOLUME": ["100"],
        }
    )
    df.columns = normalize_column_names(df.columns)
    ts_col = detect_timestamp_alias(df.columns, ["timestamp", "ts", "datetime", "date"]).selected_column
    out = map_to_canonical(df, ts_col)

    assert tuple(out.columns) == CANONICAL_COLUMNS


def test_missing_required_columns_fails_with_clear_message() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00"],
            "open": ["1"],
            "high": ["2"],
            "low": ["0.5"],
            "close": ["1.5"],
        }
    )
    with pytest.raises(ValueError, match="Could not map required field: volume"):
        map_to_canonical(df, "timestamp")
