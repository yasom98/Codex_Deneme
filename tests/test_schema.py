"""Tests for schema mapping and final gate checks."""

from __future__ import annotations

import pandas as pd
import pytest

from data.schema import (
    CANONICAL_COLUMNS,
    convert_numeric_with_drop,
    detect_timestamp_alias,
    final_gate_checks,
    map_to_canonical,
    normalize_column_names,
)


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


def test_convert_numeric_with_drop_counts_invalid_rows_and_float32() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": ["1", "bad"],
            "high": ["2", "3"],
            "low": ["0.5", "0.7"],
            "close": ["1.5", "2.5"],
            "volume": ["100", "oops"],
        }
    )
    out, dropped = convert_numeric_with_drop(df, ("open", "high", "low", "close", "volume"))
    assert dropped == 1
    assert len(out) == 1
    for col in ("open", "high", "low", "close", "volume"):
        assert str(out[col].dtype) == "float32"


def test_final_gate_checks_pass_for_clean_frame() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": pd.Series([1.0, 2.0], dtype="float32"),
            "high": pd.Series([2.0, 3.0], dtype="float32"),
            "low": pd.Series([0.5, 1.5], dtype="float32"),
            "close": pd.Series([1.5, 2.5], dtype="float32"),
            "volume": pd.Series([100.0, 110.0], dtype="float32"),
        }
    )
    gates = final_gate_checks(df)
    assert gates == {
        "schema_ok": True,
        "monotonic_ok": True,
        "unique_ts_ok": True,
        "dtype_ok": True,
        "no_nan_ok": True,
    }
