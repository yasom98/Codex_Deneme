"""Schema mapping and enforcement for OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import pandas as pd

CANONICAL_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "price_close"),
    "volume": ("volume", "vol", "v", "base_volume"),
}

DATETIME_ALIAS_PRIORITY: tuple[str, ...] = (
    "timestamp",
    "ts",
    "datetime",
    "date",
    "time",
    "open_time",
    "candle_time",
    "close_time",
)


@dataclass(frozen=True)
class TimestampAliasDetection:
    """Result of timestamp alias detection."""

    selected_column: str
    selected_alias: str
    confidence: float
    matched_aliases: tuple[str, ...]


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_column_names(columns: Sequence[str]) -> list[str]:
    """Normalize raw dataframe columns into snake_case names."""
    return [_normalize_name(col) for col in columns]


def detect_timestamp_alias(columns: Sequence[str], aliases: Sequence[str]) -> TimestampAliasDetection:
    """Detect timestamp alias with fixed priority and confidence score."""
    normalized_cols = {_normalize_name(col): col for col in columns}
    configured = {_normalize_name(alias) for alias in aliases}
    priority = [alias for alias in DATETIME_ALIAS_PRIORITY if alias in configured]

    matched = [alias for alias in priority if alias in normalized_cols]
    if not matched:
        raise ValueError(
            "No datetime alias found. Expected one of: "
            + ", ".join(DATETIME_ALIAS_PRIORITY)
        )

    selected_alias = matched[0]
    selected_column = normalized_cols[selected_alias]
    confidence = 1.0 if len(matched) == 1 else 0.85
    return TimestampAliasDetection(
        selected_column=selected_column,
        selected_alias=selected_alias,
        confidence=confidence,
        matched_aliases=tuple(matched),
    )


def detect_timestamp_column(columns: Sequence[str], aliases: Sequence[str]) -> str:
    """Backwards-compatible wrapper returning only selected column."""
    return detect_timestamp_alias(columns, aliases).selected_column


def _pick_column(columns: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """Validate required columns exist in dataframe."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def map_to_canonical(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Map dataframe columns to canonical OHLCV schema."""
    columns = set(df.columns)
    rename_map: dict[str, str] = {timestamp_col: "timestamp"}

    for canonical, candidates in _FIELD_ALIASES.items():
        picked = _pick_column(columns, candidates)
        if picked is None:
            raise ValueError(f"Could not map required field: {canonical}")
        rename_map[picked] = canonical

    mapped = df.rename(columns=rename_map)
    validate_required_columns(mapped, CANONICAL_COLUMNS)
    return mapped.loc[:, list(CANONICAL_COLUMNS)].copy()


def convert_numeric_with_drop(df: pd.DataFrame, numeric_cols: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """Convert numeric columns; drop rows where conversion fails."""
    out = df.copy()
    converted_cols: dict[str, pd.Series] = {}

    for col in numeric_cols:
        converted_cols[col] = pd.to_numeric(out[col], errors="coerce")

    invalid_mask = pd.Series(False, index=out.index)
    for col in numeric_cols:
        invalid_mask |= converted_cols[col].isna()

    dropped = int(invalid_mask.sum())
    out = out.loc[~invalid_mask].copy()

    for col in numeric_cols:
        out[col] = converted_cols[col].loc[out.index].astype("float32")

    return out, dropped


def final_gate_checks(df: pd.DataFrame) -> dict[str, bool]:
    """Run strict final gate checks for standardized output."""
    schema_ok = all(col in df.columns for col in CANONICAL_COLUMNS)
    monotonic_ok = bool(df["timestamp"].is_monotonic_increasing) if schema_ok else False
    unique_ts_ok = bool(df["timestamp"].is_unique) if schema_ok else False
    dtype_ok = bool(all(str(df[col].dtype) == "float32" for col in NUMERIC_COLUMNS)) if schema_ok else False
    no_nan_ok = bool(not df[list(CANONICAL_COLUMNS)].isna().any().any()) if schema_ok else False

    return {
        "schema_ok": schema_ok,
        "monotonic_ok": monotonic_ok,
        "unique_ts_ok": unique_ts_ok,
        "dtype_ok": dtype_ok,
        "no_nan_ok": no_nan_ok,
    }
