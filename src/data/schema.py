"""Schema mapping and enforcement for OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import pandas as pd

CANONICAL_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")

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


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce timestamp ordering/uniqueness and float32 dtypes."""
    validate_required_columns(df, CANONICAL_COLUMNS)
    out = df.copy()

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    invalid_timestamps = int(out["timestamp"].isna().sum())
    if invalid_timestamps > 0:
        raise ValueError(f"Unparseable timestamps detected: {invalid_timestamps}")

    out = out.sort_values("timestamp", kind="mergesort")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")

    if out.empty:
        raise ValueError("Dataframe is empty after schema enforcement.")

    for col in ("open", "high", "low", "close", "volume"):
        original = out[col]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_numeric = int((original.notna() & converted.isna()).sum())
        if invalid_numeric > 0:
            raise ValueError(f"Invalid numeric values in {col}: {invalid_numeric}")
        out[col] = converted.astype("float32")

    if not out["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic increasing after enforcement.")
    if out["timestamp"].duplicated().any():
        raise ValueError("Timestamps are not unique after enforcement.")

    return out.loc[:, list(CANONICAL_COLUMNS)].reset_index(drop=True)
