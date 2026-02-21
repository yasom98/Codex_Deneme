"""Adapter for user-provided pivot reference specifications."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
import pandas as pd

from core.logging import get_logger

LOGGER = get_logger(__name__)

PIVOT_REFERENCE_SOURCE: str = "indicator_specs/pivot_traditional.py"
PIVOT_REFERENCE_TYPE: str = "Traditional"
PIVOT_OUTPUT_COLUMNS: tuple[str, ...] = ("PP", "R1", "S1", "R2", "S2", "R3", "S3", "R4", "S4", "R5", "S5")


def _repo_root() -> Path:
    """Return repository root path from src/data module location."""

    return Path(__file__).resolve().parents[2]


def _pivot_reference_path() -> Path:
    """Return pivot reference module path."""

    return _repo_root() / PIVOT_REFERENCE_SOURCE


def pivot_reference_available() -> bool:
    """Return True when the user-provided pivot reference file exists."""

    return _pivot_reference_path().exists()


def _load_reference_module(path: Path) -> ModuleType:
    """Load pivot reference module from a file path."""

    spec = importlib.util.spec_from_file_location("indicator_specs.pivot_traditional", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_reference_callable() -> Callable[..., pd.DataFrame]:
    """Load the compute function from user-provided pivot reference module."""

    path = _pivot_reference_path()
    if not path.exists():
        raise FileNotFoundError(f"Pivot reference file not found: {path}")

    module = _load_reference_module(path)
    compute = getattr(module, "compute_pivots_intraday", None)
    if not callable(compute):
        raise TypeError("pivot reference module must define callable compute_pivots_intraday")
    return compute


def compute_reference_pivots_intraday(
    ohlcv: pd.DataFrame,
    *,
    pivot_tf: str = "1D",
) -> pd.DataFrame:
    """
    Compute pivot values via user-provided reference implementation.

    Adapter responsibilities:
    - validate required columns
    - normalize timestamp to UTC DatetimeIndex
    - preserve original row order/index
    """

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"Missing required columns for pivot reference adapter: {sorted(missing)}")

    frame = ohlcv.loc[:, ["timestamp", "open", "high", "low", "close"]].copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values for pivot reference adapter")
    if not frame["timestamp"].is_monotonic_increasing:
        LOGGER.warning("pivot reference adapter sorted non-monotonic timestamps before resample")
        frame = frame.sort_values("timestamp", kind="mergesort")

    indexed = frame.set_index("timestamp", drop=True)
    if not isinstance(indexed.index, pd.DatetimeIndex):
        raise ValueError("Pivot reference adapter failed to create DatetimeIndex")
    if indexed.index.tz is None:
        raise ValueError("Pivot reference adapter requires timezone-aware UTC timestamp index")

    compute = _load_reference_callable()
    pivots = compute(
        indexed.loc[:, ["open", "high", "low", "close"]],
        pivot_tf=pivot_tf,
        pivot_type=PIVOT_REFERENCE_TYPE,
    )
    if not isinstance(pivots, pd.DataFrame):
        raise TypeError("compute_pivots_intraday must return a pandas DataFrame")
    missing_outputs = set(PIVOT_OUTPUT_COLUMNS) - set(pivots.columns)
    if missing_outputs:
        raise ValueError(f"Pivot reference output missing columns: {sorted(missing_outputs)}")

    aligned = pd.DataFrame(index=indexed.index)
    aligned["_row_order"] = frame["_row_order"].to_numpy(dtype=np.int64, copy=False)
    aligned.loc[:, list(PIVOT_OUTPUT_COLUMNS)] = pivots.loc[:, list(PIVOT_OUTPUT_COLUMNS)]
    restored = aligned.reset_index(drop=True).sort_values("_row_order", kind="mergesort")
    out = restored.loc[:, list(PIVOT_OUTPUT_COLUMNS)].copy()
    out.index = ohlcv.index
    return out.astype("float32")
