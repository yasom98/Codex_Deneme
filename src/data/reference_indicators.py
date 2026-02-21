"""Adapters for user-provided AlphaTrend/SuperTrend reference specifications."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

import numpy as np
import pandas as pd

from core.logging import get_logger

LOGGER = get_logger(__name__)

ALPHATREND_REFERENCE_SOURCE: str = "indicator_specs/alphatrend.py"
SUPERTREND_REFERENCE_SOURCE: str = "indicator_specs/supertrend.py"

ALPHATREND_OUTPUT_COLUMNS: tuple[str, ...] = ("AlphaTrend", "AlphaTrend_2", "AT_buy", "AT_sell")
SUPERTREND_OUTPUT_COLUMNS: tuple[str, ...] = ("ST_trend", "ST_up", "ST_dn", "ST_buy", "ST_sell", "ST_change")


def _repo_root() -> Path:
    """Return repository root path from src/data module location."""

    return Path(__file__).resolve().parents[2]


def _alphatrend_reference_path() -> Path:
    """Return AlphaTrend reference module path."""

    return _repo_root() / ALPHATREND_REFERENCE_SOURCE


def _supertrend_reference_path() -> Path:
    """Return SuperTrend reference module path."""

    return _repo_root() / SUPERTREND_REFERENCE_SOURCE


def alphatrend_reference_available() -> bool:
    """Return True when AlphaTrend user reference file exists."""

    return _alphatrend_reference_path().exists()


def supertrend_reference_available() -> bool:
    """Return True when SuperTrend user reference file exists."""

    return _supertrend_reference_path().exists()


def _load_reference_module(path: Path, module_name: str) -> ModuleType:
    """Load a reference module from file path."""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_alphatrend_reference_callable() -> tuple[Callable[..., pd.DataFrame], type[object]]:
    """Load callable and config class for AlphaTrend reference module."""

    path = _alphatrend_reference_path()
    if not path.exists():
        raise FileNotFoundError(f"AlphaTrend reference file not found: {path}")

    module = _load_reference_module(path, "indicator_specs.alphatrend")
    compute = getattr(module, "compute_alphatrend", None)
    cfg_cls = getattr(module, "AlphaTrendConfig", None)
    if not callable(compute):
        raise TypeError("alphatrend reference module must define callable compute_alphatrend")
    if cfg_cls is None:
        raise TypeError("alphatrend reference module must define AlphaTrendConfig")
    return compute, cfg_cls


@lru_cache(maxsize=1)
def _load_supertrend_reference_callable() -> tuple[Callable[..., pd.DataFrame], type[object]]:
    """Load callable and config class for SuperTrend reference module."""

    path = _supertrend_reference_path()
    if not path.exists():
        raise FileNotFoundError(f"SuperTrend reference file not found: {path}")

    module = _load_reference_module(path, "indicator_specs.supertrend")
    compute = getattr(module, "compute_supertrend", None)
    cfg_cls = getattr(module, "SupertrendConfig", None)
    if not callable(compute):
        raise TypeError("supertrend reference module must define callable compute_supertrend")
    if cfg_cls is None:
        raise TypeError("supertrend reference module must define SupertrendConfig")
    return compute, cfg_cls


def _prepare_reference_frame(
    ohlcv: pd.DataFrame,
    required: set[str],
    optional: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate required columns and normalize timestamp/index for adapters."""

    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"Missing required columns for indicator reference adapter: {sorted(missing)}")

    cols = ["timestamp", *sorted(required - {"timestamp"})]
    cols.extend(col for col in optional if (col in ohlcv.columns and col not in cols))
    frame = ohlcv.loc[:, cols].copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values for indicator reference adapter")
    if not frame["timestamp"].is_monotonic_increasing:
        LOGGER.warning("indicator reference adapter sorted non-monotonic timestamps before computation")
        frame = frame.sort_values("timestamp", kind="mergesort")

    indexed = frame.set_index("timestamp", drop=True)
    if not isinstance(indexed.index, pd.DatetimeIndex):
        raise ValueError("Indicator reference adapter failed to create DatetimeIndex")
    if indexed.index.tz is None:
        raise ValueError("Indicator reference adapter requires timezone-aware UTC timestamp index")
    return frame, indexed


def compute_reference_alphatrend(
    ohlcv: pd.DataFrame,
    *,
    coeff: float = 3.0,
    ap: int = 11,
    use_no_volume: bool = False,
) -> pd.DataFrame:
    """Compute AlphaTrend outputs via user-provided reference implementation."""

    required = {"timestamp", "high", "low", "close"}
    optional_cols = ("volume",)
    frame, indexed = _prepare_reference_frame(ohlcv, required=required, optional=optional_cols)

    compute, cfg_cls = _load_alphatrend_reference_callable()
    input_cols = ["high", "low", "close", *[col for col in optional_cols if col in indexed.columns]]
    cfg = cfg_cls(coeff=coeff, ap=ap, use_no_volume=use_no_volume)
    out = compute(indexed.loc[:, input_cols], cfg=cfg, show_progress=False)

    if not isinstance(out, pd.DataFrame):
        raise TypeError("compute_alphatrend must return a pandas DataFrame")
    missing_outputs = set(ALPHATREND_OUTPUT_COLUMNS) - set(out.columns)
    if missing_outputs:
        raise ValueError(f"AlphaTrend reference output missing columns: {sorted(missing_outputs)}")

    aligned = pd.DataFrame(index=indexed.index)
    aligned["_row_order"] = frame["_row_order"].to_numpy(dtype=np.int64, copy=False)
    aligned.loc[:, list(ALPHATREND_OUTPUT_COLUMNS)] = out.loc[:, list(ALPHATREND_OUTPUT_COLUMNS)]
    restored = aligned.reset_index(drop=True).sort_values("_row_order", kind="mergesort")

    result = restored.loc[:, list(ALPHATREND_OUTPUT_COLUMNS)].copy()
    result.index = ohlcv.index
    result["AlphaTrend"] = result["AlphaTrend"].astype("float32")
    result["AlphaTrend_2"] = result["AlphaTrend_2"].astype("float32")
    result["AT_buy"] = result["AT_buy"].fillna(0).astype("uint8")
    result["AT_sell"] = result["AT_sell"].fillna(0).astype("uint8")
    return result


def compute_reference_supertrend(
    ohlcv: pd.DataFrame,
    *,
    periods: int = 10,
    multiplier: float = 3.0,
    source: str = "hl2",
    change_atr_method: bool = True,
) -> pd.DataFrame:
    """Compute SuperTrend outputs via user-provided reference implementation."""

    required = {"timestamp", "open", "high", "low", "close"}
    frame, indexed = _prepare_reference_frame(ohlcv, required=required)

    compute, cfg_cls = _load_supertrend_reference_callable()
    cfg = cfg_cls(
        periods=periods,
        multiplier=multiplier,
        source=source,
        change_atr_method=change_atr_method,
    )
    out = compute(indexed.loc[:, ["open", "high", "low", "close"]], cfg=cfg, show_progress=False)

    if not isinstance(out, pd.DataFrame):
        raise TypeError("compute_supertrend must return a pandas DataFrame")

    required_outputs = {"ST_trend", "ST_up", "ST_dn", "ST_buy", "ST_sell"}
    missing_outputs = required_outputs - set(out.columns)
    if missing_outputs:
        raise ValueError(f"SuperTrend reference output missing columns: {sorted(missing_outputs)}")
    if "ST_change" not in out.columns:
        out = out.copy()
        out["ST_change"] = (out["ST_trend"] != out["ST_trend"].shift(1)).astype("uint8")

    aligned = pd.DataFrame(index=indexed.index)
    aligned["_row_order"] = frame["_row_order"].to_numpy(dtype=np.int64, copy=False)
    aligned.loc[:, list(SUPERTREND_OUTPUT_COLUMNS)] = out.loc[:, list(SUPERTREND_OUTPUT_COLUMNS)]
    restored = aligned.reset_index(drop=True).sort_values("_row_order", kind="mergesort")

    result = restored.loc[:, list(SUPERTREND_OUTPUT_COLUMNS)].copy()
    result.index = ohlcv.index
    result["ST_trend"] = result["ST_trend"].fillna(0).astype("int8")
    result["ST_up"] = result["ST_up"].astype("float32")
    result["ST_dn"] = result["ST_dn"].astype("float32")
    result["ST_buy"] = result["ST_buy"].fillna(0).astype("uint8")
    result["ST_sell"] = result["ST_sell"].fillna(0).astype("uint8")
    result["ST_change"] = result["ST_change"].fillna(0).astype("uint8")
    return result
