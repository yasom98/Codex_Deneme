"""Unit tests for SuperTrend reference adapter integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.reference_indicators import SUPERTREND_OUTPUT_COLUMNS, compute_reference_supertrend


def _fixture_df(rows: int = 1200) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    base = np.linspace(100.0, 130.0, rows)
    close = base + np.sin(np.linspace(0.0, 25.0, rows))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close + 0.15).astype(np.float32),
            "high": (close + 0.7).astype(np.float32),
            "low": (close - 0.7).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.linspace(1000.0, 1400.0, rows).astype(np.float32),
        }
    )


def test_reference_supertrend_adapter_output_contract_and_index_preservation() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:05:00Z",
                "2024-01-01T00:10:00Z",
                "2024-01-01T00:15:00Z",
            ],
            "open": np.array([10.0, 10.3, 10.6, 10.9], dtype=np.float32),
            "high": np.array([10.8, 11.0, 11.2, 11.4], dtype=np.float32),
            "low": np.array([9.8, 10.0, 10.2, 10.4], dtype=np.float32),
            "close": np.array([10.2, 10.6, 10.8, 11.1], dtype=np.float32),
        },
        index=pd.Index([17, 8, 3, 1]),
    )

    out = compute_reference_supertrend(df)

    assert tuple(out.columns) == SUPERTREND_OUTPUT_COLUMNS
    assert out.index.equals(df.index)
    assert str(out["ST_trend"].dtype) == "int8"
    assert str(out["ST_up"].dtype) == "float32"
    assert str(out["ST_dn"].dtype) == "float32"
    assert str(out["ST_buy"].dtype) == "uint8"
    assert str(out["ST_sell"].dtype) == "uint8"
    assert str(out["ST_change"].dtype) == "uint8"


def test_reference_supertrend_adapter_raises_on_missing_required_columns() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
            "high": np.array([1.2, 1.3, 1.4], dtype=np.float32),
            "low": np.array([0.8, 0.9, 1.0], dtype=np.float32),
            "close": np.array([1.1, 1.2, 1.3], dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        compute_reference_supertrend(df)


def test_reference_supertrend_adapter_smoke_executes() -> None:
    df = _fixture_df()
    out = compute_reference_supertrend(df, periods=10, multiplier=3.0, source="hl2", change_atr_method=True)

    assert len(out) == len(df)
    assert out["ST_trend"].isin([-1, 1]).all()
    assert set(np.unique(out["ST_buy"].to_numpy(dtype=np.uint8, copy=False))).issubset({0, 1})
    assert set(np.unique(out["ST_sell"].to_numpy(dtype=np.uint8, copy=False))).issubset({0, 1})
    assert set(np.unique(out["ST_change"].to_numpy(dtype=np.uint8, copy=False))).issubset({0, 1})
