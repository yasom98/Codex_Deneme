"""Unit and regression tests for pivot reference adapter integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.features import (
    AlphaTrendConfig,
    FeatureBuildConfig,
    HealthPolicyConfig,
    INDICATOR_SPEC_VERSION,
    ParityPolicyConfig,
    PivotPolicyConfig,
    SuperTrendConfig,
    build_feature_artifacts,
    compute_daily_pivots_with_std_bands,
)
from data.reference_pivots import PIVOT_OUTPUT_COLUMNS, compute_reference_pivots_intraday


def _cfg() -> FeatureBuildConfig:
    return FeatureBuildConfig(
        input_root=Path("."),
        runs_root=Path("."),
        parquet_glob="*.parquet",
        seed=42,
        supertrend=SuperTrendConfig(periods=10, multiplier=3.0, source="hl2", change_atr_method=True),
        alphatrend=AlphaTrendConfig(coeff=3.0, ap=11, use_no_volume=False),
        pivot=PivotPolicyConfig(pivot_tf="1D", warmup_policy="allow_first_session_nan", first_session_fill="none"),
        parity=ParityPolicyConfig(enabled=True, sample_rows=512, float_atol=1e-6, float_rtol=1e-6),
        health=HealthPolicyConfig(warn_ratio=1.0, critical_warn_ratio=1.0, critical_columns=("EMA_200",)),
        config_hash="unit",
        indicator_spec_version=INDICATOR_SPEC_VERSION,
    )


def _intraday_fixture(rows: int = 1800) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    base = np.linspace(100.0, 130.0, rows)
    close = base + np.sin(np.linspace(0.0, 30.0, rows))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close + 0.2).astype(np.float32),
            "high": (close + 0.7).astype(np.float32),
            "low": (close - 0.7).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.linspace(1000.0, 1500.0, rows).astype(np.float32),
        }
    )


def test_reference_adapter_converts_timestamp_and_preserves_output_contract() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:05:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-02T00:05:00Z",
            ],
            "open": np.array([10.1, 10.2, 11.0, 11.2], dtype=np.float32),
            "high": np.array([10.8, 11.0, 11.8, 12.0], dtype=np.float32),
            "low": np.array([9.8, 9.9, 10.6, 10.8], dtype=np.float32),
            "close": np.array([10.4, 10.6, 11.4, 11.6], dtype=np.float32),
            "volume": np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32),
        },
        index=pd.Index([11, 7, 19, 3]),
    )
    out = compute_reference_pivots_intraday(df, pivot_tf="1D")

    assert tuple(out.columns) == PIVOT_OUTPUT_COLUMNS
    assert out.index.equals(df.index)
    assert all(str(out[col].dtype) == "float32" for col in PIVOT_OUTPUT_COLUMNS)
    assert out.iloc[:2].isna().all(axis=1).all()
    assert out.iloc[2:].notna().all(axis=1).all()


def test_reference_adapter_raises_on_missing_required_columns() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
            "open": np.array([1.0, 1.1, 1.2], dtype=np.float32),
            "high": np.array([1.2, 1.3, 1.4], dtype=np.float32),
            "close": np.array([1.1, 1.2, 1.3], dtype=np.float32),
        }
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_reference_pivots_intraday(df, pivot_tf="1D")


def test_pivot_reference_mismatch_fails_indicator_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    df = _intraday_fixture()
    original = compute_daily_pivots_with_std_bands

    def bad_pivots(
        frame: pd.DataFrame,
        warmup_policy: str = "allow_first_session_nan",
        first_session_fill: str = "none",
        pivot_tf: str = "1D",
        assume_validated: bool = False,
        indexed_ohlcv: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        out = original(
            frame,
            warmup_policy=warmup_policy,
            first_session_fill=first_session_fill,
            pivot_tf=pivot_tf,
            assume_validated=assume_validated,
            indexed_ohlcv=indexed_ohlcv,
        )
        out = out.copy()
        out["PP"] = out["PP"] + np.float32(1.0)
        return out

    monkeypatch.setattr("data.features.compute_daily_pivots_with_std_bands", bad_pivots)

    artifacts = build_feature_artifacts(df, cfg)
    assert artifacts.indicator_validation_status == "failed"
    assert artifacts.indicator_validation_details["pivot_reference_parity"] is False


def test_pivot_reference_warmup_allows_single_session_nan() -> None:
    cfg = _cfg()
    single_session = _intraday_fixture(rows=200)
    artifacts = build_feature_artifacts(single_session, cfg)

    assert artifacts.indicator_validation_status == "passed"
    assert artifacts.indicator_validation_details["pivot_reference_available"] is True
    assert artifacts.indicator_validation_details["pivot_reference_execution_ok"] is True
    assert artifacts.indicator_validation_details["pivot_reference_parity"] is True

