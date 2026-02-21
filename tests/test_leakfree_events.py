"""Tests for strict leak-free shift(1) event policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.features import (
    AlphaTrendConfig,
    EVENT_FLAG_COLUMNS,
    FeatureBuildConfig,
    HealthPolicyConfig,
    INDICATOR_SPEC_VERSION,
    ParityPolicyConfig,
    PivotPolicyConfig,
    REGIME_FLAG_COLUMNS,
    SuperTrendConfig,
    build_feature_artifacts,
    validate_shift_one,
    validate_shift_one_for_columns,
)


def _sample_ohlcv(rows: int = 2400) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
    base = np.linspace(100.0, 130.0, rows)
    wave = np.sin(np.linspace(0.0, 24.0, rows))

    close = base + wave
    open_ = close + 0.1
    high = close + 0.5
    low = close - 0.5
    volume = np.linspace(1000.0, 2000.0, rows)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_.astype(np.float32),
            "high": high.astype(np.float32),
            "low": low.astype(np.float32),
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        }
    )


def _config() -> FeatureBuildConfig:
    return FeatureBuildConfig(
        input_root=Path("."),
        runs_root=Path("."),
        parquet_glob="*.parquet",
        seed=42,
        supertrend=SuperTrendConfig(periods=10, multiplier=3.0, source="hl2", change_atr_method=True),
        alphatrend=AlphaTrendConfig(coeff=3.0, ap=11, use_no_volume=False),
        pivot=PivotPolicyConfig(pivot_tf="1D", warmup_policy="allow_first_session_nan", first_session_fill="none"),
        parity=ParityPolicyConfig(enabled=True, sample_rows=512, float_atol=1e-6, float_rtol=1e-6),
        health=HealthPolicyConfig(
            warn_ratio=0.005,
            critical_warn_ratio=0.001,
            critical_columns=("EMA_200", "EMA_600", "EMA_1200"),
        ),
        config_hash="unit",
        indicator_spec_version=INDICATOR_SPEC_VERSION,
    )


def test_event_columns_are_strict_shift_one() -> None:
    artifacts = build_feature_artifacts(_sample_ohlcv(), _config())

    assert validate_shift_one(artifacts.raw_events, artifacts.shifted_events)
    assert validate_shift_one_for_columns(
        artifacts.raw_regime_flags,
        artifacts.shifted_regime_flags,
        REGIME_FLAG_COLUMNS,
    )

    for col in EVENT_FLAG_COLUMNS:
        raw = artifacts.raw_events[col].fillna(0).astype("uint8").to_numpy(dtype=np.uint8)
        shifted = artifacts.shifted_events[col].to_numpy(dtype=np.uint8)

        assert shifted[0] == 0
        np.testing.assert_array_equal(shifted[1:], raw[:-1])

    for col in REGIME_FLAG_COLUMNS:
        raw = artifacts.raw_regime_flags[col].fillna(0).astype("uint8").to_numpy(dtype=np.uint8)
        shifted = artifacts.shifted_regime_flags[col].to_numpy(dtype=np.uint8)
        assert shifted[0] == 0
        np.testing.assert_array_equal(shifted[1:], raw[:-1])
