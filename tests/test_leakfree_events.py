"""Tests for strict leak-free shift(1) event policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.features import (
    AlphaTrendConfig,
    EventConfig,
    FeatureBuildConfig,
    HealthPolicyConfig,
    RsiConfig,
    SuperTrendConfig,
    ThresholdRule,
    build_feature_artifacts,
    validate_shift_one,
)


def _sample_ohlcv(rows: int = 240) -> pd.DataFrame:
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
        supertrend=SuperTrendConfig(period=10, multiplier=3.0),
        alphatrend=AlphaTrendConfig(
            period=14,
            atr_multiplier=1.0,
            signal_period=14,
            long_rule=ThresholdRule(signal="mfi", operator=">=", threshold=50.0),
            short_rule=ThresholdRule(signal="mfi", operator="<", threshold=50.0),
        ),
        rsi=RsiConfig(period=14, slope_lag=3, zscore_window=50),
        events=EventConfig(rsi_centerline=50.0, rsi_overbought=70.0, rsi_oversold=30.0),
        health=HealthPolicyConfig(
            warn_ratio=0.005,
            critical_warn_ratio=0.001,
            critical_columns=("supertrend", "alphatrend", "rsi"),
        ),
    )


def test_event_columns_are_strict_shift_one() -> None:
    artifacts = build_feature_artifacts(_sample_ohlcv(), _config())

    assert validate_shift_one(artifacts.raw_events, artifacts.shifted_events)

    for col in artifacts.shifted_events.columns:
        raw = artifacts.raw_events[col].fillna(False).astype("uint8").to_numpy(dtype=np.uint8)
        shifted = artifacts.shifted_events[col].to_numpy(dtype=np.uint8)

        assert shifted[0] == 0
        np.testing.assert_array_equal(shifted[1:], raw[:-1])
