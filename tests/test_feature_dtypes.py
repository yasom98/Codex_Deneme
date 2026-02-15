"""Tests for feature dtype policy enforcement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.features import (
    CONTINUOUS_FEATURE_COLUMNS,
    EVENT_FLAG_COLUMNS,
    AlphaTrendConfig,
    EventConfig,
    FeatureBuildConfig,
    HealthPolicyConfig,
    RsiConfig,
    SuperTrendConfig,
    ThresholdRule,
    build_feature_artifacts,
)


def _sample_ohlcv(rows: int = 180) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    trend = np.linspace(200.0, 260.0, rows)
    noise = np.cos(np.linspace(0.0, 16.0, rows))

    close = trend + noise
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close + 0.2).astype(np.float32),
            "high": (close + 0.6).astype(np.float32),
            "low": (close - 0.6).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.linspace(900.0, 1600.0, rows).astype(np.float32),
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
            period=11,
            atr_multiplier=3.0,
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


def test_dtype_policy_continuous_float32_flags_uint8() -> None:
    artifacts = build_feature_artifacts(_sample_ohlcv(), _config())
    out = artifacts.frame

    continuous = ("open", "high", "low", "close", "volume", *CONTINUOUS_FEATURE_COLUMNS)
    for col in continuous:
        assert str(out[col].dtype) == "float32"

    for col in EVENT_FLAG_COLUMNS:
        assert str(out[col].dtype) == "uint8"
