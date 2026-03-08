"""Tests for feature dtype policy enforcement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.features import (
    AlphaTrendConfig,
    FeatureBuildConfig,
    HealthPolicyConfig,
    INDICATOR_SPEC_VERSION,
    ParityPolicyConfig,
    PivotPolicyConfig,
    SuperTrendConfig,
    build_feature_artifacts,
)


def _sample_ohlcv(rows: int = 1800) -> pd.DataFrame:
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


def test_dtype_policy_continuous_float32_flags_uint8() -> None:
    artifacts = build_feature_artifacts(_sample_ohlcv(), _config())
    out = artifacts.frame

    continuous = ("open", "high", "low", "close", "volume", *artifacts.continuous_feature_columns)
    for col in continuous:
        assert str(out[col].dtype) == "float32"

    for col in artifacts.flag_feature_columns:
        assert str(out[col].dtype) == "uint8"


def test_body_ratio_zero_range_candle_is_zero_after_warmup() -> None:
    df = _sample_ohlcv()
    flat_index = 1400
    nearby_index = 1401
    flat_price = np.float32(df.loc[flat_index, "close"])
    df.loc[flat_index, ["open", "high", "low", "close"]] = flat_price

    artifacts = build_feature_artifacts(df, _config())
    out = artifacts.frame

    assert flat_index > artifacts.warmup_rows_by_column["EMA_1200"]
    assert float(out.loc[flat_index, "hl_range"]) == 0.0
    assert float(out.loc[flat_index, "body_ratio"]) == 0.0
    assert np.isfinite(float(out.loc[flat_index, "body_ratio"]))
    assert str(out["hl_range"].dtype) == "float32"
    assert str(out["body_ratio"].dtype) == "float32"

    expected_body_ratio = abs(float(df.loc[nearby_index, "close"]) - float(df.loc[nearby_index, "open"])) / (
        float(df.loc[nearby_index, "high"]) - float(df.loc[nearby_index, "low"])
    )
    assert float(out.loc[nearby_index, "hl_range"]) > 0.0
    assert np.isclose(float(out.loc[nearby_index, "body_ratio"]), expected_body_ratio)
