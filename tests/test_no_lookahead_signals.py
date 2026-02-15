"""No-lookahead tests for all shifted event signal columns."""

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
    SuperTrendConfig,
    build_feature_artifacts,
)


def _fixture_df(rows: int = 1800) -> pd.DataFrame:
    ts = pd.date_range("2024-02-01", periods=rows, freq="5min", tz="UTC")
    base = np.linspace(90.0, 140.0, rows)
    noise = np.sin(np.linspace(0.0, 36.0, rows))
    close = base + noise
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close + 0.2).astype(np.float32),
            "high": (close + 0.7).astype(np.float32),
            "low": (close - 0.7).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.linspace(1200.0, 2100.0, rows).astype(np.float32),
        }
    )


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
        health=HealthPolicyConfig(
            warn_ratio=1.0,
            critical_warn_ratio=1.0,
            critical_columns=("EMA_200",),
        ),
        config_hash="unit",
        indicator_spec_version=INDICATOR_SPEC_VERSION,
    )


def test_all_shifted_signal_columns_are_strict_shift_one() -> None:
    artifacts = build_feature_artifacts(_fixture_df(), _cfg())

    for col in EVENT_FLAG_COLUMNS:
        raw = artifacts.raw_events[col].to_numpy(dtype=np.uint8)
        shifted = artifacts.shifted_events[col].to_numpy(dtype=np.uint8)
        assert shifted[0] == 0
        np.testing.assert_array_equal(shifted[1:], raw[:-1])
