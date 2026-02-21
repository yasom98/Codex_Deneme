"""Reference parity tests for locked indicator implementations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import indicator_reference as ref
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
from data.reference_indicators import compute_reference_alphatrend, compute_reference_supertrend
from data.reference_pivots import compute_reference_pivots_intraday


def _fixture_df(rows: int = 2000) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    base = np.linspace(100.0, 150.0, rows)
    wave = np.sin(np.linspace(0.0, 40.0, rows))
    close = base + wave
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close + 0.1).astype(np.float32),
            "high": (close + 0.6).astype(np.float32),
            "low": (close - 0.6).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.linspace(1000.0, 1500.0, rows).astype(np.float32),
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


def test_pivot_parity_against_reference() -> None:
    df = _fixture_df()
    cfg = _cfg()
    artifacts = build_feature_artifacts(df, cfg)

    piv_ref = compute_reference_pivots_intraday(df, pivot_tf="1D")

    for col in ("PP", "R1", "S1", "R2", "S2", "R3", "S3", "R4", "S4", "R5", "S5"):
        np.testing.assert_allclose(
            artifacts.frame[col].to_numpy(dtype=np.float64),
            piv_ref[col].to_numpy(dtype=np.float64),
            atol=1e-6,
            rtol=1e-6,
            equal_nan=True,
        )


def test_ema_parity_against_reference() -> None:
    df = _fixture_df()
    cfg = _cfg()
    artifacts = build_feature_artifacts(df, cfg)

    indexed = df.set_index("timestamp")
    ema_ref = ref.compute_ema_set(indexed, price_col="close")

    for col in ("EMA_200", "EMA_600", "EMA_1200"):
        np.testing.assert_allclose(
            artifacts.frame[col].to_numpy(dtype=np.float64),
            ema_ref[col].to_numpy(dtype=np.float64),
            atol=1e-6,
            rtol=1e-6,
            equal_nan=True,
        )


def test_alphatrend_parity_against_reference() -> None:
    df = _fixture_df()
    cfg = _cfg()
    artifacts = build_feature_artifacts(df, cfg)

    at_ref = compute_reference_alphatrend(df, coeff=3.0, ap=11, use_no_volume=False)

    for col in ("AlphaTrend", "AlphaTrend_2"):
        np.testing.assert_allclose(
            artifacts.frame[col].to_numpy(dtype=np.float64),
            at_ref[col].to_numpy(dtype=np.float64),
            atol=1e-6,
            rtol=1e-6,
            equal_nan=True,
        )

    np.testing.assert_array_equal(artifacts.raw_events["evt_at_buy_raw"].to_numpy(), at_ref["AT_buy"].to_numpy())
    np.testing.assert_array_equal(artifacts.raw_events["evt_at_sell_raw"].to_numpy(), at_ref["AT_sell"].to_numpy())


def test_supertrend_parity_against_reference() -> None:
    df = _fixture_df()
    cfg = _cfg()
    artifacts = build_feature_artifacts(df, cfg)

    st_ref = compute_reference_supertrend(df, periods=10, multiplier=3.0, source="hl2", change_atr_method=True)

    np.testing.assert_allclose(
        artifacts.frame["ST_trend"].to_numpy(dtype=np.float64),
        st_ref["ST_trend"].to_numpy(dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        artifacts.frame["ST_up"].to_numpy(dtype=np.float64),
        st_ref["ST_up"].to_numpy(dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        artifacts.frame["ST_dn"].to_numpy(dtype=np.float64),
        st_ref["ST_dn"].to_numpy(dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(artifacts.raw_events["evt_st_buy"].to_numpy(), st_ref["ST_buy"].to_numpy())
    np.testing.assert_array_equal(artifacts.raw_events["evt_st_sell"].to_numpy(), st_ref["ST_sell"].to_numpy())
    st_change = (artifacts.frame["ST_trend"] != artifacts.frame["ST_trend"].shift(1)).astype("uint8")
    np.testing.assert_array_equal(st_change.to_numpy(dtype=np.uint8), st_ref["ST_change"].to_numpy(dtype=np.uint8))
