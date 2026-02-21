"""Tests for RL feature manifest payload stability."""

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
    RLFeatureConfig,
    SuperTrendConfig,
    build_feature_artifacts,
    build_feature_manifest_payload,
)


def _sample_ohlcv(rows: int = 1600) -> pd.DataFrame:
    ts = pd.date_range("2024-03-01", periods=rows, freq="5min", tz="UTC")
    base = np.linspace(30000.0, 32000.0, rows)
    wave = np.sin(np.linspace(0.0, 28.0, rows))
    close = base + wave * 40.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close + 4.0).astype(np.float32),
            "high": (close + 20.0).astype(np.float32),
            "low": (close - 20.0).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.linspace(500.0, 900.0, rows).astype(np.float32),
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
        parity=ParityPolicyConfig(enabled=True, sample_rows=256, float_atol=1e-6, float_rtol=1e-6),
        health=HealthPolicyConfig(warn_ratio=1.0, critical_warn_ratio=1.0, critical_columns=("EMA_200",)),
        config_hash="unit",
        indicator_spec_version=INDICATOR_SPEC_VERSION,
        rl_features=RLFeatureConfig(rolling_vol_windows=(20, 50), zscore_window=50, zscore_source="return", volatility_regime_bins=3),
    )


def test_manifest_payload_contains_required_groups_and_dtypes() -> None:
    cfg = _cfg()
    artifacts = build_feature_artifacts(_sample_ohlcv(), cfg)
    out = artifacts.frame

    payload = build_feature_manifest_payload(
        run_id="manifest_run",
        cfg=cfg,
        feature_groups=artifacts.feature_groups,
        column_dtypes={col: str(dtype) for col, dtype in out.dtypes.items()},
        row_count=len(out),
        date_min_utc=out["timestamp"].iloc[0].isoformat(),
        date_max_utc=out["timestamp"].iloc[-1].isoformat(),
        formula_fingerprints=artifacts.formula_fingerprints,
        formula_fingerprint_bundle=artifacts.formula_fingerprint_bundle,
    )

    assert payload["manifest_version"].startswith("features.manifest.")
    assert payload["run_id"] == "manifest_run"
    assert payload["timestamp_column"] == "timestamp"
    assert payload["row_count"] == len(out)
    assert payload["date_range"]["min_utc"] == out["timestamp"].iloc[0].isoformat()
    assert payload["date_range"]["max_utc"] == out["timestamp"].iloc[-1].isoformat()
    assert "price_derived" in payload["feature_groups"]
    assert "trend" in payload["feature_groups"]
    assert "regime" in payload["feature_groups"]
    assert "event" in payload["feature_groups"]
    assert "placeholders" in payload["feature_groups"]
    assert payload["column_dtypes"]["rolling_vol_20"] == "float32"
    assert payload["column_dtypes"]["trend_regime"] == "uint8"
    assert payload["column_dtypes"]["position_placeholder"] == "uint8"
    assert payload["column_dtypes"]["floating_pnl_placeholder"] == "float32"
    assert "position_placeholder" in payload["placeholder_columns"]
    assert payload["warmup_policy"]["rolling_vol_windows"] == [20, 50]
    assert payload["warmup_policy"]["zscore_window"] == 50
    assert payload["warmup_policy"]["zscore_source"] == "return"
