"""Benchmark-lite coverage for indicator computation on small/medium fixtures."""

from __future__ import annotations

from pathlib import Path
import time

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


def _fixture_df(rows: int) -> pd.DataFrame:
    ts = pd.date_range("2024-03-01", periods=rows, freq="1min", tz="UTC")
    base = np.linspace(100.0, 130.0, rows)
    wave = np.sin(np.linspace(0.0, 30.0, rows))
    close = base + wave
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close + 0.1).astype(np.float32),
            "high": (close + 0.6).astype(np.float32),
            "low": (close - 0.6).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.linspace(1000.0, 1700.0, rows).astype(np.float32),
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


def test_indicator_benchmark_lite_small_medium() -> None:
    cfg = _cfg()

    timings: dict[str, float] = {}
    for name, rows in (("small", 2_000), ("medium", 20_000)):
        df = _fixture_df(rows)
        t0 = time.perf_counter()
        artifacts = build_feature_artifacts(df, cfg)
        timings[name] = time.perf_counter() - t0
        assert artifacts.indicator_parity_status == "passed"
        assert len(artifacts.frame) == rows

    assert timings["small"] > 0.0
    assert timings["medium"] > 0.0
    assert timings["medium"] >= timings["small"] * 0.5
