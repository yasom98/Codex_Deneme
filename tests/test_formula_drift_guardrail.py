"""Tests for formula drift guardrail fingerprints and locked parameters."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from data.features import (
    AlphaTrendConfig,
    FeatureBuildConfig,
    HealthPolicyConfig,
    INDICATOR_SPEC_VERSION,
    ParityPolicyConfig,
    PivotPolicyConfig,
    SuperTrendConfig,
    compute_formula_fingerprint_bundle,
    compute_formula_fingerprints,
    validate_feature_config,
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
        parity=ParityPolicyConfig(enabled=True, sample_rows=128, float_atol=1e-6, float_rtol=1e-6),
        health=HealthPolicyConfig(warn_ratio=1.0, critical_warn_ratio=1.0, critical_columns=("EMA_200",)),
        config_hash="unit",
        indicator_spec_version=INDICATOR_SPEC_VERSION,
    )


def test_formula_fingerprints_are_stable() -> None:
    fingerprints = compute_formula_fingerprints(_cfg())
    bundle = compute_formula_fingerprint_bundle(fingerprints)

    expected = {
        "pivot_traditional": "81d8d9d788f5c1a2",
        "ema_set": "aa55f2254ca6982c",
        "alphatrend": "fac20ec70c1a42d7",
        "supertrend": "2f8a5eb24c0520d2",
        "event_shift_policy": "88f0d99bf92ffa4c",
    }
    expected_bundle = "5e9cc14340f9bc5c"

    assert fingerprints == expected
    assert bundle == expected_bundle


def test_locked_parameter_modification_is_rejected() -> None:
    cfg = _cfg()

    with pytest.raises(ValueError, match="supertrend\\.source must be fixed at hl2"):
        validate_feature_config(replace(cfg, supertrend=replace(cfg.supertrend, source="close")))

    with pytest.raises(ValueError, match="alphatrend\\.coeff must be fixed at 3.0"):
        validate_feature_config(replace(cfg, alphatrend=replace(cfg.alphatrend, coeff=2.5)))

    with pytest.raises(ValueError, match="pivot\\.pivot_tf must be fixed at 1D"):
        validate_feature_config(replace(cfg, pivot=replace(cfg.pivot, pivot_tf="4H")))
