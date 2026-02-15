"""Regression tests for strict indicator parameter lock in config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.features import load_feature_config


def _write_config(tmp_path: Path, coeff_line: str, ap_line: str, supertrend_periods_line: str = "  periods: 10") -> Path:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "features.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {input_root}",
                f"runs_root: {runs_root}",
                'parquet_glob: "*.parquet"',
                "seed: 42",
                "",
                "supertrend:",
                supertrend_periods_line,
                "  multiplier: 3.0",
                "  source: hl2",
                "  change_atr_method: true",
                "",
                "alphatrend:",
                coeff_line,
                ap_line,
                "  use_no_volume: false",
                "",
                "pivot:",
                "  pivot_tf: 1D",
                "  warmup_policy: allow_first_session_nan",
                "  first_session_fill: none",
                "",
                "parity:",
                "  enabled: true",
                "  sample_rows: 128",
                "  float_atol: 1.0e-6",
                "  float_rtol: 1.0e-6",
                "",
                "health:",
                "  warn_ratio: 0.005",
                "  critical_warn_ratio: 0.001",
                "  critical_columns:",
                "    - EMA_200",
                "    - AlphaTrend",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_indicator_lock_accepts_only_alphatrend_3_11(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "  coeff: 3.0", "  ap: 11")
    cfg = load_feature_config(config_path)
    assert cfg.alphatrend.coeff == 3.0
    assert cfg.alphatrend.ap == 11


def test_indicator_lock_rejects_wrong_alphatrend_ap(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "  coeff: 3.0", "  ap: 10")
    with pytest.raises(ValueError, match="alphatrend\\.ap must be fixed at 11"):
        load_feature_config(config_path)


def test_indicator_lock_requires_alphatrend_coeff_field(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "  # coeff intentionally missing", "  ap: 11")
    with pytest.raises(ValueError, match="alphatrend\\.coeff is required"):
        load_feature_config(config_path)


def test_indicator_lock_rejects_supertrend_override(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "  coeff: 3.0", "  ap: 11", supertrend_periods_line="  periods: 12")
    with pytest.raises(ValueError, match="supertrend\\.periods must be fixed at 10"):
        load_feature_config(config_path)
