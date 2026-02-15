"""Regression tests for strict AlphaTrend parameter lock in config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.features import load_feature_config


def _write_config(tmp_path: Path, period_line: str, atr_line: str) -> Path:
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
                "  period: 10",
                "  multiplier: 3.0",
                "",
                "alphatrend:",
                period_line,
                atr_line,
                "  signal_period: 14",
                "  long_rule:",
                "    signal: mfi",
                "    op: \">=\"",
                "    threshold: 50.0",
                "  short_rule:",
                "    signal: mfi",
                "    op: \"<\"",
                "    threshold: 50.0",
                "",
                "rsi:",
                "  period: 14",
                "  slope_lag: 3",
                "  zscore_window: 50",
                "",
                "events:",
                "  rsi_centerline: 50.0",
                "  rsi_overbought: 70.0",
                "  rsi_oversold: 30.0",
                "",
                "health:",
                "  warn_ratio: 0.005",
                "  critical_warn_ratio: 0.001",
                "  critical_columns:",
                "    - supertrend",
                "    - alphatrend",
                "    - rsi",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_alphatrend_lock_accepts_only_period_11_multiplier_3(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "  period: 11", "  atr_multiplier: 3.0")
    cfg = load_feature_config(config_path)
    assert cfg.alphatrend.period == 11
    assert cfg.alphatrend.atr_multiplier == 3.0


def test_alphatrend_lock_rejects_wrong_period(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "  period: 10", "  atr_multiplier: 3.0")
    with pytest.raises(ValueError, match="alphatrend\\.period must be fixed at 11"):
        load_feature_config(config_path)


def test_alphatrend_lock_requires_atr_multiplier_field(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "  period: 11", "  # atr_multiplier intentionally missing")
    with pytest.raises(ValueError, match="alphatrend\\.atr_multiplier is required"):
        load_feature_config(config_path)
