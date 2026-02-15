"""Integration-style tests for feature build CLI and health-gated writes."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "make_features.py"


def _healthy_df(rows: int = 200) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC")
    base = np.linspace(100.0, 140.0, rows)
    wiggle = np.sin(np.linspace(0.0, 12.0, rows))
    close = base + wiggle

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close + 0.2).astype(np.float32),
            "high": (close + 0.7).astype(np.float32),
            "low": (close - 0.7).astype(np.float32),
            "close": close.astype(np.float32),
            "volume": np.linspace(1000.0, 1700.0, rows).astype(np.float32),
        }
    )


def _nan_df(rows: int = 60) -> pd.DataFrame:
    df = _healthy_df(rows)
    df.loc[:9, "close"] = np.nan
    return df


def _write_config(config_path: Path, input_root: Path, runs_root: Path) -> None:
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
                "  period: 14",
                "  atr_multiplier: 1.0",
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


def _load_main():
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_make_features_cli_writes_outputs_and_reports(monkeypatch: object, tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    (input_root / "sample.parquet").write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "features.yaml"
    _write_config(config_path, input_root=input_root, runs_root=runs_root)

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        del path
        return _healthy_df()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    main = _load_main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_features.py", "--config", str(config_path), "--run-id", "unit_run"],
    )

    exit_code = int(main())
    assert exit_code == 0

    run_root = runs_root / "unit_run" / "data_features"
    out_parquet = run_root / "parquet" / "sample.parquet"
    per_file_report = run_root / "reports" / "per_file" / "sample.json"
    summary_report = run_root / "reports" / "summary.json"

    assert out_parquet.exists()
    assert per_file_report.exists()
    assert summary_report.exists()

    per_file_payload = json.loads(per_file_report.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_report.read_text(encoding="utf-8"))

    assert per_file_payload["status"] == "success"
    assert summary_payload["succeeded_files"] == 1
    assert summary_payload["failed_files"] == 0
    assert not list(run_root.rglob("*.tmp"))


def test_make_features_cli_blocks_output_when_health_fails(monkeypatch: object, tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    (input_root / "broken.parquet").write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "features.yaml"
    _write_config(config_path, input_root=input_root, runs_root=runs_root)

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        del path
        return _nan_df()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    main = _load_main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_features.py", "--config", str(config_path), "--run-id", "fail_run"],
    )

    exit_code = int(main())
    assert exit_code == 1

    run_root = runs_root / "fail_run" / "data_features"
    out_parquet = run_root / "parquet" / "broken.parquet"
    per_file_report = run_root / "reports" / "per_file" / "broken.json"
    summary_report = run_root / "reports" / "summary.json"

    assert not out_parquet.exists()
    assert per_file_report.exists()
    assert summary_report.exists()

    per_file_payload = json.loads(per_file_report.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_report.read_text(encoding="utf-8"))

    assert per_file_payload["status"] == "failed"
    assert any(error["code"] == "NAN_RATIO_TOO_HIGH" for error in per_file_payload["errors"])
    assert summary_payload["failed_files"] == 1
