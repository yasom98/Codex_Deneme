"""Integration-style tests for feature build CLI and health-gated writes."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.features import PIVOT_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "make_features.py"


def _healthy_df(rows: int = 420) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="1D", tz="UTC")
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


def _invalid_df(rows: int = 420) -> pd.DataFrame:
    df = _healthy_df(rows)
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    return df


def _write_config(config_path: Path, input_root: Path, runs_root: Path, first_session_fill: str = "none") -> None:
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {input_root}",
                f"runs_root: {runs_root}",
                'parquet_glob: "*.parquet"',
                "seed: 42",
                "",
                "supertrend:",
                "  periods: 10",
                "  multiplier: 3.0",
                "  source: hl2",
                "  change_atr_method: true",
                "",
                "alphatrend:",
                "  coeff: 3.0",
                "  ap: 11",
                "  use_no_volume: false",
                "",
                "pivot:",
                "  pivot_tf: 1D",
                "  warmup_policy: allow_first_session_nan",
                f"  first_session_fill: {first_session_fill}",
                "",
                "parity:",
                "  enabled: true",
                "  sample_rows: 128",
                "  float_atol: 1.0e-6",
                "  float_rtol: 1.0e-6",
                "",
                "health:",
                "  warn_ratio: 1.0",
                "  critical_warn_ratio: 1.0",
                "  critical_columns:",
                "    - EMA_200",
                "    - AlphaTrend",
            ]
        ),
        encoding="utf-8",
    )


def _load_main() -> object:
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
    monkeypatch.setattr(sys, "argv", ["make_features.py", "--config", str(config_path), "--run-id", "unit_run"])

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
    assert per_file_payload["indicator_parity_status"] == "passed"
    assert per_file_payload["pivot_first_session_allowed_nan"] is True
    assert per_file_payload["pivot_first_session_rows"] == 1
    assert per_file_payload["pivot_nonnull_ratio_after_first_session"] == 1.0
    assert per_file_payload["pivot_fill_strategy_applied"] == "none"
    assert isinstance(per_file_payload["formula_fingerprints"], dict)
    assert "alphatrend" in per_file_payload["formula_fingerprints"]
    assert isinstance(per_file_payload["formula_fingerprint_bundle"], str)
    assert per_file_payload["strict_parity_enabled"] is True

    assert summary_payload["succeeded_files"] == 1
    assert summary_payload["failed_files"] == 0
    assert summary_payload["parity_status_overall"] is True
    assert summary_payload["strict_parity"] is True
    assert isinstance(summary_payload["formula_fingerprints"], dict)
    assert isinstance(summary_payload["formula_fingerprint_bundle"], str)
    assert "indicator_spec_version" in summary_payload
    assert "config_hash" in summary_payload
    assert not list(run_root.rglob("*.tmp"))


def test_make_features_cli_fails_on_empty_pivot_mapping(monkeypatch: object, tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    (input_root / "pivot_bug.parquet").write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "features.yaml"
    _write_config(config_path, input_root=input_root, runs_root=runs_root)

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        del path
        return _healthy_df()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("ok", encoding="utf-8")

    def fake_all_nan_pivots(
        df: pd.DataFrame,
        warmup_policy: str = "allow_first_session_nan",
        first_session_fill: str = "none",
        pivot_tf: str = "1D",
        assume_validated: bool = False,
        indexed_ohlcv: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del warmup_policy, first_session_fill, pivot_tf, assume_validated, indexed_ohlcv
        return pd.DataFrame({col: np.full(len(df), np.nan, dtype=np.float32) for col in PIVOT_FEATURE_COLUMNS}, index=df.index)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr("data.features.compute_daily_pivots_with_std_bands", fake_all_nan_pivots)

    main = _load_main()
    monkeypatch.setattr(sys, "argv", ["make_features.py", "--config", str(config_path), "--run-id", "pivot_fail_run"])

    exit_code = int(main())
    assert exit_code == 1

    run_root = runs_root / "pivot_fail_run" / "data_features"
    out_parquet = run_root / "parquet" / "pivot_bug.parquet"
    per_file_report = run_root / "reports" / "per_file" / "pivot_bug.json"
    summary_report = run_root / "reports" / "summary.json"

    assert not out_parquet.exists()
    assert per_file_report.exists()
    assert summary_report.exists()

    per_file_payload = json.loads(per_file_report.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_report.read_text(encoding="utf-8"))

    assert per_file_payload["status"] == "failed"
    assert any(error["code"] == "PIVOT_MAPPING_EMPTY" for error in per_file_payload["errors"])
    assert summary_payload["failed_files"] == 1


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
        return _invalid_df()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    main = _load_main()
    monkeypatch.setattr(sys, "argv", ["make_features.py", "--config", str(config_path), "--run-id", "fail_run"])

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
    assert len(per_file_payload["errors"]) > 0
    assert summary_payload["failed_files"] == 1


def test_make_features_cli_fails_on_parity_mismatch_when_strict(monkeypatch: object, tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    (input_root / "parity_strict.parquet").write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "features.yaml"
    _write_config(config_path, input_root=input_root, runs_root=runs_root)

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        del path
        return _healthy_df()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("ok", encoding="utf-8")

    def fake_parity(*_: object, **__: object) -> tuple[str, dict[str, bool]]:
        return "failed", {"EMA_200": False}

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr("data.features.evaluate_indicator_parity", fake_parity)

    main = _load_main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_features.py", "--config", str(config_path), "--run-id", "parity_strict_run"],
    )

    exit_code = int(main())
    assert exit_code == 1

    run_root = runs_root / "parity_strict_run" / "data_features"
    per_file_report = run_root / "reports" / "per_file" / "parity_strict.json"
    summary_report = run_root / "reports" / "summary.json"

    per_file_payload = json.loads(per_file_report.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_report.read_text(encoding="utf-8"))

    assert per_file_payload["status"] == "failed"
    assert per_file_payload["parity_ok"] is False
    assert per_file_payload["parity_gate_ok"] is False
    assert per_file_payload["strict_parity_enabled"] is True
    assert any(error["code"] == "INDICATOR_PARITY_FAILED" for error in per_file_payload["errors"])
    assert summary_payload["failed_files"] == 1


def test_make_features_cli_allows_parity_mismatch_when_not_strict(monkeypatch: object, tmp_path: Path) -> None:
    input_root = tmp_path / "in"
    runs_root = tmp_path / "runs"
    input_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    (input_root / "parity_relaxed.parquet").write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "features.yaml"
    _write_config(config_path, input_root=input_root, runs_root=runs_root)

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        del path
        return _healthy_df()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("ok", encoding="utf-8")

    def fake_parity(*_: object, **__: object) -> tuple[str, dict[str, bool]]:
        return "failed", {"EMA_200": False}

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr("data.features.evaluate_indicator_parity", fake_parity)

    main = _load_main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_features.py",
            "--config",
            str(config_path),
            "--run-id",
            "parity_relaxed_run",
            "--strict-parity",
            "false",
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    run_root = runs_root / "parity_relaxed_run" / "data_features"
    per_file_report = run_root / "reports" / "per_file" / "parity_relaxed.json"
    summary_report = run_root / "reports" / "summary.json"

    per_file_payload = json.loads(per_file_report.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_report.read_text(encoding="utf-8"))

    assert per_file_payload["status"] == "success"
    assert per_file_payload["parity_ok"] is False
    assert per_file_payload["parity_gate_ok"] is True
    assert per_file_payload["strict_parity_enabled"] is False
    assert not any(error["code"] == "INDICATOR_PARITY_FAILED" for error in per_file_payload["errors"])
    assert summary_payload["failed_files"] == 0
    assert summary_payload["strict_parity"] is False
