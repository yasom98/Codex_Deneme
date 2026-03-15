"""Contract tests for canonical tail refresh Phase A and Phase B."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from data.tail_refresh import RefreshOptions, run_phase_a, run_phase_b


def _write_data_config(config_path: Path, input_root: Path, runs_root: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {input_root}",
                f"runs_root: {runs_root}",
                'csv_glob: "*.csv"',
                "timestamp_aliases:",
                "  - timestamp",
                "  - ts",
                "  - datetime",
                "  - time",
                "required_columns:",
                "  - open",
                "  - high",
                "  - low",
                "  - close",
                "  - volume",
                "float_columns:",
                "  - open",
                "  - high",
                "  - low",
                "  - close",
                "  - volume",
                "duplicate_policy: last",
                "seed: 42",
            ]
        ),
        encoding="utf-8",
    )


def _write_features_config(config_path: Path, runs_root: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {runs_root / 'placeholder' / 'data_standardized' / 'parquet'}",
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
                "  first_session_fill: ffill_from_second_session",
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
            ]
        ),
        encoding="utf-8",
    )


def _write_raw_csv(path: Path, timeframe: str) -> None:
    if timeframe == "1m":
        path.write_text(
            "\n".join(
                [
                    "ts,open,high,low,close,volume",
                    "2024-01-01 00:00:00+00:00,10,11,9,10.5,100",
                    "2024-01-01 00:01:00+00:00,11,12,10,11.5,101",
                ]
            ),
            encoding="utf-8",
        )
        return
    if timeframe == "5m":
        path.write_text(
            "\n".join(
                [
                    "timestamp,open,high,low,close,volume",
                    "2024-01-01 00:00:00+00:00,20,21,19,20.5,200",
                    "2024-01-01 00:05:00+00:00,21,22,20,21.5,201",
                ]
            ),
            encoding="utf-8",
        )
        return
    if timeframe == "15m":
        path.write_text(
            "\n".join(
                [
                    "ts;open;high;low;close;volume",
                    "2024-01-01 00:00:00+00:00;30;31;29;30.5;300",
                    "2024-01-01 00:15:00+00:00;31;32;30;31.5;301",
                ]
            ),
            encoding="utf-8",
        )
        return
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _accepted_feature_manifest(run_id: str) -> dict[str, object]:
    return {
        "manifest_version": "features.manifest.v1",
        "run_id": run_id,
        "feature_groups": {"raw_ohlcv": ["timestamp", "open", "high", "low", "close", "volume"]},
        "column_dtypes": {
            "timestamp": "datetime64[ns, UTC]",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float32",
        },
        "event_columns": [],
        "continuous_columns": ["open", "high", "low", "close", "volume"],
        "placeholder_columns": [],
        "warmup_policy": {"pivot_warmup_policy": "allow_first_session_nan"},
        "indicator_spec_version": "indicators.vtest",
        "config_hash": "cfg-hash",
        "formula_fingerprint_bundle": "bundle-hash",
    }


def _seed_project(tmp_path: Path, *, accepted_run_id: str = "accepted_ref") -> tuple[Path, Path, Path]:
    project_root = tmp_path
    configs_root = project_root / "configs"
    runs_root = project_root / "runs"
    configs_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    data_config = configs_root / "data.yaml"
    features_config = configs_root / "features.yaml"
    _write_data_config(data_config, project_root, runs_root)
    _write_features_config(features_config, runs_root)

    for timeframe in ("1m", "5m", "15m"):
        _write_raw_csv(project_root / f"BTC_USDT_{timeframe}_price_data.csv", timeframe)

    standardized_root = runs_root / accepted_run_id / "data_standardized"
    (standardized_root / "reports" / "per_file").mkdir(parents=True, exist_ok=True)
    (standardized_root / "parquet").mkdir(parents=True, exist_ok=True)
    (standardized_root / "reports" / "summary.json").write_text(
        json.dumps({"total_files": 3, "succeeded_files": 3, "failed_files": 0, "run_id": accepted_run_id}),
        encoding="utf-8",
    )

    report_rows = {
        "1m": ("2024-01-01T00:00:00+00:00", "2024-01-01T00:01:00+00:00"),
        "5m": ("2024-01-01T00:00:00+00:00", "2024-01-01T00:05:00+00:00"),
        "15m": ("2024-01-01T00:00:00+00:00", "2024-01-01T00:15:00+00:00"),
    }
    for timeframe, (ts_min, ts_max) in report_rows.items():
        parquet_path = standardized_root / "parquet" / f"BTC_USDT_{timeframe}_price_data.parquet"
        parquet_path.write_text("placeholder", encoding="utf-8")
        report_payload = {
            "input_file": str((project_root / f"BTC_USDT_{timeframe}_price_data.csv").resolve()),
            "output_file": str(parquet_path.resolve()),
            "rows_in": 2,
            "rows_out": 2,
            "status": "success",
            "timestamp_min": ts_min,
            "timestamp_max": ts_max,
        }
        (standardized_root / "reports" / "per_file" / f"BTC_USDT_{timeframe}_price_data.json").write_text(
            json.dumps(report_payload),
            encoding="utf-8",
        )

    feature_reports_root = runs_root / accepted_run_id / "data_features" / "reports"
    feature_reports_root.mkdir(parents=True, exist_ok=True)
    feature_manifest = _accepted_feature_manifest(accepted_run_id)
    (feature_reports_root / "feature_manifest.json").write_text(json.dumps(feature_manifest), encoding="utf-8")
    (feature_reports_root / "train_input_validation_report.json").write_text(
        json.dumps({"run_id": accepted_run_id, "train_input_validation_overall": True}),
        encoding="utf-8",
    )
    (feature_reports_root / "split_validation_report.json").write_text(
        json.dumps(
            {
                "run_id": accepted_run_id,
                "split_mode": "ratio_chrono",
                "invocation_args": {
                    "require_train_input_validation": True,
                    "min_train_rows": 1,
                    "min_val_rows": 1,
                    "min_test_rows": 1,
                    "warmup_rows": 0,
                    "split_overrides": {"train_ratio": "0.70", "val_ratio": "0.15", "test_ratio": "0.15"},
                },
            }
        ),
        encoding="utf-8",
    )

    datasets_reports_root = runs_root / accepted_run_id / "data_datasets" / "reports"
    datasets_reports_root.mkdir(parents=True, exist_ok=True)
    (datasets_reports_root / "dataset_build_report.json").write_text(
        json.dumps(
            {
                "run_id": accepted_run_id,
                "invocation_args": {
                    "overwrite": False,
                    "require_train_input_validation": True,
                    "require_split_validation": True,
                    "aggregate_walk_forward": False,
                    "execution_price_column": "close",
                    "mark_to_market_column": "close",
                },
            }
        ),
        encoding="utf-8",
    )

    states_reports_root = runs_root / accepted_run_id / "data_states" / "reports"
    states_reports_root.mkdir(parents=True, exist_ok=True)
    (states_reports_root / "state_build_report.json").write_text(
        json.dumps(
            {
                "run_id": accepted_run_id,
                "invocation_args": {
                    "overwrite": False,
                    "enable_scaling": False,
                    "scaler_type": "none",
                    "build_mode": "materialize_only",
                    "strict_column_selection": True,
                    "sequence_mode": False,
                    "aggregate_walk_forward": False,
                    "execution_price_column": "close",
                    "mark_to_market_column": "close",
                },
            }
        ),
        encoding="utf-8",
    )
    return project_root, data_config, features_config


def _options(project_root: Path, data_config: Path, features_config: Path, *, accepted_run_id: str = "accepted_ref", refresh_session_id: str = "refresh_001", exchange: str | None = "okx", market_type: str | None = "spot", symbol: str | None = "BTC/USDT") -> RefreshOptions:
    probe_report_path = project_root / "runs" / "probe_gate" / "data_tail_refresh" / "reports" / "provider_capability_report.json"
    probe_report_path.parent.mkdir(parents=True, exist_ok=True)
    probe_report_path.write_text(
        json.dumps(
            {
                "source_of_truth_verdict": {"status": "proven_canonical_raw"},
                "canonical_exchange_verdict": {"status": "explicit_override", "value": "okx"},
                "market_type_verdict": {"status": "explicit_override", "value": "spot"},
                "symbol_verdict": {"status": "explicit_override", "exchange_symbol": "BTC/USDT"},
                "provider_results": [
                    {
                        "provider": "okx",
                        "probe_status": "success",
                        "retrieval_worked_in_practice": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return RefreshOptions(
        project_root=project_root,
        accepted_run_id=accepted_run_id,
        refresh_session_id=refresh_session_id,
        data_config_path=data_config,
        features_config_path=features_config,
        provider_probe_report_path=probe_report_path,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        fallback_exchanges=(),
        request_limit=100,
        max_retries=2,
        retry_backoff_seconds=0.0,
        overlap_abs_tolerance=0.0,
        python_executable=Path("/tmp/fake-python"),
        log_level="INFO",
    )


def _phase_a_frames() -> dict[str, pd.DataFrame]:
    return {
        "1m": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"],
                    utc=True,
                ),
                "open": [10.0, 11.0, 12.0],
                "high": [11.0, 12.0, 13.0],
                "low": [9.0, 10.0, 11.0],
                "close": [10.5, 11.5, 12.5],
                "volume": [100.0, 101.0, 102.0],
            }
        ),
        "5m": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z", "2024-01-01T00:10:00Z"],
                    utc=True,
                ),
                "open": [20.0, 21.0, 22.0],
                "high": [21.0, 22.0, 23.0],
                "low": [19.0, 20.0, 21.0],
                "close": [20.5, 21.5, 22.5],
                "volume": [200.0, 201.0, 202.0],
            }
        ),
        "15m": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01T00:00:00Z", "2024-01-01T00:15:00Z", "2024-01-01T00:30:00Z"],
                    utc=True,
                ),
                "open": [30.0, 31.0, 32.0],
                "high": [31.0, 32.0, 33.0],
                "low": [29.0, 30.0, 31.0],
                "close": [30.5, 31.5, 32.5],
                "volume": [300.0, 301.0, 302.0],
            }
        ),
    }


def test_phase_a_fails_closed_when_market_provenance_missing(tmp_path: Path) -> None:
    project_root, data_config, features_config = _seed_project(tmp_path)
    options = _options(project_root, data_config, features_config, exchange=None, market_type=None, symbol=None)

    result = run_phase_a(options)

    assert result["status"] == "failed"
    report = json.loads((project_root / "runs" / "refresh_001" / "data_tail_refresh" / "reports" / "data_tail_refresh_report.json").read_text(encoding="utf-8"))
    error_codes = {item["code"] for item in report["errors"]}
    assert report["source_of_truth_status"] == "proven_canonical_raw"
    assert {"EXCHANGE_PROVENANCE_MISSING", "MARKET_TYPE_PROVENANCE_MISSING", "SYMBOL_PROVENANCE_MISSING"} <= error_codes


def test_phase_a_creates_versioned_snapshot_after_successful_validation(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config, features_config = _seed_project(tmp_path)
    options = _options(project_root, data_config, features_config)
    original_1m = (project_root / "BTC_USDT_1m_price_data.csv").read_text(encoding="utf-8")

    monkeypatch.setattr(
        "data.tail_refresh._download_timeframes",
        lambda *args, **kwargs: {"status": "success", "exchange_used": "okx", "attempts": [{"exchange": "okx", "status": "success"}], "frames": _phase_a_frames()},
    )
    monkeypatch.setattr(
        "data.tail_refresh._last_closed_candle_start",
        lambda now_utc, timeframe: {
            "1m": pd.Timestamp("2024-01-01T00:02:00Z"),
            "5m": pd.Timestamp("2024-01-01T00:10:00Z"),
            "15m": pd.Timestamp("2024-01-01T00:30:00Z"),
        }[timeframe],
    )

    result = run_phase_a(options)

    assert result["status"] == "success"
    snapshot_root = project_root / "runs" / "refresh_001" / "data_tail_refresh" / "canonical_raw_snapshot"
    assert snapshot_root.exists()
    assert (snapshot_root / "BTC_USDT_1m_price_data.csv").exists()
    assert (snapshot_root / "BTC_USDT_5m_price_data.csv").exists()
    assert (snapshot_root / "BTC_USDT_15m_price_data.csv").exists()
    assert (project_root / "BTC_USDT_1m_price_data.csv").read_text(encoding="utf-8") == original_1m

    snapshot_15m = (snapshot_root / "BTC_USDT_15m_price_data.csv").read_text(encoding="utf-8")
    assert ";" in snapshot_15m.splitlines()[0]
    assert "2024-01-01 00:30:00+00:00;32.0;33.0;31.0;32.5;302.0" in snapshot_15m

    report = json.loads((project_root / "runs" / "refresh_001" / "data_tail_refresh" / "reports" / "data_tail_refresh_report.json").read_text(encoding="utf-8"))
    assert report["phase_a_status"] == "success"
    assert report["exchange_used"] == "okx"
    assert report["market_type"] == "spot"
    assert report["merged_rows"] == 3


def test_phase_a_fails_closed_on_material_overlap_mismatch(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config, features_config = _seed_project(tmp_path)
    options = _options(project_root, data_config, features_config)
    bad_frames = _phase_a_frames()
    bad_frames["1m"].loc[1, "close"] = 999.0

    monkeypatch.setattr(
        "data.tail_refresh._download_timeframes",
        lambda *args, **kwargs: {"status": "success", "exchange_used": "okx", "attempts": [{"exchange": "okx", "status": "success"}], "frames": bad_frames},
    )
    monkeypatch.setattr(
        "data.tail_refresh._last_closed_candle_start",
        lambda now_utc, timeframe: {
            "1m": pd.Timestamp("2024-01-01T00:02:00Z"),
            "5m": pd.Timestamp("2024-01-01T00:10:00Z"),
            "15m": pd.Timestamp("2024-01-01T00:30:00Z"),
        }[timeframe],
    )

    result = run_phase_a(options)

    assert result["status"] == "failed"
    snapshot_root = project_root / "runs" / "refresh_001" / "data_tail_refresh" / "canonical_raw_snapshot"
    assert not snapshot_root.exists()

    merge_report = json.loads((project_root / "runs" / "refresh_001" / "data_tail_refresh" / "reports" / "canonical_merge_report.json").read_text(encoding="utf-8"))
    assert merge_report["status"] == "failed"
    assert merge_report["errors"][0]["code"] == "OVERLAP_MISMATCH"


def test_phase_b_runs_explicit_rebuild_chain_and_checks_feature_contract(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config, features_config = _seed_project(tmp_path)
    options = _options(project_root, data_config, features_config)

    refresh_root = project_root / "runs" / "refresh_001"
    raw_snapshot_root = refresh_root / "data_tail_refresh" / "canonical_raw_snapshot"
    raw_snapshot_root.mkdir(parents=True, exist_ok=True)
    for timeframe in ("1m", "5m", "15m"):
        shutil.copyfile(project_root / f"BTC_USDT_{timeframe}_price_data.csv", raw_snapshot_root / f"BTC_USDT_{timeframe}_price_data.csv")

    phase_a_report_path = refresh_root / "data_tail_refresh" / "reports" / "data_tail_refresh_report.json"
    phase_a_report_path.parent.mkdir(parents=True, exist_ok=True)
    phase_a_report_path.write_text(
        json.dumps(
            {
                "refresh_session_id": "refresh_001",
                "phase_a_status": "success",
                "phase_b_status": "not_started",
                "rebuild_status": "not_started",
            }
        ),
        encoding="utf-8",
    )
    phase_a_result = {
        "status": "success",
        "phase_a_status": "success",
        "phase_b_status": "not_started",
        "main_report_path": phase_a_report_path,
        "raw_snapshot_root": raw_snapshot_root,
    }

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    accepted_manifest = _accepted_feature_manifest("accepted_ref")

    def fake_run_subprocess(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        del cwd
        script_name = Path(command[1]).name
        if script_name == "make_features.py":
            feature_reports_root = project_root / "runs" / "refresh_001" / "data_features" / "reports"
            feature_reports_root.mkdir(parents=True, exist_ok=True)
            (feature_reports_root / "summary.json").write_text(json.dumps({"run_id": "refresh_001"}), encoding="utf-8")
            refreshed_manifest = dict(accepted_manifest)
            refreshed_manifest["run_id"] = "refresh_001"
            (feature_reports_root / "feature_manifest.json").write_text(json.dumps(refreshed_manifest), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="feature ok", stderr="")
        if script_name == "validate_train_inputs.py":
            feature_reports_root = project_root / "runs" / "refresh_001" / "data_features" / "reports"
            (feature_reports_root / "train_input_validation_report.json").write_text(
                json.dumps({"run_id": "refresh_001", "train_input_validation_overall": True}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="train input ok", stderr="")
        if script_name == "validate_splits.py":
            feature_reports_root = project_root / "runs" / "refresh_001" / "data_features" / "reports"
            (feature_reports_root / "split_validation_report.json").write_text(
                json.dumps({"run_id": "refresh_001", "split_validation_overall": True}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="split ok", stderr="")
        if script_name == "build_datasets.py":
            dataset_reports_root = project_root / "runs" / "refresh_001" / "data_datasets" / "reports"
            dataset_reports_root.mkdir(parents=True, exist_ok=True)
            (dataset_reports_root / "dataset_build_report.json").write_text(
                json.dumps({"run_id": "refresh_001", "dataset_build_overall": True}),
                encoding="utf-8",
            )
            (dataset_reports_root / "dataset_manifest.json").write_text(
                json.dumps({"run_id": "refresh_001", "manifest_version": "datasets.manifest.v1"}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="dataset ok", stderr="")
        if script_name == "build_states.py":
            state_reports_root = project_root / "runs" / "refresh_001" / "data_states" / "reports"
            state_reports_root.mkdir(parents=True, exist_ok=True)
            (state_reports_root / "state_build_report.json").write_text(
                json.dumps({"run_id": "refresh_001", "state_build_overall": True}),
                encoding="utf-8",
            )
            (state_reports_root / "state_manifest.json").write_text(
                json.dumps({"run_id": "refresh_001", "manifest_version": "states.manifest.v1"}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="state ok", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("data.tail_refresh._run_subprocess", fake_run_subprocess)

    result = run_phase_b(options, phase_a_result)

    assert result["status"] == "success"
    rebuild_summary = json.loads((project_root / "runs" / "refresh_001" / "data_tail_refresh" / "reports" / "rebuild_summary.json").read_text(encoding="utf-8"))
    assert rebuild_summary["status"] == "success"
    assert rebuild_summary["feature_contract_compatibility"]["status"] == "success"
    stage_statuses = {item["stage"]: item["status"] for item in rebuild_summary["stages"]}
    assert stage_statuses == {
        "standardized_build": "success",
        "feature_build": "success",
        "train_input_validation": "success",
        "split_validation": "success",
        "dataset_build": "success",
        "state_build": "success",
    }
    standardize_command = next(item["command"] for item in rebuild_summary["stages"] if item["stage"] == "standardized_build")
    assert "--input-root" in standardize_command
