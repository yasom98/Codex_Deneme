"""Unit tests for train-input contract validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.train_input_validation import (
    TRAIN_INPUT_COLUMN_ORDER_DRIFT,
    TRAIN_INPUT_DTYPE_MISMATCH,
    TRAIN_INPUT_DUPLICATE_TIMESTAMP,
    TRAIN_INPUT_EMPTY_FILE,
    TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID,
    TRAIN_INPUT_MANIFEST_INVALID_JSON,
    TRAIN_INPUT_MANIFEST_MISSING,
    TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
    TRAIN_INPUT_NON_MONOTONIC,
    TRAIN_INPUT_REQUIRED_COLUMN_MISSING,
    TRAIN_INPUT_UNEXPECTED_COLUMNS,
    TrainInputValidationOptions,
    validate_train_inputs,
)


def _all_columns() -> list[str]:
    return [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "PP",
        "R1",
        "S1",
        "R2",
        "S2",
        "R3",
        "S3",
        "R4",
        "S4",
        "R5",
        "S5",
        "EMA_200",
        "EMA_600",
        "EMA_1200",
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_up",
        "ST_dn",
        "log_return",
        "hl_range",
        "body_ratio",
        "rolling_vol_20",
        "rolling_vol_50",
        "zscore_return",
        "normalized_range",
        "trend_strength",
        "squeeze_proxy",
        "floating_pnl_placeholder",
        "drawdown_placeholder",
        "evt_at_buy_raw",
        "evt_at_sell_raw",
        "evt_at_buy",
        "evt_at_sell",
        "evt_st_buy",
        "evt_st_sell",
        "trend_regime",
        "volatility_regime",
        "position_placeholder",
        "market_state",
    ]


def _continuous_columns() -> list[str]:
    return [
        "PP",
        "R1",
        "S1",
        "R2",
        "S2",
        "R3",
        "S3",
        "R4",
        "S4",
        "R5",
        "S5",
        "EMA_200",
        "EMA_600",
        "EMA_1200",
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_up",
        "ST_dn",
        "log_return",
        "hl_range",
        "body_ratio",
        "rolling_vol_20",
        "rolling_vol_50",
        "zscore_return",
        "normalized_range",
        "trend_strength",
        "squeeze_proxy",
        "floating_pnl_placeholder",
        "drawdown_placeholder",
    ]


def _event_columns() -> list[str]:
    return [
        "evt_at_buy_raw",
        "evt_at_sell_raw",
        "evt_at_buy",
        "evt_at_sell",
        "evt_st_buy",
        "evt_st_sell",
        "trend_regime",
        "volatility_regime",
        "position_placeholder",
        "market_state",
    ]


def _placeholder_columns() -> list[str]:
    return [
        "position_placeholder",
        "floating_pnl_placeholder",
        "drawdown_placeholder",
        "market_state",
    ]


def _feature_groups() -> dict[str, list[str]]:
    return {
        "raw_ohlcv": ["timestamp", "open", "high", "low", "close", "volume"],
        "price_derived": ["log_return", "hl_range", "body_ratio", "rolling_vol_20", "rolling_vol_50", "zscore_return", "normalized_range"],
        "trend": ["EMA_200", "EMA_600", "EMA_1200", "AlphaTrend", "AlphaTrend_2", "ST_trend", "ST_up", "ST_dn", "trend_strength"],
        "regime": ["trend_regime", "volatility_regime", "squeeze_proxy"],
        "event": ["evt_at_buy_raw", "evt_at_sell_raw", "evt_at_buy", "evt_at_sell", "evt_st_buy", "evt_st_sell"],
        "placeholders": ["position_placeholder", "floating_pnl_placeholder", "drawdown_placeholder", "market_state"],
    }


def _base_frame(rows: int = 64) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    out = pd.DataFrame({"timestamp": ts})
    float_columns = [col for col in _all_columns() if col not in {"timestamp", *_event_columns()}]
    for idx, col in enumerate(float_columns):
        out[col] = pd.Series(np.linspace(1.0 + idx, 2.0 + idx, rows), dtype="float32")

    for idx, col in enumerate(_event_columns()):
        pattern = np.zeros(rows, dtype=np.uint8)
        pattern[idx % 5 :: 7] = np.uint8(1)
        out[col] = pd.Series(pattern, dtype="uint8")

    return out.loc[:, _all_columns()]


def _base_manifest(run_id: str, frame: pd.DataFrame) -> dict[str, object]:
    canonical_order = list(frame.columns)
    return {
        "manifest_version": "features.manifest.v1",
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "feature_groups": _feature_groups(),
        "column_dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "event_columns": _event_columns(),
        "continuous_columns": _continuous_columns(),
        "placeholder_columns": _placeholder_columns(),
        "indicator_spec_version": "indicators.v2026-02-15.1",
        "config_hash": "abc123",
        "formula_fingerprint_bundle": "bundle123",
        "timestamp_column": "timestamp",
        "canonical_column_order": canonical_order,
    }


def _setup_run(tmp_path: Path, run_id: str = "unit_run") -> tuple[Path, Path, Path, Path]:
    input_root = tmp_path / "runs" / run_id / "data_features" / "parquet"
    reports_root = tmp_path / "runs" / run_id / "data_features" / "reports"
    input_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    parquet_path = input_root / "sample.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")
    manifest_path = reports_root / "feature_manifest.json"
    return input_root, reports_root, manifest_path, parquet_path


def _patch_read_parquet(monkeypatch: object, frame: pd.DataFrame) -> None:
    def fake_read_parquet(path: Path) -> pd.DataFrame:
        del path
        return frame.copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)


def _validate(tmp_path: Path, run_id: str, *, strict_extra_columns: bool = True, strict_column_order: bool = False) -> object:
    input_root = tmp_path / "runs" / run_id / "data_features" / "parquet"
    reports_root = tmp_path / "runs" / run_id / "data_features" / "reports"
    options = TrainInputValidationOptions(
        run_id=run_id,
        input_root=input_root,
        reports_root=reports_root,
        strict_extra_columns=strict_extra_columns,
        strict_column_order=strict_column_order,
    )
    return validate_train_inputs(options)


def _error_codes(report: object) -> set[str]:
    payload = report.to_dict()
    codes = {item["code"] for item in payload["errors"]}
    for file_report in payload["file_reports"]:
        for err in file_report["errors"]:
            codes.add(err["code"])
    return codes


def test_manifest_missing_fails(tmp_path: Path) -> None:
    run_id = "missing_manifest_run"
    _setup_run(tmp_path, run_id=run_id)

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_MANIFEST_MISSING in _error_codes(report)


def test_manifest_invalid_json_fails(tmp_path: Path) -> None:
    run_id = "invalid_json_manifest_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    manifest_path.write_text("{invalid-json", encoding="utf-8")

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_MANIFEST_INVALID_JSON in _error_codes(report)


def test_manifest_missing_required_fields_fails(tmp_path: Path) -> None:
    run_id = "missing_fields_manifest_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    manifest_path.write_text(json.dumps({"manifest_version": "features.manifest.v1"}), encoding="utf-8")

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_MANIFEST_SCHEMA_INVALID in _error_codes(report)


def test_happy_path_passes(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "happy_path_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    manifest_path.write_text(json.dumps(_base_manifest(run_id, frame)), encoding="utf-8")
    _patch_read_parquet(monkeypatch, frame)

    report = _validate(tmp_path, run_id, strict_column_order=True)
    assert report.train_input_validation_overall is True
    assert report.failed_files == 0


def test_required_column_missing_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "missing_column_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame().drop(columns=["EMA_200"])
    manifest_path.write_text(json.dumps(_base_manifest(run_id, _base_frame())), encoding="utf-8")
    _patch_read_parquet(monkeypatch, frame)

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_REQUIRED_COLUMN_MISSING in _error_codes(report)


def test_dtype_mismatch_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dtype_mismatch_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    frame["EMA_600"] = frame["EMA_600"].astype("float64")
    manifest_path.write_text(json.dumps(_base_manifest(run_id, _base_frame())), encoding="utf-8")
    _patch_read_parquet(monkeypatch, frame)

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_DTYPE_MISMATCH in _error_codes(report)


def test_duplicate_timestamp_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "duplicate_ts_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    manifest_path.write_text(json.dumps(_base_manifest(run_id, _base_frame())), encoding="utf-8")
    _patch_read_parquet(monkeypatch, frame)

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_DUPLICATE_TIMESTAMP in _error_codes(report)


def test_non_monotonic_timestamp_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "non_monotonic_ts_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    frame.loc[0, "timestamp"], frame.loc[1, "timestamp"] = frame.loc[1, "timestamp"], frame.loc[0, "timestamp"]
    manifest_path.write_text(json.dumps(_base_manifest(run_id, _base_frame())), encoding="utf-8")
    _patch_read_parquet(monkeypatch, frame)

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_NON_MONOTONIC in _error_codes(report)


def test_empty_parquet_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "empty_file_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame().iloc[0:0].copy()
    manifest_path.write_text(json.dumps(_base_manifest(run_id, _base_frame())), encoding="utf-8")
    _patch_read_parquet(monkeypatch, frame)

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_EMPTY_FILE in _error_codes(report)


def test_extra_column_strict_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "extra_column_strict_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    frame["extra_feature"] = pd.Series(np.ones(len(frame)), dtype="float32")
    manifest_path.write_text(json.dumps(_base_manifest(run_id, _base_frame())), encoding="utf-8")
    _patch_read_parquet(monkeypatch, frame)

    report = _validate(tmp_path, run_id, strict_extra_columns=True)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_UNEXPECTED_COLUMNS in _error_codes(report)


def test_extra_column_non_strict_warns_and_passes(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "extra_column_non_strict_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    frame["extra_feature"] = pd.Series(np.ones(len(frame)), dtype="float32")
    manifest_path.write_text(json.dumps(_base_manifest(run_id, _base_frame())), encoding="utf-8")
    _patch_read_parquet(monkeypatch, frame)

    report = _validate(tmp_path, run_id, strict_extra_columns=False)
    payload = report.to_dict()
    assert report.train_input_validation_overall is True
    assert payload["file_reports"][0]["warnings"]
    assert payload["file_reports"][0]["errors"] == []


def test_feature_group_subset_violation_fails(tmp_path: Path) -> None:
    run_id = "subset_violation_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    manifest = _base_manifest(run_id, frame)
    manifest["event_columns"] = ["evt_at_buy_raw", "missing_evt_col"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID in _error_codes(report)


def test_feature_groups_missing_required_key_fails(tmp_path: Path) -> None:
    run_id = "missing_feature_group_key_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    manifest = _base_manifest(run_id, frame)
    feature_groups = dict(manifest["feature_groups"])
    del feature_groups["trend"]
    manifest["feature_groups"] = feature_groups
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_MANIFEST_SCHEMA_INVALID in _error_codes(report)


def test_forbidden_overlap_fails(tmp_path: Path) -> None:
    run_id = "forbidden_overlap_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    manifest = _base_manifest(run_id, frame)
    manifest["continuous_columns"] = [*manifest["continuous_columns"], "evt_at_buy_raw"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _validate(tmp_path, run_id)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID in _error_codes(report)


def test_column_order_strict_off_warn_only(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "column_order_warn_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    drift_frame = frame.loc[:, [*frame.columns[:2], *frame.columns[3:6], frame.columns[2], *frame.columns[6:]]]
    manifest_path.write_text(json.dumps(_base_manifest(run_id, frame)), encoding="utf-8")
    _patch_read_parquet(monkeypatch, drift_frame)

    report = _validate(tmp_path, run_id, strict_column_order=False)
    payload = report.to_dict()
    assert report.train_input_validation_overall is True
    assert payload["file_reports"][0]["column_order_severity"] == "warning"
    assert payload["file_reports"][0]["column_order_evaluated"] is True


def test_column_order_strict_on_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "column_order_error_run"
    _, _, manifest_path, _ = _setup_run(tmp_path, run_id=run_id)
    frame = _base_frame()
    drift_frame = frame.loc[:, [*frame.columns[:2], *frame.columns[3:6], frame.columns[2], *frame.columns[6:]]]
    manifest_path.write_text(json.dumps(_base_manifest(run_id, frame)), encoding="utf-8")
    _patch_read_parquet(monkeypatch, drift_frame)

    report = _validate(tmp_path, run_id, strict_column_order=True)
    assert report.train_input_validation_overall is False
    assert TRAIN_INPUT_COLUMN_ORDER_DRIFT in _error_codes(report)

