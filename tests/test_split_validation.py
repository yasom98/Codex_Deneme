"""Unit tests for split contract validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.split_validation import (
    SPLIT_EMBARGO_VIOLATION,
    SPLIT_EXPLICIT_RANGE_INVALID,
    SPLIT_MIN_ROWS_NOT_MET,
    SPLIT_MODE_UNSUPPORTED,
    SPLIT_OVERLAP_DETECTED,
    SPLIT_TRAIN_INPUT_VALIDATION_FAILED,
    SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_MISSING,
    SPLIT_WARMUP_INSUFFICIENT,
    SplitValidationOptions,
    validate_splits,
)


def _base_frame(rows: int = 64, freq: str = "15min") -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq=freq, tz="UTC")
    out = pd.DataFrame({"timestamp": ts})
    out["feat_cont"] = pd.Series(np.linspace(1.0, 2.0, rows), dtype="float32")
    event = np.zeros(rows, dtype=np.uint8)
    event[::7] = np.uint8(1)
    out["evt_flag"] = pd.Series(event, dtype="uint8")
    return out


def _base_manifest(run_id: str) -> dict[str, object]:
    return {
        "manifest_version": "features.manifest.v1",
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "timestamp_column": "timestamp",
        "continuous_columns": ["feat_cont"],
        "event_columns": ["evt_flag"],
    }


def _setup_run(tmp_path: Path, run_id: str, file_names: list[str]) -> tuple[Path, Path, list[Path]]:
    input_root = tmp_path / "runs" / run_id / "data_features" / "parquet"
    reports_root = tmp_path / "runs" / run_id / "data_features" / "reports"
    input_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for name in file_names:
        path = input_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
        paths.append(path)
    return input_root, reports_root, paths


def _write_manifest(reports_root: Path, run_id: str) -> None:
    manifest_path = reports_root / "feature_manifest.json"
    manifest_path.write_text(json.dumps(_base_manifest(run_id)), encoding="utf-8")


def _write_train_input_report(reports_root: Path, run_id: str, overall: bool) -> None:
    payload = {
        "run_id": run_id,
        "train_input_validation_overall": bool(overall),
    }
    path = reports_root / "train_input_validation_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _patch_read_parquet(monkeypatch: object, mapping: dict[str, pd.DataFrame]) -> None:
    def fake_read_parquet(path: Path) -> pd.DataFrame:
        key = str(path)
        if key not in mapping:
            raise ValueError(f"unexpected path: {key}")
        return mapping[key].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)


def _error_codes(payload: dict[str, object]) -> set[str]:
    codes: set[str] = set()
    for item in payload.get("errors", []):
        if isinstance(item, dict) and isinstance(item.get("code"), str):
            codes.add(item["code"])
    for level in ("file_reports", "fold_reports"):
        reports = payload.get(level, [])
        if not isinstance(reports, list):
            continue
        for report in reports:
            if not isinstance(report, dict):
                continue
            for item in report.get("errors", []):
                if isinstance(item, dict) and isinstance(item.get("code"), str):
                    codes.add(item["code"])
    return codes


def test_ratio_chrono_happy_path_passes(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "ratio_happy_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    frame = _base_frame(rows=100, freq="1min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="ratio_chrono",
            split_overrides={"train_ratio": "0.7", "val_ratio": "0.2", "test_ratio": "0.1"},
        )
    )
    payload = report.to_dict()

    assert payload["split_validation_overall"] is True
    assert payload["errors"] == []
    assert payload["file_reports"][0]["status"] == "success"


def test_ratio_chrono_min_rows_gate_fails_even_when_spec_valid(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "ratio_min_rows_fail_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    frame = _base_frame(rows=10, freq="1min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="ratio_chrono",
            split_overrides={"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
            min_val_rows=2,
            min_test_rows=2,
        )
    )

    assert report.split_validation_overall is False
    assert SPLIT_MIN_ROWS_NOT_MET in _error_codes(report.to_dict())


def test_invalid_mode_fails(tmp_path: Path) -> None:
    run_id = "invalid_mode_run"
    input_root, reports_root, _ = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="invalid_mode",
        )
    )
    assert report.split_validation_overall is False
    assert SPLIT_MODE_UNSUPPORTED in _error_codes(report.to_dict())


def test_explicit_ranges_partial_pair_fails(tmp_path: Path) -> None:
    run_id = "explicit_partial_pair_run"
    input_root, reports_root, _ = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="explicit_ranges",
            split_overrides={
                "train_start": "2024-01-01T00:00:00Z",
                "train_end": "2024-01-01T01:00:00Z",
                "val_start": "2024-01-01T01:05:00Z",
            },
        )
    )

    assert report.split_validation_overall is False
    assert SPLIT_EXPLICIT_RANGE_INVALID in _error_codes(report.to_dict())


def test_explicit_ranges_overlap_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "explicit_overlap_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    frame = _base_frame(rows=8, freq="5min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="explicit_ranges",
            split_overrides={
                "train_start": frame.iloc[0]["timestamp"].isoformat(),
                "train_end": frame.iloc[3]["timestamp"].isoformat(),
                "val_start": frame.iloc[3]["timestamp"].isoformat(),
                "val_end": frame.iloc[5]["timestamp"].isoformat(),
                "test_start": frame.iloc[6]["timestamp"].isoformat(),
                "test_end": frame.iloc[7]["timestamp"].isoformat(),
            },
        )
    )

    assert report.split_validation_overall is False
    assert SPLIT_OVERLAP_DETECTED in _error_codes(report.to_dict())


def test_embargo_violation_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "embargo_fail_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    frame = _base_frame(rows=20, freq="1min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="ratio_chrono",
            split_overrides={"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25},
            embargo_bars=1,
        )
    )

    assert report.split_validation_overall is False
    assert SPLIT_EMBARGO_VIOLATION in _error_codes(report.to_dict())


def test_warmup_insufficient_reports_val_and_test_context(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "warmup_fail_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    frame = _base_frame(rows=10, freq="1min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="ratio_chrono",
            split_overrides={"train_ratio": 0.2, "val_ratio": 0.4, "test_ratio": 0.4},
            warmup_rows=9,
        )
    )

    payload = report.to_dict()
    assert report.split_validation_overall is False
    assert SPLIT_WARMUP_INSUFFICIENT in _error_codes(payload)

    warmup_detail = payload["file_reports"][0]["warmup_detail"]
    assert warmup_detail["val"]["checked"] is True
    assert warmup_detail["test"]["checked"] is True


def test_walk_forward_happy_path_passes(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "walk_forward_happy_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    frame = _base_frame(rows=40, freq="1min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="walk_forward",
            split_overrides={
                "min_train_bars": 12,
                "val_window_bars": 6,
                "test_window_bars": 4,
                "step_bars": 4,
                "max_folds": 3,
            },
        )
    )

    payload = report.to_dict()
    assert payload["split_validation_overall"] is True
    assert len(payload["fold_reports"]) == 3
    assert payload["file_reports"][0]["fold_count"] == 3


def test_mixed_timeframe_duration_spec_is_file_local(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "mixed_tf_run"
    input_root, reports_root, paths = _setup_run(
        tmp_path,
        run_id,
        ["btc_1m.parquet", "btc_5m.parquet", "btc_15m.parquet"],
    )
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)

    mapping = {
        str(paths[0]): _base_frame(rows=180, freq="1min"),
        str(paths[1]): _base_frame(rows=120, freq="5min"),
        str(paths[2]): _base_frame(rows=80, freq="15min"),
    }
    _patch_read_parquet(monkeypatch, mapping)

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="walk_forward",
            split_overrides={
                "min_train_duration": "30m",
                "val_window_duration": "15m",
                "test_window_duration": "15m",
                "step_duration": "15m",
                "max_folds": 2,
            },
        )
    )

    payload = report.to_dict()
    assert payload["split_validation_overall"] is True
    observed = [item["observed_median_delta_seconds"] for item in payload["file_reports"]]
    assert sorted(observed) == [60.0, 300.0, 900.0]


def test_train_input_required_missing_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "prior_missing_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)

    frame = _base_frame(rows=30, freq="1min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="ratio_chrono",
            split_overrides={"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
            require_train_input_validation=True,
        )
    )

    assert report.split_validation_overall is False
    assert SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_MISSING in _error_codes(report.to_dict())


def test_train_input_required_false_missing_warns_and_contract_passes(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "prior_optional_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)

    frame = _base_frame(rows=30, freq="1min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="ratio_chrono",
            split_overrides={"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
            require_train_input_validation=False,
        )
    )

    payload = report.to_dict()
    assert payload["split_validation_overall"] is True
    warning_codes = {item["code"] for item in payload["warnings"]}
    assert SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_MISSING in warning_codes


def test_train_input_failed_report_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "prior_failed_run"
    input_root, reports_root, paths = _setup_run(tmp_path, run_id, ["sample.parquet"])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=False)

    frame = _base_frame(rows=30, freq="1min")
    _patch_read_parquet(monkeypatch, {str(paths[0]): frame})

    report = validate_splits(
        SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode="ratio_chrono",
            split_overrides={"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
            require_train_input_validation=True,
        )
    )

    assert report.split_validation_overall is False
    assert SPLIT_TRAIN_INPUT_VALIDATION_FAILED in _error_codes(report.to_dict())
