"""Unit tests for dataset builder contract (Milestone 4.3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.dataset_builder import (
    DATASET_BUILD_COLUMN_SELECTION_INVALID,
    DATASET_BUILD_FEATURE_MANIFEST_MISSING,
    DATASET_BUILD_LINEAGE_MISMATCH,
    DATASET_BUILD_ORDERING_CONTRACT_VIOLATION,
    DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
    DATASET_BUILD_OUTPUT_ROOT_EXISTS,
    DATASET_BUILD_PARTITION_TIMESTAMP_DUPLICATE,
    DATASET_BUILD_RUN_ID_MISMATCH,
    DATASET_BUILD_SPLIT_NOT_PASSED,
    DATASET_BUILD_SPLIT_PARTITION_DEFS_MISSING,
    DATASET_BUILD_SPLIT_REPORT_MISSING,
    DATASET_BUILD_TRAIN_INPUT_NOT_PASSED,
    DATASET_BUILD_TRAIN_INPUT_REPORT_MISSING,
    DatasetBuildOptions,
    build_datasets,
)


def _base_frame(rows: int = 20, freq: str = "1min") -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq=freq, tz="UTC")
    frame = pd.DataFrame({"timestamp": ts})
    frame["feat_cont"] = pd.Series(np.linspace(1.0, 2.0, rows), dtype="float32")
    events = np.zeros(rows, dtype=np.uint8)
    events[::5] = np.uint8(1)
    frame["evt_flag"] = pd.Series(events, dtype="uint8")
    return frame


def _range_payload(ts: pd.Series, start: int, end: int) -> dict[str, Any]:
    assert end > start
    end_exclusive = ts.iloc[end].isoformat() if end < len(ts) else None
    return {
        "start_utc": ts.iloc[start].isoformat(),
        "end_exclusive_utc": end_exclusive,
        "end_inclusive_utc": ts.iloc[end - 1].isoformat(),
        "row_count": int(end - start),
        "internal_interval": "[start, end)",
    }


def _fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _setup_run(tmp_path: Path, run_id: str, files: list[tuple[str, pd.DataFrame]]) -> tuple[Path, Path, list[Path], dict[str, pd.DataFrame]]:
    input_root = tmp_path / "runs" / run_id / "data_features" / "parquet"
    reports_root = tmp_path / "runs" / run_id / "data_features" / "reports"
    input_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    source_paths: list[Path] = []
    frame_map: dict[str, pd.DataFrame] = {}
    for rel_name, frame in files:
        src = input_root / rel_name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("source-placeholder", encoding="utf-8")
        source_paths.append(src)
        frame_map[str(src.resolve())] = frame.copy()

    return input_root, reports_root, source_paths, frame_map


def _write_manifest(reports_root: Path, run_id: str) -> Path:
    payload = {
        "manifest_version": "features.manifest.v1",
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "timestamp_column": "timestamp",
        "continuous_columns": ["feat_cont"],
        "event_columns": ["evt_flag"],
        "column_dtypes": {
            "timestamp": "datetime64[ns, UTC]",
            "feat_cont": "float32",
            "evt_flag": "uint8",
        },
        "feature_groups": {
            "raw_ohlcv": ["timestamp"],
            "price_derived": [],
            "trend": ["feat_cont"],
            "regime": [],
            "event": ["evt_flag"],
            "placeholders": [],
        },
    }
    path = reports_root / "feature_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_train_input_report(reports_root: Path, run_id: str, *, overall: bool = True) -> Path:
    payload = {
        "run_id": run_id,
        "train_input_validation_overall": bool(overall),
    }
    path = reports_root / "train_input_validation_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_split_report_standard(
    reports_root: Path,
    run_id: str,
    source_paths: list[Path],
    frame_map: dict[str, pd.DataFrame],
    *,
    overall: bool = True,
) -> Path:
    file_reports: list[dict[str, Any]] = []
    for src in source_paths:
        frame = frame_map[str(src.resolve())]
        ts = pd.to_datetime(frame["timestamp"], utc=True)
        file_reports.append(
            {
                "input_file": str(src.resolve()),
                "status": "success",
                "train_range": _range_payload(ts, 0, 12),
                "val_range": _range_payload(ts, 12, 16),
                "test_range": _range_payload(ts, 16, 20),
            }
        )

    payload = {
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "split_mode": "ratio_chrono",
        "split_validation_overall": bool(overall),
        "manifest_path": str(reports_root / "feature_manifest.json"),
        "train_input_validation_report_path": str(reports_root / "train_input_validation_report.json"),
        "file_reports": file_reports,
        "fold_reports": [],
    }
    path = reports_root / "split_validation_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_split_report_walk_forward(
    reports_root: Path,
    run_id: str,
    source_paths: list[Path],
    frame_map: dict[str, pd.DataFrame],
) -> Path:
    fold_reports: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []

    for src in source_paths:
        frame = frame_map[str(src.resolve())]
        ts = pd.to_datetime(frame["timestamp"], utc=True)

        fold_reports.append(
            {
                "fold_id": 0,
                "input_file": str(src.resolve()),
                "train_range": _range_payload(ts, 0, 10),
                "val_range": _range_payload(ts, 10, 14),
                "test_range": _range_payload(ts, 14, 18),
            }
        )
        fold_reports.append(
            {
                "fold_id": 1,
                "input_file": str(src.resolve()),
                "train_range": _range_payload(ts, 0, 12),
                "val_range": _range_payload(ts, 12, 16),
                "test_range": _range_payload(ts, 16, 20),
            }
        )

        file_reports.append(
            {
                "input_file": str(src.resolve()),
                "status": "success",
                "fold_count": 2,
                "failed_fold_count": 0,
                "train_range": _range_payload(ts, 0, 12),
                "val_range": _range_payload(ts, 12, 16),
                "test_range": _range_payload(ts, 16, 20),
            }
        )

    payload = {
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "split_mode": "walk_forward",
        "split_validation_overall": True,
        "manifest_path": str(reports_root / "feature_manifest.json"),
        "train_input_validation_report_path": str(reports_root / "train_input_validation_report.json"),
        "file_reports": file_reports,
        "fold_reports": fold_reports,
    }
    path = reports_root / "split_validation_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _patch_parquet_io(monkeypatch: object, frame_map: dict[str, pd.DataFrame]) -> None:
    def fake_read_parquet(path: Path) -> pd.DataFrame:
        key = str(Path(path).resolve())
        if key not in frame_map:
            raise ValueError(f"Unexpected parquet path: {key}")
        return frame_map[key].copy()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del index
        payload = {
            "rows": int(len(self)),
            "columns": list(self.columns),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)


def _error_codes(payload: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for issue in payload.get("errors", []):
        if isinstance(issue, dict) and isinstance(issue.get("code"), str):
            codes.add(str(issue["code"]))
    return codes


def test_precondition_manifest_missing_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_manifest_missing"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame())])
    _write_train_input_report(reports_root, run_id)
    _write_split_report_standard(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert result.report_payload["dataset_build_overall"] is False
    assert DATASET_BUILD_FEATURE_MANIFEST_MISSING in _error_codes(result.report_payload)


def test_precondition_train_input_missing_or_failed(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_train_input_missing"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame())])
    _write_manifest(reports_root, run_id)
    _write_split_report_standard(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    result_missing = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert DATASET_BUILD_TRAIN_INPUT_REPORT_MISSING in _error_codes(result_missing.report_payload)

    _write_train_input_report(reports_root, run_id, overall=False)
    result_failed = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert DATASET_BUILD_TRAIN_INPUT_NOT_PASSED in _error_codes(result_failed.report_payload)


def test_precondition_split_missing_or_failed(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_split_missing"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame())])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    _patch_parquet_io(monkeypatch, frame_map)

    result_missing = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert DATASET_BUILD_SPLIT_REPORT_MISSING in _error_codes(result_missing.report_payload)

    _write_split_report_standard(reports_root, run_id, source_paths, frame_map, overall=False)
    result_failed = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert DATASET_BUILD_SPLIT_NOT_PASSED in _error_codes(result_failed.report_payload)


def test_split_partition_defs_missing_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_missing_partition_defs"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame())])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    split_path = _write_split_report_standard(reports_root, run_id, source_paths, frame_map, overall=True)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["file_reports"][0]["val_range"] = None
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert DATASET_BUILD_SPLIT_PARTITION_DEFS_MISSING in _error_codes(result.report_payload)


def test_run_id_and_lineage_mismatch_fail(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_lineage_mismatch"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame())])
    manifest_path = _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    split_path = _write_split_report_standard(reports_root, run_id, source_paths, frame_map, overall=True)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["run_id"] = "other"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_payload["manifest_path"] = str(reports_root / "other_manifest.json")
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")

    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    codes = _error_codes(result.report_payload)
    assert DATASET_BUILD_RUN_ID_MISMATCH in codes
    assert DATASET_BUILD_LINEAGE_MISMATCH in codes


def test_output_root_collision_and_overwrite(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_overwrite_policy"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame())])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    _write_split_report_standard(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    output_root = tmp_path / "runs" / run_id / "data_datasets"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "existing.txt").write_text("occupied", encoding="utf-8")

    result_fail = build_datasets(
        DatasetBuildOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            output_root=output_root,
            overwrite=False,
        )
    )
    assert DATASET_BUILD_OUTPUT_ROOT_EXISTS in _error_codes(result_fail.report_payload)

    result_ok = build_datasets(
        DatasetBuildOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            output_root=output_root,
            overwrite=True,
        )
    )
    assert result_ok.report_payload["dataset_build_overall"] is True
    assert (output_root / "reports" / "dataset_manifest.json").exists()


def test_success_path_records_ranges_and_order_contract(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_success"
    input_root, reports_root, source_paths, frame_map = _setup_run(
        tmp_path,
        run_id,
        [("a.parquet", _base_frame(rows=20, freq="1min"))],
    )
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    _write_split_report_standard(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert result.report_payload["dataset_build_overall"] is True
    assert result.report_payload["output_completeness_ok"] is True
    assert result.manifest_payload is not None
    assert result.manifest_payload["row_order_policy"]["name"] == "timestamp_ascending"
    assert result.manifest_payload["column_selection_contract"]["column_order_hash"]
    assert result.manifest_payload["column_selection_contract"]["dtype_hash"]


def test_persisted_metadata_paths_are_promoted_not_staging(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_promoted_paths"
    input_root, reports_root, source_paths, frame_map = _setup_run(
        tmp_path,
        run_id,
        [("nested/a.parquet", _base_frame(rows=20, freq="1min"))],
    )
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    _write_split_report_walk_forward(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(
        DatasetBuildOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            aggregate_walk_forward=True,
        )
    )
    assert result.report_payload["dataset_build_overall"] is True
    assert result.manifest_payload is not None

    output_root = (tmp_path / "runs" / run_id / "data_datasets").resolve()
    report_path = output_root / "reports" / "dataset_build_report.json"
    manifest_path = output_root / "reports" / "dataset_manifest.json"

    report_text = report_path.read_text(encoding="utf-8")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "__staging__" not in report_text
    assert "__staging__" not in manifest_text

    persisted_manifest = json.loads(manifest_text)
    assert persisted_manifest["partition_metadata"]
    assert persisted_manifest["walk_forward_fold_metadata"]

    for item in persisted_manifest["partition_metadata"]:
        output_path = str(item["output_path"])
        assert "__staging__" not in output_path
        assert output_path.startswith(str(output_root))

    for item in persisted_manifest["walk_forward_fold_metadata"]:
        output_path = str(item["output_path"])
        assert "__staging__" not in output_path
        assert output_path.startswith(str(output_root))


def test_mixed_timeframe_build_path(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_mixed_tf"
    files = [
        ("btc_1m.parquet", _base_frame(rows=20, freq="1min")),
        ("btc_5m.parquet", _base_frame(rows=20, freq="5min")),
        ("btc_15m.parquet", _base_frame(rows=20, freq="15min")),
    ]
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, files)
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    _write_split_report_standard(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert result.report_payload["dataset_build_overall"] is True
    assert result.report_payload["totals"]["files_processed"] == 3


def test_walk_forward_default_fold_only_and_aggregate_option(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_walk_forward"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame(rows=20, freq="1min"))])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    _write_split_report_walk_forward(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    result_fold_only = build_datasets(
        DatasetBuildOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            aggregate_walk_forward=False,
        )
    )
    assert result_fold_only.report_payload["dataset_build_overall"] is True
    assert result_fold_only.manifest_payload is not None
    assert result_fold_only.manifest_payload["output_semantics"]["mode"] == "walk_forward_fold_only"

    result_agg = build_datasets(
        DatasetBuildOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            aggregate_walk_forward=True,
            overwrite=True,
        )
    )
    assert result_agg.report_payload["dataset_build_overall"] is True
    assert result_agg.manifest_payload is not None
    assert result_agg.manifest_payload["output_semantics"]["mode"] == "walk_forward_fold_plus_aggregate"
    dup_policy = result_agg.manifest_payload["duplicate_timestamp_policy"]
    assert dup_policy["aggregate_walk_forward_allowed"] is True


def test_column_selection_valid_and_invalid(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_column_selection"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame(rows=20, freq="1min"))])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    _write_split_report_standard(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    result_valid = build_datasets(
        DatasetBuildOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            include_feature_groups=("trend",),
        )
    )
    assert result_valid.report_payload["dataset_build_overall"] is True

    result_invalid = build_datasets(
        DatasetBuildOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            include_feature_groups=("does_not_exist",),
            overwrite=True,
        )
    )
    assert DATASET_BUILD_COLUMN_SELECTION_INVALID in _error_codes(result_invalid.report_payload)


def test_output_completeness_mismatch_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_completeness_fail"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame(rows=20, freq="1min"))])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    split_path = _write_split_report_standard(reports_root, run_id, source_paths, frame_map)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["file_reports"][0]["train_range"]["row_count"] = 999
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH in _error_codes(result.report_payload)


def test_partition_timestamp_uniqueness_and_ordering_checks(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_ts_uniqueness"
    bad = _base_frame(rows=20, freq="1min")
    bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]

    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", bad)])
    _write_manifest(reports_root, run_id)
    _write_train_input_report(reports_root, run_id, overall=True)
    _write_split_report_standard(reports_root, run_id, source_paths, frame_map)
    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    codes = _error_codes(result.report_payload)
    assert DATASET_BUILD_PARTITION_TIMESTAMP_DUPLICATE in codes
    assert DATASET_BUILD_ORDERING_CONTRACT_VIOLATION in codes


def test_no_mutation_source_artifacts(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_no_mutation"
    input_root, reports_root, source_paths, frame_map = _setup_run(tmp_path, run_id, [("a.parquet", _base_frame(rows=20, freq="1min"))])
    manifest_path = _write_manifest(reports_root, run_id)
    train_input_path = _write_train_input_report(reports_root, run_id, overall=True)
    split_path = _write_split_report_standard(reports_root, run_id, source_paths, frame_map)

    before_manifest = _fingerprint(manifest_path)
    before_train = _fingerprint(train_input_path)
    before_split = _fingerprint(split_path)
    before_source = _fingerprint(source_paths[0])

    _patch_parquet_io(monkeypatch, frame_map)

    result = build_datasets(DatasetBuildOptions(run_id=run_id, input_root=input_root, reports_root=reports_root))
    assert result.report_payload["dataset_build_overall"] is True

    assert _fingerprint(manifest_path) == before_manifest
    assert _fingerprint(train_input_path) == before_train
    assert _fingerprint(split_path) == before_split
    assert _fingerprint(source_paths[0]) == before_source
