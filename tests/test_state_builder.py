"""Unit tests for RL state builder contract (Milestone 4.4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.state_builder import (
    STATE_BUILD_AGGREGATE_WALK_FORWARD_DEFERRED,
    STATE_BUILD_COLUMN_SELECTION_INVALID,
    STATE_BUILD_DATASET_MANIFEST_MISSING,
    STATE_BUILD_DATASET_NOT_PASSED,
    STATE_BUILD_DATASET_REPORT_MISSING,
    STATE_BUILD_LINEAGE_MISMATCH,
    STATE_BUILD_ORDERING_CONTRACT_VIOLATION,
    STATE_BUILD_OUTPUT_COMPLETENESS_FAILED_UPSTREAM,
    STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
    STATE_BUILD_OUTPUT_ROOT_EXISTS,
    STATE_BUILD_RUNTIME_PRICE_CONTRACT_REQUIRED,
    STATE_BUILD_RUN_ID_MISMATCH,
    STATE_BUILD_SCALER_TYPE_UNSUPPORTED,
    STATE_BUILD_SEQUENCE_MODE_DEFERRED,
    STATE_BUILD_STAGING_ROOT_COLLISION,
    STATE_BUILD_TIMESTAMP_DUPLICATES,
    STATE_MANIFEST_VERSION,
    STATE_BUILDER_VERSION,
    StateBuildOptions,
    build_states,
)


def _base_frame(rows: int = 20, freq: str = "1min", start: str = "2024-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=rows, freq=freq, tz="UTC")
    frame = pd.DataFrame({"timestamp": ts})
    frame["close"] = pd.Series(np.linspace(100.0, 100.0 + float(rows - 1), rows), dtype="float32")
    frame["feat_a"] = pd.Series(np.linspace(1.0, float(rows), rows), dtype="float32")
    frame["feat_b"] = pd.Series(np.linspace(10.0, float(rows + 9), rows), dtype="float32")
    event = np.zeros(rows, dtype=np.uint8)
    event[::4] = np.uint8(1)
    frame["evt_flag"] = pd.Series(event, dtype="uint8")
    return frame


def _fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_path(dataset_root: Path, rel_name: str, *, scope: str, partition: str, fold_id: int | None = None) -> Path:
    stem = Path(rel_name)
    if scope == "fold":
        if fold_id is None:
            raise ValueError("fold scope requires fold_id")
        return dataset_root / "parquet" / "folds" / f"fold_{fold_id:03d}" / partition / stem
    return dataset_root / "parquet" / "partitions" / partition / stem


def _partition_entry(
    *,
    dataset_root: Path,
    rel_name: str,
    scope: str,
    partition: str,
    frame: pd.DataFrame,
    fold_id: int | None = None,
) -> tuple[dict[str, Any], Path]:
    path = _source_path(dataset_root, rel_name, scope=scope, partition=partition, fold_id=fold_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dataset-placeholder", encoding="utf-8")

    ts = pd.to_datetime(frame["timestamp"], utc=True)
    duplicate_count = int(ts.duplicated().sum())

    payload: dict[str, Any] = {
        "scope": scope,
        "source_rel": rel_name,
        "partition": partition,
        "fold_id": fold_id,
        "output_path": str(path.resolve()),
        "row_count": int(len(frame)),
        "timestamp_min_utc": ts.iloc[0].isoformat() if len(frame) > 0 else None,
        "timestamp_max_utc": ts.iloc[-1].isoformat() if len(frame) > 0 else None,
        "duplicate_timestamp_count": duplicate_count,
        "timestamp_unique_ok": duplicate_count == 0,
        "file_sha256": "placeholder",
    }
    return payload, path


def _write_dataset_artifacts(
    *,
    tmp_path: Path,
    run_id: str,
    entries: list[dict[str, Any]],
    split_mode: str,
    output_semantics_mode: str,
    dataset_build_overall: bool = True,
    output_completeness_ok: bool = True,
    dataset_build_id: str = "dataset-build-id-1",
    selected_columns: list[str] | None = None,
    selected_dtypes: dict[str, str] | None = None,
) -> tuple[Path, Path, Path, Path]:
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"
    reports_root = dataset_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = reports_root / "dataset_manifest.json"
    dataset_report_path = reports_root / "dataset_build_report.json"

    effective_selected_columns = list(selected_columns) if selected_columns is not None else ["timestamp", "feat_a", "feat_b", "evt_flag"]
    effective_selected_dtypes = (
        dict(selected_dtypes)
        if selected_dtypes is not None
        else {
            "timestamp": "datetime64[ns, UTC]",
            "feat_a": "float32",
            "feat_b": "float32",
            "evt_flag": "uint8",
        }
    )

    dataset_manifest = {
        "manifest_version": "datasets.manifest.v1",
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "builder_version": "dataset_builder.v1",
        "dataset_build_id": dataset_build_id,
        "split_mode": split_mode,
        "output_semantics": {
            "mode": output_semantics_mode,
            "fold_outputs_generated": output_semantics_mode.startswith("walk_forward"),
            "top_level_partitions_generated": output_semantics_mode == "standard_partitions",
            "aggregate_walk_forward": output_semantics_mode == "walk_forward_fold_plus_aggregate",
        },
        "column_selection_contract": {
            "timestamp_column": "timestamp",
            "selected_columns": effective_selected_columns,
            "selected_dtypes": effective_selected_dtypes,
            "column_selection_hash": "colhash",
            "dtype_hash": "dtypehash",
        },
        "source_hashes": {
            "feature_manifest_hash": "feat-hash",
            "train_input_report_hash": "train-hash",
            "split_report_hash": "split-hash",
        },
        "partition_metadata": entries,
        "walk_forward_fold_metadata": [],
        "row_order_policy": {
            "name": "timestamp_ascending",
            "stable_tie_breaker": "source_row_position",
        },
        "duplicate_timestamp_policy": {
            "default_partition_uniqueness_required": True,
        },
        "output_completeness_ok": True,
    }
    dataset_manifest_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")

    dataset_report = {
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "builder_version": "dataset_builder.v1",
        "dataset_build_id": dataset_build_id,
        "dataset_build_overall": bool(dataset_build_overall),
        "output_completeness_ok": bool(output_completeness_ok),
        "split_mode": split_mode,
        "output_semantics": dataset_manifest["output_semantics"],
        "dataset_manifest_path": str(dataset_manifest_path.resolve()),
        "source_hashes": {
            "feature_manifest_hash": "feat-hash",
            "train_input_report_hash": "train-hash",
            "split_report_hash": "split-hash",
        },
    }
    dataset_report_path.write_text(json.dumps(dataset_report), encoding="utf-8")

    features_reports_root = tmp_path / "runs" / run_id / "data_features" / "reports"
    features_reports_root.mkdir(parents=True, exist_ok=True)
    (features_reports_root / "summary.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    return dataset_root, reports_root, dataset_manifest_path, dataset_report_path


def _seed_standard_run(tmp_path: Path, run_id: str) -> tuple[Path, dict[str, pd.DataFrame], Path, Path, Path]:
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"

    train = _base_frame(rows=8, freq="1min")
    val = _base_frame(rows=4, freq="1min", start="2024-01-01 00:08:00")
    test = _base_frame(rows=4, freq="1min", start="2024-01-01 00:12:00")

    entries: list[dict[str, Any]] = []
    frame_map: dict[str, pd.DataFrame] = {}

    for partition, frame in (("train", train), ("val", val), ("test", test)):
        entry, src = _partition_entry(dataset_root=dataset_root, rel_name="a.parquet", scope="partition", partition=partition, frame=frame)
        entries.append(entry)
        frame_map[str(src.resolve())] = frame.copy()

    dataset_root, _, manifest_path, report_path = _write_dataset_artifacts(
        tmp_path=tmp_path,
        run_id=run_id,
        entries=entries,
        split_mode="ratio_chrono",
        output_semantics_mode="standard_partitions",
    )
    return dataset_root, frame_map, manifest_path, report_path, (tmp_path / "runs" / run_id / "data_features" / "reports" / "summary.json")


def _seed_mixed_timeframe_run(tmp_path: Path, run_id: str) -> tuple[Path, dict[str, pd.DataFrame]]:
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"
    entries: list[dict[str, Any]] = []
    frame_map: dict[str, pd.DataFrame] = {}

    configs = [
        ("btc_1m.parquet", "1min"),
        ("btc_5m.parquet", "5min"),
        ("btc_15m.parquet", "15min"),
    ]

    for rel_name, freq in configs:
        for partition, start in (("train", "2024-01-01"), ("val", "2024-01-02"), ("test", "2024-01-03")):
            frame = _base_frame(rows=5, freq=freq, start=start)
            entry, src = _partition_entry(dataset_root=dataset_root, rel_name=rel_name, scope="partition", partition=partition, frame=frame)
            entries.append(entry)
            frame_map[str(src.resolve())] = frame.copy()

    _write_dataset_artifacts(
        tmp_path=tmp_path,
        run_id=run_id,
        entries=entries,
        split_mode="ratio_chrono",
        output_semantics_mode="standard_partitions",
    )
    return dataset_root, frame_map


def _seed_walk_forward_run(tmp_path: Path, run_id: str) -> tuple[Path, dict[str, pd.DataFrame]]:
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"
    entries: list[dict[str, Any]] = []
    frame_map: dict[str, pd.DataFrame] = {}

    fold_frames = {
        0: {
            "train": _base_frame(rows=6, freq="1min", start="2024-01-01 00:00:00"),
            "val": _base_frame(rows=2, freq="1min", start="2024-01-01 00:06:00"),
            "test": _base_frame(rows=2, freq="1min", start="2024-01-01 00:08:00"),
        },
        1: {
            "train": _base_frame(rows=8, freq="1min", start="2024-01-02 00:00:00"),
            "val": _base_frame(rows=2, freq="1min", start="2024-01-02 00:08:00"),
            "test": _base_frame(rows=2, freq="1min", start="2024-01-02 00:10:00"),
        },
    }

    for fold_id, parts in fold_frames.items():
        for partition, frame in parts.items():
            entry, src = _partition_entry(
                dataset_root=dataset_root,
                rel_name="wf.parquet",
                scope="fold",
                partition=partition,
                fold_id=fold_id,
                frame=frame,
            )
            entries.append(entry)
            frame_map[str(src.resolve())] = frame.copy()

    _write_dataset_artifacts(
        tmp_path=tmp_path,
        run_id=run_id,
        entries=entries,
        split_mode="walk_forward",
        output_semantics_mode="walk_forward_fold_only",
    )
    return dataset_root, frame_map


def _patch_parquet_io(
    monkeypatch: object,
    *,
    frame_map: dict[str, pd.DataFrame],
    written_frames: dict[str, pd.DataFrame],
) -> None:
    def fake_read_parquet(path: Path) -> pd.DataFrame:
        key = str(Path(path).resolve())
        if key not in frame_map:
            raise ValueError(f"Unexpected parquet path: {key}")
        return frame_map[key].copy()

    def fake_atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        written_frames[str(dest.resolve())] = df.copy()
        payload = {
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
        dest.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr("data.state_builder.atomic_write_parquet", fake_atomic_write_parquet)


def _error_codes(payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for issue in payload.get("errors", []):
        if isinstance(issue, dict) and isinstance(issue.get("code"), str):
            out.add(str(issue["code"]))
    return out


def _state_build_options(
    *,
    run_id: str,
    dataset_root: Path,
    output_root: Path,
    **overrides: Any,
) -> StateBuildOptions:
    base: dict[str, Any] = {
        "run_id": run_id,
        "input_root": dataset_root,
        "output_root": output_root,
        "execution_price_column": "close",
        "mark_to_market_column": "close",
    }
    base.update(overrides)
    return StateBuildOptions(**base)


def test_precondition_dataset_manifest_missing_fails(tmp_path: Path) -> None:
    run_id = "state_missing_manifest"
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"
    reports_root = dataset_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "dataset_build_report.json").write_text(json.dumps({"run_id": run_id, "dataset_build_overall": True, "output_completeness_ok": True}), encoding="utf-8")

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    assert result.report_payload["state_build_overall"] is False
    assert STATE_BUILD_DATASET_MANIFEST_MISSING in _error_codes(result.report_payload)


def test_precondition_dataset_report_missing_or_failed(tmp_path: Path) -> None:
    run_id = "state_missing_report"
    dataset_root, _, manifest_path, _, _ = _seed_standard_run(tmp_path, run_id)
    (dataset_root / "reports" / "dataset_build_report.json").unlink()

    result_missing = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    assert STATE_BUILD_DATASET_REPORT_MISSING in _error_codes(result_missing.report_payload)

    report_payload = json.loads((dataset_root / "reports" / "dataset_build_report.json").read_text(encoding="utf-8") if (dataset_root / "reports" / "dataset_build_report.json").exists() else "{}")
    if not report_payload:
        report_payload = {
            "run_id": run_id,
            "dataset_build_overall": False,
            "output_completeness_ok": True,
        }
    report_payload["dataset_build_overall"] = False
    (dataset_root / "reports" / "dataset_build_report.json").write_text(json.dumps(report_payload), encoding="utf-8")

    result_failed = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    assert STATE_BUILD_DATASET_NOT_PASSED in _error_codes(result_failed.report_payload)
    assert manifest_path.exists()


def test_upstream_output_completeness_failure_fails(tmp_path: Path) -> None:
    run_id = "state_upstream_completeness_fail"
    dataset_root, _, _, report_path, _ = _seed_standard_run(tmp_path, run_id)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["output_completeness_ok"] = False
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    assert STATE_BUILD_OUTPUT_COMPLETENESS_FAILED_UPSTREAM in _error_codes(result.report_payload)


def test_run_id_and_lineage_mismatch_fail(tmp_path: Path) -> None:
    run_id = "state_lineage_mismatch"
    dataset_root, _, manifest_path, report_path, _ = _seed_standard_run(tmp_path, run_id)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "other"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dataset_build_id"] = "different-id"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    codes = _error_codes(result.report_payload)
    assert STATE_BUILD_RUN_ID_MISMATCH in codes
    assert STATE_BUILD_LINEAGE_MISMATCH in codes


def test_output_root_collision_and_overwrite(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_overwrite"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    output_root = tmp_path / "runs" / run_id / "data_states"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "existing.txt").write_text("occupied", encoding="utf-8")

    result_fail = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=output_root,
            overwrite=False,
        )
    )
    assert STATE_BUILD_OUTPUT_ROOT_EXISTS in _error_codes(result_fail.report_payload)

    result_ok = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=output_root,
            overwrite=True,
        )
    )
    assert result_ok.report_payload["state_build_overall"] is True


def test_standard_success_and_manifest_contract(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_standard_success"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            enable_scaling=False,
            scaler_type="none",
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )
    assert result.report_payload["state_build_overall"] is True
    assert result.manifest_payload is not None
    assert result.manifest_payload["manifest_version"] == STATE_MANIFEST_VERSION
    assert result.manifest_payload["builder_version"] == STATE_BUILDER_VERSION
    assert result.manifest_payload["output_completeness_ok"] is True
    assert result.manifest_payload["observation_contract"]["scaling_policy"]["enabled"] is False
    assert result.manifest_payload["observation_contract"]["state_feature_columns"] == ["feat_a", "feat_b", "evt_flag"]
    assert result.manifest_payload["runtime_price_contract"] == {
        "timestamp_column": "timestamp",
        "execution_price_column": "close",
        "mark_to_market_column": "close",
        "required_runtime_columns": ["close"],
        "runtime_price_dtypes": {"close": "float32"},
        "artifact_columns": ["timestamp", "feat_a", "feat_b", "evt_flag", "close"],
    }
    first_artifact = result.manifest_payload["partition_metadata"][0]
    assert first_artifact["warmup_contract"] == {
        "enabled": False,
        "required_observation_columns": [],
        "policy": "drop_head_until_all_required_obs_numeric",
        "valid_from_row": 0,
        "valid_from_timestamp": first_artifact["timestamp_min_utc"],
        "post_valid_nan_policy": "fail_closed",
        "head_nan_profile": {},
    }
    assert result.manifest_payload["warmup_contract_summary"] == {
        "enabled": False,
        "policy": "drop_head_until_all_required_obs_numeric",
        "post_valid_nan_policy": "fail_closed",
        "artifacts_total": 3,
        "artifacts_with_warmup": 0,
        "max_valid_from_row": 0,
    }
    written = next(iter(written_frames.values()))
    assert list(written.columns) == ["timestamp", "feat_a", "feat_b", "evt_flag", "close"]


def test_supertrend_geometry_contract_replaces_conditional_raw_bands(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_supertrend_geometry"
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
            "AlphaTrend": pd.Series([95.0, 96.0, 97.0, 98.0, 99.0], dtype="float32"),
            "AlphaTrend_2": pd.Series([94.0, 95.0, 96.0, 97.0, 98.0], dtype="float32"),
            "ST_trend": pd.Series([1, 1, -1, -1, -1], dtype="int8"),
            "ST_up": pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan], dtype="float32"),
            "ST_dn": pd.Series([np.nan, np.nan, 101.0, 102.0, 103.0], dtype="float32"),
            "evt_at_buy": pd.Series([0, 0, 0, 0, 0], dtype="uint8"),
            "evt_at_sell": pd.Series([0, 0, 0, 0, 0], dtype="uint8"),
            "evt_st_buy": pd.Series([0, 1, 0, 0, 0], dtype="uint8"),
            "evt_st_sell": pd.Series([0, 0, 1, 0, 0], dtype="uint8"),
            "close": pd.Series([99.0, 101.0, 100.0, 101.0, 104.0], dtype="float32"),
        }
    )

    entry, src = _partition_entry(dataset_root=dataset_root, rel_name="st_geometry.parquet", scope="partition", partition="train", frame=frame)
    frame_map = {str(src.resolve()): frame.copy()}
    _write_dataset_artifacts(
        tmp_path=tmp_path,
        run_id=run_id,
        entries=[entry],
        split_mode="ratio_chrono",
        output_semantics_mode="standard_partitions",
        selected_columns=[
            "timestamp",
            "AlphaTrend",
            "AlphaTrend_2",
            "ST_trend",
            "ST_up",
            "ST_dn",
            "evt_at_buy",
            "evt_at_sell",
            "evt_st_buy",
            "evt_st_sell",
        ],
        selected_dtypes={
            "timestamp": "datetime64[ns, UTC]",
            "AlphaTrend": "float32",
            "AlphaTrend_2": "float32",
            "ST_trend": "int8",
            "ST_up": "float32",
            "ST_dn": "float32",
            "evt_at_buy": "uint8",
            "evt_at_sell": "uint8",
            "evt_st_buy": "uint8",
            "evt_st_sell": "uint8",
        },
    )

    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            enable_scaling=False,
            scaler_type="none",
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )

    assert result.report_payload["state_build_overall"] is True
    assert result.manifest_payload is not None

    observation_contract = result.manifest_payload["observation_contract"]
    assert observation_contract["state_feature_columns"] == [
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_active_line",
        "ST_distance_to_active_line",
        "evt_at_buy",
        "evt_at_sell",
        "evt_st_buy",
        "evt_st_sell",
    ]
    assert observation_contract["event_columns"] == ["evt_at_buy", "evt_at_sell", "evt_st_buy", "evt_st_sell"]
    assert observation_contract["regime_columns"] == ["ST_trend"]
    assert observation_contract["geometry_columns"] == [
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_active_line",
        "ST_distance_to_active_line",
    ]
    assert observation_contract["strict_post_valid_numeric_columns"] == observation_contract["state_feature_columns"]
    assert observation_contract["conditional_raw_columns"] == ["ST_up", "ST_dn"]
    assert observation_contract["conditional_column_policy"] == "exclude_from_core_and_replace_with_geometry"
    assert observation_contract["conditional_column_replacements"] == {
        "ST_up": ["ST_active_line", "ST_distance_to_active_line"],
        "ST_dn": ["ST_active_line", "ST_distance_to_active_line"],
    }
    assert observation_contract["geometry_feature_version"] == "geometry.features.v1"
    assert observation_contract["geometry_feature_formulas"] == {
        "ST_active_line_formula": "deterministic_single_finite_band_with_trend_consistency",
        "ST_distance_to_active_line_formula": "close_minus_active_line",
    }
    assert observation_contract["future_feature_hooks"]["trend_age_context"]["implemented"] is False

    artifact = result.manifest_payload["partition_metadata"][0]
    assert artifact["warmup_contract"]["required_observation_columns"] == ["ST_active_line", "ST_distance_to_active_line"]
    assert artifact["warmup_contract"]["valid_from_row"] == 1

    written = next(iter(written_frames.values()))
    assert list(written.columns) == [
        "timestamp",
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_active_line",
        "ST_distance_to_active_line",
        "evt_at_buy",
        "evt_at_sell",
        "evt_st_buy",
        "evt_st_sell",
        "close",
    ]
    assert "ST_up" not in written.columns
    assert "ST_dn" not in written.columns
    np.testing.assert_allclose(
        written["ST_active_line"].to_numpy(dtype=np.float32),
        np.asarray([np.nan, 100.0, 101.0, 102.0, 103.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        written["ST_distance_to_active_line"].to_numpy(dtype=np.float32),
        np.asarray([np.nan, 1.0, -1.0, -1.0, 1.0], dtype=np.float32),
        equal_nan=True,
    )
    assert str(written["evt_st_buy"].dtype) == "uint8"


def test_supertrend_geometry_contract_fails_closed_on_trend_band_mismatch(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_supertrend_mismatch"
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC"),
            "ST_trend": pd.Series([1, 1, 1, -1], dtype="int8"),
            "ST_up": pd.Series([np.nan, 100.0, np.nan, np.nan], dtype="float32"),
            "ST_dn": pd.Series([np.nan, np.nan, 101.0, 102.0], dtype="float32"),
            "evt_st_buy": pd.Series([0, 1, 0, 0], dtype="uint8"),
            "close": pd.Series([99.0, 101.0, 100.0, 99.0], dtype="float32"),
        }
    )

    entry, src = _partition_entry(dataset_root=dataset_root, rel_name="st_mismatch.parquet", scope="partition", partition="train", frame=frame)
    frame_map = {str(src.resolve()): frame.copy()}
    _write_dataset_artifacts(
        tmp_path=tmp_path,
        run_id=run_id,
        entries=[entry],
        split_mode="ratio_chrono",
        output_semantics_mode="standard_partitions",
        selected_columns=["timestamp", "ST_trend", "ST_up", "ST_dn", "evt_st_buy"],
        selected_dtypes={
            "timestamp": "datetime64[ns, UTC]",
            "ST_trend": "int8",
            "ST_up": "float32",
            "ST_dn": "float32",
            "evt_st_buy": "uint8",
        },
    )

    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            enable_scaling=False,
            scaler_type="none",
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )

    assert result.report_payload["state_build_overall"] is False
    assert any("ST_trend must agree with the selected SuperTrend active band." == item["message"] for item in result.report_payload["errors"])


def test_warmup_contract_generation_counts_leading_non_finite_only(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_warmup_contract"
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC"),
            "EMA_200": pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32"),
            "EMA_600": pd.Series([np.nan, np.nan, 10.0, 11.0, np.nan, 13.0], dtype="float32"),
            "EMA_1200": pd.Series([np.nan, np.nan, np.nan, 20.0, 21.0, 22.0], dtype="float32"),
            "feat_gap": pd.Series([30.0, 31.0, np.nan, 33.0, 34.0, 35.0], dtype="float32"),
            "evt_flag": pd.Series([0, 1, 0, 1, 0, 1], dtype="uint8"),
            "close": pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], dtype="float32"),
        }
    )

    entry, src = _partition_entry(dataset_root=dataset_root, rel_name="warmup.parquet", scope="partition", partition="train", frame=frame)
    frame_map = {str(src.resolve()): frame.copy()}
    _write_dataset_artifacts(
        tmp_path=tmp_path,
        run_id=run_id,
        entries=[entry],
        split_mode="ratio_chrono",
        output_semantics_mode="standard_partitions",
        selected_columns=["timestamp", "EMA_200", "EMA_600", "EMA_1200", "feat_gap", "evt_flag"],
        selected_dtypes={
            "timestamp": "datetime64[ns, UTC]",
            "EMA_200": "float32",
            "EMA_600": "float32",
            "EMA_1200": "float32",
            "feat_gap": "float32",
            "evt_flag": "uint8",
        },
    )

    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )

    assert result.report_payload["state_build_overall"] is True
    assert result.manifest_payload is not None

    artifact = result.manifest_payload["partition_metadata"][0]
    assert artifact["row_count"] == 6
    assert artifact["warmup_contract"] == {
        "enabled": True,
        "required_observation_columns": ["EMA_200", "EMA_600", "EMA_1200"],
        "policy": "drop_head_until_all_required_obs_numeric",
        "valid_from_row": 3,
        "valid_from_timestamp": "2024-01-01T00:03:00+00:00",
        "post_valid_nan_policy": "fail_closed",
        "head_nan_profile": {
            "EMA_200": 1,
            "EMA_600": 2,
            "EMA_1200": 3,
        },
    }
    assert result.manifest_payload["warmup_contract_summary"] == {
        "enabled": True,
        "policy": "drop_head_until_all_required_obs_numeric",
        "post_valid_nan_policy": "fail_closed",
        "artifacts_total": 1,
        "artifacts_with_warmup": 1,
        "max_valid_from_row": 3,
    }
    written = next(iter(written_frames.values()))
    assert len(written) == 6


def test_mixed_timeframe_per_file_correctness(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_mixed_tf"
    dataset_root, frame_map = _seed_mixed_timeframe_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    assert result.report_payload["state_build_overall"] is True
    assert result.report_payload["totals"]["files_processed"] == 9
    assert result.report_payload["totals"]["artifacts_written"] == 9


def test_row_ordering_and_timestamp_uniqueness_checks(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_ordering"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)
    bad = frame_map[next(iter(frame_map.keys()))].copy()
    bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
    first_key = next(iter(frame_map.keys()))
    frame_map[first_key] = bad

    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    codes = _error_codes(result.report_payload)
    assert STATE_BUILD_TIMESTAMP_DUPLICATES in codes
    assert STATE_BUILD_ORDERING_CONTRACT_VIOLATION in codes


def test_scaling_standard_fit_on_train_only(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_scaling_standard"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            enable_scaling=True,
            scaler_type="standard",
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )
    assert result.report_payload["state_build_overall"] is True
    assert result.manifest_payload is not None

    scaling_policy = result.manifest_payload["observation_contract"]["scaling_policy"]
    assert scaling_policy["enabled"] is True
    assert scaling_policy["fit_scope_policy"] == "train_only"
    assert result.scaler_stats_payload is not None

    groups = result.scaler_stats_payload["groups"]
    assert groups
    assert groups[0]["scope"]["partition"] == "train"


def test_scaling_walk_forward_fit_per_fold_train(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_scaling_walk_forward"
    dataset_root, frame_map = _seed_walk_forward_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            enable_scaling=True,
            scaler_type="standard",
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )
    assert result.report_payload["state_build_overall"] is True
    assert result.scaler_stats_payload is not None

    groups = result.scaler_stats_payload["groups"]
    scopes = {(item["scope"]["source_rel"], item["scope"].get("fold_id"), item["scope"]["partition"]) for item in groups}
    assert ("wf.parquet", 0, "train") in scopes
    assert ("wf.parquet", 1, "train") in scopes


def test_state_build_id_is_deterministic(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_build_id_det"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)

    written_frames_1: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames_1)
    result_1 = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            overwrite=True,
        )
    )

    written_frames_2: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames_2)
    result_2 = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            overwrite=True,
        )
    )

    assert result_1.report_payload["state_build_id"] == result_2.report_payload["state_build_id"]


def test_output_completeness_expected_vs_actual(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_completeness"
    dataset_root, frame_map, manifest_path, _, _ = _seed_standard_run(tmp_path, run_id)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["partition_metadata"][0]["row_count"] = 999
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    assert STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH in _error_codes(result.report_payload)


def test_no_mutation_of_upstream_artifacts(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_no_mutation"
    dataset_root, frame_map, manifest_path, report_path, _ = _seed_standard_run(tmp_path, run_id)
    source_path = Path(next(iter(frame_map.keys())))

    before_manifest = _fingerprint(manifest_path)
    before_report = _fingerprint(report_path)
    before_source = _fingerprint(source_path)

    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )
    assert result.report_payload["state_build_overall"] is True

    assert _fingerprint(manifest_path) == before_manifest
    assert _fingerprint(report_path) == before_report
    assert _fingerprint(source_path) == before_source


def test_persisted_metadata_has_no_staging_paths(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_no_staging_metadata"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    output_root = tmp_path / "runs" / run_id / "data_states"
    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=output_root,
        )
    )
    assert result.report_payload["state_build_overall"] is True

    report_text = (output_root / "reports" / "state_build_report.json").read_text(encoding="utf-8")
    manifest_text = (output_root / "reports" / "state_manifest.json").read_text(encoding="utf-8")
    assert "__staging__" not in report_text
    assert "__staging__" not in manifest_text

    report_payload = json.loads(report_text)
    manifest_payload = json.loads(manifest_text)
    assert report_payload["staging_root"] is None

    output_root_resolved = output_root.resolve()
    for item in manifest_payload["partition_metadata"]:
        output_path = Path(str(item["output_path"])).resolve()
        assert output_path.is_relative_to(output_root_resolved)
        assert "__staging__" not in str(output_path)


def test_persisted_failure_report_has_no_staging_paths(tmp_path: Path) -> None:
    run_id = "state_no_staging_failure_report"
    dataset_root, _, _, _, _ = _seed_standard_run(tmp_path, run_id)

    output_root = tmp_path / "runs" / run_id / "data_states"
    staging_root = output_root.parent / f"{output_root.name}.__staging__"
    staging_root.mkdir(parents=True, exist_ok=True)
    (staging_root / "occupied.txt").write_text("occupied", encoding="utf-8")

    result = build_states(
        _state_build_options(
            run_id=run_id,
            dataset_root=dataset_root,
            output_root=output_root,
            overwrite=False,
        )
    )
    assert result.report_payload["state_build_overall"] is False
    assert STATE_BUILD_STAGING_ROOT_COLLISION in _error_codes(result.report_payload)

    report_path = output_root / "reports" / "state_build_report.json"
    report_text = report_path.read_text(encoding="utf-8")
    assert "__staging__" not in report_text


def test_scaler_type_unsupported_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_scaler_unsupported"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            enable_scaling=True,
            scaler_type="unknown",
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )
    assert STATE_BUILD_SCALER_TYPE_UNSUPPORTED in _error_codes(result.report_payload)


def test_sequence_and_aggregate_defer_fail(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_defer_flags"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result_seq = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            sequence_mode=True,
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )
    assert STATE_BUILD_SEQUENCE_MODE_DEFERRED in _error_codes(result_seq.report_payload)

    result_agg = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            aggregate_walk_forward=True,
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )
    assert STATE_BUILD_AGGREGATE_WALK_FORWARD_DEFERRED in _error_codes(result_agg.report_payload)


def test_state_columns_invalid_fails(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_columns_invalid"
    dataset_root, frame_map, _, _, _ = _seed_standard_run(tmp_path, run_id)
    written_frames: dict[str, pd.DataFrame] = {}
    _patch_parquet_io(monkeypatch, frame_map=frame_map, written_frames=written_frames)

    result = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
            state_columns=("timestamp", "missing_col"),
            execution_price_column="close",
            mark_to_market_column="close",
        )
    )
    assert STATE_BUILD_COLUMN_SELECTION_INVALID in _error_codes(result.report_payload)


def test_runtime_price_columns_are_explicitly_required(tmp_path: Path) -> None:
    run_id = "state_runtime_price_required"
    dataset_root, _, _, _, _ = _seed_standard_run(tmp_path, run_id)

    result = build_states(
        StateBuildOptions(
            run_id=run_id,
            input_root=dataset_root,
            output_root=tmp_path / "runs" / run_id / "data_states",
        )
    )

    assert result.report_payload["state_build_overall"] is False
    assert STATE_BUILD_RUNTIME_PRICE_CONTRACT_REQUIRED in _error_codes(result.report_payload)
