"""Dataset build subsystem for Milestone 4.3.

This module materializes train/val/test datasets from validated feature artifacts.
It is fail-closed and consumes split_validation_report.json as source-of-truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from core.io_atomic import atomic_write_json, atomic_write_parquet
from core.logging import get_logger
from core.paths import ensure_within_root

LOGGER = get_logger(__name__)

DATASET_BUILDER_VERSION = "dataset_builder.v1"
DATASET_MANIFEST_VERSION = "datasets.manifest.v1"
DEFAULT_BUILD_MODE = "materialize_only"

DATASET_BUILD_PRECONDITION_FAILED = "DATASET_BUILD_PRECONDITION_FAILED"
DATASET_BUILD_FEATURE_MANIFEST_MISSING = "DATASET_BUILD_FEATURE_MANIFEST_MISSING"
DATASET_BUILD_FEATURE_MANIFEST_INVALID = "DATASET_BUILD_FEATURE_MANIFEST_INVALID"
DATASET_BUILD_TRAIN_INPUT_REPORT_MISSING = "DATASET_BUILD_TRAIN_INPUT_REPORT_MISSING"
DATASET_BUILD_TRAIN_INPUT_REPORT_INVALID = "DATASET_BUILD_TRAIN_INPUT_REPORT_INVALID"
DATASET_BUILD_TRAIN_INPUT_NOT_PASSED = "DATASET_BUILD_TRAIN_INPUT_NOT_PASSED"
DATASET_BUILD_SPLIT_REPORT_MISSING = "DATASET_BUILD_SPLIT_REPORT_MISSING"
DATASET_BUILD_SPLIT_REPORT_INVALID = "DATASET_BUILD_SPLIT_REPORT_INVALID"
DATASET_BUILD_SPLIT_NOT_PASSED = "DATASET_BUILD_SPLIT_NOT_PASSED"
DATASET_BUILD_SPLIT_PARTITION_DEFS_MISSING = "DATASET_BUILD_SPLIT_PARTITION_DEFS_MISSING"
DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT = "DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT"
DATASET_BUILD_RUN_ID_MISMATCH = "DATASET_BUILD_RUN_ID_MISMATCH"
DATASET_BUILD_LINEAGE_MISMATCH = "DATASET_BUILD_LINEAGE_MISMATCH"
DATASET_BUILD_OUTPUT_ROOT_EXISTS = "DATASET_BUILD_OUTPUT_ROOT_EXISTS"
DATASET_BUILD_STAGING_ROOT_COLLISION = "DATASET_BUILD_STAGING_ROOT_COLLISION"
DATASET_BUILD_COLUMN_SELECTION_INVALID = "DATASET_BUILD_COLUMN_SELECTION_INVALID"
DATASET_BUILD_SOURCE_FILE_MISSING = "DATASET_BUILD_SOURCE_FILE_MISSING"
DATASET_BUILD_PARTITION_EMPTY = "DATASET_BUILD_PARTITION_EMPTY"
DATASET_BUILD_PARTITION_TIMESTAMP_DUPLICATE = "DATASET_BUILD_PARTITION_TIMESTAMP_DUPLICATE"
DATASET_BUILD_ORDERING_CONTRACT_VIOLATION = "DATASET_BUILD_ORDERING_CONTRACT_VIOLATION"
DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH = "DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH"
DATASET_BUILD_WRITE_FAILED = "DATASET_BUILD_WRITE_FAILED"
DATASET_BUILD_RUNTIME_ERROR = "DATASET_BUILD_RUNTIME_ERROR"

DATASET_BUILD_REPORT_WRITE_FAILED = "DATASET_BUILD_REPORT_WRITE_FAILED"
DATASET_BUILD_SUMMARY_UPDATE_FAILED = "DATASET_BUILD_SUMMARY_UPDATE_FAILED"


@dataclass
class ValidationIssue:
    """Machine-readable issue payload."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetBuildOptions:
    """Runtime options for dataset materialization."""

    run_id: str
    input_root: Path
    reports_root: Path
    output_root: Path | None = None
    feature_manifest_path: Path | None = None
    train_input_report_path: Path | None = None
    split_report_path: Path | None = None
    dataset_config_path: Path | None = None
    include_feature_groups: tuple[str, ...] = ()
    exclude_columns: tuple[str, ...] = ()
    timestamp_column_override: str | None = None
    require_train_input_validation: bool = True
    require_split_validation: bool = True
    aggregate_walk_forward: bool = False
    overwrite: bool = False
    strict_column_selection: bool = True
    build_mode: str = DEFAULT_BUILD_MODE

    def to_invocation_args(self) -> dict[str, Any]:
        """Serialize invocation arguments for report payload."""

        return {
            "run_id": self.run_id,
            "input_root": str(self.input_root),
            "reports_root": str(self.reports_root),
            "output_root": str(self.output_root) if self.output_root is not None else None,
            "feature_manifest_path": str(self.feature_manifest_path) if self.feature_manifest_path is not None else None,
            "train_input_report_path": str(self.train_input_report_path) if self.train_input_report_path is not None else None,
            "split_report_path": str(self.split_report_path) if self.split_report_path is not None else None,
            "dataset_config_path": str(self.dataset_config_path) if self.dataset_config_path is not None else None,
            "include_feature_groups": list(self.include_feature_groups),
            "exclude_columns": list(self.exclude_columns),
            "timestamp_column_override": self.timestamp_column_override,
            "require_train_input_validation": bool(self.require_train_input_validation),
            "require_split_validation": bool(self.require_split_validation),
            "aggregate_walk_forward": bool(self.aggregate_walk_forward),
            "overwrite": bool(self.overwrite),
            "strict_column_selection": bool(self.strict_column_selection),
            "build_mode": self.build_mode,
        }


@dataclass(frozen=True)
class MaterializationSpec:
    """Expected artifact spec from split source-of-truth."""

    scope: str
    source_file: Path
    source_rel: str
    partition: str
    fold_id: int | None
    start_utc: str
    end_inclusive_utc: str
    expected_rows: int
    allow_duplicate_timestamps: bool = False


@dataclass
class MaterializedArtifact:
    """Produced artifact metadata."""

    scope: str
    source_rel: str
    partition: str
    fold_id: int | None
    output_path: str
    row_count: int
    timestamp_min_utc: str | None
    timestamp_max_utc: str | None
    duplicate_timestamp_count: int
    timestamp_unique_ok: bool
    file_sha256: str

    def key(self) -> tuple[str, str, str, int | None]:
        """Return deterministic artifact key."""

        return (self.scope, self.source_rel, self.partition, self.fold_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize into JSON payload."""

        return asdict(self)


@dataclass
class DatasetBuildResult:
    """Result payload for dataset build call."""

    report_payload: dict[str, Any]
    manifest_payload: dict[str, Any] | None
    report_path: Path
    manifest_path: Path


def build_datasets(options: DatasetBuildOptions) -> DatasetBuildResult:
    """Build dataset artifacts from validated split and train-input reports."""

    if not options.run_id.strip():
        raise ValueError("run_id must be non-empty")

    run_id = options.run_id.strip()
    input_root = options.input_root.resolve()
    reports_root = options.reports_root.resolve()
    output_root = options.output_root.resolve() if options.output_root is not None else _default_output_root(reports_root)

    feature_manifest_path = (
        options.feature_manifest_path.resolve() if options.feature_manifest_path is not None else (reports_root / "feature_manifest.json").resolve()
    )
    train_input_report_path = (
        options.train_input_report_path.resolve()
        if options.train_input_report_path is not None
        else (reports_root / "train_input_validation_report.json").resolve()
    )
    split_report_path = (
        options.split_report_path.resolve() if options.split_report_path is not None else (reports_root / "split_validation_report.json").resolve()
    )

    report_path = output_root / "reports" / "dataset_build_report.json"
    manifest_path = output_root / "reports" / "dataset_manifest.json"
    staging_root = output_root.parent / f"{output_root.name}.__staging__"

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    split_mode: str | None = None
    output_semantics: dict[str, Any] = {
        "mode": "unknown",
        "fold_outputs_generated": False,
        "top_level_partitions_generated": False,
        "aggregate_walk_forward": bool(options.aggregate_walk_forward),
    }

    report_payload = _base_report_payload(
        run_id=run_id,
        input_root=input_root,
        output_root=output_root,
        report_path=report_path,
        manifest_path=manifest_path,
        staging_root=staging_root,
        invocation_args=options.to_invocation_args(),
        source_paths={
            "feature_manifest_path": str(feature_manifest_path),
            "train_input_validation_report_path": str(train_input_report_path),
            "split_validation_report_path": str(split_report_path),
        },
    )
    report_payload["build_mode"] = options.build_mode

    manifest_payload: dict[str, Any] | None = None

    feature_manifest = _load_json_object(
        feature_manifest_path,
        missing_code=DATASET_BUILD_FEATURE_MANIFEST_MISSING,
        invalid_code=DATASET_BUILD_FEATURE_MANIFEST_INVALID,
        missing_message="feature_manifest.json not found.",
        invalid_message="feature_manifest.json is invalid.",
        errors=errors,
    )
    train_input_report = _load_json_object(
        train_input_report_path,
        missing_code=DATASET_BUILD_TRAIN_INPUT_REPORT_MISSING,
        invalid_code=DATASET_BUILD_TRAIN_INPUT_REPORT_INVALID,
        missing_message="train_input_validation_report.json not found.",
        invalid_message="train_input_validation_report.json is invalid.",
        errors=errors,
    )
    split_report = _load_json_object(
        split_report_path,
        missing_code=DATASET_BUILD_SPLIT_REPORT_MISSING,
        invalid_code=DATASET_BUILD_SPLIT_REPORT_INVALID,
        missing_message="split_validation_report.json not found.",
        invalid_message="split_validation_report.json is invalid.",
        errors=errors,
    )

    if feature_manifest is not None:
        _require_run_id(field_name="feature_manifest.run_id", payload=feature_manifest, run_id=run_id, errors=errors)
    if train_input_report is not None:
        _require_run_id(field_name="train_input_validation_report.run_id", payload=train_input_report, run_id=run_id, errors=errors)
    if split_report is not None:
        _require_run_id(field_name="split_validation_report.run_id", payload=split_report, run_id=run_id, errors=errors)

    if train_input_report is not None:
        overall = train_input_report.get("train_input_validation_overall")
        if not isinstance(overall, bool):
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_TRAIN_INPUT_REPORT_INVALID,
                    message="train_input_validation_overall must be boolean.",
                    context={"value": overall},
                )
            )
        elif not overall and options.require_train_input_validation:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_TRAIN_INPUT_NOT_PASSED,
                    message="train_input_validation_overall must be true before dataset build.",
                    context={"train_input_validation_overall": overall},
                )
            )
        elif not overall:
            warnings.append(
                ValidationIssue(
                    code=DATASET_BUILD_TRAIN_INPUT_NOT_PASSED,
                    message="train_input_validation_overall=false; continuing because requirement is disabled.",
                    context={"require_train_input_validation": False},
                )
            )

    if split_report is not None:
        split_overall = split_report.get("split_validation_overall")
        split_mode_raw = split_report.get("split_mode")
        if isinstance(split_mode_raw, str) and split_mode_raw.strip():
            split_mode = split_mode_raw.strip()
        if not isinstance(split_overall, bool):
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_SPLIT_REPORT_INVALID,
                    message="split_validation_overall must be boolean.",
                    context={"value": split_overall},
                )
            )
        elif not split_overall:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_SPLIT_NOT_PASSED,
                    message="split_validation_overall must be true before dataset build.",
                    context={"split_validation_overall": split_overall},
                )
            )

    if split_report is not None:
        _validate_lineage_paths(
            split_report=split_report,
            manifest_path=feature_manifest_path,
            train_input_report_path=train_input_report_path,
            errors=errors,
        )

    source_files = discover_parquet_files(input_root)
    if not source_files:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_SOURCE_FILE_MISSING,
                message="No source parquet files found under input_root.",
                context={"input_root": str(input_root)},
            )
        )

    selected_columns: list[str] = []
    selected_dtype_map: dict[str, str] = {}
    timestamp_column = "timestamp"
    column_selection_contract: dict[str, Any] = {}

    if feature_manifest is not None:
        selected_columns, selected_dtype_map, timestamp_column, column_selection_contract = _resolve_column_selection(
            feature_manifest=feature_manifest,
            options=options,
            errors=errors,
            warnings=warnings,
        )

    expected_specs: list[MaterializationSpec] = []
    expected_coverage: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}

    if split_report is not None and split_mode is not None:
        expected_specs, expected_coverage = _build_expected_specs(
            split_report=split_report,
            split_mode=split_mode,
            input_root=input_root,
            source_files=source_files,
            aggregate_walk_forward=bool(options.aggregate_walk_forward),
            errors=errors,
        )

    output_semantics = _resolve_output_semantics(split_mode=split_mode, aggregate_walk_forward=bool(options.aggregate_walk_forward))

    overwrite_policy = _evaluate_overwrite_policy(
        output_root=output_root,
        staging_root=staging_root,
        overwrite=bool(options.overwrite),
        errors=errors,
    )

    report_payload["split_mode"] = split_mode
    report_payload["output_semantics"] = output_semantics
    report_payload["overwrite_policy"] = overwrite_policy
    report_payload["column_selection_contract"] = column_selection_contract

    if errors:
        report_payload["errors"] = [asdict(issue) for issue in errors]
        report_payload["warnings"] = [asdict(issue) for issue in warnings]
        report_payload["dataset_build_overall"] = False
        report_payload["error_code"] = DATASET_BUILD_PRECONDITION_FAILED
        _write_report_best_effort(report_payload, report_path, warnings)
        return DatasetBuildResult(
            report_payload=report_payload,
            manifest_payload=None,
            report_path=report_path,
            manifest_path=manifest_path,
        )

    feature_manifest_hash = _sha256_file(feature_manifest_path)
    train_input_report_hash = _sha256_file(train_input_report_path)
    split_report_hash = _sha256_file(split_report_path)
    source_inventory_hash = _hash_sequence(sorted(str(path.resolve().relative_to(input_root)) for path in source_files))
    column_selection_hash = _hash_sequence(selected_columns)

    report_payload["source_hashes"] = {
        "feature_manifest_hash": feature_manifest_hash,
        "train_input_report_hash": train_input_report_hash,
        "split_report_hash": split_report_hash,
        "source_file_inventory_hash": source_inventory_hash,
    }

    dataset_build_id = _compute_dataset_build_id(
        run_id=run_id,
        split_mode=split_mode,
        output_semantics_mode=str(output_semantics["mode"]),
        aggregate_walk_forward=bool(options.aggregate_walk_forward),
        timestamp_column=timestamp_column,
        column_selection_hash=column_selection_hash,
        build_mode=options.build_mode,
        feature_manifest_hash=feature_manifest_hash,
        train_input_report_hash=train_input_report_hash,
        split_report_hash=split_report_hash,
        source_file_inventory_hash=source_inventory_hash,
    )
    dataset_build_id_policy = {
        "algorithm": "sha256",
        "canonical_json": {"sort_keys": True, "separators": [",", ":"], "ensure_ascii": True},
        "hash_inputs_order": [
            "run_id",
            "builder_version",
            "build_mode",
            "split_mode",
            "output_semantics_mode",
            "aggregate_walk_forward",
            "timestamp_column",
            "column_selection_hash",
            "feature_manifest_hash",
            "train_input_report_hash",
            "split_report_hash",
            "source_file_inventory_hash",
        ],
    }

    artifacts: list[MaterializedArtifact] = []
    expected_specs_by_source = _group_specs_by_source(expected_specs)
    partition_frames_for_aggregate: dict[tuple[str, str], list[tuple[int, pd.DataFrame]]] = {}

    rows_read = 0
    rows_written = 0
    files_processed = 0
    files_failed = 0

    _prepare_staging_root(staging_root=staging_root, overwrite=bool(options.overwrite), errors=errors)
    if errors:
        report_payload["errors"] = [asdict(issue) for issue in errors]
        report_payload["warnings"] = [asdict(issue) for issue in warnings]
        report_payload["dataset_build_overall"] = False
        report_payload["error_code"] = DATASET_BUILD_PRECONDITION_FAILED
        _write_report_best_effort(report_payload, report_path, warnings)
        return DatasetBuildResult(report_payload=report_payload, manifest_payload=None, report_path=report_path, manifest_path=manifest_path)

    for source_path, specs in sorted(expected_specs_by_source.items(), key=lambda item: str(item[0])):
        files_processed += 1
        try:
            frame = pd.read_parquet(source_path)
            rows_read += int(len(frame))
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_SOURCE_FILE_MISSING,
                    message="Failed to read source parquet for materialization.",
                    context={"input_file": str(source_path), "error": str(exc)},
                )
            )
            files_failed += 1
            continue

        if timestamp_column not in frame.columns:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                    message="Timestamp column is missing in source parquet.",
                    context={"input_file": str(source_path), "timestamp_column": timestamp_column},
                )
            )
            files_failed += 1
            continue

        missing_cols = [col for col in selected_columns if col not in frame.columns]
        if missing_cols:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                    message="Selected columns are missing in source parquet.",
                    context={"input_file": str(source_path), "missing_columns": missing_cols},
                )
            )
            files_failed += 1
            continue

        frame_with_pos = frame.copy()
        frame_with_pos["__row_position"] = pd.Series(range(len(frame_with_pos)), dtype="int64")

        parsed_ts = pd.to_datetime(frame_with_pos[timestamp_column], utc=True, errors="coerce")
        if parsed_ts.isna().any():
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_ORDERING_CONTRACT_VIOLATION,
                    message="Timestamp parsing failed in source parquet.",
                    context={"input_file": str(source_path), "timestamp_column": timestamp_column},
                )
            )
            files_failed += 1
            continue
        frame_with_pos[timestamp_column] = parsed_ts

        for col in selected_columns:
            expected_dtype = selected_dtype_map.get(col)
            actual_dtype = str(frame_with_pos[col].dtype)
            if expected_dtype is not None and expected_dtype != actual_dtype:
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                        message="Source dtype does not match selected contract dtype.",
                        context={"input_file": str(source_path), "column": col, "expected_dtype": expected_dtype, "actual_dtype": actual_dtype},
                    )
                )

        source_rel = str(source_path.resolve().relative_to(input_root))

        for spec in sorted(specs, key=lambda item: (item.scope, item.fold_id if item.fold_id is not None else -1, item.partition)):
            subset = _slice_partition(
                frame=frame_with_pos,
                timestamp_column=timestamp_column,
                start_utc=spec.start_utc,
                end_inclusive_utc=spec.end_inclusive_utc,
            )
            subset = subset.sort_values([timestamp_column, "__row_position"], kind="mergesort").reset_index(drop=True)
            out_df = subset.loc[:, selected_columns].copy()

            if len(out_df) == 0:
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_PARTITION_EMPTY,
                        message="Materialized partition produced zero rows.",
                        context={"source_file": source_rel, "scope": spec.scope, "partition": spec.partition, "fold_id": spec.fold_id},
                    )
                )

            ts_out = pd.to_datetime(out_df[timestamp_column], utc=True, errors="coerce") if len(out_df) > 0 else pd.Series(dtype="datetime64[ns, UTC]")
            if len(out_df) > 0 and (not ts_out.is_monotonic_increasing):
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_ORDERING_CONTRACT_VIOLATION,
                        message="Materialized partition is not monotonic by timestamp.",
                        context={"source_file": source_rel, "scope": spec.scope, "partition": spec.partition, "fold_id": spec.fold_id},
                    )
                )

            duplicate_count = int(ts_out.duplicated().sum()) if len(out_df) > 0 else 0
            unique_ok = duplicate_count == 0
            if duplicate_count > 0 and not spec.allow_duplicate_timestamps:
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_PARTITION_TIMESTAMP_DUPLICATE,
                        message="Duplicate timestamps detected in materialized partition.",
                        context={
                            "source_file": source_rel,
                            "scope": spec.scope,
                            "partition": spec.partition,
                            "fold_id": spec.fold_id,
                            "duplicate_timestamp_count": duplicate_count,
                        },
                    )
                )
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_ORDERING_CONTRACT_VIOLATION,
                        message="Ordering contract violation due to duplicate timestamps in partition.",
                        context={
                            "source_file": source_rel,
                            "scope": spec.scope,
                            "partition": spec.partition,
                            "fold_id": spec.fold_id,
                            "duplicate_timestamp_count": duplicate_count,
                        },
                    )
                )

            if spec.scope == "aggregate":
                out_path = _build_aggregate_output_path(staging_root, source_rel, spec.partition)
            elif spec.scope == "fold":
                if spec.fold_id is None:
                    raise ValueError("fold scope requires fold_id")
                out_path = _build_fold_output_path(staging_root, source_rel, spec.partition, spec.fold_id)
            else:
                out_path = _build_partition_output_path(staging_root, source_rel, spec.partition)

            ensure_within_root(out_path, staging_root)
            try:
                atomic_write_parquet(out_df, out_path)
            except RuntimeError as exc:
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_WRITE_FAILED,
                        message="Failed to write partition parquet atomically.",
                        context={"output_path": str(out_path), "error": str(exc)},
                    )
                )
                continue

            rows_written += int(len(out_df))
            file_sha = _sha256_file(out_path)
            timestamp_min = ts_out.iloc[0].isoformat() if len(out_df) > 0 else None
            timestamp_max = ts_out.iloc[-1].isoformat() if len(out_df) > 0 else None
            artifacts.append(
                MaterializedArtifact(
                    scope=spec.scope,
                    source_rel=source_rel,
                    partition=spec.partition,
                    fold_id=spec.fold_id,
                    output_path=str(out_path),
                    row_count=int(len(out_df)),
                    timestamp_min_utc=timestamp_min,
                    timestamp_max_utc=timestamp_max,
                    duplicate_timestamp_count=duplicate_count,
                    timestamp_unique_ok=unique_ok,
                    file_sha256=file_sha,
                )
            )

            if split_mode == "walk_forward" and bool(options.aggregate_walk_forward) and spec.scope == "fold":
                key = (source_rel, spec.partition)
                if spec.fold_id is None:
                    raise ValueError("walk-forward fold materialization requires fold_id")
                partition_frames_for_aggregate.setdefault(key, []).append((spec.fold_id, out_df.copy()))

    if split_mode == "walk_forward" and bool(options.aggregate_walk_forward):
        aggregate_specs = _build_walk_forward_aggregate_specs(expected_specs)
        for spec in aggregate_specs:
            key = (spec.source_rel, spec.partition)
            fold_frames = partition_frames_for_aggregate.get(key, [])
            if not fold_frames:
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                        message="Aggregate walk-forward frames are missing for source/partition.",
                        context={"source_file": spec.source_rel, "partition": spec.partition},
                    )
                )
                continue

            ordered = [frame for _, frame in sorted(fold_frames, key=lambda item: item[0])]
            out_df = pd.concat(ordered, axis=0, ignore_index=True)
            ts_out = pd.to_datetime(out_df[timestamp_column], utc=True, errors="coerce") if len(out_df) > 0 else pd.Series(dtype="datetime64[ns, UTC]")
            if len(out_df) > 0 and ts_out.isna().any():
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_ORDERING_CONTRACT_VIOLATION,
                        message="Aggregate walk-forward output has invalid timestamps.",
                        context={"source_file": spec.source_rel, "partition": spec.partition},
                    )
                )
                continue

            duplicate_count = int(ts_out.duplicated().sum()) if len(out_df) > 0 else 0
            out_path = _build_aggregate_output_path(staging_root, spec.source_rel, spec.partition)
            ensure_within_root(out_path, staging_root)

            try:
                atomic_write_parquet(out_df, out_path)
            except RuntimeError as exc:
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_WRITE_FAILED,
                        message="Failed to write aggregate walk-forward parquet atomically.",
                        context={"output_path": str(out_path), "error": str(exc)},
                    )
                )
                continue

            rows_written += int(len(out_df))
            file_sha = _sha256_file(out_path)
            timestamp_min = ts_out.iloc[0].isoformat() if len(out_df) > 0 else None
            timestamp_max = ts_out.iloc[-1].isoformat() if len(out_df) > 0 else None
            artifacts.append(
                MaterializedArtifact(
                    scope="aggregate",
                    source_rel=spec.source_rel,
                    partition=spec.partition,
                    fold_id=None,
                    output_path=str(out_path),
                    row_count=int(len(out_df)),
                    timestamp_min_utc=timestamp_min,
                    timestamp_max_utc=timestamp_max,
                    duplicate_timestamp_count=duplicate_count,
                    timestamp_unique_ok=(duplicate_count == 0),
                    file_sha256=file_sha,
                )
            )

    if split_mode == "walk_forward" and bool(options.aggregate_walk_forward):
        expected_coverage = _maybe_extend_with_aggregate_coverage(expected_coverage, expected_specs)

    output_completeness_ok, completeness_errors = _evaluate_output_completeness(expected_coverage, artifacts)
    if not output_completeness_ok:
        errors.extend(completeness_errors)

    totals = {
        "files_processed": int(files_processed),
        "files_failed": int(files_failed),
        "rows_read": int(rows_read),
        "rows_written": int(rows_written),
        "artifacts_written": int(len(artifacts)),
    }

    partition_summaries = _build_partition_summaries(artifacts)
    fold_summaries = _build_fold_summaries(artifacts)

    report_payload["totals"] = totals
    report_payload["partition_summaries"] = partition_summaries
    report_payload["fold_summaries"] = fold_summaries
    report_payload["output_completeness_ok"] = bool(output_completeness_ok)
    report_payload["dataset_build_id"] = dataset_build_id
    report_payload["dataset_build_id_policy"] = dataset_build_id_policy

    duplicate_timestamp_policy = {
        "default_partition_uniqueness_required": True,
        "aggregate_walk_forward_allowed": bool(split_mode == "walk_forward" and options.aggregate_walk_forward),
        "aggregate_scope": "top_level_partitions_only" if (split_mode == "walk_forward" and options.aggregate_walk_forward) else "not_applicable",
        "reason": "fold overlap is expected" if (split_mode == "walk_forward" and options.aggregate_walk_forward) else "timestamps must be unique per artifact",
    }

    row_order_policy = {
        "name": "timestamp_ascending",
        "stable_tie_breaker": "source_row_position",
        "aggregate_walk_forward_order": "fold_id_ascending_then_timestamp_ascending",
    }

    try:
        persisted_artifacts = _build_promoted_artifact_metadata(
            artifacts=artifacts,
            staging_root=staging_root,
            output_root=output_root,
        )
    except RuntimeError as exc:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_WRITE_FAILED,
                message="Failed to map staged artifact metadata to promoted output paths.",
                context={"error": str(exc)},
            )
        )
        persisted_artifacts = []

    manifest_payload = {
        "manifest_version": DATASET_MANIFEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "builder_version": DATASET_BUILDER_VERSION,
        "dataset_build_id": dataset_build_id,
        "dataset_build_id_policy": dataset_build_id_policy,
        "build_mode": options.build_mode,
        "source_lineage": {
            "feature_manifest_path": str(feature_manifest_path),
            "train_input_validation_report_path": str(train_input_report_path),
            "split_validation_report_path": str(split_report_path),
        },
        "source_hashes": {
            "feature_manifest_hash": feature_manifest_hash,
            "train_input_report_hash": train_input_report_hash,
            "split_report_hash": split_report_hash,
        },
        "split_mode": split_mode,
        "output_semantics": output_semantics,
        "column_selection_contract": {
            **column_selection_contract,
            "column_selection_hash": column_selection_hash,
        },
        "partition_metadata": [item.to_dict() for item in persisted_artifacts],
        "walk_forward_fold_metadata": _build_walk_forward_fold_metadata(persisted_artifacts),
        "row_order_policy": row_order_policy,
        "duplicate_timestamp_policy": duplicate_timestamp_policy,
        "output_completeness_ok": bool(output_completeness_ok),
    }

    report_payload["duplicate_timestamp_policy"] = duplicate_timestamp_policy
    report_payload["row_order_policy"] = row_order_policy
    report_payload["staging_root"] = None
    report_payload["errors"] = [asdict(issue) for issue in errors]
    report_payload["warnings"] = [asdict(issue) for issue in warnings]
    report_payload["dataset_build_overall"] = bool(len(errors) == 0 and output_completeness_ok and files_failed == 0)

    success = bool(report_payload["dataset_build_overall"])

    if success:
        staging_report_path = staging_root / "reports" / "dataset_build_report.json"
        staging_manifest_path = staging_root / "reports" / "dataset_manifest.json"
        try:
            atomic_write_json(report_payload, staging_report_path)
            atomic_write_json(manifest_payload, staging_manifest_path)
            _promote_staging_to_output(staging_root=staging_root, output_root=output_root, overwrite=bool(options.overwrite))
        except RuntimeError as exc:
            success = False
            report_payload["dataset_build_overall"] = False
            report_payload["errors"].append(
                asdict(
                    ValidationIssue(
                        code=DATASET_BUILD_WRITE_FAILED,
                        message="Failed while finalizing staged outputs.",
                        context={"error": str(exc)},
                    )
                )
            )
            _cleanup_staging_root(staging_root)
    else:
        _cleanup_staging_root(staging_root)

    if not success:
        _write_report_best_effort(report_payload, report_path, warnings)
        manifest_payload = None

    return DatasetBuildResult(
        report_payload=report_payload,
        manifest_payload=manifest_payload,
        report_path=report_path,
        manifest_path=manifest_path,
    )


def discover_parquet_files(input_root: Path) -> list[Path]:
    """Discover parquet files recursively under input root."""

    if not input_root.exists() or not input_root.is_dir():
        return []
    files = [path.resolve() for path in input_root.glob("**/*.parquet") if path.is_file() and path.suffix.lower() == ".parquet"]
    return sorted(files)


def _base_report_payload(
    *,
    run_id: str,
    input_root: Path,
    output_root: Path,
    report_path: Path,
    manifest_path: Path,
    staging_root: Path,
    invocation_args: dict[str, Any],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    """Build baseline report payload."""

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "builder_version": DATASET_BUILDER_VERSION,
        "dataset_build_overall": False,
        "dataset_build_id": None,
        "build_mode": DEFAULT_BUILD_MODE,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "staging_root": str(staging_root),
        "dataset_build_report_path": str(report_path),
        "dataset_manifest_path": str(manifest_path),
        "source_paths": source_paths,
        "split_mode": None,
        "output_semantics": {},
        "totals": {
            "files_processed": 0,
            "files_failed": 0,
            "rows_read": 0,
            "rows_written": 0,
            "artifacts_written": 0,
        },
        "partition_summaries": {},
        "fold_summaries": {},
        "output_completeness_ok": False,
        "invocation_args": invocation_args,
        "overwrite_policy": {},
        "column_selection_contract": {},
        "source_hashes": {},
        "errors": [],
        "warnings": [],
    }


def _default_output_root(reports_root: Path) -> Path:
    """Resolve default output root from reports root."""

    return (reports_root.parent.parent / "data_datasets").resolve()


def _load_json_object(
    path: Path,
    *,
    missing_code: str,
    invalid_code: str,
    missing_message: str,
    invalid_message: str,
    errors: list[ValidationIssue],
) -> dict[str, Any] | None:
    """Load JSON object and append deterministic errors on failure."""

    if not path.exists():
        errors.append(ValidationIssue(code=missing_code, message=missing_message, context={"path": str(path)}))
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(
            ValidationIssue(
                code=invalid_code,
                message=invalid_message,
                context={"path": str(path), "error": str(exc)},
            )
        )
        return None

    if not isinstance(payload, dict):
        errors.append(
            ValidationIssue(
                code=invalid_code,
                message="JSON payload must be an object.",
                context={"path": str(path)},
            )
        )
        return None
    return payload


def _require_run_id(*, field_name: str, payload: Mapping[str, Any], run_id: str, errors: list[ValidationIssue]) -> None:
    """Enforce run_id match against expected run id."""

    seen = payload.get("run_id")
    if seen != run_id:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_RUN_ID_MISMATCH,
                message="run_id mismatch across lineage artifacts.",
                context={"field": field_name, "expected_run_id": run_id, "seen_run_id": seen},
            )
        )


def _validate_lineage_paths(
    *,
    split_report: Mapping[str, Any],
    manifest_path: Path,
    train_input_report_path: Path,
    errors: list[ValidationIssue],
) -> None:
    """Cross-check split report lineage references."""

    split_manifest_path = split_report.get("manifest_path")
    if isinstance(split_manifest_path, str) and split_manifest_path.strip():
        if Path(split_manifest_path).resolve() != manifest_path:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_LINEAGE_MISMATCH,
                    message="split_validation_report manifest_path does not match build manifest path.",
                    context={"split_manifest_path": split_manifest_path, "expected_manifest_path": str(manifest_path)},
                )
            )

    split_train_input_path = split_report.get("train_input_validation_report_path")
    if isinstance(split_train_input_path, str) and split_train_input_path.strip():
        if Path(split_train_input_path).resolve() != train_input_report_path:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_LINEAGE_MISMATCH,
                    message="split_validation_report train_input_validation_report_path does not match build train input report path.",
                    context={
                        "split_train_input_validation_report_path": split_train_input_path,
                        "expected_train_input_validation_report_path": str(train_input_report_path),
                    },
                )
            )


def _resolve_column_selection(
    *,
    feature_manifest: Mapping[str, Any],
    options: DatasetBuildOptions,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> tuple[list[str], dict[str, str], str, dict[str, Any]]:
    """Resolve deterministic column selection contract."""

    timestamp_column = options.timestamp_column_override if options.timestamp_column_override else feature_manifest.get("timestamp_column")
    if not isinstance(timestamp_column, str) or not timestamp_column.strip():
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                message="timestamp_column is missing or invalid in manifest/override.",
                context={"timestamp_column": timestamp_column},
            )
        )
        return [], {}, "timestamp", {}
    timestamp_column = timestamp_column.strip()

    continuous_columns = _parse_string_list(feature_manifest.get("continuous_columns"))
    event_columns = _parse_string_list(feature_manifest.get("event_columns"))
    if continuous_columns is None or event_columns is None:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                message="Manifest continuous_columns/event_columns must be list[str].",
                context={},
            )
        )
        return [], {}, timestamp_column, {}

    dtype_map_raw = feature_manifest.get("column_dtypes")
    if not isinstance(dtype_map_raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in dtype_map_raw.items()):
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                message="Manifest column_dtypes must be dict[str, str].",
                context={},
            )
        )
        return [], {}, timestamp_column, {}

    feature_groups_raw = feature_manifest.get("feature_groups")
    if not isinstance(feature_groups_raw, dict):
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                message="Manifest feature_groups must be dict[str, list[str]].",
                context={},
            )
        )
        return [], {}, timestamp_column, {}

    feature_groups: dict[str, list[str]] = {}
    for group_name, cols in feature_groups_raw.items():
        parsed = _parse_string_list(cols)
        if parsed is None:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                    message="Manifest feature_groups contains non-string members.",
                    context={"group": group_name},
                )
            )
            return [], {}, timestamp_column, {}
        feature_groups[str(group_name)] = parsed

    default_feature_columns = _stable_unique([*continuous_columns, *event_columns])

    selected_feature_columns: list[str]
    include_groups = [item.strip() for item in options.include_feature_groups if item.strip()]
    if include_groups:
        selected_feature_columns = []
        for group in include_groups:
            cols = feature_groups.get(group)
            if cols is None:
                issue = ValidationIssue(
                    code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                    message="Unknown include feature group.",
                    context={"group": group, "available_groups": sorted(feature_groups.keys())},
                )
                if options.strict_column_selection:
                    errors.append(issue)
                else:
                    warnings.append(issue)
                continue
            selected_feature_columns.extend(cols)
        selected_feature_columns = _stable_unique([col for col in selected_feature_columns if col != timestamp_column])
    else:
        selected_feature_columns = list(default_feature_columns)

    excludes = [item.strip() for item in options.exclude_columns if item.strip()]
    unknown_excludes = [col for col in excludes if col not in dtype_map_raw]
    if unknown_excludes:
        issue = ValidationIssue(
            code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
            message="Unknown exclude columns.",
            context={"unknown_excludes": unknown_excludes},
        )
        if options.strict_column_selection:
            errors.append(issue)
        else:
            warnings.append(issue)

    if timestamp_column in excludes:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                message="timestamp_column cannot be excluded.",
                context={"timestamp_column": timestamp_column},
            )
        )

    selected_feature_columns = [col for col in selected_feature_columns if col not in set(excludes)]
    selected_columns = [timestamp_column, *selected_feature_columns]
    selected_columns = _stable_unique(selected_columns)

    if len(selected_columns) <= 1:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                message="Selected column set is empty after include/exclude contract.",
                context={"selected_columns": selected_columns},
            )
        )

    missing_dtypes = [col for col in selected_columns if col not in dtype_map_raw]
    if missing_dtypes:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_COLUMN_SELECTION_INVALID,
                message="Selected columns missing dtype declaration in manifest.",
                context={"missing_dtype_columns": missing_dtypes},
            )
        )

    selected_dtype_map = {col: str(dtype_map_raw[col]) for col in selected_columns if col in dtype_map_raw}
    column_order_hash = _hash_sequence(selected_columns)
    dtype_hash = _hash_mapping(selected_dtype_map)

    contract = {
        "timestamp_column": timestamp_column,
        "include_feature_groups": include_groups,
        "exclude_columns": excludes,
        "selected_columns": selected_columns,
        "selected_dtypes": selected_dtype_map,
        "column_order_hash": column_order_hash,
        "dtype_hash": dtype_hash,
    }
    return selected_columns, selected_dtype_map, timestamp_column, contract


def _build_expected_specs(
    *,
    split_report: Mapping[str, Any],
    split_mode: str,
    input_root: Path,
    source_files: Sequence[Path],
    aggregate_walk_forward: bool,
    errors: list[ValidationIssue],
) -> tuple[list[MaterializationSpec], dict[tuple[str, str, str, int | None], dict[str, Any]]]:
    """Build expected materialization specs from split report source-of-truth."""

    source_file_set = {path.resolve() for path in source_files}
    expected_specs: list[MaterializationSpec] = []
    expected_coverage: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}

    if split_mode == "walk_forward":
        fold_reports_raw = split_report.get("fold_reports")
        if not isinstance(fold_reports_raw, list) or not fold_reports_raw:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_SPLIT_PARTITION_DEFS_MISSING,
                    message="walk_forward split requires non-empty fold_reports.",
                    context={"split_mode": split_mode},
                )
            )
            return [], {}

        for fold_report in fold_reports_raw:
            if not isinstance(fold_report, dict):
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT,
                        message="fold_reports entries must be objects.",
                        context={},
                    )
                )
                continue

            fold_id = fold_report.get("fold_id")
            if not isinstance(fold_id, int) or fold_id < 0:
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT,
                        message="fold_id must be non-negative integer.",
                        context={"fold_id": fold_id},
                    )
                )
                continue

            source_path = _resolve_source_path(
                raw_input_file=fold_report.get("input_file"),
                input_root=input_root,
                source_file_set=source_file_set,
                errors=errors,
            )
            if source_path is None:
                continue

            source_rel = str(source_path.resolve().relative_to(input_root))
            for partition in ("train", "val", "test"):
                rng = _parse_partition_range(
                    raw_range=fold_report.get(f"{partition}_range"),
                    partition=partition,
                    errors=errors,
                )
                if rng is None:
                    continue
                spec = MaterializationSpec(
                    scope="fold",
                    source_file=source_path,
                    source_rel=source_rel,
                    partition=partition,
                    fold_id=fold_id,
                    start_utc=rng["start_utc"],
                    end_inclusive_utc=rng["end_inclusive_utc"],
                    expected_rows=rng["row_count"],
                )
                expected_specs.append(spec)
                expected_coverage[(spec.scope, spec.source_rel, spec.partition, spec.fold_id)] = {
                    "expected_rows": spec.expected_rows,
                    "expected_start_utc": spec.start_utc,
                    "expected_end_inclusive_utc": spec.end_inclusive_utc,
                }

        if aggregate_walk_forward:
            for agg_spec in _build_walk_forward_aggregate_specs(expected_specs):
                expected_coverage[(agg_spec.scope, agg_spec.source_rel, agg_spec.partition, agg_spec.fold_id)] = {
                    "expected_rows": agg_spec.expected_rows,
                    "expected_start_utc": agg_spec.start_utc,
                    "expected_end_inclusive_utc": agg_spec.end_inclusive_utc,
                }

    else:
        file_reports_raw = split_report.get("file_reports")
        if not isinstance(file_reports_raw, list) or not file_reports_raw:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_SPLIT_PARTITION_DEFS_MISSING,
                    message="split report must include non-empty file_reports.",
                    context={"split_mode": split_mode},
                )
            )
            return [], {}

        for file_report in file_reports_raw:
            if not isinstance(file_report, dict):
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT,
                        message="file_reports entries must be objects.",
                        context={},
                    )
                )
                continue

            source_path = _resolve_source_path(
                raw_input_file=file_report.get("input_file"),
                input_root=input_root,
                source_file_set=source_file_set,
                errors=errors,
            )
            if source_path is None:
                continue

            source_rel = str(source_path.resolve().relative_to(input_root))
            for partition in ("train", "val", "test"):
                rng = _parse_partition_range(
                    raw_range=file_report.get(f"{partition}_range"),
                    partition=partition,
                    errors=errors,
                )
                if rng is None:
                    continue
                spec = MaterializationSpec(
                    scope="partition",
                    source_file=source_path,
                    source_rel=source_rel,
                    partition=partition,
                    fold_id=None,
                    start_utc=rng["start_utc"],
                    end_inclusive_utc=rng["end_inclusive_utc"],
                    expected_rows=rng["row_count"],
                )
                expected_specs.append(spec)
                expected_coverage[(spec.scope, spec.source_rel, spec.partition, spec.fold_id)] = {
                    "expected_rows": spec.expected_rows,
                    "expected_start_utc": spec.start_utc,
                    "expected_end_inclusive_utc": spec.end_inclusive_utc,
                }

    expected_sources = {item.source_file.resolve() for item in expected_specs}
    missing_sources = sorted(str(path) for path in source_file_set.difference(expected_sources))
    if missing_sources:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT,
                message="split report does not provide partition defs for all source files.",
                context={"missing_sources": missing_sources},
            )
        )

    if not expected_specs:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT,
                message="No valid partition specs could be extracted from split report.",
                context={"split_mode": split_mode},
            )
        )

    return expected_specs, expected_coverage


def _resolve_source_path(
    *,
    raw_input_file: Any,
    input_root: Path,
    source_file_set: set[Path],
    errors: list[ValidationIssue],
) -> Path | None:
    """Resolve and validate source parquet path from split report."""

    if not isinstance(raw_input_file, str) or not raw_input_file.strip():
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT,
                message="input_file must be a non-empty string in split report.",
                context={"input_file": raw_input_file},
            )
        )
        return None

    source_path = Path(raw_input_file).resolve()
    try:
        source_path.relative_to(input_root)
    except ValueError:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_LINEAGE_MISMATCH,
                message="split report input_file escapes input_root.",
                context={"input_file": str(source_path), "input_root": str(input_root)},
            )
        )
        return None

    if source_path not in source_file_set:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_SOURCE_FILE_MISSING,
                message="split report references source parquet that is not present under input_root.",
                context={"input_file": str(source_path)},
            )
        )
        return None

    return source_path


def _parse_partition_range(
    *,
    raw_range: Any,
    partition: str,
    errors: list[ValidationIssue],
) -> dict[str, Any] | None:
    """Parse a deterministic partition range payload from split report."""

    if not isinstance(raw_range, dict):
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_SPLIT_PARTITION_DEFS_MISSING,
                message="Partition range is missing in split report.",
                context={"partition": partition},
            )
        )
        return None

    start_utc = raw_range.get("start_utc")
    end_inclusive_utc = raw_range.get("end_inclusive_utc")
    row_count = raw_range.get("row_count")

    if not isinstance(start_utc, str) or not start_utc.strip() or not isinstance(end_inclusive_utc, str) or not end_inclusive_utc.strip():
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT,
                message="Partition range must include start_utc and end_inclusive_utc.",
                context={"partition": partition, "range": raw_range},
            )
        )
        return None

    if not isinstance(row_count, int):
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_SPLIT_PARTITION_DEFS_INSUFFICIENT,
                message="Partition range row_count must be integer.",
                context={"partition": partition, "row_count": row_count},
            )
        )
        return None

    if row_count <= 0:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_PARTITION_EMPTY,
                message="Partition range row_count must be > 0.",
                context={"partition": partition, "row_count": row_count},
            )
        )
        return None

    return {
        "start_utc": start_utc.strip(),
        "end_inclusive_utc": end_inclusive_utc.strip(),
        "row_count": int(row_count),
    }


def _resolve_output_semantics(*, split_mode: str | None, aggregate_walk_forward: bool) -> dict[str, Any]:
    """Resolve output semantics contract payload."""

    if split_mode == "walk_forward":
        if aggregate_walk_forward:
            return {
                "mode": "walk_forward_fold_plus_aggregate",
                "fold_outputs_generated": True,
                "top_level_partitions_generated": True,
                "aggregate_walk_forward": True,
            }
        return {
            "mode": "walk_forward_fold_only",
            "fold_outputs_generated": True,
            "top_level_partitions_generated": False,
            "aggregate_walk_forward": False,
        }

    return {
        "mode": "standard_partitions",
        "fold_outputs_generated": False,
        "top_level_partitions_generated": True,
        "aggregate_walk_forward": False,
    }


def _evaluate_overwrite_policy(*, output_root: Path, staging_root: Path, overwrite: bool, errors: list[ValidationIssue]) -> dict[str, Any]:
    """Evaluate overwrite/collision contract and emit deterministic errors."""

    output_exists = output_root.exists()
    output_non_empty = output_exists and _is_non_empty_dir(output_root)
    output_empty = output_exists and output_root.is_dir() and not output_non_empty

    if output_exists and (not output_root.is_dir()):
        if not overwrite:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_OUTPUT_ROOT_EXISTS,
                    message="Output root exists and is not a directory; overwrite=false.",
                    context={"output_root": str(output_root)},
                )
            )
        else:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_OUTPUT_ROOT_EXISTS,
                    message="Output root exists as file and cannot be used for directory outputs.",
                    context={"output_root": str(output_root)},
                )
            )

    if output_non_empty and not overwrite:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_OUTPUT_ROOT_EXISTS,
                message="Output root exists and is non-empty while overwrite=false.",
                context={"output_root": str(output_root)},
            )
        )

    staging_exists = staging_root.exists()
    staging_non_empty = staging_exists and _is_non_empty_dir(staging_root)

    if staging_non_empty and not overwrite:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_STAGING_ROOT_COLLISION,
                message="Staging root exists and is non-empty while overwrite=false.",
                context={"staging_root": str(staging_root)},
            )
        )

    return {
        "overwrite_requested": overwrite,
        "output_root_exists": output_exists,
        "output_root_empty": output_empty,
        "output_root_non_empty": output_non_empty,
        "staging_root_exists": staging_exists,
        "staging_root_non_empty": staging_non_empty,
    }


def _prepare_staging_root(*, staging_root: Path, overwrite: bool, errors: list[ValidationIssue]) -> None:
    """Prepare staging root deterministically."""

    if staging_root.exists():
        if _is_non_empty_dir(staging_root):
            if not overwrite:
                errors.append(
                    ValidationIssue(
                        code=DATASET_BUILD_STAGING_ROOT_COLLISION,
                        message="Staging root exists and non-empty while overwrite=false.",
                        context={"staging_root": str(staging_root)},
                    )
                )
                return
            shutil.rmtree(staging_root)
        else:
            if staging_root.is_file():
                if not overwrite:
                    errors.append(
                        ValidationIssue(
                            code=DATASET_BUILD_STAGING_ROOT_COLLISION,
                            message="Staging root exists as file while overwrite=false.",
                            context={"staging_root": str(staging_root)},
                        )
                    )
                    return
                staging_root.unlink()
            else:
                shutil.rmtree(staging_root)

    staging_root.mkdir(parents=True, exist_ok=False)


def _promote_staging_to_output(*, staging_root: Path, output_root: Path, overwrite: bool) -> None:
    """Promote staging output to final output root."""

    try:
        if output_root.exists():
            if output_root.is_dir():
                if _is_non_empty_dir(output_root):
                    if not overwrite:
                        raise RuntimeError("output_root exists and non-empty while overwrite=false")
                    shutil.rmtree(output_root)
                else:
                    output_root.rmdir()
            else:
                if not overwrite:
                    raise RuntimeError("output_root exists as file while overwrite=false")
                output_root.unlink()

        os.replace(staging_root, output_root)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to promote staging root to output root: {exc}") from exc


def _cleanup_staging_root(staging_root: Path) -> None:
    """Best-effort cleanup for staging root."""

    if not staging_root.exists():
        return
    try:
        if staging_root.is_dir():
            shutil.rmtree(staging_root)
        else:
            staging_root.unlink()
    except OSError:
        LOGGER.info("Failed to cleanup staging root (best-effort) | path=%s", staging_root)


def _group_specs_by_source(specs: Sequence[MaterializationSpec]) -> dict[Path, list[MaterializationSpec]]:
    """Group materialization specs by source parquet path."""

    grouped: dict[Path, list[MaterializationSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.source_file.resolve(), []).append(spec)
    return grouped


def _slice_partition(
    *,
    frame: pd.DataFrame,
    timestamp_column: str,
    start_utc: str,
    end_inclusive_utc: str,
) -> pd.DataFrame:
    """Slice partition deterministically from timestamp bounds."""

    start_ts = pd.to_datetime(start_utc, utc=True, errors="raise")
    end_ts = pd.to_datetime(end_inclusive_utc, utc=True, errors="raise")
    mask = (frame[timestamp_column] >= start_ts) & (frame[timestamp_column] <= end_ts)
    return frame.loc[mask].copy()


def _build_partition_output_path(staging_root: Path, source_rel: str, partition: str) -> Path:
    """Build output path for standard partition artifacts."""

    stem = _source_rel_to_stem(source_rel)
    out_path = staging_root / "parquet" / "partitions" / partition / f"{stem}.parquet"
    return out_path


def _build_fold_output_path(staging_root: Path, source_rel: str, partition: str, fold_id: int) -> Path:
    """Build output path for fold-aware artifacts."""

    stem = _source_rel_to_stem(source_rel)
    out_path = staging_root / "parquet" / "folds" / f"fold_{fold_id:03d}" / partition / f"{stem}.parquet"
    return out_path


def _build_aggregate_output_path(staging_root: Path, source_rel: str, partition: str) -> Path:
    """Build output path for aggregate walk-forward artifacts."""

    stem = _source_rel_to_stem(source_rel)
    out_path = staging_root / "parquet" / "partitions" / partition / f"{stem}.parquet"
    return out_path


def _to_promoted_output_path(*, staged_output_path: str, staging_root: Path, output_root: Path) -> str:
    """Map staged artifact path into final promoted output root."""

    staging_root_resolved = staging_root.resolve()
    staged_output_resolved = Path(staged_output_path).resolve()

    try:
        relative_path = staged_output_resolved.relative_to(staging_root_resolved)
    except ValueError as exc:
        raise RuntimeError(
            "Artifact metadata output_path is outside staging root; cannot promote metadata safely."
        ) from exc

    promoted_output = output_root.resolve() / relative_path
    return str(promoted_output)


def _build_promoted_artifact_metadata(
    *,
    artifacts: Sequence[MaterializedArtifact],
    staging_root: Path,
    output_root: Path,
) -> list[MaterializedArtifact]:
    """Create immutable artifact metadata list with promoted output paths."""

    promoted: list[MaterializedArtifact] = []
    for item in artifacts:
        promoted_output = _to_promoted_output_path(
            staged_output_path=item.output_path,
            staging_root=staging_root,
            output_root=output_root,
        )
        promoted.append(replace(item, output_path=promoted_output))
    return promoted


def _source_rel_to_stem(source_rel: str) -> str:
    """Convert source relative path into deterministic output stem."""

    rel = Path(source_rel)
    return "__".join(rel.with_suffix("").parts)


def _build_walk_forward_aggregate_specs(specs: Sequence[MaterializationSpec]) -> list[MaterializationSpec]:
    """Build aggregate expected specs from fold specs."""

    grouped: dict[tuple[str, str], list[MaterializationSpec]] = {}
    for spec in specs:
        if spec.scope != "fold":
            continue
        grouped.setdefault((spec.source_rel, spec.partition), []).append(spec)

    out: list[MaterializationSpec] = []
    for (source_rel, partition), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item.fold_id if item.fold_id is not None else -1)
        total_rows = int(sum(item.expected_rows for item in ordered))
        start_utc = min(item.start_utc for item in ordered)
        end_inclusive_utc = max(item.end_inclusive_utc for item in ordered)
        out.append(
            MaterializationSpec(
                scope="aggregate",
                source_file=ordered[0].source_file,
                source_rel=source_rel,
                partition=partition,
                fold_id=None,
                start_utc=start_utc,
                end_inclusive_utc=end_inclusive_utc,
                expected_rows=total_rows,
                allow_duplicate_timestamps=True,
            )
        )
    return out


def _maybe_extend_with_aggregate_coverage(
    expected_coverage: dict[tuple[str, str, str, int | None], dict[str, Any]],
    specs: Sequence[MaterializationSpec],
) -> dict[tuple[str, str, str, int | None], dict[str, Any]]:
    """Extend expected coverage map with aggregate walk-forward coverage specs."""

    out = dict(expected_coverage)
    aggregate_specs = _build_walk_forward_aggregate_specs(specs)
    for spec in aggregate_specs:
        out[(spec.scope, spec.source_rel, spec.partition, spec.fold_id)] = {
            "expected_rows": spec.expected_rows,
            "expected_start_utc": spec.start_utc,
            "expected_end_inclusive_utc": spec.end_inclusive_utc,
        }
    return out


def _evaluate_output_completeness(
    expected_coverage: Mapping[tuple[str, str, str, int | None], Mapping[str, Any]],
    artifacts: Sequence[MaterializedArtifact],
) -> tuple[bool, list[ValidationIssue]]:
    """Evaluate expected-vs-actual materialization coverage."""

    errors: list[ValidationIssue] = []
    actual = {item.key(): item for item in artifacts}
    expected_keys = set(expected_coverage.keys())
    actual_keys = set(actual.keys())

    missing = sorted(expected_keys.difference(actual_keys))
    extra = sorted(actual_keys.difference(expected_keys))

    if missing:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                message="Expected materialization entries are missing.",
                context={"missing": [list(item) for item in missing]},
            )
        )
    if extra:
        errors.append(
            ValidationIssue(
                code=DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                message="Unexpected materialization entries were produced.",
                context={"extra": [list(item) for item in extra]},
            )
        )

    for key in sorted(expected_keys.intersection(actual_keys)):
        expected_entry = expected_coverage[key]
        actual_entry = actual[key]
        expected_rows = expected_entry.get("expected_rows")
        if isinstance(expected_rows, int) and expected_rows != actual_entry.row_count:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                    message="Expected/actual row_count mismatch.",
                    context={"key": list(key), "expected_rows": expected_rows, "actual_rows": actual_entry.row_count},
                )
            )

        expected_start = expected_entry.get("expected_start_utc")
        if isinstance(expected_start, str) and actual_entry.timestamp_min_utc is not None and expected_start != actual_entry.timestamp_min_utc:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                    message="Expected/actual timestamp_min mismatch.",
                    context={"key": list(key), "expected_start_utc": expected_start, "actual_start_utc": actual_entry.timestamp_min_utc},
                )
            )

        expected_end = expected_entry.get("expected_end_inclusive_utc")
        if isinstance(expected_end, str) and actual_entry.timestamp_max_utc is not None and expected_end != actual_entry.timestamp_max_utc:
            errors.append(
                ValidationIssue(
                    code=DATASET_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                    message="Expected/actual timestamp_max mismatch.",
                    context={"key": list(key), "expected_end_inclusive_utc": expected_end, "actual_end_inclusive_utc": actual_entry.timestamp_max_utc},
                )
            )

    return len(errors) == 0, errors


def _build_partition_summaries(artifacts: Sequence[MaterializedArtifact]) -> dict[str, dict[str, Any]]:
    """Build partition-level aggregate summaries."""

    out: dict[str, dict[str, Any]] = {}
    for partition in ("train", "val", "test"):
        items = [item for item in artifacts if item.partition == partition]
        if not items:
            continue
        timestamp_min = min(item.timestamp_min_utc for item in items if item.timestamp_min_utc is not None)
        timestamp_max = max(item.timestamp_max_utc for item in items if item.timestamp_max_utc is not None)
        out[partition] = {
            "artifacts": len(items),
            "rows": int(sum(item.row_count for item in items)),
            "timestamp_min_utc": timestamp_min,
            "timestamp_max_utc": timestamp_max,
            "duplicate_timestamp_count_total": int(sum(item.duplicate_timestamp_count for item in items)),
        }
    return out


def _build_fold_summaries(artifacts: Sequence[MaterializedArtifact]) -> dict[str, Any]:
    """Build fold summaries for walk-forward outputs."""

    folds: dict[int, dict[str, Any]] = {}
    for item in artifacts:
        if item.scope != "fold" or item.fold_id is None:
            continue
        fold = folds.setdefault(item.fold_id, {"rows": 0, "artifacts": 0})
        fold["rows"] += int(item.row_count)
        fold["artifacts"] += 1

    return {
        "total_folds": len(folds),
        "folds": {str(fold_id): payload for fold_id, payload in sorted(folds.items())},
    }


def _build_walk_forward_fold_metadata(artifacts: Sequence[MaterializedArtifact]) -> list[dict[str, Any]]:
    """Build per-fold metadata payload for manifest."""

    out: list[dict[str, Any]] = []
    for item in artifacts:
        if item.scope != "fold":
            continue
        out.append(
            {
                "fold_id": item.fold_id,
                "source_rel": item.source_rel,
                "partition": item.partition,
                "output_path": item.output_path,
                "row_count": item.row_count,
                "timestamp_min_utc": item.timestamp_min_utc,
                "timestamp_max_utc": item.timestamp_max_utc,
            }
        )
    return out


def _compute_dataset_build_id(
    *,
    run_id: str,
    split_mode: str | None,
    output_semantics_mode: str,
    aggregate_walk_forward: bool,
    timestamp_column: str,
    column_selection_hash: str,
    build_mode: str,
    feature_manifest_hash: str,
    train_input_report_hash: str,
    split_report_hash: str,
    source_file_inventory_hash: str,
) -> str:
    """Compute deterministic dataset_build_id.

    Hash inputs are fixed and ordered:
    run_id, builder_version, build_mode, split_mode, output_semantics.mode,
    aggregate_walk_forward, timestamp_column, column_selection_hash,
    feature_manifest_hash, train_input_report_hash, split_report_hash,
    source_file_inventory_hash.
    """

    payload = {
        "run_id": run_id,
        "builder_version": DATASET_BUILDER_VERSION,
        "build_mode": build_mode,
        "split_mode": split_mode,
        "output_semantics_mode": output_semantics_mode,
        "aggregate_walk_forward": bool(aggregate_walk_forward),
        "timestamp_column": timestamp_column,
        "column_selection_hash": column_selection_hash,
        "feature_manifest_hash": feature_manifest_hash,
        "train_input_report_hash": train_input_report_hash,
        "split_report_hash": split_report_hash,
        "source_file_inventory_hash": source_file_inventory_hash,
    }
    return _hash_canonical_json(payload)


def _hash_canonical_json(payload: Mapping[str, Any]) -> str:
    """Hash JSON payload with canonical deterministic encoding."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_sequence(values: Sequence[Any]) -> str:
    """Hash sequence deterministically."""

    payload = [str(value) for value in values]
    return _hash_canonical_json({"items": payload})


def _hash_mapping(values: Mapping[str, Any]) -> str:
    """Hash mapping deterministically."""

    normalized = {str(key): str(value) for key, value in sorted(values.items())}
    return _hash_canonical_json(normalized)


def _sha256_file(path: Path) -> str:
    """Compute sha256 hash of file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_string_list(value: Any) -> list[str] | None:
    """Parse list[str] helper."""

    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        out.append(item)
    return out


def _stable_unique(values: Iterable[str]) -> list[str]:
    """Return stable unique sequence."""

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _is_non_empty_dir(path: Path) -> bool:
    """Return True if path is a non-empty directory."""

    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())


def _write_report_best_effort(payload: dict[str, Any], report_path: Path, warnings: list[ValidationIssue]) -> None:
    """Write report best-effort for failure paths."""

    try:
        atomic_write_json(payload, report_path)
    except RuntimeError as exc:
        warnings.append(
            ValidationIssue(
                code=DATASET_BUILD_REPORT_WRITE_FAILED,
                message="dataset_build_report.json write failed (best-effort).",
                context={"report_path": str(report_path), "error": str(exc)},
            )
        )
        LOGGER.info("Dataset build report write failed (best-effort) | path=%s error=%s", report_path, exc)
