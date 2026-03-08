"""State build subsystem for Milestone 4.4.

This module materializes deterministic RL-ready state artifacts from Milestone 4.3
validated dataset artifacts.
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

import numpy as np
import pandas as pd

from core.io_atomic import atomic_write_json, atomic_write_parquet
from core.logging import get_logger
from core.paths import ensure_within_root

LOGGER = get_logger(__name__)

STATE_BUILDER_VERSION = "state_builder.v1"
STATE_MANIFEST_VERSION = "states.manifest.v1"
DEFAULT_BUILD_MODE = "materialize_only"

STATE_BUILD_PRECONDITION_FAILED = "STATE_BUILD_PRECONDITION_FAILED"
STATE_BUILD_DATASET_MANIFEST_MISSING = "STATE_BUILD_DATASET_MANIFEST_MISSING"
STATE_BUILD_DATASET_MANIFEST_INVALID = "STATE_BUILD_DATASET_MANIFEST_INVALID"
STATE_BUILD_DATASET_REPORT_MISSING = "STATE_BUILD_DATASET_REPORT_MISSING"
STATE_BUILD_DATASET_REPORT_INVALID = "STATE_BUILD_DATASET_REPORT_INVALID"
STATE_BUILD_DATASET_NOT_PASSED = "STATE_BUILD_DATASET_NOT_PASSED"
STATE_BUILD_OUTPUT_COMPLETENESS_FAILED_UPSTREAM = "STATE_BUILD_OUTPUT_COMPLETENESS_FAILED_UPSTREAM"
STATE_BUILD_RUN_ID_MISMATCH = "STATE_BUILD_RUN_ID_MISMATCH"
STATE_BUILD_LINEAGE_MISMATCH = "STATE_BUILD_LINEAGE_MISMATCH"
STATE_BUILD_OUTPUT_SEMANTICS_UNSUPPORTED = "STATE_BUILD_OUTPUT_SEMANTICS_UNSUPPORTED"
STATE_BUILD_BUILD_MODE_UNSUPPORTED = "STATE_BUILD_BUILD_MODE_UNSUPPORTED"
STATE_BUILD_OUTPUT_ROOT_EXISTS = "STATE_BUILD_OUTPUT_ROOT_EXISTS"
STATE_BUILD_STAGING_ROOT_COLLISION = "STATE_BUILD_STAGING_ROOT_COLLISION"
STATE_BUILD_SOURCE_FILE_MISSING = "STATE_BUILD_SOURCE_FILE_MISSING"
STATE_BUILD_COLUMN_SELECTION_INVALID = "STATE_BUILD_COLUMN_SELECTION_INVALID"
STATE_BUILD_ORDERING_CONTRACT_VIOLATION = "STATE_BUILD_ORDERING_CONTRACT_VIOLATION"
STATE_BUILD_TIMESTAMP_DUPLICATES = "STATE_BUILD_TIMESTAMP_DUPLICATES"
STATE_BUILD_SCALER_TYPE_UNSUPPORTED = "STATE_BUILD_SCALER_TYPE_UNSUPPORTED"
STATE_BUILD_SCALER_FIT_FAILED = "STATE_BUILD_SCALER_FIT_FAILED"
STATE_BUILD_RUNTIME_PRICE_CONTRACT_REQUIRED = "STATE_BUILD_RUNTIME_PRICE_CONTRACT_REQUIRED"
STATE_BUILD_RUNTIME_PRICE_CONTRACT_INVALID = "STATE_BUILD_RUNTIME_PRICE_CONTRACT_INVALID"
STATE_BUILD_RUNTIME_PRICE_COLUMN_MISSING = "STATE_BUILD_RUNTIME_PRICE_COLUMN_MISSING"
STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH = "STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH"
STATE_BUILD_SEQUENCE_MODE_DEFERRED = "STATE_BUILD_SEQUENCE_MODE_DEFERRED"
STATE_BUILD_AGGREGATE_WALK_FORWARD_DEFERRED = "STATE_BUILD_AGGREGATE_WALK_FORWARD_DEFERRED"
STATE_BUILD_WRITE_FAILED = "STATE_BUILD_WRITE_FAILED"
STATE_BUILD_RUNTIME_ERROR = "STATE_BUILD_RUNTIME_ERROR"

STATE_BUILD_OPTIONAL_LINEAGE_HASH_MISSING = "STATE_BUILD_OPTIONAL_LINEAGE_HASH_MISSING"
STATE_BUILD_LINEAGE_CHECK_PARTIAL = "STATE_BUILD_LINEAGE_CHECK_PARTIAL"
STATE_BUILD_REPORT_WRITE_FAILED = "STATE_BUILD_REPORT_WRITE_FAILED"
STATE_BUILD_SUMMARY_UPDATE_FAILED = "STATE_BUILD_SUMMARY_UPDATE_FAILED"

SUPPORTED_SCALER_TYPES = {"none", "standard"}
WARMUP_POLICY_DROP_HEAD = "drop_head_until_all_required_obs_numeric"
WARMUP_POST_VALID_NAN_POLICY = "fail_closed"
CONDITIONAL_COLUMN_POLICY_EXCLUDE_AND_REPLACE = "exclude_from_core_and_replace_with_geometry"
GEOMETRY_FEATURE_VERSION = "geometry.features.v1"
ST_ACTIVE_LINE_COLUMN = "ST_active_line"
ST_DISTANCE_TO_ACTIVE_LINE_COLUMN = "ST_distance_to_active_line"
ST_CONDITIONAL_RAW_COLUMNS = ("ST_up", "ST_dn")
TREND_AGE_FUTURE_COLUMNS = ("bars_since_AT_flip", "bars_since_ST_flip")


@dataclass
class ValidationIssue:
    """Machine-readable issue payload."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateBuildOptions:
    """Runtime options for state materialization."""

    run_id: str
    input_root: Path
    output_root: Path | None = None
    dataset_manifest_path: Path | None = None
    dataset_build_report_path: Path | None = None
    overwrite: bool = False
    enable_scaling: bool = False
    scaler_type: str = "none"
    timestamp_column_override: str | None = None
    build_mode: str = DEFAULT_BUILD_MODE
    strict_column_selection: bool = True
    state_columns: tuple[str, ...] = ()
    sequence_mode: bool = False
    lookback: int | None = None
    aggregate_walk_forward: bool = False
    execution_price_column: str | None = None
    mark_to_market_column: str | None = None

    def to_invocation_args(self) -> dict[str, Any]:
        """Serialize invocation arguments for report payload."""

        return {
            "run_id": self.run_id,
            "input_root": str(self.input_root),
            "output_root": str(self.output_root) if self.output_root is not None else None,
            "dataset_manifest_path": str(self.dataset_manifest_path) if self.dataset_manifest_path is not None else None,
            "dataset_build_report_path": str(self.dataset_build_report_path)
            if self.dataset_build_report_path is not None
            else None,
            "overwrite": bool(self.overwrite),
            "enable_scaling": bool(self.enable_scaling),
            "scaler_type": str(self.scaler_type),
            "timestamp_column_override": self.timestamp_column_override,
            "build_mode": self.build_mode,
            "strict_column_selection": bool(self.strict_column_selection),
            "state_columns": list(self.state_columns),
            "sequence_mode": bool(self.sequence_mode),
            "lookback": self.lookback,
            "aggregate_walk_forward": bool(self.aggregate_walk_forward),
            "execution_price_column": self.execution_price_column,
            "mark_to_market_column": self.mark_to_market_column,
        }


@dataclass(frozen=True)
class SourceSpec:
    """Expected source materialization spec from dataset manifest."""

    scope: str
    source_file: Path
    source_rel: str
    partition: str
    fold_id: int | None
    expected_rows: int
    expected_timestamp_min_utc: str | None
    expected_timestamp_max_utc: str | None

    def key(self) -> tuple[str, str, str, int | None]:
        """Return deterministic key."""

        return (self.scope, self.source_rel, self.partition, self.fold_id)


@dataclass
class StateArtifact:
    """Produced state artifact metadata."""

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
    warmup_contract: dict[str, Any]

    def key(self) -> tuple[str, str, str, int | None]:
        """Return deterministic key."""

        return (self.scope, self.source_rel, self.partition, self.fold_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize into JSON payload."""

        return asdict(self)


@dataclass
class StateBuildResult:
    """Result payload for state build call."""

    report_payload: dict[str, Any]
    manifest_payload: dict[str, Any] | None
    scaler_stats_payload: dict[str, Any] | None
    report_path: Path
    manifest_path: Path
    scaler_stats_path: Path


@dataclass(frozen=True)
class ResolvedObservationContract:
    """Resolved final observation schema for one state build."""

    input_state_feature_columns: tuple[str, ...]
    state_feature_columns: tuple[str, ...]
    selected_input_columns: tuple[str, ...]
    selected_dtypes: dict[str, str]
    event_columns: tuple[str, ...]
    regime_columns: tuple[str, ...]
    geometry_columns: tuple[str, ...]
    strict_post_valid_numeric_columns: tuple[str, ...]
    conditional_raw_columns: tuple[str, ...]
    conditional_column_policy: str
    conditional_column_replacements: dict[str, list[str]]
    geometry_feature_version: str
    geometry_feature_formulas: dict[str, str]
    future_feature_hooks: dict[str, Any]
    source_price_column: str


def build_states(options: StateBuildOptions) -> StateBuildResult:
    """Build deterministic RL states from dataset artifacts."""

    if not options.run_id.strip():
        raise ValueError("run_id must be non-empty")

    run_id = options.run_id.strip()
    input_root = options.input_root.resolve()
    output_root = options.output_root.resolve() if options.output_root is not None else _default_output_root(input_root)

    dataset_manifest_path = (
        options.dataset_manifest_path.resolve()
        if options.dataset_manifest_path is not None
        else (input_root / "reports" / "dataset_manifest.json").resolve()
    )
    dataset_build_report_path = (
        options.dataset_build_report_path.resolve()
        if options.dataset_build_report_path is not None
        else (input_root / "reports" / "dataset_build_report.json").resolve()
    )

    report_path = output_root / "reports" / "state_build_report.json"
    manifest_path = output_root / "reports" / "state_manifest.json"
    scaler_stats_path = output_root / "reports" / "scaler_stats.json"
    staging_root = output_root.parent / f"{output_root.name}.__staging__"

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    report_payload = _base_report_payload(
        run_id=run_id,
        input_root=input_root,
        output_root=output_root,
        report_path=report_path,
        manifest_path=manifest_path,
        scaler_stats_path=scaler_stats_path,
        staging_root=staging_root,
        invocation_args=options.to_invocation_args(),
        source_paths={
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_build_report_path": str(dataset_build_report_path),
        },
    )
    report_payload["build_mode"] = options.build_mode
    report_payload["staging_root"] = None

    if options.build_mode != DEFAULT_BUILD_MODE:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_BUILD_MODE_UNSUPPORTED,
                message="Unsupported build_mode for state_builder.v1.",
                context={"build_mode": options.build_mode, "supported": [DEFAULT_BUILD_MODE]},
            )
        )

    if options.sequence_mode:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_SEQUENCE_MODE_DEFERRED,
                message="sequence mode is deferred in state_builder.v1.",
                context={"sequence_mode": True},
            )
        )

    if options.aggregate_walk_forward:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_AGGREGATE_WALK_FORWARD_DEFERRED,
                message="aggregate walk-forward outputs are deferred in state_builder.v1.",
                context={"aggregate_walk_forward": True},
            )
        )

    scaler_type_input = str(options.scaler_type).strip().lower()
    if scaler_type_input not in SUPPORTED_SCALER_TYPES:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_SCALER_TYPE_UNSUPPORTED,
                message="Unsupported scaler_type requested.",
                context={"scaler_type": options.scaler_type, "supported": sorted(SUPPORTED_SCALER_TYPES)},
            )
        )
    elif bool(options.enable_scaling) and scaler_type_input != "standard":
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_SCALER_TYPE_UNSUPPORTED,
                message="enable_scaling=true requires scaler_type=standard in state_builder.v1.",
                context={"enable_scaling": True, "scaler_type": scaler_type_input, "required_scaler_type": "standard"},
            )
        )

    dataset_manifest = _load_json_object(
        dataset_manifest_path,
        missing_code=STATE_BUILD_DATASET_MANIFEST_MISSING,
        invalid_code=STATE_BUILD_DATASET_MANIFEST_INVALID,
        missing_message="dataset_manifest.json not found.",
        invalid_message="dataset_manifest.json is invalid.",
        errors=errors,
    )
    dataset_build_report = _load_json_object(
        dataset_build_report_path,
        missing_code=STATE_BUILD_DATASET_REPORT_MISSING,
        invalid_code=STATE_BUILD_DATASET_REPORT_INVALID,
        missing_message="dataset_build_report.json not found.",
        invalid_message="dataset_build_report.json is invalid.",
        errors=errors,
    )

    if dataset_manifest is not None:
        _require_run_id(field_name="dataset_manifest.run_id", payload=dataset_manifest, run_id=run_id, errors=errors)

    if dataset_build_report is not None:
        _require_run_id(field_name="dataset_build_report.run_id", payload=dataset_build_report, run_id=run_id, errors=errors)

    if dataset_build_report is not None:
        upstream_overall = dataset_build_report.get("dataset_build_overall")
        if not isinstance(upstream_overall, bool):
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_REPORT_INVALID,
                    message="dataset_build_overall must be boolean.",
                    context={"value": upstream_overall},
                )
            )
        elif not upstream_overall:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_NOT_PASSED,
                    message="dataset_build_overall must be true before state build.",
                    context={"dataset_build_overall": upstream_overall},
                )
            )

        upstream_completeness = dataset_build_report.get("output_completeness_ok")
        if not isinstance(upstream_completeness, bool):
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_REPORT_INVALID,
                    message="output_completeness_ok must be boolean in dataset_build_report.",
                    context={"value": upstream_completeness},
                )
            )
        elif not upstream_completeness:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_OUTPUT_COMPLETENESS_FAILED_UPSTREAM,
                    message="Upstream dataset output_completeness_ok must be true.",
                    context={"output_completeness_ok": upstream_completeness},
                )
            )

    split_mode: str | None = None
    upstream_output_semantics: dict[str, Any] = {}
    if dataset_manifest is not None:
        split_mode_raw = dataset_manifest.get("split_mode")
        if isinstance(split_mode_raw, str) and split_mode_raw.strip():
            split_mode = split_mode_raw.strip()
    if split_mode is None and dataset_build_report is not None:
        split_mode_raw = dataset_build_report.get("split_mode")
        if isinstance(split_mode_raw, str) and split_mode_raw.strip():
            split_mode = split_mode_raw.strip()

    if dataset_manifest is not None:
        output_semantics_raw = dataset_manifest.get("output_semantics")
        if isinstance(output_semantics_raw, dict):
            upstream_output_semantics = dict(output_semantics_raw)
    if not upstream_output_semantics and dataset_build_report is not None:
        output_semantics_raw = dataset_build_report.get("output_semantics")
        if isinstance(output_semantics_raw, dict):
            upstream_output_semantics = dict(output_semantics_raw)

    _validate_lineage(
        dataset_manifest=dataset_manifest,
        dataset_build_report=dataset_build_report,
        warnings=warnings,
        errors=errors,
    )

    selected_columns: list[str] = []
    selected_dtype_map: dict[str, str] = {}
    timestamp_column = "timestamp"
    column_selection_contract: dict[str, Any] = {}
    state_feature_columns: list[str] = []
    observation_contract_spec: ResolvedObservationContract | None = None
    runtime_price_contract: dict[str, Any] = {}
    runtime_price_columns: list[str] = []
    artifact_columns: list[str] = []

    if dataset_manifest is not None:
        selected_columns, selected_dtype_map, timestamp_column, column_selection_contract = _resolve_column_selection(
            dataset_manifest=dataset_manifest,
            options=options,
            warnings=warnings,
            errors=errors,
        )
        input_state_feature_columns = [col for col in selected_columns if col != timestamp_column]
        observation_contract_spec = _resolve_observation_contract(
            timestamp_column=timestamp_column,
            input_state_feature_columns=input_state_feature_columns,
            input_selected_dtypes=selected_dtype_map,
            execution_price_column=options.execution_price_column,
            errors=errors,
        )
        state_feature_columns = (
            list(observation_contract_spec.state_feature_columns) if observation_contract_spec is not None else []
        )
        runtime_price_columns, artifact_columns, runtime_price_contract = _resolve_runtime_price_contract(
            options=options,
            timestamp_column=timestamp_column,
            state_feature_columns=state_feature_columns,
            errors=errors,
        )

    expected_specs: list[SourceSpec] = []
    expected_coverage: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}

    if dataset_manifest is not None:
        expected_specs, expected_coverage = _build_expected_specs(
            dataset_manifest=dataset_manifest,
            split_mode=split_mode,
            warnings=warnings,
            errors=errors,
        )

    output_semantics = _resolve_state_output_semantics(split_mode=split_mode)

    overwrite_policy = _evaluate_overwrite_policy(
        output_root=output_root,
        staging_root=staging_root,
        overwrite=bool(options.overwrite),
        errors=errors,
    )

    scaling_contract = _resolve_scaling_contract(
        split_mode=split_mode,
        enable_scaling=bool(options.enable_scaling),
        scaler_type=scaler_type_input,
    )

    report_payload["split_mode"] = split_mode
    report_payload["upstream_output_semantics"] = upstream_output_semantics
    report_payload["output_semantics"] = output_semantics
    report_payload["overwrite_policy"] = overwrite_policy
    report_payload["column_selection_contract"] = column_selection_contract
    report_payload["scaling_contract"] = scaling_contract
    report_payload["runtime_price_contract"] = runtime_price_contract

    manifest_payload: dict[str, Any] | None = None
    scaler_stats_payload: dict[str, Any] | None = None

    if errors:
        report_payload["errors"] = [asdict(item) for item in errors]
        report_payload["warnings"] = [asdict(item) for item in warnings]
        report_payload["state_build_overall"] = False
        report_payload["error_code"] = STATE_BUILD_PRECONDITION_FAILED
        report_payload = _sanitize_persisted_metadata_paths(
            payload=report_payload,
            staging_root=staging_root,
            output_root=output_root,
        )
        _write_report_best_effort(report_payload, report_path, warnings)
        return StateBuildResult(
            report_payload=report_payload,
            manifest_payload=None,
            scaler_stats_payload=None,
            report_path=report_path,
            manifest_path=manifest_path,
            scaler_stats_path=scaler_stats_path,
        )

    dataset_manifest_hash = _sha256_file(dataset_manifest_path)
    dataset_build_report_hash = _sha256_file(dataset_build_report_path)
    source_inventory_hash = _hash_sequence(sorted({str(item.source_file.resolve()) for item in expected_specs}))
    state_column_selection_hash = _hash_sequence(selected_columns)
    if observation_contract_spec is None:
        raise ValueError("observation_contract must be resolved before computing state build metadata")
    observation_contract_hash = _hash_canonical_json(
        _normalize_mapping_for_hash(
            {
                "state_feature_columns": list(observation_contract_spec.state_feature_columns),
                "event_columns": list(observation_contract_spec.event_columns),
                "regime_columns": list(observation_contract_spec.regime_columns),
                "geometry_columns": list(observation_contract_spec.geometry_columns),
                "strict_post_valid_numeric_columns": list(observation_contract_spec.strict_post_valid_numeric_columns),
                "conditional_raw_columns": list(observation_contract_spec.conditional_raw_columns),
                "conditional_column_policy": observation_contract_spec.conditional_column_policy,
                "conditional_column_replacements": observation_contract_spec.conditional_column_replacements,
                "geometry_feature_version": observation_contract_spec.geometry_feature_version,
                "geometry_feature_formulas": observation_contract_spec.geometry_feature_formulas,
            }
        )
    )
    scaling_contract_hash = _hash_canonical_json(_normalize_mapping_for_hash(scaling_contract))
    runtime_price_contract_hash = _hash_canonical_json(
        _normalize_mapping_for_hash(
            {
                "timestamp_column": runtime_price_contract.get("timestamp_column"),
                "execution_price_column": runtime_price_contract.get("execution_price_column"),
                "mark_to_market_column": runtime_price_contract.get("mark_to_market_column"),
                "required_runtime_columns": runtime_price_contract.get("required_runtime_columns", []),
                "artifact_columns": runtime_price_contract.get("artifact_columns", []),
            }
        )
    )

    report_payload["source_hashes"] = {
        "dataset_manifest_hash": dataset_manifest_hash,
        "dataset_build_report_hash": dataset_build_report_hash,
        "source_file_inventory_hash": source_inventory_hash,
    }

    state_build_id = _compute_state_build_id(
        run_id=run_id,
        build_mode=options.build_mode,
        output_semantics_mode=str(output_semantics.get("mode")),
        dataset_manifest_hash=dataset_manifest_hash,
        dataset_build_report_hash=dataset_build_report_hash,
        scaling_contract_hash=scaling_contract_hash,
        state_column_selection_hash=state_column_selection_hash,
        observation_contract_hash=observation_contract_hash,
        runtime_price_contract_hash=runtime_price_contract_hash,
        source_inventory_hash=source_inventory_hash,
    )

    state_build_id_policy = {
        "algorithm": "sha256",
        "canonical_json": {"sort_keys": True, "separators": [",", ":"], "ensure_ascii": True},
        "hash_inputs_order": [
            "run_id",
            "builder_version",
            "build_mode",
            "output_semantics_mode",
            "dataset_manifest_hash",
            "dataset_build_report_hash",
            "scaling_contract_hash",
            "state_column_selection_hash",
            "observation_contract_hash",
            "runtime_price_contract_hash",
            "source_inventory_hash",
        ],
    }

    _prepare_staging_root(staging_root=staging_root, overwrite=bool(options.overwrite), errors=errors)
    if errors:
        report_payload["errors"] = [asdict(item) for item in errors]
        report_payload["warnings"] = [asdict(item) for item in warnings]
        report_payload["state_build_overall"] = False
        report_payload["error_code"] = STATE_BUILD_PRECONDITION_FAILED
        report_payload = _sanitize_persisted_metadata_paths(
            payload=report_payload,
            staging_root=staging_root,
            output_root=output_root,
        )
        _write_report_best_effort(report_payload, report_path, warnings)
        return StateBuildResult(
            report_payload=report_payload,
            manifest_payload=None,
            scaler_stats_payload=None,
            report_path=report_path,
            manifest_path=manifest_path,
            scaler_stats_path=scaler_stats_path,
        )

    artifacts: list[StateArtifact] = []
    rows_read = 0
    rows_written = 0
    files_processed = 0
    files_failed = 0

    scaled_columns = [
        col
        for col in state_feature_columns
        if str(observation_contract_spec.selected_dtypes.get(col, "")).lower().startswith("float")
    ]

    scaler_registry: dict[tuple[str, int | None], dict[str, dict[str, Any]]] = {}
    scaler_group_rows: dict[tuple[str, int | None], int] = {}
    runtime_price_dtypes: dict[str, str] = {}

    sorted_specs = sorted(
        expected_specs,
        key=lambda item: (
            item.source_rel,
            item.fold_id if item.fold_id is not None else -1,
            _partition_order(item.partition),
            str(item.source_file.resolve()),
        ),
    )

    for spec in sorted_specs:
        files_processed += 1

        try:
            frame = pd.read_parquet(spec.source_file)
            rows_read += int(len(frame))
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_SOURCE_FILE_MISSING,
                    message="Failed to read source dataset parquet.",
                    context={"source_file": str(spec.source_file), "error": str(exc)},
                )
            )
            files_failed += 1
            continue

        missing_columns = [col for col in selected_columns if col not in frame.columns]
        if missing_columns:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                    message="Selected state columns are missing in source dataset parquet.",
                    context={"source_file": str(spec.source_file), "missing_columns": missing_columns},
                )
            )
            files_failed += 1
            continue

        missing_runtime_columns = [col for col in runtime_price_columns if col not in frame.columns]
        if missing_runtime_columns:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_RUNTIME_PRICE_COLUMN_MISSING,
                    message="Runtime price columns are missing in source dataset parquet.",
                    context={"source_file": str(spec.source_file), "missing_columns": missing_runtime_columns},
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
                    code=STATE_BUILD_ORDERING_CONTRACT_VIOLATION,
                    message="Timestamp parsing failed in source dataset parquet.",
                    context={"source_file": str(spec.source_file), "timestamp_column": timestamp_column},
                )
            )
            files_failed += 1
            continue

        frame_with_pos[timestamp_column] = parsed_ts

        for column in selected_columns:
            expected_dtype = selected_dtype_map.get(column)
            actual_dtype = str(frame_with_pos[column].dtype)
            if expected_dtype is not None and expected_dtype != actual_dtype:
                errors.append(
                    ValidationIssue(
                        code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                        message="Source dtype does not match state column contract dtype.",
                        context={
                            "source_file": str(spec.source_file),
                            "column": column,
                            "expected_dtype": expected_dtype,
                            "actual_dtype": actual_dtype,
                        },
                    )
                )
        for column in runtime_price_columns:
            actual_dtype = str(frame_with_pos[column].dtype)
            expected_runtime_dtype = runtime_price_dtypes.get(column)
            if expected_runtime_dtype is None:
                runtime_price_dtypes[column] = actual_dtype
            elif expected_runtime_dtype != actual_dtype:
                errors.append(
                    ValidationIssue(
                        code=STATE_BUILD_RUNTIME_PRICE_CONTRACT_INVALID,
                        message="Runtime price dtype is inconsistent across source artifacts.",
                        context={
                            "source_file": str(spec.source_file),
                            "column": column,
                            "expected_dtype": expected_runtime_dtype,
                            "actual_dtype": actual_dtype,
                        },
                    )
                )
            numeric_price = pd.to_numeric(frame_with_pos[column], errors="coerce")
            if numeric_price.isna().any():
                errors.append(
                    ValidationIssue(
                        code=STATE_BUILD_RUNTIME_PRICE_CONTRACT_INVALID,
                        message="Runtime price column must be numeric without NaN after coercion.",
                        context={"source_file": str(spec.source_file), "column": column},
                    )
                )
        # sort with tie-breaker on source row position before deterministic observation materialization
        ordered_frame = frame_with_pos.sort_values([timestamp_column, "__row_position"], kind="mergesort").reset_index(drop=True)
        out_df = _materialize_state_artifact_frame(
            frame=ordered_frame,
            observation_contract=observation_contract_spec,
            runtime_price_columns=runtime_price_columns,
            timestamp_column=timestamp_column,
            source_rel=spec.source_rel,
            scope=spec.scope,
            partition=spec.partition,
            fold_id=spec.fold_id,
            errors=errors,
        )
        if out_df is None:
            files_failed += 1
            continue
        warmup_contract = _build_warmup_contract(
            frame=out_df,
            observation_columns=state_feature_columns,
            timestamp_column=timestamp_column,
        )

        ts_out = pd.to_datetime(out_df[timestamp_column], utc=True, errors="coerce") if len(out_df) > 0 else pd.Series(dtype="datetime64[ns, UTC]")
        if len(out_df) > 0 and (not ts_out.is_monotonic_increasing):
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_ORDERING_CONTRACT_VIOLATION,
                    message="State artifact is not monotonic by timestamp.",
                    context={"source_file": spec.source_rel, "scope": spec.scope, "partition": spec.partition, "fold_id": spec.fold_id},
                )
            )

        duplicate_count = int(ts_out.duplicated().sum()) if len(out_df) > 0 else 0
        unique_ok = duplicate_count == 0
        if duplicate_count > 0:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_TIMESTAMP_DUPLICATES,
                    message="Duplicate timestamps detected in state artifact.",
                    context={
                        "source_file": spec.source_rel,
                        "scope": spec.scope,
                        "partition": spec.partition,
                        "fold_id": spec.fold_id,
                        "duplicate_timestamp_count": duplicate_count,
                    },
                )
            )
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_ORDERING_CONTRACT_VIOLATION,
                    message="Ordering contract violation due to duplicate timestamps.",
                    context={
                        "source_file": spec.source_rel,
                        "scope": spec.scope,
                        "partition": spec.partition,
                        "fold_id": spec.fold_id,
                        "duplicate_timestamp_count": duplicate_count,
                    },
                )
            )

        if scaling_contract["enabled"]:
            scaler_key = (spec.source_rel, spec.fold_id if spec.scope == "fold" else None)
            if spec.partition == "train":
                stats, failed = _fit_standard_scaler(
                    frame=out_df,
                    columns=scaled_columns,
                    source_rel=spec.source_rel,
                    fold_id=spec.fold_id if spec.scope == "fold" else None,
                )
                if failed:
                    errors.append(
                        ValidationIssue(
                            code=STATE_BUILD_SCALER_FIT_FAILED,
                            message="Failed to fit scaler on training partition.",
                            context={
                                "source_rel": spec.source_rel,
                                "fold_id": spec.fold_id if spec.scope == "fold" else None,
                                "partition": spec.partition,
                            },
                        )
                    )
                    files_failed += 1
                    continue
                scaler_registry[scaler_key] = stats
                scaler_group_rows[scaler_key] = int(len(out_df))

            stats = scaler_registry.get(scaler_key)
            if stats is None:
                errors.append(
                    ValidationIssue(
                        code=STATE_BUILD_SCALER_FIT_FAILED,
                        message="Scaler stats are unavailable for transform scope.",
                        context={
                            "source_rel": spec.source_rel,
                            "fold_id": spec.fold_id if spec.scope == "fold" else None,
                            "partition": spec.partition,
                        },
                    )
                )
                files_failed += 1
                continue

            _apply_standard_scaler(frame=out_df, stats=stats)

        if spec.scope == "fold":
            if spec.fold_id is None:
                raise ValueError("fold scope requires fold_id")
            out_path = _build_fold_output_path(staging_root, spec.source_rel, spec.partition, spec.fold_id)
        else:
            out_path = _build_partition_output_path(staging_root, spec.source_rel, spec.partition)

        ensure_within_root(out_path, staging_root)

        try:
            atomic_write_parquet(out_df, out_path)
        except RuntimeError as exc:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_WRITE_FAILED,
                    message="Failed to write state parquet atomically.",
                    context={"output_path": str(out_path), "error": str(exc)},
                )
            )
            files_failed += 1
            continue

        rows_written += int(len(out_df))
        file_sha = _sha256_file(out_path)
        artifacts.append(
            StateArtifact(
                scope=spec.scope,
                source_rel=spec.source_rel,
                partition=spec.partition,
                fold_id=spec.fold_id,
                output_path=str(out_path),
                row_count=int(len(out_df)),
                timestamp_min_utc=ts_out.iloc[0].isoformat() if len(out_df) > 0 else None,
                timestamp_max_utc=ts_out.iloc[-1].isoformat() if len(out_df) > 0 else None,
                duplicate_timestamp_count=duplicate_count,
                timestamp_unique_ok=unique_ok,
                file_sha256=file_sha,
                warmup_contract=warmup_contract,
            )
        )

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
    report_payload["state_build_id"] = state_build_id
    report_payload["state_build_id_policy"] = state_build_id_policy

    row_order_policy = {
        "name": "timestamp_ascending",
        "stable_tie_breaker": "source_row_position",
    }
    timestamp_policy = {
        "timestamp_column": timestamp_column,
        "required_timezone": "UTC",
        "monotonic_required": True,
        "uniqueness_required": True,
    }
    dtype_policy = {
        "selected_dtypes": observation_contract_spec.selected_dtypes,
        "expected_float_dtype": "float32",
        "expected_event_dtype": "uint8",
    }

    scaler_groups: list[dict[str, Any]] = []
    if scaling_contract["enabled"]:
        for scaler_key, feature_stats in sorted(scaler_registry.items(), key=lambda item: (item[0][0], item[0][1] if item[0][1] is not None else -1)):
            source_rel, fold_id = scaler_key
            features_payload: list[dict[str, Any]] = []
            for feature_name in sorted(feature_stats.keys()):
                payload = feature_stats[feature_name]
                features_payload.append(
                    {
                        "name": feature_name,
                        "mean": float(payload["mean"]),
                        "std": float(payload["std"]),
                        "count": int(payload["count"]),
                        "zero_std_replaced": bool(payload["zero_std_replaced"]),
                    }
                )

            scaler_groups.append(
                {
                    "scope": {
                        "source_rel": source_rel,
                        "fold_id": fold_id,
                        "partition": "train",
                    },
                    "row_count": int(scaler_group_rows.get(scaler_key, 0)),
                    "features": features_payload,
                    "stats_hash": _hash_canonical_json({"features": features_payload}),
                }
            )

    scaler_stats_payload = None
    if scaling_contract["enabled"]:
        scaler_stats_payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "builder_version": STATE_BUILDER_VERSION,
            "scaler_type": scaling_contract["scaler_type"],
            "fit_scope_policy": scaling_contract["fit_scope_policy"],
            "transform_scope_policy": scaling_contract["transform_scope_policy"],
            "groups": scaler_groups,
        }

    try:
        persisted_artifacts = _build_promoted_artifact_metadata(artifacts=artifacts, staging_root=staging_root, output_root=output_root)
    except RuntimeError as exc:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_WRITE_FAILED,
                message="Failed to map staged state artifact metadata to promoted paths.",
                context={"error": str(exc)},
            )
        )
        persisted_artifacts = []

    scaler_stats_ref: dict[str, Any] | None = None
    if scaler_stats_payload is not None:
        scaler_stats_ref = {
            "path": str(scaler_stats_path),
            "sha256": None,
            "hash_algorithm": "sha256",
        }

    observation_contract = {
        "selected_input_columns": list(observation_contract_spec.selected_input_columns),
        "state_feature_columns": list(state_feature_columns),
        "event_columns": list(observation_contract_spec.event_columns),
        "regime_columns": list(observation_contract_spec.regime_columns),
        "geometry_columns": list(observation_contract_spec.geometry_columns),
        "strict_post_valid_numeric_columns": list(observation_contract_spec.strict_post_valid_numeric_columns),
        "conditional_raw_columns": list(observation_contract_spec.conditional_raw_columns),
        "conditional_column_policy": observation_contract_spec.conditional_column_policy,
        "conditional_column_replacements": observation_contract_spec.conditional_column_replacements,
        "geometry_feature_version": observation_contract_spec.geometry_feature_version,
        "geometry_feature_formulas": observation_contract_spec.geometry_feature_formulas,
        "future_feature_hooks": observation_contract_spec.future_feature_hooks,
        "dtype_policy": dtype_policy,
        "row_order_policy": row_order_policy,
        "timestamp_policy": timestamp_policy,
        "scaling_policy": scaling_contract,
    }
    runtime_price_contract = {
        **runtime_price_contract,
        "runtime_price_dtypes": {key: runtime_price_dtypes[key] for key in runtime_price_columns if key in runtime_price_dtypes},
    }

    manifest_payload = {
        "manifest_version": STATE_MANIFEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "builder_version": STATE_BUILDER_VERSION,
        "state_build_id": state_build_id,
        "state_build_id_policy": state_build_id_policy,
        "build_mode": options.build_mode,
        "source_lineage": {
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_build_report_path": str(dataset_build_report_path),
        },
        "source_hashes": {
            "dataset_manifest_hash": dataset_manifest_hash,
            "dataset_build_report_hash": dataset_build_report_hash,
            "source_file_inventory_hash": source_inventory_hash,
        },
        "split_mode": split_mode,
        "output_semantics": output_semantics,
        "column_selection_contract": {
            **column_selection_contract,
            "state_column_selection_hash": state_column_selection_hash,
        },
        "observation_contract": observation_contract,
        "runtime_price_contract": runtime_price_contract,
        "warmup_contract_summary": _build_warmup_contract_summary(persisted_artifacts),
        "scaler_stats_ref": scaler_stats_ref,
        "partition_metadata": [item.to_dict() for item in persisted_artifacts],
        "walk_forward_fold_metadata": _build_walk_forward_fold_metadata(persisted_artifacts),
        "output_completeness_ok": bool(output_completeness_ok),
    }

    report_payload["row_order_policy"] = row_order_policy
    report_payload["timestamp_policy"] = timestamp_policy
    report_payload["observation_contract"] = observation_contract
    report_payload["runtime_price_contract"] = runtime_price_contract
    report_payload["warmup_contract_summary"] = _build_warmup_contract_summary(persisted_artifacts)
    report_payload["staging_root"] = None
    report_payload["errors"] = [asdict(item) for item in errors]
    report_payload["warnings"] = [asdict(item) for item in warnings]
    report_payload["state_build_overall"] = bool(len(errors) == 0 and output_completeness_ok and files_failed == 0)

    success = bool(report_payload["state_build_overall"])

    if success:
        staging_report_path = staging_root / "reports" / "state_build_report.json"
        staging_manifest_path = staging_root / "reports" / "state_manifest.json"
        staging_scaler_stats_path = staging_root / "reports" / "scaler_stats.json"
        persisted_report_payload = _sanitize_persisted_metadata_paths(
            payload=report_payload,
            staging_root=staging_root,
            output_root=output_root,
        )
        persisted_manifest_payload = _sanitize_persisted_metadata_paths(
            payload=manifest_payload,
            staging_root=staging_root,
            output_root=output_root,
        )

        try:
            if scaler_stats_payload is not None:
                atomic_write_json(scaler_stats_payload, staging_scaler_stats_path)
                if persisted_manifest_payload["scaler_stats_ref"] is not None:
                    persisted_manifest_payload["scaler_stats_ref"]["sha256"] = _sha256_file(staging_scaler_stats_path)

            atomic_write_json(persisted_report_payload, staging_report_path)
            atomic_write_json(persisted_manifest_payload, staging_manifest_path)
            _promote_staging_to_output(staging_root=staging_root, output_root=output_root, overwrite=bool(options.overwrite))
            report_payload = persisted_report_payload
            manifest_payload = persisted_manifest_payload
        except RuntimeError as exc:
            success = False
            report_payload["state_build_overall"] = False
            report_payload["errors"].append(
                asdict(
                    ValidationIssue(
                        code=STATE_BUILD_WRITE_FAILED,
                        message="Failed while finalizing staged state outputs.",
                        context={"error": str(exc)},
                    )
                )
            )
            _cleanup_staging_root(staging_root)
    else:
        _cleanup_staging_root(staging_root)

    if not success:
        report_payload = _sanitize_persisted_metadata_paths(
            payload=report_payload,
            staging_root=staging_root,
            output_root=output_root,
        )
        _write_report_best_effort(report_payload, report_path, warnings)
        manifest_payload = None
        scaler_stats_payload = None

    return StateBuildResult(
        report_payload=report_payload,
        manifest_payload=manifest_payload,
        scaler_stats_payload=scaler_stats_payload,
        report_path=report_path,
        manifest_path=manifest_path,
        scaler_stats_path=scaler_stats_path,
    )


def _base_report_payload(
    *,
    run_id: str,
    input_root: Path,
    output_root: Path,
    report_path: Path,
    manifest_path: Path,
    scaler_stats_path: Path,
    staging_root: Path,
    invocation_args: dict[str, Any],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    """Build baseline report payload."""

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "builder_version": STATE_BUILDER_VERSION,
        "state_build_overall": False,
        "state_build_id": None,
        "build_mode": DEFAULT_BUILD_MODE,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "staging_root": str(staging_root),
        "state_build_report_path": str(report_path),
        "state_manifest_path": str(manifest_path),
        "scaler_stats_path": str(scaler_stats_path),
        "source_paths": source_paths,
        "split_mode": None,
        "upstream_output_semantics": {},
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
        "runtime_price_contract": {},
        "warmup_contract_summary": _default_warmup_contract_summary(),
        "scaling_contract": {},
        "source_hashes": {},
        "errors": [],
        "warnings": [],
    }


def _default_warmup_contract(*, valid_from_row: int = 0, valid_from_timestamp: str | None = None) -> dict[str, Any]:
    """Return a stable warmup-contract payload."""

    return {
        "enabled": False,
        "required_observation_columns": [],
        "policy": WARMUP_POLICY_DROP_HEAD,
        "valid_from_row": int(valid_from_row),
        "valid_from_timestamp": valid_from_timestamp,
        "post_valid_nan_policy": WARMUP_POST_VALID_NAN_POLICY,
        "head_nan_profile": {},
    }


def _default_warmup_contract_summary() -> dict[str, Any]:
    """Return a stable warmup summary payload."""

    return {
        "enabled": False,
        "policy": WARMUP_POLICY_DROP_HEAD,
        "post_valid_nan_policy": WARMUP_POST_VALID_NAN_POLICY,
        "artifacts_total": 0,
        "artifacts_with_warmup": 0,
        "max_valid_from_row": 0,
    }


def _default_output_root(input_root: Path) -> Path:
    """Resolve default output root from dataset input root."""

    return (input_root.parent / "data_states").resolve()


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
        errors.append(ValidationIssue(code=invalid_code, message=invalid_message, context={"path": str(path), "error": str(exc)}))
        return None

    if not isinstance(payload, dict):
        errors.append(ValidationIssue(code=invalid_code, message="JSON payload must be an object.", context={"path": str(path)}))
        return None

    return payload


def _require_run_id(*, field_name: str, payload: Mapping[str, Any], run_id: str, errors: list[ValidationIssue]) -> None:
    """Enforce run_id match against expected run id."""

    seen = payload.get("run_id")
    if seen != run_id:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_RUN_ID_MISMATCH,
                message="run_id mismatch across state build lineage artifacts.",
                context={"field": field_name, "expected_run_id": run_id, "seen_run_id": seen},
            )
        )


def _validate_lineage(
    *,
    dataset_manifest: Mapping[str, Any] | None,
    dataset_build_report: Mapping[str, Any] | None,
    warnings: list[ValidationIssue],
    errors: list[ValidationIssue],
) -> None:
    """Validate hard and soft lineage policies."""

    if dataset_manifest is None or dataset_build_report is None:
        return

    manifest_build_id = dataset_manifest.get("dataset_build_id")
    report_build_id = dataset_build_report.get("dataset_build_id")
    if isinstance(manifest_build_id, str) and isinstance(report_build_id, str) and manifest_build_id != report_build_id:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_LINEAGE_MISMATCH,
                message="dataset_build_id mismatch between manifest and report.",
                context={"manifest_dataset_build_id": manifest_build_id, "report_dataset_build_id": report_build_id},
            )
        )

    manifest_split = dataset_manifest.get("split_mode")
    report_split = dataset_build_report.get("split_mode")
    if isinstance(manifest_split, str) and isinstance(report_split, str) and manifest_split != report_split:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_LINEAGE_MISMATCH,
                message="split_mode mismatch between manifest and report.",
                context={"manifest_split_mode": manifest_split, "report_split_mode": report_split},
            )
        )

    manifest_sem = dataset_manifest.get("output_semantics")
    report_sem = dataset_build_report.get("output_semantics")
    if isinstance(manifest_sem, dict) and isinstance(report_sem, dict):
        manifest_mode = manifest_sem.get("mode")
        report_mode = report_sem.get("mode")
        if isinstance(manifest_mode, str) and isinstance(report_mode, str) and manifest_mode != report_mode:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_LINEAGE_MISMATCH,
                    message="output_semantics.mode mismatch between manifest and report.",
                    context={"manifest_mode": manifest_mode, "report_mode": report_mode},
                )
            )

    manifest_hashes = dataset_manifest.get("source_hashes")
    report_hashes = dataset_build_report.get("source_hashes")

    critical_keys = ["feature_manifest_hash", "train_input_report_hash", "split_report_hash"]
    if not isinstance(manifest_hashes, dict) or not isinstance(report_hashes, dict):
        warnings.append(
            ValidationIssue(
                code=STATE_BUILD_LINEAGE_CHECK_PARTIAL,
                message="Lineage hash comparison is partial because source_hashes block is missing.",
                context={},
            )
        )
        return

    partial = False
    for key in critical_keys:
        manifest_value = manifest_hashes.get(key)
        report_value = report_hashes.get(key)

        if isinstance(manifest_value, str) and isinstance(report_value, str):
            if manifest_value != report_value:
                errors.append(
                    ValidationIssue(
                        code=STATE_BUILD_LINEAGE_MISMATCH,
                        message="Critical lineage hash mismatch.",
                        context={"key": key, "manifest_value": manifest_value, "report_value": report_value},
                    )
                )
        else:
            partial = True
            warnings.append(
                ValidationIssue(
                    code=STATE_BUILD_OPTIONAL_LINEAGE_HASH_MISSING,
                    message="Optional lineage hash is missing; continuing with partial lineage checks.",
                    context={"key": key},
                )
            )

    if partial:
        warnings.append(
            ValidationIssue(
                code=STATE_BUILD_LINEAGE_CHECK_PARTIAL,
                message="Lineage hash comparison is partial due to missing optional hash fields.",
                context={},
            )
        )


def _resolve_column_selection(
    *,
    dataset_manifest: Mapping[str, Any],
    options: StateBuildOptions,
    warnings: list[ValidationIssue],
    errors: list[ValidationIssue],
) -> tuple[list[str], dict[str, str], str, dict[str, Any]]:
    """Resolve deterministic state column selection contract."""

    selection_contract = dataset_manifest.get("column_selection_contract")
    if not isinstance(selection_contract, dict):
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="dataset_manifest.column_selection_contract must be an object.",
                context={},
            )
        )
        return [], {}, "timestamp", {}

    timestamp_column = options.timestamp_column_override if options.timestamp_column_override else selection_contract.get("timestamp_column")
    if not isinstance(timestamp_column, str) or not timestamp_column.strip():
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="timestamp_column is missing or invalid.",
                context={"timestamp_column": timestamp_column},
            )
        )
        return [], {}, "timestamp", {}
    timestamp_column = timestamp_column.strip()

    selected_columns_manifest = _parse_string_list(selection_contract.get("selected_columns"))
    if selected_columns_manifest is None or not selected_columns_manifest:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="column_selection_contract.selected_columns must be list[str].",
                context={},
            )
        )
        return [], {}, timestamp_column, {}

    selected_dtypes_raw = selection_contract.get("selected_dtypes")
    if not isinstance(selected_dtypes_raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in selected_dtypes_raw.items()):
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="column_selection_contract.selected_dtypes must be dict[str, str].",
                context={},
            )
        )
        return [], {}, timestamp_column, {}

    requested_state_columns = [item.strip() for item in options.state_columns if item.strip()]
    if requested_state_columns:
        unknown = [col for col in requested_state_columns if col not in selected_columns_manifest]
        if unknown:
            issue = ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="Requested state columns are not part of dataset manifest selection.",
                context={"unknown_columns": unknown},
            )
            if options.strict_column_selection:
                errors.append(issue)
            else:
                warnings.append(issue)

        selected_columns = [col for col in requested_state_columns if col in selected_columns_manifest]
    else:
        selected_columns = list(selected_columns_manifest)

    if timestamp_column not in selected_columns:
        selected_columns = [timestamp_column, *selected_columns]

    selected_columns = _stable_unique(selected_columns)

    missing_dtype_columns = [col for col in selected_columns if col not in selected_dtypes_raw]
    if missing_dtype_columns:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="Selected columns are missing dtype declarations.",
                context={"missing_dtype_columns": missing_dtype_columns},
            )
        )

    selected_dtype_map = {col: str(selected_dtypes_raw[col]) for col in selected_columns if col in selected_dtypes_raw}

    contract = {
        "timestamp_column": timestamp_column,
        "input_selected_columns": selected_columns_manifest,
        "requested_state_columns": requested_state_columns,
        "selected_state_columns": selected_columns,
        "selected_state_dtypes": selected_dtype_map,
        "column_selection_hash": _hash_sequence(selected_columns),
        "dtype_hash": _hash_mapping(selected_dtype_map),
    }
    return selected_columns, selected_dtype_map, timestamp_column, contract


def _build_expected_specs(
    *,
    dataset_manifest: Mapping[str, Any],
    split_mode: str | None,
    warnings: list[ValidationIssue],
    errors: list[ValidationIssue],
) -> tuple[list[SourceSpec], dict[tuple[str, str, str, int | None], dict[str, Any]]]:
    """Build expected specs from dataset manifest partition metadata."""

    if split_mode not in {"ratio_chrono", "explicit_ranges", "walk_forward"}:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_OUTPUT_SEMANTICS_UNSUPPORTED,
                message="split_mode is unsupported for state_builder.v1.",
                context={"split_mode": split_mode},
            )
        )
        return [], {}

    output_semantics = dataset_manifest.get("output_semantics")
    if not isinstance(output_semantics, dict):
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_OUTPUT_SEMANTICS_UNSUPPORTED,
                message="dataset_manifest.output_semantics must be an object.",
                context={},
            )
        )
        return [], {}

    mode_raw = output_semantics.get("mode")
    mode = mode_raw.strip() if isinstance(mode_raw, str) else ""
    supported_modes = {"standard_partitions", "walk_forward_fold_only", "walk_forward_fold_plus_aggregate"}
    if mode not in supported_modes:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_OUTPUT_SEMANTICS_UNSUPPORTED,
                message="dataset output semantics are unsupported for state builder v1.",
                context={"mode": mode, "supported": sorted(supported_modes)},
            )
        )

    raw_entries = dataset_manifest.get("partition_metadata")
    if not isinstance(raw_entries, list):
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_DATASET_MANIFEST_INVALID,
                message="dataset_manifest.partition_metadata must be a list.",
                context={},
            )
        )
        return [], {}

    specs: list[SourceSpec] = []
    coverage: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}

    for idx, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_MANIFEST_INVALID,
                    message="partition_metadata entries must be objects.",
                    context={"index": idx},
                )
            )
            continue

        scope = str(item.get("scope", "")).strip()
        source_rel = str(item.get("source_rel", "")).strip()
        partition = str(item.get("partition", "")).strip()
        fold_id_raw = item.get("fold_id")
        output_path_raw = item.get("output_path")
        row_count_raw = item.get("row_count")

        if partition not in {"train", "val", "test"}:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_MANIFEST_INVALID,
                    message="partition_metadata.partition must be one of train/val/test.",
                    context={"index": idx, "partition": partition},
                )
            )
            continue

        if not source_rel:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_MANIFEST_INVALID,
                    message="partition_metadata.source_rel is required.",
                    context={"index": idx},
                )
            )
            continue

        if not isinstance(output_path_raw, str) or not output_path_raw.strip():
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_MANIFEST_INVALID,
                    message="partition_metadata.output_path is required.",
                    context={"index": idx, "source_rel": source_rel},
                )
            )
            continue

        if not isinstance(row_count_raw, int) or row_count_raw < 0:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_MANIFEST_INVALID,
                    message="partition_metadata.row_count must be non-negative integer.",
                    context={"index": idx, "row_count": row_count_raw},
                )
            )
            continue

        if split_mode == "walk_forward":
            if scope == "aggregate":
                warnings.append(
                    ValidationIssue(
                        code=STATE_BUILD_LINEAGE_CHECK_PARTIAL,
                        message="Aggregate upstream entries were skipped in state_builder.v1.",
                        context={"source_rel": source_rel, "partition": partition},
                    )
                )
                continue
            if scope != "fold":
                errors.append(
                    ValidationIssue(
                        code=STATE_BUILD_OUTPUT_SEMANTICS_UNSUPPORTED,
                        message="walk_forward state build requires fold scope entries.",
                        context={"index": idx, "scope": scope},
                    )
                )
                continue

            if not isinstance(fold_id_raw, int) or fold_id_raw < 0:
                errors.append(
                    ValidationIssue(
                        code=STATE_BUILD_DATASET_MANIFEST_INVALID,
                        message="fold scope requires fold_id >= 0.",
                        context={"index": idx, "fold_id": fold_id_raw},
                    )
                )
                continue
            fold_id = int(fold_id_raw)
        else:
            if scope != "partition":
                errors.append(
                    ValidationIssue(
                        code=STATE_BUILD_OUTPUT_SEMANTICS_UNSUPPORTED,
                        message="standard split_mode requires partition scope entries.",
                        context={"index": idx, "scope": scope},
                    )
                )
                continue
            fold_id = None

        source_file = Path(output_path_raw).resolve()
        if not source_file.exists():
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_SOURCE_FILE_MISSING,
                    message="Source dataset parquet referenced by manifest is missing.",
                    context={"source_file": str(source_file), "source_rel": source_rel},
                )
            )
            continue

        spec = SourceSpec(
            scope=scope,
            source_file=source_file,
            source_rel=source_rel,
            partition=partition,
            fold_id=fold_id,
            expected_rows=int(row_count_raw),
            expected_timestamp_min_utc=item.get("timestamp_min_utc") if isinstance(item.get("timestamp_min_utc"), str) else None,
            expected_timestamp_max_utc=item.get("timestamp_max_utc") if isinstance(item.get("timestamp_max_utc"), str) else None,
        )

        key = spec.key()
        if key in coverage:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_DATASET_MANIFEST_INVALID,
                    message="Duplicate partition_metadata key encountered.",
                    context={"key": list(key)},
                )
            )
            continue

        coverage[key] = {
            "expected_rows": spec.expected_rows,
            "expected_timestamp_min_utc": spec.expected_timestamp_min_utc,
            "expected_timestamp_max_utc": spec.expected_timestamp_max_utc,
        }
        specs.append(spec)

    if not specs:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_OUTPUT_SEMANTICS_UNSUPPORTED,
                message="No eligible source specs found for state materialization.",
                context={"split_mode": split_mode, "output_mode": mode},
            )
        )

    return specs, coverage


def _resolve_runtime_price_contract(
    *,
    options: StateBuildOptions,
    timestamp_column: str,
    state_feature_columns: Sequence[str],
    errors: list[ValidationIssue],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Resolve deterministic runtime price contract."""

    execution_price_column = str(options.execution_price_column).strip() if options.execution_price_column is not None else ""
    mark_to_market_column = str(options.mark_to_market_column).strip() if options.mark_to_market_column is not None else ""

    if not execution_price_column or not mark_to_market_column:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_RUNTIME_PRICE_CONTRACT_REQUIRED,
                message="execution_price_column and mark_to_market_column must be provided explicitly.",
                context={
                    "execution_price_column": options.execution_price_column,
                    "mark_to_market_column": options.mark_to_market_column,
                },
            )
        )
        return [], [], {}

    required_runtime_columns = _stable_unique([execution_price_column, mark_to_market_column])
    invalid_columns = [col for col in required_runtime_columns if col == timestamp_column or col in state_feature_columns]
    if invalid_columns:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_RUNTIME_PRICE_CONTRACT_INVALID,
                message="Runtime price columns must be distinct from timestamp and observation feature columns.",
                context={
                    "invalid_columns": invalid_columns,
                    "timestamp_column": timestamp_column,
                    "state_feature_columns": list(state_feature_columns),
                },
            )
        )
        return [], [], {}

    artifact_columns = [timestamp_column, *state_feature_columns, *required_runtime_columns]
    contract = {
        "timestamp_column": timestamp_column,
        "execution_price_column": execution_price_column,
        "mark_to_market_column": mark_to_market_column,
        "required_runtime_columns": required_runtime_columns,
        "runtime_price_dtypes": {},
        "artifact_columns": artifact_columns,
    }
    return required_runtime_columns, artifact_columns, contract


def _resolve_observation_contract(
    *,
    timestamp_column: str,
    input_state_feature_columns: Sequence[str],
    input_selected_dtypes: Mapping[str, str],
    execution_price_column: str | None,
    errors: list[ValidationIssue],
) -> ResolvedObservationContract | None:
    """Resolve final observation schema and semantic roles."""

    final_state_feature_columns: list[str] = []
    final_selected_dtypes: dict[str, str] = {}
    conditional_raw_columns: list[str] = []
    conditional_column_replacements: dict[str, list[str]] = {}
    geometry_feature_formulas: dict[str, str] = {}

    timestamp_dtype = input_selected_dtypes.get(timestamp_column)
    if timestamp_dtype is None:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="Observation contract timestamp dtype is missing.",
                context={"timestamp_column": timestamp_column},
            )
        )
        return None
    final_selected_dtypes[timestamp_column] = str(timestamp_dtype)

    execution_price = str(execution_price_column).strip() if execution_price_column is not None else ""
    st_conditional_selected = any(column in input_state_feature_columns for column in ST_CONDITIONAL_RAW_COLUMNS)
    if st_conditional_selected:
        missing_band_columns = [column for column in ST_CONDITIONAL_RAW_COLUMNS if column not in input_state_feature_columns]
        if missing_band_columns:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                    message="SuperTrend conditional raw columns must be selected together.",
                    context={
                        "required_columns": list(ST_CONDITIONAL_RAW_COLUMNS),
                        "missing_columns": missing_band_columns,
                        "selected_state_columns": list(input_state_feature_columns),
                    },
                )
            )
            return None
        conflicting_columns = [
            column
            for column in (ST_ACTIVE_LINE_COLUMN, ST_DISTANCE_TO_ACTIVE_LINE_COLUMN)
            if column in input_state_feature_columns
        ]
        if conflicting_columns:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                    message="Derived SuperTrend geometry columns cannot collide with selected raw state columns.",
                    context={"conflicting_columns": conflicting_columns},
                )
            )
            return None
        if not execution_price:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_RUNTIME_PRICE_CONTRACT_REQUIRED,
                    message="execution_price_column is required when deriving SuperTrend geometry features.",
                    context={"execution_price_column": execution_price_column},
                )
            )
            return None

    st_geometry_inserted = False
    for column in input_state_feature_columns:
        if column in ST_CONDITIONAL_RAW_COLUMNS:
            conditional_raw_columns.append(column)
            conditional_column_replacements[column] = [ST_ACTIVE_LINE_COLUMN, ST_DISTANCE_TO_ACTIVE_LINE_COLUMN]
            if not st_geometry_inserted:
                final_state_feature_columns.extend([ST_ACTIVE_LINE_COLUMN, ST_DISTANCE_TO_ACTIVE_LINE_COLUMN])
                final_selected_dtypes[ST_ACTIVE_LINE_COLUMN] = "float32"
                final_selected_dtypes[ST_DISTANCE_TO_ACTIVE_LINE_COLUMN] = "float32"
                st_geometry_inserted = True
            continue

        dtype_name = input_selected_dtypes.get(column)
        if dtype_name is None:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                    message="Observation contract dtype is missing for selected state column.",
                    context={"column": column},
                )
            )
            return None
        final_state_feature_columns.append(column)
        final_selected_dtypes[column] = str(dtype_name)

    final_state_feature_columns = _stable_unique(final_state_feature_columns)
    event_columns = tuple(column for column in final_state_feature_columns if _is_event_observation_column(column))
    regime_columns = tuple(column for column in final_state_feature_columns if _is_regime_observation_column(column))
    geometry_columns = tuple(column for column in final_state_feature_columns if _is_geometry_observation_column(column))

    if st_conditional_selected:
        geometry_feature_formulas = {
            "ST_active_line_formula": "deterministic_single_finite_band_with_trend_consistency",
            "ST_distance_to_active_line_formula": f"{execution_price}_minus_active_line",
        }

    return ResolvedObservationContract(
        input_state_feature_columns=tuple(str(column) for column in input_state_feature_columns),
        state_feature_columns=tuple(final_state_feature_columns),
        selected_input_columns=(timestamp_column, *final_state_feature_columns),
        selected_dtypes=final_selected_dtypes,
        event_columns=event_columns,
        regime_columns=regime_columns,
        geometry_columns=geometry_columns,
        strict_post_valid_numeric_columns=tuple(final_state_feature_columns),
        conditional_raw_columns=tuple(_stable_unique(conditional_raw_columns)),
        conditional_column_policy=CONDITIONAL_COLUMN_POLICY_EXCLUDE_AND_REPLACE,
        conditional_column_replacements=conditional_column_replacements,
        geometry_feature_version=GEOMETRY_FEATURE_VERSION,
        geometry_feature_formulas=geometry_feature_formulas,
        future_feature_hooks={
            "trend_age_context": {
                "implemented": False,
                "planned_columns": list(TREND_AGE_FUTURE_COLUMNS),
                "note": "Reserved metadata hook for future bars_since_flip style regime-age features.",
            }
        },
        source_price_column=execution_price,
    )


def _resolve_state_output_semantics(*, split_mode: str | None) -> dict[str, Any]:
    """Resolve deterministic output semantics for state builder v1."""

    if split_mode == "walk_forward":
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


def _resolve_scaling_contract(*, split_mode: str | None, enable_scaling: bool, scaler_type: str) -> dict[str, Any]:
    """Resolve deterministic scaling contract payload."""

    enabled = bool(enable_scaling)
    effective_scaler_type = scaler_type if enabled else "none"

    fit_scope = "train_only"
    transform_scope = "train_stats_to_val_test"
    if split_mode == "walk_forward":
        fit_scope = "fold_train_only"
        transform_scope = "fold_train_stats_to_fold_val_test"

    return {
        "enabled": enabled,
        "scaler_type": effective_scaler_type,
        "fit_scope_policy": fit_scope,
        "transform_scope_policy": transform_scope,
        "no_leakage_enforced": True,
    }


def _is_event_observation_column(column: str) -> bool:
    """Return True when column is an event-style observation feature."""

    return str(column).startswith("evt_")


def _is_regime_observation_column(column: str) -> bool:
    """Return True when column is a persistent regime/state feature."""

    name = str(column)
    return name == "ST_trend" or name == "AT_state" or name == "market_state" or name.endswith("_regime")


def _is_geometry_observation_column(column: str) -> bool:
    """Return True when column is a line-geometry observation feature."""

    return str(column) in {
        "AlphaTrend",
        "AlphaTrend_2",
        ST_ACTIVE_LINE_COLUMN,
        ST_DISTANCE_TO_ACTIVE_LINE_COLUMN,
    }


def _materialize_state_artifact_frame(
    *,
    frame: pd.DataFrame,
    observation_contract: ResolvedObservationContract,
    runtime_price_columns: Sequence[str],
    timestamp_column: str,
    source_rel: str,
    scope: str,
    partition: str,
    fold_id: int | None,
    errors: list[ValidationIssue],
) -> pd.DataFrame | None:
    """Materialize final state artifact columns from selected source columns."""

    output = pd.DataFrame(index=frame.index)
    output[timestamp_column] = frame[timestamp_column].copy()

    derived_geometry = _materialize_supertrend_geometry_columns(
        frame=frame,
        observation_contract=observation_contract,
        errors=errors,
        source_rel=source_rel,
        scope=scope,
        partition=partition,
        fold_id=fold_id,
    )
    if derived_geometry is None and any(
        column in observation_contract.state_feature_columns
        for column in (ST_ACTIVE_LINE_COLUMN, ST_DISTANCE_TO_ACTIVE_LINE_COLUMN)
    ):
        return None

    for column in observation_contract.state_feature_columns:
        if derived_geometry is not None and column in derived_geometry.columns:
            output[column] = derived_geometry[column]
            continue
        if column not in frame.columns:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                    message="Resolved observation column is missing from source frame.",
                    context={
                        "column": column,
                        "source_rel": source_rel,
                        "scope": scope,
                        "partition": partition,
                        "fold_id": fold_id,
                    },
                )
            )
            return None
        output[column] = frame[column].copy()

    for column in runtime_price_columns:
        output[column] = frame[column].copy()

    return output.loc[:, [timestamp_column, *observation_contract.state_feature_columns, *runtime_price_columns]]


def _materialize_supertrend_geometry_columns(
    *,
    frame: pd.DataFrame,
    observation_contract: ResolvedObservationContract,
    errors: list[ValidationIssue],
    source_rel: str,
    scope: str,
    partition: str,
    fold_id: int | None,
) -> pd.DataFrame | None:
    """Derive fail-closed SuperTrend geometry features from conditional bands."""

    if not any(column in observation_contract.conditional_raw_columns for column in ST_CONDITIONAL_RAW_COLUMNS):
        return pd.DataFrame(index=frame.index)

    missing_columns = [
        column
        for column in (*ST_CONDITIONAL_RAW_COLUMNS, observation_contract.source_price_column)
        if column not in frame.columns
    ]
    if missing_columns:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="SuperTrend geometry derivation inputs are missing from source frame.",
                context={
                    "missing_columns": missing_columns,
                    "source_rel": source_rel,
                    "scope": scope,
                    "partition": partition,
                    "fold_id": fold_id,
                },
            )
        )
        return None

    st_up = pd.to_numeric(frame["ST_up"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    st_dn = pd.to_numeric(frame["ST_dn"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    price = pd.to_numeric(frame[observation_contract.source_price_column], errors="coerce").to_numpy(dtype=np.float64, copy=True)

    up_finite = np.isfinite(st_up)
    dn_finite = np.isfinite(st_dn)
    any_band_finite = up_finite | dn_finite
    if not bool(any_band_finite.any()):
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="SuperTrend geometry derivation requires at least one valid raw band row.",
                context={"source_rel": source_rel, "scope": scope, "partition": partition, "fold_id": fold_id},
            )
        )
        return None

    first_valid_pos = int(np.flatnonzero(any_band_finite)[0])
    post_warmup_mask = np.arange(len(frame)) >= first_valid_pos
    invalid_exactly_one_mask = post_warmup_mask & (up_finite == dn_finite)
    if bool(invalid_exactly_one_mask.any()):
        first_invalid = int(np.flatnonzero(invalid_exactly_one_mask)[0])
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                message="SuperTrend raw bands must have exactly one finite active band after warmup.",
                context={
                    "source_rel": source_rel,
                    "scope": scope,
                    "partition": partition,
                    "fold_id": fold_id,
                    "row_index": first_invalid,
                },
            )
        )
        return None

    active_line = np.full(len(frame), np.nan, dtype=np.float32)
    selected_up_mask = post_warmup_mask & up_finite
    selected_dn_mask = post_warmup_mask & dn_finite
    active_line[selected_up_mask] = st_up[selected_up_mask].astype(np.float32, copy=False)
    active_line[selected_dn_mask] = st_dn[selected_dn_mask].astype(np.float32, copy=False)

    if "ST_trend" in observation_contract.input_state_feature_columns and "ST_trend" in frame.columns:
        trend = pd.to_numeric(frame["ST_trend"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        trend_up_mismatch = selected_up_mask & (trend != 1.0)
        trend_dn_mismatch = selected_dn_mask & (trend != -1.0)
        if bool(trend_up_mismatch.any()) or bool(trend_dn_mismatch.any()):
            first_invalid = int(
                np.flatnonzero(trend_up_mismatch | trend_dn_mismatch)[0]
            )
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_COLUMN_SELECTION_INVALID,
                    message="ST_trend must agree with the selected SuperTrend active band.",
                    context={
                        "source_rel": source_rel,
                        "scope": scope,
                        "partition": partition,
                        "fold_id": fold_id,
                        "row_index": first_invalid,
                    },
                )
            )
            return None

    distance = np.full(len(frame), np.nan, dtype=np.float32)
    finite_active_mask = np.isfinite(active_line.astype(np.float64, copy=False))
    distance[finite_active_mask] = (
        price[finite_active_mask] - active_line[finite_active_mask].astype(np.float64, copy=False)
    ).astype(np.float32, copy=False)

    return pd.DataFrame(
        {
            ST_ACTIVE_LINE_COLUMN: pd.Series(active_line, index=frame.index, dtype="float32"),
            ST_DISTANCE_TO_ACTIVE_LINE_COLUMN: pd.Series(distance, index=frame.index, dtype="float32"),
        }
    )


def _count_leading_non_finite(series: pd.Series) -> int:
    """Count only the leading non-finite values in one observation column."""

    numeric = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(numeric.astype("float64", copy=False).to_numpy())
    leading = 0
    for is_finite in finite_mask:
        if bool(is_finite):
            break
        leading += 1
    return int(leading)


def _build_warmup_contract(
    *,
    frame: pd.DataFrame,
    observation_columns: Sequence[str],
    timestamp_column: str,
) -> dict[str, Any]:
    """Build additive warmup metadata for one materialized artifact."""

    timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    head_nan_profile: dict[str, int] = {}
    required_observation_columns: list[str] = []

    for column in observation_columns:
        leading_non_finite = _count_leading_non_finite(frame[column])
        if leading_non_finite > 0:
            head_nan_profile[column] = leading_non_finite
            required_observation_columns.append(column)

    valid_from_row = int(max(head_nan_profile.values(), default=0))
    valid_from_timestamp: str | None = None
    if valid_from_row < len(timestamps):
        valid_item = timestamps.iloc[valid_from_row]
        if pd.notna(valid_item):
            valid_from_timestamp = pd.Timestamp(valid_item).isoformat()

    contract = _default_warmup_contract(
        valid_from_row=valid_from_row,
        valid_from_timestamp=valid_from_timestamp,
    )
    contract["enabled"] = bool(required_observation_columns)
    contract["required_observation_columns"] = list(required_observation_columns)
    contract["head_nan_profile"] = dict(head_nan_profile)
    return contract


def _build_warmup_contract_summary(artifacts: Sequence[StateArtifact]) -> dict[str, Any]:
    """Summarize artifact-level warmup metadata for reports."""

    summary = _default_warmup_contract_summary()
    summary["artifacts_total"] = int(len(artifacts))
    summary["artifacts_with_warmup"] = int(sum(1 for item in artifacts if bool(item.warmup_contract.get("enabled", False))))
    summary["enabled"] = bool(summary["artifacts_with_warmup"])
    summary["max_valid_from_row"] = int(
        max((int(item.warmup_contract.get("valid_from_row", 0)) for item in artifacts), default=0)
    )
    return summary


def _fit_standard_scaler(
    *,
    frame: pd.DataFrame,
    columns: Sequence[str],
    source_rel: str,
    fold_id: int | None,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Fit standard scaler stats from train scope."""

    stats: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in frame.columns:
            return {}, True
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.isna().any():
            return {}, True

        mean = float(series.mean())
        std = float(series.std(ddof=0))
        zero_std_replaced = False
        if not pd.notna(std) or std == 0.0:
            std = 1.0
            zero_std_replaced = True

        stats[column] = {
            "mean": mean,
            "std": std,
            "count": int(series.shape[0]),
            "zero_std_replaced": zero_std_replaced,
            "source_rel": source_rel,
            "fold_id": fold_id,
        }

    return stats, False


def _apply_standard_scaler(*, frame: pd.DataFrame, stats: Mapping[str, Mapping[str, Any]]) -> None:
    """Apply fitted standard-scaler stats in-place."""

    for column, item in stats.items():
        mean = float(item["mean"])
        std = float(item["std"])
        frame[column] = ((pd.to_numeric(frame[column], errors="raise") - mean) / std).astype("float32")


def _evaluate_overwrite_policy(*, output_root: Path, staging_root: Path, overwrite: bool, errors: list[ValidationIssue]) -> dict[str, Any]:
    """Evaluate overwrite/collision contract and emit deterministic errors."""

    output_exists = output_root.exists()
    output_non_empty = output_exists and _is_non_empty_dir(output_root)
    output_empty = output_exists and output_root.is_dir() and not output_non_empty

    if output_exists and (not output_root.is_dir()):
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_OUTPUT_ROOT_EXISTS,
                message="Output root exists and is not a directory.",
                context={"output_root": str(output_root), "overwrite": bool(overwrite)},
            )
        )

    if output_non_empty and not overwrite:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_OUTPUT_ROOT_EXISTS,
                message="Output root exists and is non-empty while overwrite=false.",
                context={"output_root": str(output_root)},
            )
        )

    staging_exists = staging_root.exists()
    staging_non_empty = staging_exists and _is_non_empty_dir(staging_root)

    if staging_non_empty and not overwrite:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_STAGING_ROOT_COLLISION,
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
                        code=STATE_BUILD_STAGING_ROOT_COLLISION,
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
                            code=STATE_BUILD_STAGING_ROOT_COLLISION,
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


def _build_partition_output_path(staging_root: Path, source_rel: str, partition: str) -> Path:
    """Build output path for partition artifact."""

    stem = _source_rel_to_stem(source_rel)
    return staging_root / "parquet" / "partitions" / partition / f"{stem}.parquet"


def _build_fold_output_path(staging_root: Path, source_rel: str, partition: str, fold_id: int) -> Path:
    """Build output path for fold artifact."""

    stem = _source_rel_to_stem(source_rel)
    return staging_root / "parquet" / "folds" / f"fold_{fold_id:03d}" / partition / f"{stem}.parquet"


def _to_promoted_output_path(*, staged_output_path: str, staging_root: Path, output_root: Path) -> str:
    """Map staged artifact path into final promoted output root."""

    staging_root_resolved = staging_root.resolve()
    staged_output_resolved = Path(staged_output_path).resolve()

    try:
        relative_path = staged_output_resolved.relative_to(staging_root_resolved)
    except ValueError as exc:
        raise RuntimeError("Artifact metadata output_path is outside staging root.") from exc

    promoted = output_root.resolve() / relative_path
    return str(promoted)


def _build_promoted_artifact_metadata(
    *,
    artifacts: Sequence[StateArtifact],
    staging_root: Path,
    output_root: Path,
) -> list[StateArtifact]:
    """Create immutable artifact metadata list with promoted output paths."""

    promoted: list[StateArtifact] = []
    for item in artifacts:
        promoted_output = _to_promoted_output_path(
            staged_output_path=item.output_path,
            staging_root=staging_root,
            output_root=output_root,
        )
        promoted.append(replace(item, output_path=promoted_output))
    return promoted


def _map_staging_path_to_promoted(*, value: str, staging_root: Path, output_root: Path) -> str:
    """Map staging-root path fragments in a string to final promoted output root."""

    mapped = value
    path_pairs = (
        (str(staging_root), str(output_root)),
        (str(staging_root.resolve()), str(output_root.resolve())),
    )
    for staged_root, promoted_root in path_pairs:
        if not staged_root:
            continue
        mapped = mapped.replace(staged_root, promoted_root)
    return mapped


def _sanitize_persisted_metadata_paths(*, payload: Mapping[str, Any], staging_root: Path, output_root: Path) -> dict[str, Any]:
    """Remove staging-root leaks from persisted report/manifest metadata payloads."""

    def _sanitize(value: Any, key: str | None = None) -> Any:
        if key == "staging_root":
            return None

        if isinstance(value, dict):
            return {str(item_key): _sanitize(item_value, str(item_key)) for item_key, item_value in value.items()}

        if isinstance(value, list):
            return [_sanitize(item) for item in value]

        if isinstance(value, str):
            return _map_staging_path_to_promoted(value=value, staging_root=staging_root, output_root=output_root)

        return value

    sanitized = _sanitize(dict(payload))
    if not isinstance(sanitized, dict):
        raise RuntimeError("Persisted metadata sanitizer must return an object payload.")
    return sanitized


def _evaluate_output_completeness(
    expected_coverage: Mapping[tuple[str, str, str, int | None], Mapping[str, Any]],
    artifacts: Sequence[StateArtifact],
) -> tuple[bool, list[ValidationIssue]]:
    """Evaluate expected-vs-actual output coverage."""

    errors: list[ValidationIssue] = []
    actual = {item.key(): item for item in artifacts}
    expected_keys = set(expected_coverage.keys())
    actual_keys = set(actual.keys())

    missing = sorted(expected_keys.difference(actual_keys))
    extra = sorted(actual_keys.difference(expected_keys))

    if missing:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                message="Expected state materialization entries are missing.",
                context={"missing": [list(item) for item in missing]},
            )
        )

    if extra:
        errors.append(
            ValidationIssue(
                code=STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                message="Unexpected state materialization entries were produced.",
                context={"extra": [list(item) for item in extra]},
            )
        )

    for key in sorted(expected_keys.intersection(actual_keys)):
        expected_item = expected_coverage[key]
        actual_item = actual[key]

        expected_rows = expected_item.get("expected_rows")
        if isinstance(expected_rows, int) and expected_rows != actual_item.row_count:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                    message="Expected/actual row_count mismatch.",
                    context={"key": list(key), "expected_rows": expected_rows, "actual_rows": actual_item.row_count},
                )
            )

        expected_min = expected_item.get("expected_timestamp_min_utc")
        if isinstance(expected_min, str) and actual_item.timestamp_min_utc is not None and expected_min != actual_item.timestamp_min_utc:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                    message="Expected/actual timestamp_min mismatch.",
                    context={"key": list(key), "expected_timestamp_min_utc": expected_min, "actual_timestamp_min_utc": actual_item.timestamp_min_utc},
                )
            )

        expected_max = expected_item.get("expected_timestamp_max_utc")
        if isinstance(expected_max, str) and actual_item.timestamp_max_utc is not None and expected_max != actual_item.timestamp_max_utc:
            errors.append(
                ValidationIssue(
                    code=STATE_BUILD_OUTPUT_COMPLETENESS_MISMATCH,
                    message="Expected/actual timestamp_max mismatch.",
                    context={"key": list(key), "expected_timestamp_max_utc": expected_max, "actual_timestamp_max_utc": actual_item.timestamp_max_utc},
                )
            )

    return len(errors) == 0, errors


def _build_partition_summaries(artifacts: Sequence[StateArtifact]) -> dict[str, dict[str, Any]]:
    """Build partition aggregate summaries."""

    out: dict[str, dict[str, Any]] = {}
    for partition in ("train", "val", "test"):
        items = [item for item in artifacts if item.partition == partition]
        if not items:
            continue

        mins = [item.timestamp_min_utc for item in items if item.timestamp_min_utc is not None]
        maxs = [item.timestamp_max_utc for item in items if item.timestamp_max_utc is not None]

        out[partition] = {
            "artifacts": len(items),
            "rows": int(sum(item.row_count for item in items)),
            "timestamp_min_utc": min(mins) if mins else None,
            "timestamp_max_utc": max(maxs) if maxs else None,
            "duplicate_timestamp_count_total": int(sum(item.duplicate_timestamp_count for item in items)),
        }
    return out


def _build_fold_summaries(artifacts: Sequence[StateArtifact]) -> dict[str, Any]:
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


def _build_walk_forward_fold_metadata(artifacts: Sequence[StateArtifact]) -> list[dict[str, Any]]:
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


def _compute_state_build_id(
    *,
    run_id: str,
    build_mode: str,
    output_semantics_mode: str,
    dataset_manifest_hash: str,
    dataset_build_report_hash: str,
    scaling_contract_hash: str,
    state_column_selection_hash: str,
    observation_contract_hash: str,
    runtime_price_contract_hash: str,
    source_inventory_hash: str,
) -> str:
    """Compute deterministic state_build_id."""

    payload = {
        "run_id": run_id,
        "builder_version": STATE_BUILDER_VERSION,
        "build_mode": build_mode,
        "output_semantics_mode": output_semantics_mode,
        "dataset_manifest_hash": dataset_manifest_hash,
        "dataset_build_report_hash": dataset_build_report_hash,
        "scaling_contract_hash": scaling_contract_hash,
        "state_column_selection_hash": state_column_selection_hash,
        "observation_contract_hash": observation_contract_hash,
        "runtime_price_contract_hash": runtime_price_contract_hash,
        "source_inventory_hash": source_inventory_hash,
    }
    return _hash_canonical_json(payload)


def _partition_order(partition: str) -> int:
    """Return deterministic partition ordering key."""

    mapping = {"train": 0, "val": 1, "test": 2}
    return mapping.get(partition, 99)


def _source_rel_to_stem(source_rel: str) -> str:
    """Convert source relative path into deterministic output stem."""

    rel = Path(source_rel)
    return "__".join(rel.with_suffix("").parts)


def _hash_canonical_json(payload: Mapping[str, Any]) -> str:
    """Hash JSON payload with canonical deterministic encoding."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_sequence(values: Sequence[Any]) -> str:
    """Hash sequence deterministically."""

    payload = [str(item) for item in values]
    return _hash_canonical_json({"items": payload})


def _hash_mapping(values: Mapping[str, Any]) -> str:
    """Hash mapping deterministically."""

    normalized = {str(key): str(value) for key, value in sorted(values.items())}
    return _hash_canonical_json(normalized)


def _normalize_mapping_for_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize mapping values for deterministic hashing."""

    out: dict[str, Any] = {}
    for key, value in sorted(payload.items()):
        if isinstance(value, dict):
            out[str(key)] = _normalize_mapping_for_hash(value)
        elif isinstance(value, list):
            out[str(key)] = [str(item) for item in value]
        else:
            out[str(key)] = value
    return out


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
    """Write state build report best-effort for failure paths."""

    try:
        atomic_write_json(payload, report_path)
    except RuntimeError as exc:
        warnings.append(
            ValidationIssue(
                code=STATE_BUILD_REPORT_WRITE_FAILED,
                message="state_build_report.json write failed (best-effort).",
                context={"report_path": str(report_path), "error": str(exc)},
            )
        )
        LOGGER.info("State build report write failed (best-effort) | path=%s error=%s", report_path, exc)
