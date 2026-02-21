"""Validation layer for RL training input contract checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from core.logging import get_logger

LOGGER = get_logger(__name__)

VALIDATOR_VERSION = "train_input_validator.v1"
SUPPORTED_MANIFEST_VERSIONS: tuple[str, ...] = ("features.manifest.v1",)
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "manifest_version",
    "run_id",
    "feature_groups",
    "column_dtypes",
    "event_columns",
    "continuous_columns",
    "placeholder_columns",
    "timestamp_column",
    "indicator_spec_version",
    "config_hash",
    "formula_fingerprint_bundle",
)
REQUIRED_FEATURE_GROUP_KEYS: tuple[str, ...] = (
    "raw_ohlcv",
    "price_derived",
    "trend",
    "regime",
    "event",
    "placeholders",
)

TIMEFRAME_TOLERANCE_RATIO = 0.05
TIMEFRAME_PATTERN = re.compile(r"^\s*(\d+)\s*([smhdSMHD])\s*$")
TIMEFRAME_INFER_PATTERN = re.compile(r"(?<!\d)(\d+)([smhdSMHD])(?![a-zA-Z])")

TRAIN_INPUT_MANIFEST_MISSING = "TRAIN_INPUT_MANIFEST_MISSING"
TRAIN_INPUT_MANIFEST_INVALID_JSON = "TRAIN_INPUT_MANIFEST_INVALID_JSON"
TRAIN_INPUT_MANIFEST_SCHEMA_INVALID = "TRAIN_INPUT_MANIFEST_SCHEMA_INVALID"
TRAIN_INPUT_NO_PARQUET_FILES = "TRAIN_INPUT_NO_PARQUET_FILES"
TRAIN_INPUT_REQUIRED_COLUMN_MISSING = "TRAIN_INPUT_REQUIRED_COLUMN_MISSING"
TRAIN_INPUT_UNEXPECTED_COLUMNS = "TRAIN_INPUT_UNEXPECTED_COLUMNS"
TRAIN_INPUT_DTYPE_MISMATCH = "TRAIN_INPUT_DTYPE_MISMATCH"
TRAIN_INPUT_TIMESTAMP_INVALID = "TRAIN_INPUT_TIMESTAMP_INVALID"
TRAIN_INPUT_NON_MONOTONIC = "TRAIN_INPUT_NON_MONOTONIC"
TRAIN_INPUT_DUPLICATE_TIMESTAMP = "TRAIN_INPUT_DUPLICATE_TIMESTAMP"
TRAIN_INPUT_EMPTY_FILE = "TRAIN_INPUT_EMPTY_FILE"
TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID = "TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID"
TRAIN_INPUT_COLUMN_ORDER_DRIFT = "TRAIN_INPUT_COLUMN_ORDER_DRIFT"
TRAIN_INPUT_TIMEFRAME_INCONSISTENT = "TRAIN_INPUT_TIMEFRAME_INCONSISTENT"
TRAIN_INPUT_RUNTIME_ERROR = "TRAIN_INPUT_RUNTIME_ERROR"
TRAIN_INPUT_TIMEFRAME_UNKNOWN = "TRAIN_INPUT_TIMEFRAME_UNKNOWN"
TRAIN_INPUT_COLUMN_ORDER_NOT_EVALUATED = "TRAIN_INPUT_COLUMN_ORDER_NOT_EVALUATED"
TRAIN_INPUT_HASH_MISSING = "TRAIN_INPUT_HASH_MISSING"


@dataclass
class ValidationIssue:
    """Machine-readable issue payload."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManifestContract:
    """Manifest contract normalized for validation."""

    manifest_version: str
    run_id: str
    feature_groups: dict[str, tuple[str, ...]]
    column_dtypes: dict[str, str]
    event_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    placeholder_columns: tuple[str, ...]
    timestamp_column: str
    overlap_whitelist_event_continuous: set[str] = field(default_factory=set)
    overlap_whitelist_placeholder_event: set[str] = field(default_factory=set)
    canonical_column_order: tuple[str, ...] | None = None
    expected_timeframe: str | None = None


@dataclass
class TrainInputValidationOptions:
    """Runtime options for train input validation."""

    run_id: str
    input_root: Path
    reports_root: Path
    strict_extra_columns: bool = True
    strict_column_order: bool = False
    expected_timeframe: str | None = None

    def to_invocation_args(self) -> dict[str, Any]:
        """Serialize invocation args into JSON-friendly payload."""

        return {
            "run_id": self.run_id,
            "input_root": str(self.input_root),
            "reports_root": str(self.reports_root),
            "strict_extra_columns": bool(self.strict_extra_columns),
            "strict_column_order": bool(self.strict_column_order),
            "expected_timeframe": self.expected_timeframe,
        }


@dataclass
class FileValidationReport:
    """Per-file validation report payload."""

    input_file: str
    status: str = "failed"
    rows: int = 0
    timestamp_min_utc: str | None = None
    timestamp_max_utc: str | None = None
    schema_ok: bool = False
    dtype_ok: bool = False
    timestamp_ok: bool = False
    monotonic_ok: bool = False
    unique_ts_ok: bool = False
    required_columns_ok: bool = False
    unexpected_columns_ok: bool = True
    column_order_ok: bool | None = None
    column_order_evaluated: bool = False
    column_order_severity: str = "none"
    feature_group_contract_ok: bool = False
    timeframe_consistency_ok: bool | None = None
    observed_median_delta_seconds: float | None = None
    schema_hash_actual: str | None = None
    dtype_hash_actual: str | None = None
    column_order_hash_actual: str | None = None
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dict."""

        return asdict(self)


@dataclass
class TrainInputValidationReport:
    """Run-level train input validation report payload."""

    generated_at_utc: str
    run_id: str
    manifest_path: str
    manifest_loaded: bool = False
    manifest_valid: bool = False
    manifest_version: str | None = None
    validator_version: str = VALIDATOR_VERSION
    invocation_args: dict[str, Any] = field(default_factory=dict)
    strict_extra_columns: bool = True
    strict_column_order: bool = False
    total_files: int = 0
    succeeded_files: int = 0
    failed_files: int = 0
    train_input_validation_overall: bool = False
    schema_hash_actual: str | None = None
    dtype_hash_actual: str | None = None
    column_order_hash_actual: str | None = None
    file_reports: list[FileValidationReport] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dict."""

        return asdict(self)


def validate_train_inputs(options: TrainInputValidationOptions) -> TrainInputValidationReport:
    """Validate run-scoped feature artifacts for RL training readiness."""

    report = TrainInputValidationReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_id=options.run_id,
        manifest_path=str(options.reports_root / "feature_manifest.json"),
        invocation_args=options.to_invocation_args(),
        strict_extra_columns=bool(options.strict_extra_columns),
        strict_column_order=bool(options.strict_column_order),
    )

    manifest_path = Path(report.manifest_path)
    manifest_payload, manifest_contract, manifest_errors, manifest_warnings = _load_and_validate_manifest(manifest_path)
    report.manifest_loaded = manifest_payload is not None
    report.manifest_valid = manifest_contract is not None and len(manifest_errors) == 0
    report.manifest_version = manifest_payload.get("manifest_version") if isinstance(manifest_payload, dict) else None
    report.errors.extend(manifest_errors)
    report.warnings.extend(manifest_warnings)

    parquet_files = discover_parquet_files(options.input_root)
    report.total_files = len(parquet_files)
    if not parquet_files:
        report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_NO_PARQUET_FILES,
                message="No parquet files found under input root.",
                context={"input_root": str(options.input_root)},
            )
        )

    if manifest_contract is None:
        for path in parquet_files:
            file_report = _build_manifest_invalid_file_report(path)
            report.file_reports.append(file_report)
        _finalize_report(report)
        return report

    canonical_order, canonical_order_source, canonical_warnings = _resolve_canonical_column_order(manifest_contract)
    report.warnings.extend(canonical_warnings)

    expected_timeframe_seconds_global, expected_timeframe_source_global = _resolve_expected_timeframe_seconds(
        cli_expected_timeframe=options.expected_timeframe,
        manifest=manifest_contract,
        file_path=None,
    )

    for parquet_path in parquet_files:
        file_expected_timeframe_seconds = expected_timeframe_seconds_global
        file_expected_timeframe_source = expected_timeframe_source_global
        if file_expected_timeframe_seconds is None:
            file_expected_timeframe_seconds, file_expected_timeframe_source = _resolve_expected_timeframe_seconds(
                cli_expected_timeframe=None,
                manifest=manifest_contract,
                file_path=parquet_path,
            )

        file_report = _validate_single_parquet(
            parquet_path=parquet_path,
            manifest=manifest_contract,
            strict_extra_columns=bool(options.strict_extra_columns),
            strict_column_order=bool(options.strict_column_order),
            canonical_order=canonical_order,
            canonical_order_source=canonical_order_source,
            expected_timeframe_seconds=file_expected_timeframe_seconds,
            expected_timeframe_source=file_expected_timeframe_source,
        )
        report.file_reports.append(file_report)

    _populate_aggregate_hashes(report)
    _validate_manifest_hash_comparisons(report, manifest_payload=manifest_payload)
    _finalize_report(report)
    return report


def discover_parquet_files(input_root: Path) -> list[Path]:
    """Discover parquet files recursively under input root."""

    if not input_root.exists() or not input_root.is_dir():
        return []
    files = [path for path in input_root.glob("**/*.parquet") if path.is_file() and path.suffix.lower() == ".parquet"]
    return sorted(files)


def _build_manifest_invalid_file_report(parquet_path: Path) -> FileValidationReport:
    file_report = FileValidationReport(input_file=str(parquet_path))
    file_report.errors.append(
        ValidationIssue(
            code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
            message="Manifest invalid; file validation skipped.",
            context={"input_file": str(parquet_path)},
        )
    )
    return file_report


def _load_and_validate_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any] | None, ManifestContract | None, list[ValidationIssue], list[ValidationIssue]]:
    """Load manifest and validate manifest-level contract rules."""

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if not manifest_path.exists():
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_MISSING,
                message="feature_manifest.json not found.",
                context={"manifest_path": str(manifest_path)},
            )
        )
        return None, None, errors, warnings

    try:
        raw_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_INVALID_JSON,
                message="feature_manifest.json is not valid JSON.",
                context={"manifest_path": str(manifest_path), "error": str(exc)},
            )
        )
        return None, None, errors, warnings

    if not isinstance(raw_payload, dict):
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="Manifest payload must be a JSON object.",
                context={"manifest_path": str(manifest_path)},
            )
        )
        return raw_payload, None, errors, warnings

    missing_fields = [field_name for field_name in REQUIRED_MANIFEST_FIELDS if field_name not in raw_payload]
    if missing_fields:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="Manifest missing required fields.",
                context={"missing_fields": missing_fields},
            )
        )
        return raw_payload, None, errors, warnings

    manifest_version = raw_payload.get("manifest_version")
    if not isinstance(manifest_version, str) or manifest_version not in SUPPORTED_MANIFEST_VERSIONS:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="Manifest version is not recognized.",
                context={"manifest_version": manifest_version, "supported_versions": list(SUPPORTED_MANIFEST_VERSIONS)},
            )
        )

    run_id = raw_payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="Manifest run_id must be a non-empty string.",
                context={"run_id": run_id},
            )
        )

    feature_groups = _parse_feature_groups(raw_payload.get("feature_groups"), errors)
    column_dtypes = _parse_string_map(raw_payload.get("column_dtypes"), key="column_dtypes", errors=errors)
    event_columns = _parse_string_list(raw_payload.get("event_columns"), key="event_columns", errors=errors)
    continuous_columns = _parse_string_list(raw_payload.get("continuous_columns"), key="continuous_columns", errors=errors)
    placeholder_columns = _parse_string_list(raw_payload.get("placeholder_columns"), key="placeholder_columns", errors=errors)
    timestamp_column = raw_payload.get("timestamp_column")
    if not isinstance(timestamp_column, str) or not timestamp_column.strip():
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="timestamp_column must be a non-empty string.",
                context={"timestamp_column": timestamp_column},
            )
        )
        timestamp_column = "timestamp"

    overlap_whitelist = raw_payload.get("overlap_whitelist")
    overlap_event_continuous, overlap_placeholder_event = _parse_overlap_whitelist(overlap_whitelist, warnings)

    canonical_column_order = _parse_optional_string_list(raw_payload.get("canonical_column_order"), key="canonical_column_order")
    expected_timeframe = _parse_optional_string(raw_payload.get("expected_timeframe"))
    if expected_timeframe is None:
        expected_timeframe = _parse_optional_string(raw_payload.get("timeframe"))

    if errors:
        return raw_payload, None, errors, warnings

    assert feature_groups is not None
    assert column_dtypes is not None
    assert event_columns is not None
    assert continuous_columns is not None
    assert placeholder_columns is not None
    assert isinstance(timestamp_column, str)

    manifest_columns_listed = set(_union_feature_columns(feature_groups))
    manifest_columns_listed.update(event_columns)
    manifest_columns_listed.update(continuous_columns)
    manifest_columns_listed.update(placeholder_columns)
    missing_dtype_columns = sorted(
        col for col in manifest_columns_listed if col not in column_dtypes and col != timestamp_column
    )
    if missing_dtype_columns:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="Manifest-listed columns missing from column_dtypes.",
                context={"missing_dtype_columns": missing_dtype_columns},
            )
        )

    declared_columns = set(column_dtypes)
    if timestamp_column not in declared_columns:
        declared_columns.add(timestamp_column)

    subset_failures: dict[str, list[str]] = {}
    for group_name, cols in (
        ("event_columns", event_columns),
        ("continuous_columns", continuous_columns),
        ("placeholder_columns", placeholder_columns),
    ):
        missing = sorted(col for col in cols if col not in declared_columns)
        if missing:
            subset_failures[group_name] = missing
    if subset_failures:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID,
                message="Manifest subset consistency failed.",
                context={"subset_failures": subset_failures},
            )
        )

    event_set = set(event_columns)
    continuous_set = set(continuous_columns)
    placeholder_set = set(placeholder_columns)
    overlap_event_continuous_actual = sorted(event_set.intersection(continuous_set).difference(overlap_event_continuous))
    implicit_placeholder_event_whitelist = {
        col
        for col in placeholder_set.intersection(event_set)
        if _normalize_dtype_name(column_dtypes.get(col, "")) == _normalize_dtype_name("uint8")
    }
    allowed_placeholder_event = overlap_placeholder_event.union(implicit_placeholder_event_whitelist)
    overlap_placeholder_event_actual = sorted(placeholder_set.intersection(event_set).difference(allowed_placeholder_event))
    if overlap_event_continuous_actual or overlap_placeholder_event_actual:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID,
                message="Forbidden overlap detected between manifest feature groups.",
                context={
                    "event_continuous_overlap": overlap_event_continuous_actual,
                    "placeholder_event_overlap": overlap_placeholder_event_actual,
                },
            )
        )

    illegal_timestamp_membership = sorted(
        {
            group_name
            for group_name, cols in (
                ("event_columns", event_columns),
                ("continuous_columns", continuous_columns),
                ("placeholder_columns", placeholder_columns),
            )
            if timestamp_column in cols
        }
    )
    if illegal_timestamp_membership:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID,
                message="timestamp_column must not belong to event/continuous/placeholder groups.",
                context={"groups": illegal_timestamp_membership, "timestamp_column": timestamp_column},
            )
        )

    if errors:
        return raw_payload, None, errors, warnings

    return (
        raw_payload,
        ManifestContract(
            manifest_version=str(manifest_version),
            run_id=str(run_id),
            feature_groups=feature_groups,
            column_dtypes=column_dtypes,
            event_columns=tuple(event_columns),
            continuous_columns=tuple(continuous_columns),
            placeholder_columns=tuple(placeholder_columns),
            timestamp_column=timestamp_column,
            overlap_whitelist_event_continuous=overlap_event_continuous,
            overlap_whitelist_placeholder_event=overlap_placeholder_event,
            canonical_column_order=tuple(canonical_column_order) if canonical_column_order is not None else None,
            expected_timeframe=expected_timeframe,
        ),
        errors,
        warnings,
    )


def _parse_feature_groups(
    value: Any,
    errors: list[ValidationIssue],
) -> dict[str, tuple[str, ...]] | None:
    if not isinstance(value, dict):
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="feature_groups must be an object.",
                context={"feature_groups_type": type(value).__name__},
            )
        )
        return None

    missing_keys = [key for key in REQUIRED_FEATURE_GROUP_KEYS if key not in value]
    if missing_keys:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="feature_groups missing required keys.",
                context={"missing_feature_group_keys": missing_keys},
            )
        )
        return None

    parsed: dict[str, tuple[str, ...]] = {}
    for key, raw_cols in value.items():
        cols = _parse_optional_string_list(raw_cols, key=f"feature_groups.{key}")
        if cols is None:
            errors.append(
                ValidationIssue(
                    code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                    message="feature_groups values must be list[str].",
                    context={"group_name": key},
                )
            )
            continue
        parsed[key] = tuple(cols)

    return parsed if not errors else None


def _parse_string_map(value: Any, key: str, errors: list[ValidationIssue]) -> dict[str, str] | None:
    if not isinstance(value, dict):
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message=f"{key} must be an object of string keys and values.",
                context={"key": key, "value_type": type(value).__name__},
            )
        )
        return None
    out: dict[str, str] = {}
    for map_key, map_value in value.items():
        if not isinstance(map_key, str) or not isinstance(map_value, str):
            errors.append(
                ValidationIssue(
                    code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                    message=f"{key} must contain only string keys and string values.",
                    context={"key": key},
                )
            )
            return None
        out[map_key] = map_value
    return out


def _parse_string_list(value: Any, key: str, errors: list[ValidationIssue]) -> list[str] | None:
    parsed = _parse_optional_string_list(value, key=key)
    if parsed is None:
        errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message=f"{key} must be list[str].",
                context={"key": key},
            )
        )
        return None
    return parsed


def _parse_optional_string_list(value: Any, key: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        out.append(item)
    if len(out) != len(set(out)):
        LOGGER.info("Duplicate entries found in %s; keeping first-seen order.", key)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in out:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped
    return out


def _parse_optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_overlap_whitelist(value: Any, warnings: list[ValidationIssue]) -> tuple[set[str], set[str]]:
    if value is None:
        return set(), set()
    if not isinstance(value, dict):
        warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="overlap_whitelist ignored: expected object.",
                context={"overlap_whitelist_type": type(value).__name__},
            )
        )
        return set(), set()

    event_continuous = _parse_optional_string_list(value.get("event_continuous"), key="overlap_whitelist.event_continuous")
    placeholder_event = _parse_optional_string_list(value.get("placeholder_event"), key="overlap_whitelist.placeholder_event")
    return set(event_continuous or []), set(placeholder_event or [])


def _union_feature_columns(feature_groups: Mapping[str, Sequence[str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in REQUIRED_FEATURE_GROUP_KEYS:
        for col in feature_groups.get(key, ()):
            if col in seen:
                continue
            seen.add(col)
            out.append(col)
    for group_name in sorted(feature_groups):
        if group_name in REQUIRED_FEATURE_GROUP_KEYS:
            continue
        for col in feature_groups[group_name]:
            if col in seen:
                continue
            seen.add(col)
            out.append(col)
    return out


def _resolve_canonical_column_order(
    manifest: ManifestContract,
) -> tuple[tuple[str, ...] | None, str | None, list[ValidationIssue]]:
    warnings: list[ValidationIssue] = []
    declared_columns = list(manifest.column_dtypes.keys())
    declared_set = set(declared_columns)
    if manifest.timestamp_column not in declared_set:
        declared_set.add(manifest.timestamp_column)
        declared_columns = [manifest.timestamp_column, *declared_columns]

    if manifest.canonical_column_order is not None:
        canonical = tuple(col for col in manifest.canonical_column_order if col in declared_set)
        missing = sorted(col for col in declared_set if col not in canonical)
        if missing:
            warnings.append(
                ValidationIssue(
                    code=TRAIN_INPUT_COLUMN_ORDER_NOT_EVALUATED,
                    message="Manifest canonical_column_order missing declared columns; appending fallback tail order.",
                    context={"missing_columns": missing},
                )
            )
            canonical = (*canonical, *missing)
        return canonical, "manifest.canonical_column_order", warnings

    raw_group = manifest.feature_groups.get("raw_ohlcv")
    if raw_group is None or manifest.timestamp_column not in raw_group:
        warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_COLUMN_ORDER_NOT_EVALUATED,
                message="Canonical column order source unavailable: raw_ohlcv is missing or does not include timestamp.",
                context={"timestamp_column": manifest.timestamp_column},
            )
        )
        return None, None, warnings

    if not manifest.continuous_columns or not manifest.event_columns:
        warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_COLUMN_ORDER_NOT_EVALUATED,
                message="Canonical column order source unavailable: manifest continuous_columns/event_columns missing.",
                context={},
            )
        )
        return None, None, warnings

    canonical_order: list[str] = [manifest.timestamp_column]
    canonical_order.extend(col for col in raw_group if col != manifest.timestamp_column)
    canonical_order.extend(manifest.continuous_columns)
    canonical_order.extend(manifest.event_columns)
    canonical_order.extend(_union_feature_columns(manifest.feature_groups))
    canonical_order.extend(sorted(declared_set))
    canonical = _stable_unique(canonical_order)
    canonical = [col for col in canonical if col in declared_set]

    missing = sorted(col for col in declared_set if col not in set(canonical))
    if missing:
        warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_COLUMN_ORDER_NOT_EVALUATED,
                message="Canonical fallback order could not include all declared columns.",
                context={"missing_columns": missing},
            )
        )
        return None, None, warnings

    return tuple(canonical), "repo_convention_fallback", warnings


def _validate_single_parquet(
    *,
    parquet_path: Path,
    manifest: ManifestContract,
    strict_extra_columns: bool,
    strict_column_order: bool,
    canonical_order: tuple[str, ...] | None,
    canonical_order_source: str | None,
    expected_timeframe_seconds: int | None,
    expected_timeframe_source: str | None,
) -> FileValidationReport:
    file_report = FileValidationReport(input_file=str(parquet_path))
    declared_columns = list(manifest.column_dtypes.keys())
    declared_set = set(declared_columns)
    if manifest.timestamp_column not in declared_set:
        declared_set.add(manifest.timestamp_column)
        declared_columns = [manifest.timestamp_column, *declared_columns]

    try:
        frame = pd.read_parquet(parquet_path)
    except (OSError, ValueError, RuntimeError) as exc:
        file_report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_RUNTIME_ERROR,
                message="Failed to read parquet file.",
                context={"input_file": str(parquet_path), "error": str(exc)},
            )
        )
        return file_report

    file_report.rows = int(len(frame))
    observed_columns = list(frame.columns)
    file_report.schema_hash_actual = _hash_sequence(sorted(observed_columns))
    file_report.column_order_hash_actual = _hash_sequence(observed_columns)
    file_report.dtype_hash_actual = _hash_mapping({col: str(dtype) for col, dtype in frame.dtypes.items()})

    if frame.empty:
        file_report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_EMPTY_FILE,
                message="Parquet dataset is empty.",
                context={"input_file": str(parquet_path)},
            )
        )

    missing_required = sorted(col for col in declared_set if col not in frame.columns)
    file_report.required_columns_ok = len(missing_required) == 0
    if missing_required:
        file_report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_REQUIRED_COLUMN_MISSING,
                message="Required manifest columns missing in parquet file.",
                context={"missing_columns": missing_required},
            )
        )

    unexpected_columns = sorted(col for col in frame.columns if col not in declared_set)
    file_report.unexpected_columns_ok = len(unexpected_columns) == 0
    if unexpected_columns:
        issue = ValidationIssue(
            code=TRAIN_INPUT_UNEXPECTED_COLUMNS,
            message="Unexpected columns detected in parquet file.",
            context={"unexpected_columns": unexpected_columns, "strict_extra_columns": strict_extra_columns},
        )
        if strict_extra_columns:
            file_report.errors.append(issue)
        else:
            file_report.warnings.append(issue)

    timestamp_series: pd.Series | None = None
    if manifest.timestamp_column not in frame.columns:
        file_report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_TIMESTAMP_INVALID,
                message="Timestamp column missing from parquet file.",
                context={"timestamp_column": manifest.timestamp_column},
            )
        )
    else:
        parsed_timestamp = pd.to_datetime(frame[manifest.timestamp_column], utc=True, errors="coerce")
        file_report.timestamp_ok = bool(parsed_timestamp.notna().all())
        if not file_report.timestamp_ok:
            file_report.errors.append(
                ValidationIssue(
                    code=TRAIN_INPUT_TIMESTAMP_INVALID,
                    message="Timestamp values are invalid or not UTC-compatible.",
                    context={"timestamp_column": manifest.timestamp_column},
                )
            )
        else:
            timestamp_series = parsed_timestamp
            file_report.timestamp_min_utc = parsed_timestamp.min().isoformat()
            file_report.timestamp_max_utc = parsed_timestamp.max().isoformat()
            file_report.monotonic_ok = bool(parsed_timestamp.is_monotonic_increasing)
            file_report.unique_ts_ok = bool(parsed_timestamp.is_unique)
            if not file_report.monotonic_ok:
                file_report.errors.append(
                    ValidationIssue(
                        code=TRAIN_INPUT_NON_MONOTONIC,
                        message="Timestamp column is not monotonic increasing.",
                        context={"timestamp_column": manifest.timestamp_column},
                    )
                )
            if not file_report.unique_ts_ok:
                file_report.errors.append(
                    ValidationIssue(
                        code=TRAIN_INPUT_DUPLICATE_TIMESTAMP,
                        message="Timestamp column contains duplicates.",
                        context={"timestamp_column": manifest.timestamp_column},
                    )
                )

    dtype_mismatches: dict[str, dict[str, str]] = {}
    for column_name, expected_dtype in manifest.column_dtypes.items():
        if column_name not in frame.columns:
            continue
        observed_dtype = str(frame[column_name].dtype)
        if _normalize_dtype_name(observed_dtype) != _normalize_dtype_name(expected_dtype):
            dtype_mismatches[column_name] = {"expected": expected_dtype, "observed": observed_dtype}

    file_report.dtype_ok = len(dtype_mismatches) == 0
    if dtype_mismatches:
        file_report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_DTYPE_MISMATCH,
                message="Parquet dtype drift detected against manifest contract.",
                context={"dtype_mismatches": dtype_mismatches},
            )
        )

    group_missing_by_name: dict[str, list[str]] = {}
    for group_name, group_columns in manifest.feature_groups.items():
        missing_group_columns = sorted(col for col in group_columns if col not in frame.columns)
        if missing_group_columns:
            group_missing_by_name[group_name] = missing_group_columns

    file_report.feature_group_contract_ok = len(group_missing_by_name) == 0
    if not file_report.feature_group_contract_ok:
        file_report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_FEATURE_GROUP_CONTRACT_INVALID,
                message="Feature group contract inconsistent with parquet columns.",
                context={"missing_by_feature_group": group_missing_by_name},
            )
        )

    file_report.schema_ok = file_report.required_columns_ok and (
        file_report.unexpected_columns_ok or not strict_extra_columns
    )

    _evaluate_column_order(
        file_report=file_report,
        observed_columns=observed_columns,
        declared_columns=declared_columns,
        extra_columns=unexpected_columns,
        strict_extra_columns=strict_extra_columns,
        strict_column_order=strict_column_order,
        canonical_order=canonical_order,
        canonical_order_source=canonical_order_source,
    )

    _evaluate_timeframe(
        file_report=file_report,
        timestamp_series=timestamp_series,
        expected_timeframe_seconds=expected_timeframe_seconds,
        expected_timeframe_source=expected_timeframe_source,
    )

    file_report.status = "success" if len(file_report.errors) == 0 else "failed"
    return file_report


def _evaluate_column_order(
    *,
    file_report: FileValidationReport,
    observed_columns: list[str],
    declared_columns: list[str],
    extra_columns: list[str],
    strict_extra_columns: bool,
    strict_column_order: bool,
    canonical_order: tuple[str, ...] | None,
    canonical_order_source: str | None,
) -> None:
    if canonical_order is None:
        file_report.column_order_ok = None
        file_report.column_order_evaluated = False
        file_report.column_order_severity = "warning"
        file_report.warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_COLUMN_ORDER_NOT_EVALUATED,
                message="Column order check skipped because canonical source is unavailable.",
                context={"strict_column_order": strict_column_order},
            )
        )
        return

    if not file_report.required_columns_ok:
        file_report.column_order_ok = None
        file_report.column_order_evaluated = False
        file_report.column_order_severity = "warning"
        file_report.warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_COLUMN_ORDER_NOT_EVALUATED,
                message="Column order check skipped due to missing required columns.",
                context={},
            )
        )
        return

    expected_order = list(canonical_order)
    if not strict_extra_columns and extra_columns:
        expected_order = [*expected_order, *sorted(extra_columns)]

    file_report.column_order_evaluated = True
    file_report.column_order_ok = observed_columns == expected_order
    if file_report.column_order_ok:
        file_report.column_order_severity = "none"
        return

    issue = ValidationIssue(
        code=TRAIN_INPUT_COLUMN_ORDER_DRIFT,
        message="Parquet column order drift detected.",
        context={
            "canonical_order_source": canonical_order_source,
            "expected_order": expected_order,
            "observed_order": observed_columns,
            "strict_column_order": strict_column_order,
        },
    )
    if strict_column_order:
        file_report.column_order_severity = "error"
        file_report.errors.append(issue)
    else:
        file_report.column_order_severity = "warning"
        file_report.warnings.append(issue)


def _evaluate_timeframe(
    *,
    file_report: FileValidationReport,
    timestamp_series: pd.Series | None,
    expected_timeframe_seconds: int | None,
    expected_timeframe_source: str | None,
) -> None:
    if timestamp_series is None:
        file_report.timeframe_consistency_ok = None
        file_report.observed_median_delta_seconds = None
        return

    deltas = timestamp_series.diff().dt.total_seconds().dropna()
    if deltas.empty:
        file_report.timeframe_consistency_ok = None
        file_report.observed_median_delta_seconds = None
        return

    median_delta = float(deltas.median())
    file_report.observed_median_delta_seconds = median_delta

    if expected_timeframe_seconds is None:
        file_report.timeframe_consistency_ok = None
        file_report.warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_TIMEFRAME_UNKNOWN,
                message="Expected timeframe is unknown; consistency check skipped.",
                context={"observed_median_delta_seconds": median_delta},
            )
        )
        return

    tolerance = max(1.0, float(expected_timeframe_seconds) * TIMEFRAME_TOLERANCE_RATIO)
    consistent = abs(median_delta - float(expected_timeframe_seconds)) <= tolerance
    file_report.timeframe_consistency_ok = bool(consistent)
    if not consistent:
        file_report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_TIMEFRAME_INCONSISTENT,
                message="Observed median delta is inconsistent with expected timeframe.",
                context={
                    "expected_timeframe_seconds": expected_timeframe_seconds,
                    "expected_timeframe_source": expected_timeframe_source,
                    "observed_median_delta_seconds": median_delta,
                    "tolerance_seconds": tolerance,
                },
            )
        )


def _resolve_expected_timeframe_seconds(
    *,
    cli_expected_timeframe: str | None,
    manifest: ManifestContract,
    file_path: Path | None,
) -> tuple[int | None, str | None]:
    if cli_expected_timeframe:
        parsed = _parse_timeframe_to_seconds(cli_expected_timeframe)
        if parsed is not None:
            return parsed, "cli"

    if manifest.expected_timeframe:
        parsed = _parse_timeframe_to_seconds(manifest.expected_timeframe)
        if parsed is not None:
            return parsed, "manifest"

    if file_path is not None:
        inferred = _infer_timeframe_seconds_from_path(file_path)
        if inferred is not None:
            return inferred, "filename_or_folder"

    return None, None


def _parse_timeframe_to_seconds(value: str) -> int | None:
    match = TIMEFRAME_PATTERN.match(value)
    if not match:
        return None
    number = int(match.group(1))
    unit = match.group(2).lower()
    unit_seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return number * unit_seconds[unit]


def _infer_timeframe_seconds_from_path(path: Path) -> int | None:
    path_text = "/".join(path.resolve().parts).lower()
    matches = TIMEFRAME_INFER_PATTERN.findall(path_text)
    if not matches:
        return None
    number_str, unit = matches[-1]
    number = int(number_str)
    unit_seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return number * unit_seconds[unit.lower()]


def _normalize_dtype_name(dtype_name: str) -> str:
    return dtype_name.replace(" ", "").lower()


def _stable_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _hash_sequence(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), sort_keys=False, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_mapping(values: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(values.items())), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _populate_aggregate_hashes(report: TrainInputValidationReport) -> None:
    if not report.file_reports:
        report.schema_hash_actual = None
        report.dtype_hash_actual = None
        report.column_order_hash_actual = None
        return

    schema_hash_map = {item.input_file: item.schema_hash_actual for item in report.file_reports if item.schema_hash_actual}
    dtype_hash_map = {item.input_file: item.dtype_hash_actual for item in report.file_reports if item.dtype_hash_actual}
    order_hash_map = {item.input_file: item.column_order_hash_actual for item in report.file_reports if item.column_order_hash_actual}

    report.schema_hash_actual = _hash_mapping({k: v for k, v in sorted(schema_hash_map.items())}) if schema_hash_map else None
    report.dtype_hash_actual = _hash_mapping({k: v for k, v in sorted(dtype_hash_map.items())}) if dtype_hash_map else None
    report.column_order_hash_actual = _hash_mapping({k: v for k, v in sorted(order_hash_map.items())}) if order_hash_map else None


def _validate_manifest_hash_comparisons(report: TrainInputValidationReport, manifest_payload: dict[str, Any] | None) -> None:
    if not isinstance(manifest_payload, dict):
        return

    schema_manifest_hash = _pick_first_string(manifest_payload, ("schema_hash", "schema_hash_actual"))
    dtype_manifest_hash = _pick_first_string(manifest_payload, ("dtype_hash", "dtype_hash_actual"))
    order_manifest_hash = _pick_first_string(manifest_payload, ("column_order_hash", "column_order_hash_actual"))

    if schema_manifest_hash is None:
        report.warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_HASH_MISSING,
                message="Manifest schema hash is not present; schema hash comparison skipped.",
                context={},
            )
        )
    elif report.schema_hash_actual is not None and schema_manifest_hash != report.schema_hash_actual:
        report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_MANIFEST_SCHEMA_INVALID,
                message="Manifest schema hash mismatch detected.",
                context={"expected": schema_manifest_hash, "observed": report.schema_hash_actual},
            )
        )

    if dtype_manifest_hash is None:
        report.warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_HASH_MISSING,
                message="Manifest dtype hash is not present; dtype hash comparison skipped.",
                context={},
            )
        )
    elif report.dtype_hash_actual is not None and dtype_manifest_hash != report.dtype_hash_actual:
        report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_DTYPE_MISMATCH,
                message="Manifest dtype hash mismatch detected.",
                context={"expected": dtype_manifest_hash, "observed": report.dtype_hash_actual},
            )
        )

    if order_manifest_hash is None:
        report.warnings.append(
            ValidationIssue(
                code=TRAIN_INPUT_HASH_MISSING,
                message="Manifest column order hash is not present; order hash comparison skipped.",
                context={},
            )
        )
    elif report.column_order_hash_actual is not None and order_manifest_hash != report.column_order_hash_actual:
        report.errors.append(
            ValidationIssue(
                code=TRAIN_INPUT_COLUMN_ORDER_DRIFT,
                message="Manifest column order hash mismatch detected.",
                context={"expected": order_manifest_hash, "observed": report.column_order_hash_actual},
            )
        )


def _pick_first_string(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _finalize_report(report: TrainInputValidationReport) -> None:
    report.succeeded_files = sum(1 for item in report.file_reports if item.status == "success")
    report.failed_files = report.total_files - report.succeeded_files
    report.train_input_validation_overall = bool(
        report.manifest_valid
        and report.total_files > 0
        and report.failed_files == 0
        and len(report.errors) == 0
    )
