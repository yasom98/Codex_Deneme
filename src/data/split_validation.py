"""Validation layer for RL dataset split contract checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from core.logging import get_logger

LOGGER = get_logger(__name__)

VALIDATOR_VERSION = "split_validator.v1"
SUPPORTED_SPLIT_MODES: tuple[str, ...] = ("ratio_chrono", "explicit_ranges", "walk_forward")
TIMEFRAME_TOLERANCE_RATIO = 0.05
TIMEFRAME_PATTERN = re.compile(r"^\s*(\d+)\s*([smhdSMHD])\s*$")
TIMEFRAME_INFER_PATTERN = re.compile(r"(?<!\d)(\d+)([smhdSMHD])(?![a-zA-Z])")

SPLIT_SPEC_INVALID = "SPLIT_SPEC_INVALID"
SPLIT_MODE_UNSUPPORTED = "SPLIT_MODE_UNSUPPORTED"
SPLIT_RATIO_INVALID = "SPLIT_RATIO_INVALID"
SPLIT_EXPLICIT_RANGE_INVALID = "SPLIT_EXPLICIT_RANGE_INVALID"
SPLIT_WALKFORWARD_PARAM_INVALID = "SPLIT_WALKFORWARD_PARAM_INVALID"
SPLIT_MANIFEST_MISSING = "SPLIT_MANIFEST_MISSING"
SPLIT_MANIFEST_INVALID = "SPLIT_MANIFEST_INVALID"
SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_MISSING = "SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_MISSING"
SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_INVALID = "SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_INVALID"
SPLIT_TRAIN_INPUT_VALIDATION_RUN_ID_MISMATCH = "SPLIT_TRAIN_INPUT_VALIDATION_RUN_ID_MISMATCH"
SPLIT_TRAIN_INPUT_VALIDATION_FAILED = "SPLIT_TRAIN_INPUT_VALIDATION_FAILED"
SPLIT_NO_PARQUET_FILES = "SPLIT_NO_PARQUET_FILES"
SPLIT_TIMESTAMP_COLUMN_MISSING = "SPLIT_TIMESTAMP_COLUMN_MISSING"
SPLIT_TIMESTAMP_INVALID = "SPLIT_TIMESTAMP_INVALID"
SPLIT_EMPTY_PARTITION = "SPLIT_EMPTY_PARTITION"
SPLIT_MIN_ROWS_NOT_MET = "SPLIT_MIN_ROWS_NOT_MET"
SPLIT_ORDERING_INVALID = "SPLIT_ORDERING_INVALID"
SPLIT_OVERLAP_DETECTED = "SPLIT_OVERLAP_DETECTED"
SPLIT_EMBARGO_VIOLATION = "SPLIT_EMBARGO_VIOLATION"
SPLIT_WARMUP_INSUFFICIENT = "SPLIT_WARMUP_INSUFFICIENT"
SPLIT_COVERAGE_INVALID = "SPLIT_COVERAGE_INVALID"
SPLIT_FOLD_NON_MONOTONIC = "SPLIT_FOLD_NON_MONOTONIC"
SPLIT_TIMEFRAME_INCONSISTENT = "SPLIT_TIMEFRAME_INCONSISTENT"
SPLIT_RUNTIME_ERROR = "SPLIT_RUNTIME_ERROR"


@dataclass
class ValidationIssue:
    """Machine-readable issue payload."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestContract:
    """Manifest fields required by split validation."""

    run_id: str
    timestamp_column: str
    core_feature_columns: tuple[str, ...]
    expected_timeframe: str | None


@dataclass
class SplitValidationOptions:
    """Runtime options for split validation."""

    run_id: str
    input_root: Path
    reports_root: Path
    split_mode: str | None = None
    split_config_path: Path | None = None
    split_overrides: dict[str, Any] = field(default_factory=dict)
    require_train_input_validation: bool = True
    min_train_rows: int = 1
    min_val_rows: int = 1
    min_test_rows: int = 1
    embargo_bars: int | None = None
    embargo_seconds: int | None = None
    warmup_rows: int = 0
    timestamp_column_override: str | None = None
    expected_timeframe: str | None = None

    def to_invocation_args(self) -> dict[str, Any]:
        """Serialize invocation args into JSON-friendly payload."""

        return {
            "run_id": self.run_id,
            "input_root": str(self.input_root),
            "reports_root": str(self.reports_root),
            "split_mode": self.split_mode,
            "split_config_path": str(self.split_config_path) if self.split_config_path is not None else None,
            "split_overrides": dict(self.split_overrides),
            "require_train_input_validation": bool(self.require_train_input_validation),
            "min_train_rows": int(self.min_train_rows),
            "min_val_rows": int(self.min_val_rows),
            "min_test_rows": int(self.min_test_rows),
            "embargo_bars": self.embargo_bars,
            "embargo_seconds": self.embargo_seconds,
            "warmup_rows": int(self.warmup_rows),
            "timestamp_column_override": self.timestamp_column_override,
            "expected_timeframe": self.expected_timeframe,
        }


@dataclass(frozen=True)
class SplitSpec:
    """Normalized split specification payload."""

    mode: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class PartitionWindow:
    """Half-open row-window representation [start, end)."""

    start: int
    end: int


@dataclass
class FileSplitReport:
    """Per-file split validation payload."""

    input_file: str
    status: str = "failed"
    timestamp_column: str | None = None
    rows_total: int = 0
    timestamp_min_utc: str | None = None
    timestamp_max_utc: str | None = None
    split_mode: str | None = None
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    train_range: dict[str, Any] | None = None
    val_range: dict[str, Any] | None = None
    test_range: dict[str, Any] | None = None
    ordering_ok: bool = False
    non_overlap_ok: bool = False
    embargo_ok: bool = False
    warmup_ok: bool = False
    min_rows_ok: bool = False
    split_coverage_ok: bool = False
    timeframe_consistency_ok: bool | None = None
    observed_median_delta_seconds: float | None = None
    warmup_detail: dict[str, Any] = field(default_factory=dict)
    fold_count: int = 0
    failed_fold_count: int = 0
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dict."""

        return asdict(self)


@dataclass
class FoldSplitReport:
    """Per-fold payload for walk-forward validation."""

    fold_id: int
    input_file: str
    train_range: dict[str, Any] | None = None
    val_range: dict[str, Any] | None = None
    test_range: dict[str, Any] | None = None
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    ordering_ok: bool = False
    non_overlap_ok: bool = False
    embargo_ok: bool = False
    warmup_ok: bool = False
    min_rows_ok: bool = False
    fold_ok: bool = False
    warmup_detail: dict[str, Any] = field(default_factory=dict)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dict."""

        return asdict(self)


@dataclass
class SplitValidationReport:
    """Run-level split validation payload."""

    generated_at_utc: str
    run_id: str
    validator_version: str = VALIDATOR_VERSION
    split_mode: str | None = None
    split_spec: dict[str, Any] = field(default_factory=dict)
    manifest_path: str = ""
    manifest_loaded: bool = False
    manifest_valid: bool = False
    train_input_validation_report_path: str | None = None
    train_input_validation_checked: bool = False
    train_input_validation_required: bool = True
    train_input_validation_overall_seen: bool | None = None
    total_files: int = 0
    succeeded_files: int = 0
    failed_files: int = 0
    split_validation_overall: bool = False
    embargo_policy: dict[str, Any] = field(default_factory=dict)
    warmup_policy: dict[str, Any] = field(default_factory=dict)
    boundary_policy: dict[str, Any] = field(default_factory=dict)
    invocation_args: dict[str, Any] = field(default_factory=dict)
    file_reports: list[FileSplitReport] = field(default_factory=list)
    fold_reports: list[FoldSplitReport] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dict."""

        return asdict(self)


def validate_splits(options: SplitValidationOptions) -> SplitValidationReport:
    """Validate dataset split definitions against leak-free RL contract rules."""

    _validate_options(options)
    report = SplitValidationReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_id=options.run_id,
        manifest_path=str(options.reports_root / "feature_manifest.json"),
        train_input_validation_report_path=str(options.reports_root / "train_input_validation_report.json"),
        train_input_validation_required=bool(options.require_train_input_validation),
        invocation_args=options.to_invocation_args(),
        embargo_policy=_build_embargo_policy(options.embargo_bars, options.embargo_seconds),
        warmup_policy={
            "warmup_rows": int(options.warmup_rows),
            "note": "Warmup is availability-only and not counted in partition row totals.",
        },
        boundary_policy={
            "input_start": "inclusive",
            "input_end": "inclusive",
            "normalized_internal": "[start, end)",
            "boundary_equal_timestamp_assignment": "left_partition_only; overlap is invalid",
        },
    )

    manifest_payload, manifest_contract, manifest_errors, manifest_warnings = _load_manifest_contract(
        manifest_path=Path(report.manifest_path),
        expected_run_id=options.run_id,
        timestamp_override=options.timestamp_column_override,
    )
    report.manifest_loaded = manifest_payload is not None
    report.manifest_valid = manifest_contract is not None and len(manifest_errors) == 0
    report.errors.extend(manifest_errors)
    report.warnings.extend(manifest_warnings)

    split_spec, spec_errors, spec_warnings = _load_and_normalize_split_spec(options)
    report.errors.extend(spec_errors)
    report.warnings.extend(spec_warnings)
    if split_spec is not None:
        report.split_mode = split_spec.mode
        report.split_spec = dict(split_spec.payload)

    (
        train_input_checked,
        train_input_overall,
        train_input_errors,
        train_input_warnings,
    ) = _check_train_input_validation_report(
        report_path=Path(report.train_input_validation_report_path),
        required=bool(options.require_train_input_validation),
        expected_run_id=options.run_id,
    )
    report.train_input_validation_checked = train_input_checked
    report.train_input_validation_overall_seen = train_input_overall
    report.errors.extend(train_input_errors)
    report.warnings.extend(train_input_warnings)

    parquet_files = discover_parquet_files(options.input_root)
    report.total_files = len(parquet_files)
    if not parquet_files:
        report.errors.append(
            ValidationIssue(
                code=SPLIT_NO_PARQUET_FILES,
                message="No parquet files found under input root.",
                context={"input_root": str(options.input_root)},
            )
        )

    for parquet_path in parquet_files:
        if manifest_contract is None or split_spec is None:
            file_report = FileSplitReport(
                input_file=str(parquet_path),
                split_mode=split_spec.mode if split_spec is not None else None,
                timestamp_column=(manifest_contract.timestamp_column if manifest_contract is not None else options.timestamp_column_override),
            )
            file_report.errors.append(
                ValidationIssue(
                    code=SPLIT_SPEC_INVALID,
                    message="Validation preconditions not met; per-file split checks skipped.",
                    context={"manifest_valid": bool(manifest_contract is not None), "split_spec_valid": bool(split_spec is not None)},
                )
            )
            report.file_reports.append(file_report)
            continue

        file_report, fold_reports = _validate_single_file(
            parquet_path=parquet_path,
            manifest=manifest_contract,
            split_spec=split_spec,
            min_train_rows=int(options.min_train_rows),
            min_val_rows=int(options.min_val_rows),
            min_test_rows=int(options.min_test_rows),
            embargo_bars=options.embargo_bars,
            embargo_seconds=options.embargo_seconds,
            warmup_rows=int(options.warmup_rows),
            cli_expected_timeframe=options.expected_timeframe,
        )
        report.file_reports.append(file_report)
        report.fold_reports.extend(fold_reports)

    _finalize_report(report)
    return report


def discover_parquet_files(input_root: Path) -> list[Path]:
    """Discover parquet files recursively under input root."""

    if not input_root.exists() or not input_root.is_dir():
        return []
    files = [path for path in input_root.glob("**/*.parquet") if path.is_file() and path.suffix.lower() == ".parquet"]
    return sorted(files)


def _validate_options(options: SplitValidationOptions) -> None:
    if not options.run_id.strip():
        raise ValueError("run_id must be non-empty")
    if options.min_train_rows < 0 or options.min_val_rows < 0 or options.min_test_rows < 0:
        raise ValueError("min row thresholds must be >= 0")
    if options.warmup_rows < 0:
        raise ValueError("warmup_rows must be >= 0")
    if options.embargo_bars is not None and options.embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")
    if options.embargo_seconds is not None and options.embargo_seconds < 0:
        raise ValueError("embargo_seconds must be >= 0")
    if options.embargo_bars is not None and options.embargo_seconds is not None:
        raise ValueError("embargo_bars and embargo_seconds are mutually exclusive")


def _build_embargo_policy(embargo_bars: int | None, embargo_seconds: int | None) -> dict[str, Any]:
    if embargo_bars is not None:
        return {"mode": "bars", "value": int(embargo_bars)}
    if embargo_seconds is not None:
        return {"mode": "seconds", "value": int(embargo_seconds)}
    return {"mode": "none", "value": 0}


def _load_manifest_contract(
    *,
    manifest_path: Path,
    expected_run_id: str,
    timestamp_override: str | None,
) -> tuple[dict[str, Any] | None, ManifestContract | None, list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if not manifest_path.exists():
        errors.append(
            ValidationIssue(
                code=SPLIT_MANIFEST_MISSING,
                message="feature_manifest.json not found.",
                context={"manifest_path": str(manifest_path)},
            )
        )
        return None, None, errors, warnings

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(
            ValidationIssue(
                code=SPLIT_MANIFEST_INVALID,
                message="feature_manifest.json is invalid.",
                context={"manifest_path": str(manifest_path), "error": str(exc)},
            )
        )
        return None, None, errors, warnings

    if not isinstance(payload, dict):
        errors.append(
            ValidationIssue(
                code=SPLIT_MANIFEST_INVALID,
                message="Manifest payload must be a JSON object.",
                context={"manifest_path": str(manifest_path)},
            )
        )
        return payload, None, errors, warnings

    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append(
            ValidationIssue(
                code=SPLIT_MANIFEST_INVALID,
                message="Manifest run_id is missing or invalid.",
                context={"run_id": run_id},
            )
        )
        return payload, None, errors, warnings
    if run_id != expected_run_id:
        errors.append(
            ValidationIssue(
                code=SPLIT_MANIFEST_INVALID,
                message="Manifest run_id mismatch.",
                context={"expected_run_id": expected_run_id, "manifest_run_id": run_id},
            )
        )

    timestamp_column_raw = payload.get("timestamp_column")
    if timestamp_override is not None and timestamp_override.strip():
        timestamp_column = timestamp_override.strip()
    elif isinstance(timestamp_column_raw, str) and timestamp_column_raw.strip():
        timestamp_column = timestamp_column_raw.strip()
    else:
        errors.append(
            ValidationIssue(
                code=SPLIT_MANIFEST_INVALID,
                message="Manifest timestamp_column is missing.",
                context={"timestamp_column": timestamp_column_raw},
            )
        )
        return payload, None, errors, warnings

    continuous_columns = _parse_string_list(payload.get("continuous_columns"))
    event_columns = _parse_string_list(payload.get("event_columns"))
    if continuous_columns is None or event_columns is None:
        errors.append(
            ValidationIssue(
                code=SPLIT_MANIFEST_INVALID,
                message="Manifest continuous_columns/event_columns must be list[str].",
                context={},
            )
        )
        return payload, None, errors, warnings

    core_columns = tuple(_stable_unique([*continuous_columns, *event_columns]))
    if not core_columns:
        errors.append(
            ValidationIssue(
                code=SPLIT_MANIFEST_INVALID,
                message="Manifest core feature columns cannot be empty.",
                context={},
            )
        )
        return payload, None, errors, warnings

    expected_timeframe = _parse_optional_string(payload.get("expected_timeframe"))
    if expected_timeframe is None:
        expected_timeframe = _parse_optional_string(payload.get("timeframe"))

    if errors:
        return payload, None, errors, warnings

    return (
        payload,
        ManifestContract(
            run_id=run_id,
            timestamp_column=timestamp_column,
            core_feature_columns=core_columns,
            expected_timeframe=expected_timeframe,
        ),
        errors,
        warnings,
    )


def _load_and_normalize_split_spec(
    options: SplitValidationOptions,
) -> tuple[SplitSpec | None, list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    raw_spec: dict[str, Any] = {}
    if options.split_config_path is not None:
        config_payload, config_error = _load_split_config(options.split_config_path)
        if config_error is not None:
            errors.append(config_error)
            return None, errors, warnings
        raw_spec.update(config_payload)

    raw_spec.update({k: v for k, v in options.split_overrides.items() if v is not None})

    mode_value = options.split_mode if options.split_mode is not None else raw_spec.get("split_mode", raw_spec.get("mode"))
    mode = _parse_optional_string(mode_value)
    if mode is None:
        errors.append(
            ValidationIssue(
                code=SPLIT_SPEC_INVALID,
                message="split_mode is required.",
                context={"supported_modes": list(SUPPORTED_SPLIT_MODES)},
            )
        )
        return None, errors, warnings

    if mode not in SUPPORTED_SPLIT_MODES:
        errors.append(
            ValidationIssue(
                code=SPLIT_MODE_UNSUPPORTED,
                message="split_mode is not supported.",
                context={"split_mode": mode, "supported_modes": list(SUPPORTED_SPLIT_MODES)},
            )
        )
        return None, errors, warnings

    if mode == "ratio_chrono":
        return _normalize_ratio_spec(mode, raw_spec)
    if mode == "explicit_ranges":
        return _normalize_explicit_spec(mode, raw_spec)
    return _normalize_walk_forward_spec(mode, raw_spec)


def _load_split_config(path: Path) -> tuple[dict[str, Any], ValidationIssue | None]:
    if not path.exists():
        return {}, ValidationIssue(
            code=SPLIT_SPEC_INVALID,
            message="split config file not found.",
            context={"split_config_path": str(path)},
        )

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, ValidationIssue(
            code=SPLIT_SPEC_INVALID,
            message="split config file could not be read.",
            context={"split_config_path": str(path), "error": str(exc)},
        )

    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(raw)
        else:
            payload = yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        return {}, ValidationIssue(
            code=SPLIT_SPEC_INVALID,
            message="split config parse failed.",
            context={"split_config_path": str(path), "error": str(exc)},
        )

    if not isinstance(payload, dict):
        return {}, ValidationIssue(
            code=SPLIT_SPEC_INVALID,
            message="split config must be an object.",
            context={"split_config_path": str(path)},
        )
    return dict(payload), None


def _normalize_ratio_spec(
    mode: str,
    raw_spec: Mapping[str, Any],
) -> tuple[SplitSpec | None, list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    train_ratio = _parse_float(raw_spec.get("train_ratio"))
    val_ratio = _parse_float(raw_spec.get("val_ratio"))
    test_ratio = _parse_float(raw_spec.get("test_ratio"))

    if train_ratio is None or val_ratio is None or test_ratio is None:
        errors.append(
            ValidationIssue(
                code=SPLIT_RATIO_INVALID,
                message="train_ratio, val_ratio, test_ratio are required numeric values.",
                context={
                    "train_ratio": raw_spec.get("train_ratio"),
                    "val_ratio": raw_spec.get("val_ratio"),
                    "test_ratio": raw_spec.get("test_ratio"),
                },
            )
        )
        return None, errors, warnings

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(value < 0.0 for value in ratios.values()):
        errors.append(
            ValidationIssue(
                code=SPLIT_RATIO_INVALID,
                message="Ratios must be >= 0.",
                context={"ratios": ratios},
            )
        )
        return None, errors, warnings

    ratio_sum = train_ratio + val_ratio + test_ratio
    tolerance = 1e-6
    if abs(ratio_sum - 1.0) > tolerance:
        errors.append(
            ValidationIssue(
                code=SPLIT_RATIO_INVALID,
                message="Ratios must sum to 1.0 within tolerance.",
                context={"ratio_sum": ratio_sum, "tolerance": tolerance},
            )
        )
        return None, errors, warnings

    payload = {
        "mode": mode,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "ratio_sum": ratio_sum,
        "ratio_tolerance": tolerance,
        "rounding_policy": "floor_then_largest_remainder",
        "remainder_tie_breaker": ["train", "val", "test"],
        "coverage_policy": "full_required",
    }
    return SplitSpec(mode=mode, payload=payload), errors, warnings


def _normalize_explicit_spec(
    mode: str,
    raw_spec: Mapping[str, Any],
) -> tuple[SplitSpec | None, list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    train_start = _parse_utc_timestamp(raw_spec.get("train_start"))
    train_end = _parse_utc_timestamp(raw_spec.get("train_end"))
    if train_start is None or train_end is None:
        errors.append(
            ValidationIssue(
                code=SPLIT_EXPLICIT_RANGE_INVALID,
                message="train_start and train_end are required UTC-compatible timestamps.",
                context={"train_start": raw_spec.get("train_start"), "train_end": raw_spec.get("train_end")},
            )
        )
        return None, errors, warnings

    payload: dict[str, Any] = {
        "mode": mode,
        "boundary_policy": "input_inclusive_normalized_to_half_open",
        "coverage_policy": "full_not_required",
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
    }

    for name in ("val", "test"):
        start_key = f"{name}_start"
        end_key = f"{name}_end"
        raw_start = raw_spec.get(start_key)
        raw_end = raw_spec.get(end_key)
        has_start = raw_start is not None
        has_end = raw_end is not None
        if has_start != has_end:
            errors.append(
                ValidationIssue(
                    code=SPLIT_EXPLICIT_RANGE_INVALID,
                    message="Explicit range requires complete start/end pair.",
                    context={"partition": name, "start": raw_start, "end": raw_end},
                )
            )
            return None, errors, warnings
        if not has_start:
            payload[start_key] = None
            payload[end_key] = None
            continue

        start_ts = _parse_utc_timestamp(raw_start)
        end_ts = _parse_utc_timestamp(raw_end)
        if start_ts is None or end_ts is None:
            errors.append(
                ValidationIssue(
                    code=SPLIT_EXPLICIT_RANGE_INVALID,
                    message="Explicit range timestamps must be UTC-compatible.",
                    context={"partition": name, "start": raw_start, "end": raw_end},
                )
            )
            return None, errors, warnings
        payload[start_key] = start_ts.isoformat()
        payload[end_key] = end_ts.isoformat()

    ordered_ranges = [
        ("train", train_start, train_end),
    ]
    val_start_raw = payload.get("val_start")
    val_end_raw = payload.get("val_end")
    test_start_raw = payload.get("test_start")
    test_end_raw = payload.get("test_end")

    if isinstance(val_start_raw, str) and isinstance(val_end_raw, str):
        val_start = _parse_utc_timestamp(val_start_raw)
        val_end = _parse_utc_timestamp(val_end_raw)
        if val_start is None or val_end is None:
            raise ValueError("internal explicit timestamp parsing failed for val range")
        ordered_ranges.append(("val", val_start, val_end))
    if isinstance(test_start_raw, str) and isinstance(test_end_raw, str):
        test_start = _parse_utc_timestamp(test_start_raw)
        test_end = _parse_utc_timestamp(test_end_raw)
        if test_start is None or test_end is None:
            raise ValueError("internal explicit timestamp parsing failed for test range")
        ordered_ranges.append(("test", test_start, test_end))

    for name, start_ts, end_ts in ordered_ranges:
        if end_ts < start_ts:
            errors.append(
                ValidationIssue(
                    code=SPLIT_EXPLICIT_RANGE_INVALID,
                    message="Range end must be >= start.",
                    context={"partition": name, "start": start_ts.isoformat(), "end": end_ts.isoformat()},
                )
            )
            return None, errors, warnings

    return SplitSpec(mode=mode, payload=payload), errors, warnings


def _normalize_walk_forward_spec(
    mode: str,
    raw_spec: Mapping[str, Any],
) -> tuple[SplitSpec | None, list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    min_train = _parse_window_value(
        bars_raw=raw_spec.get("min_train_bars", raw_spec.get("train_window_bars")),
        duration_raw=raw_spec.get("min_train_duration", raw_spec.get("train_window_duration")),
        label="min_train",
        errors=errors,
    )
    val_window = _parse_window_value(
        bars_raw=raw_spec.get("val_window_bars"),
        duration_raw=raw_spec.get("val_window_duration"),
        label="val_window",
        errors=errors,
    )
    test_window = _parse_window_value(
        bars_raw=raw_spec.get("test_window_bars"),
        duration_raw=raw_spec.get("test_window_duration"),
        label="test_window",
        errors=errors,
    )
    step_window = _parse_window_value(
        bars_raw=raw_spec.get("step_bars"),
        duration_raw=raw_spec.get("step_duration"),
        label="step",
        errors=errors,
    )

    max_folds_raw = raw_spec.get("max_folds")
    max_folds = _parse_optional_positive_int(max_folds_raw)
    if max_folds_raw is not None and max_folds is None:
        errors.append(
            ValidationIssue(
                code=SPLIT_WALKFORWARD_PARAM_INVALID,
                message="max_folds must be a positive integer.",
                context={"max_folds": max_folds_raw},
            )
        )

    if errors:
        return None, errors, warnings

    payload = {
        "mode": mode,
        "flavor": "expanding_train_fixed_val_test.v1",
        "min_train": min_train,
        "val_window": val_window,
        "test_window": test_window,
        "step": step_window,
        "max_folds": max_folds,
        "coverage_policy": "full_not_required",
    }
    return SplitSpec(mode=mode, payload=payload), errors, warnings


def _parse_window_value(
    *,
    bars_raw: Any,
    duration_raw: Any,
    label: str,
    errors: list[ValidationIssue],
) -> dict[str, Any] | None:
    has_bars = bars_raw is not None
    has_duration = duration_raw is not None
    if has_bars == has_duration:
        errors.append(
            ValidationIssue(
                code=SPLIT_WALKFORWARD_PARAM_INVALID,
                message="Exactly one of bars or duration must be provided for a walk-forward window.",
                context={"window": label, "bars": bars_raw, "duration": duration_raw},
            )
        )
        return None

    if has_bars:
        bars = _parse_optional_positive_int(bars_raw)
        if bars is None:
            errors.append(
                ValidationIssue(
                    code=SPLIT_WALKFORWARD_PARAM_INVALID,
                    message="Window bars must be a positive integer.",
                    context={"window": label, "bars": bars_raw},
                )
            )
            return None
        return {"unit": "bars", "value": bars}

    assert duration_raw is not None
    seconds = _parse_duration_to_seconds(str(duration_raw))
    if seconds is None or seconds <= 0:
        errors.append(
            ValidationIssue(
                code=SPLIT_WALKFORWARD_PARAM_INVALID,
                message="Window duration must follow deterministic pattern like 15m, 1h, 3d.",
                context={"window": label, "duration": duration_raw},
            )
        )
        return None
    return {"unit": "seconds", "value": seconds, "raw": str(duration_raw)}


def _check_train_input_validation_report(
    *,
    report_path: Path,
    required: bool,
    expected_run_id: str,
) -> tuple[bool, bool | None, list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if not report_path.exists():
        if required:
            errors.append(
                ValidationIssue(
                    code=SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_MISSING,
                    message="train_input_validation_report.json is required but missing.",
                    context={"report_path": str(report_path)},
                )
            )
        else:
            warnings.append(
                ValidationIssue(
                    code=SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_MISSING,
                    message="train_input_validation_report.json not found; continuing because requirement is disabled.",
                    context={"report_path": str(report_path), "required": False},
                )
            )
        return True, None, errors, warnings

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issue = ValidationIssue(
            code=SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_INVALID,
            message="train_input_validation_report.json is unreadable or invalid JSON.",
            context={"report_path": str(report_path), "error": str(exc)},
        )
        if required:
            errors.append(issue)
        else:
            warnings.append(issue)
        return True, None, errors, warnings

    if not isinstance(payload, dict):
        issue = ValidationIssue(
            code=SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_INVALID,
            message="train_input_validation_report payload must be an object.",
            context={"report_path": str(report_path)},
        )
        if required:
            errors.append(issue)
        else:
            warnings.append(issue)
        return True, None, errors, warnings

    report_run_id = payload.get("run_id")
    if report_run_id != expected_run_id:
        issue = ValidationIssue(
            code=SPLIT_TRAIN_INPUT_VALIDATION_RUN_ID_MISMATCH,
            message="train_input_validation_report run_id mismatch.",
            context={"expected_run_id": expected_run_id, "report_run_id": report_run_id},
        )
        if required:
            errors.append(issue)
        else:
            warnings.append(issue)

    # TODO: Future hardening: optionally cross-check manifest fingerprints/config hash for stronger prior-report integrity.
    overall = payload.get("train_input_validation_overall")
    if not isinstance(overall, bool):
        issue = ValidationIssue(
            code=SPLIT_TRAIN_INPUT_VALIDATION_REQUIRED_INVALID,
            message="train_input_validation_overall must be boolean.",
            context={"value": overall},
        )
        if required:
            errors.append(issue)
        else:
            warnings.append(issue)
        return True, None, errors, warnings

    if not overall:
        issue = ValidationIssue(
            code=SPLIT_TRAIN_INPUT_VALIDATION_FAILED,
            message="Prior train-input validation failed.",
            context={"train_input_validation_overall": overall},
        )
        if required:
            errors.append(issue)
        else:
            warnings.append(issue)

    return True, overall, errors, warnings


def _validate_single_file(
    *,
    parquet_path: Path,
    manifest: ManifestContract,
    split_spec: SplitSpec,
    min_train_rows: int,
    min_val_rows: int,
    min_test_rows: int,
    embargo_bars: int | None,
    embargo_seconds: int | None,
    warmup_rows: int,
    cli_expected_timeframe: str | None,
) -> tuple[FileSplitReport, list[FoldSplitReport]]:
    file_report = FileSplitReport(
        input_file=str(parquet_path),
        timestamp_column=manifest.timestamp_column,
        split_mode=split_spec.mode,
        non_overlap_ok=True,
        ordering_ok=True,
        embargo_ok=True,
        warmup_ok=True,
        min_rows_ok=True,
        split_coverage_ok=True,
    )
    fold_reports: list[FoldSplitReport] = []

    try:
        frame = pd.read_parquet(parquet_path)
    except (OSError, ValueError, RuntimeError) as exc:
        file_report.errors.append(
            ValidationIssue(
                code=SPLIT_RUNTIME_ERROR,
                message="Failed to read parquet file.",
                context={"input_file": str(parquet_path), "error": str(exc)},
            )
        )
        return file_report, fold_reports

    file_report.rows_total = int(len(frame))

    missing_core_columns = [
        col for col in (manifest.timestamp_column, *manifest.core_feature_columns) if col not in frame.columns
    ]
    if missing_core_columns:
        if manifest.timestamp_column in missing_core_columns:
            file_report.errors.append(
                ValidationIssue(
                    code=SPLIT_TIMESTAMP_COLUMN_MISSING,
                    message="Timestamp column missing from parquet file.",
                    context={"timestamp_column": manifest.timestamp_column},
                )
            )
        remaining = [col for col in missing_core_columns if col != manifest.timestamp_column]
        if remaining:
            file_report.errors.append(
                ValidationIssue(
                    code=SPLIT_MANIFEST_INVALID,
                    message="Core manifest feature columns missing from parquet file.",
                    context={"missing_core_columns": remaining},
                )
            )

    timestamps = _parse_timestamp_series(frame.get(manifest.timestamp_column))
    if timestamps is None:
        file_report.errors.append(
            ValidationIssue(
                code=SPLIT_TIMESTAMP_INVALID,
                message="Timestamp values are invalid or not UTC-compatible.",
                context={"timestamp_column": manifest.timestamp_column},
            )
        )
        return _finalize_file_report(file_report), fold_reports

    if not timestamps.is_monotonic_increasing or not timestamps.is_unique:
        file_report.errors.append(
            ValidationIssue(
                code=SPLIT_TIMESTAMP_INVALID,
                message="Timestamp column must be monotonic increasing and unique.",
                context={"monotonic": bool(timestamps.is_monotonic_increasing), "unique": bool(timestamps.is_unique)},
            )
        )

    if len(timestamps) > 0:
        file_report.timestamp_min_utc = timestamps.iloc[0].isoformat()
        file_report.timestamp_max_utc = timestamps.iloc[-1].isoformat()

    median_delta = _median_delta_seconds(timestamps)
    file_report.observed_median_delta_seconds = median_delta

    timeframe_issue = _evaluate_timeframe_consistency(
        parquet_path=parquet_path,
        observed_median_delta_seconds=median_delta,
        expected_timeframe_cli=cli_expected_timeframe,
        expected_timeframe_manifest=manifest.expected_timeframe,
    )
    if timeframe_issue is None:
        file_report.timeframe_consistency_ok = True if cli_expected_timeframe or manifest.expected_timeframe else None
    else:
        file_report.timeframe_consistency_ok = False
        file_report.errors.append(timeframe_issue)

    if split_spec.mode == "ratio_chrono":
        windows, ratio_context, ratio_error = _derive_ratio_windows(total_rows=len(timestamps), payload=split_spec.payload)
        if ratio_error is not None:
            file_report.errors.append(ratio_error)
            return _finalize_file_report(file_report), fold_reports

        _apply_single_split_checks(
            file_report=file_report,
            timestamps=timestamps,
            train_window=windows.get("train"),
            val_window=windows.get("val"),
            test_window=windows.get("test"),
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
            min_test_rows=min_test_rows,
            embargo_bars=embargo_bars,
            embargo_seconds=embargo_seconds,
            warmup_rows=warmup_rows,
            coverage_required=True,
            coverage_rows=ratio_context.get("coverage_rows", 0),
        )
        file_report.warnings.append(
            ValidationIssue(
                code="SPLIT_RATIO_ALLOCATION_INFO",
                message="Deterministic ratio allocation applied.",
                context=ratio_context,
            )
        )
        return _finalize_file_report(file_report), fold_reports

    if split_spec.mode == "explicit_ranges":
        windows, explicit_error = _derive_explicit_windows(timestamps=timestamps, payload=split_spec.payload)
        if explicit_error is not None:
            file_report.errors.append(explicit_error)
            return _finalize_file_report(file_report), fold_reports

        _apply_single_split_checks(
            file_report=file_report,
            timestamps=timestamps,
            train_window=windows.get("train"),
            val_window=windows.get("val"),
            test_window=windows.get("test"),
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
            min_test_rows=min_test_rows,
            embargo_bars=embargo_bars,
            embargo_seconds=embargo_seconds,
            warmup_rows=warmup_rows,
            coverage_required=False,
            coverage_rows=int(_count_union_rows(windows.values())),
        )
        return _finalize_file_report(file_report), fold_reports

    fold_windows, walk_error, bars_context = _derive_walk_forward_windows(
        timestamps=timestamps,
        payload=split_spec.payload,
    )
    if walk_error is not None:
        file_report.errors.append(walk_error)
        return _finalize_file_report(file_report), fold_reports

    if not fold_windows:
        file_report.errors.append(
            ValidationIssue(
                code=SPLIT_EMPTY_PARTITION,
                message="No valid walk-forward folds could be produced.",
                context={"rows_total": len(timestamps), "resolved_windows": bars_context},
            )
        )
        return _finalize_file_report(file_report), fold_reports

    prior_test_end_ts: pd.Timestamp | None = None
    for fold_id, fold in enumerate(fold_windows):
        fold_report = FoldSplitReport(fold_id=fold_id, input_file=str(parquet_path))
        fold_report.train_rows = _window_rows(fold["train"])
        fold_report.val_rows = _window_rows(fold["val"])
        fold_report.test_rows = _window_rows(fold["test"])
        fold_report.train_range = _window_to_range_payload(timestamps, fold["train"])
        fold_report.val_range = _window_to_range_payload(timestamps, fold["val"])
        fold_report.test_range = _window_to_range_payload(timestamps, fold["test"])

        _apply_partition_relationship_checks(
            errors=fold_report.errors,
            timestamps=timestamps,
            train_window=fold["train"],
            val_window=fold["val"],
            test_window=fold["test"],
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
            min_test_rows=min_test_rows,
            embargo_bars=embargo_bars,
            embargo_seconds=embargo_seconds,
            warmup_rows=warmup_rows,
            warmup_detail_target=fold_report,
        )
        _populate_partition_ok_flags(fold_report)

        current_test_range = fold_report.test_range or {}
        current_test_end = _parse_utc_timestamp(current_test_range.get("end_inclusive_utc"))
        if current_test_end is not None and prior_test_end_ts is not None and current_test_end <= prior_test_end_ts:
            fold_report.errors.append(
                ValidationIssue(
                    code=SPLIT_FOLD_NON_MONOTONIC,
                    message="Fold sequence is non-monotonic by test end timestamp.",
                    context={"prior_test_end": prior_test_end_ts.isoformat(), "current_test_end": current_test_end.isoformat()},
                )
            )
        if current_test_end is not None:
            prior_test_end_ts = current_test_end

        fold_report.fold_ok = len(fold_report.errors) == 0
        fold_reports.append(fold_report)

    file_report.fold_count = len(fold_reports)
    file_report.failed_fold_count = sum(1 for fold in fold_reports if not fold.fold_ok)
    file_report.warnings.append(
        ValidationIssue(
            code="SPLIT_WALKFORWARD_RESOLVED_BARS",
            message="Walk-forward window values resolved per-file.",
            context=bars_context,
        )
    )

    if file_report.failed_fold_count > 0:
        file_report.errors.append(
            ValidationIssue(
                code=SPLIT_FOLD_NON_MONOTONIC,
                message="One or more walk-forward folds failed validation.",
                context={"failed_fold_count": file_report.failed_fold_count, "total_folds": file_report.fold_count},
            )
        )

    last_fold = fold_reports[-1]
    file_report.train_rows = last_fold.train_rows
    file_report.val_rows = last_fold.val_rows
    file_report.test_rows = last_fold.test_rows
    file_report.train_range = last_fold.train_range
    file_report.val_range = last_fold.val_range
    file_report.test_range = last_fold.test_range
    file_report.warmup_detail = {
        "val": {
            "failed_folds": sum(
                1
                for fold in fold_reports
                for issue in fold.errors
                if issue.code == SPLIT_WARMUP_INSUFFICIENT and issue.context.get("partition") == "val"
            ),
            "total_folds": len(fold_reports),
        },
        "test": {
            "failed_folds": sum(
                1
                for fold in fold_reports
                for issue in fold.errors
                if issue.code == SPLIT_WARMUP_INSUFFICIENT and issue.context.get("partition") == "test"
            ),
            "total_folds": len(fold_reports),
        },
    }
    file_report.ordering_ok = all(fold.ordering_ok for fold in fold_reports)
    file_report.non_overlap_ok = all(fold.non_overlap_ok for fold in fold_reports)
    file_report.embargo_ok = all(fold.embargo_ok for fold in fold_reports)
    file_report.warmup_ok = all(fold.warmup_ok for fold in fold_reports)
    file_report.min_rows_ok = all(fold.min_rows_ok for fold in fold_reports)
    file_report.split_coverage_ok = True

    return _finalize_file_report(file_report), fold_reports


def _derive_ratio_windows(
    *,
    total_rows: int,
    payload: Mapping[str, Any],
) -> tuple[dict[str, PartitionWindow], dict[str, Any], ValidationIssue | None]:
    train_ratio = _parse_float(payload.get("train_ratio"))
    val_ratio = _parse_float(payload.get("val_ratio"))
    test_ratio = _parse_float(payload.get("test_ratio"))
    if train_ratio is None or val_ratio is None or test_ratio is None:
        return {}, {}, ValidationIssue(
            code=SPLIT_RATIO_INVALID,
            message="Ratio values are missing from normalized split spec.",
            context={"payload": dict(payload)},
        )

    allocations, allocation_context = _allocate_ratio_counts(
        total_rows=total_rows,
        ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
    )

    train_end = allocations["train"]
    val_end = train_end + allocations["val"]
    windows = {
        "train": PartitionWindow(0, train_end),
        "val": PartitionWindow(train_end, val_end),
        "test": PartitionWindow(val_end, val_end + allocations["test"]),
    }

    coverage_rows = allocations["train"] + allocations["val"] + allocations["test"]
    allocation_context["coverage_rows"] = coverage_rows
    return windows, allocation_context, None


def _derive_explicit_windows(
    *,
    timestamps: pd.Series,
    payload: Mapping[str, Any],
) -> tuple[dict[str, PartitionWindow | None], ValidationIssue | None]:
    windows: dict[str, PartitionWindow | None] = {"train": None, "val": None, "test": None}

    for name in ("train", "val", "test"):
        start_raw = payload.get(f"{name}_start")
        end_raw = payload.get(f"{name}_end")
        if start_raw is None and end_raw is None:
            continue

        start_ts = _parse_utc_timestamp(start_raw)
        end_ts = _parse_utc_timestamp(end_raw)
        if start_ts is None or end_ts is None:
            return {}, ValidationIssue(
                code=SPLIT_EXPLICIT_RANGE_INVALID,
                message="Failed to parse explicit range timestamps.",
                context={"partition": name, "start": start_raw, "end": end_raw},
            )

        if end_ts < start_ts:
            return {}, ValidationIssue(
                code=SPLIT_EXPLICIT_RANGE_INVALID,
                message="Explicit range end must be >= start.",
                context={"partition": name, "start": start_ts.isoformat(), "end": end_ts.isoformat()},
            )

        start_idx = int(timestamps.searchsorted(start_ts, side="left"))
        end_exclusive_idx = int(timestamps.searchsorted(end_ts, side="right"))
        windows[name] = PartitionWindow(start=start_idx, end=end_exclusive_idx)

    return windows, None


def _derive_walk_forward_windows(
    *,
    timestamps: pd.Series,
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, PartitionWindow]], ValidationIssue | None, dict[str, Any]]:
    median_delta = _median_delta_seconds(timestamps)

    min_train_bars = _resolve_window_to_bars(payload.get("min_train"), median_delta)
    val_bars = _resolve_window_to_bars(payload.get("val_window"), median_delta)
    test_bars = _resolve_window_to_bars(payload.get("test_window"), median_delta)
    step_bars = _resolve_window_to_bars(payload.get("step"), median_delta)

    if min_train_bars is None or val_bars is None or test_bars is None or step_bars is None:
        return [], ValidationIssue(
            code=SPLIT_WALKFORWARD_PARAM_INVALID,
            message="Walk-forward bars could not be resolved (duration requires observed median delta).",
            context={"observed_median_delta_seconds": median_delta},
        ), {}

    if min_train_bars <= 0 or val_bars <= 0 or test_bars <= 0 or step_bars <= 0:
        return [], ValidationIssue(
            code=SPLIT_WALKFORWARD_PARAM_INVALID,
            message="Walk-forward windows must be positive.",
            context={
                "min_train_bars": min_train_bars,
                "val_window_bars": val_bars,
                "test_window_bars": test_bars,
                "step_bars": step_bars,
            },
        ), {}

    resolved_context = {
        "min_train_bars": min_train_bars,
        "val_window_bars": val_bars,
        "test_window_bars": test_bars,
        "step_bars": step_bars,
        "observed_median_delta_seconds": median_delta,
    }

    total_rows = len(timestamps)
    max_folds_raw = payload.get("max_folds")
    max_folds = int(max_folds_raw) if isinstance(max_folds_raw, int) else None

    folds: list[dict[str, PartitionWindow]] = []
    train_end = min_train_bars

    while train_end < total_rows:
        val_start = train_end
        val_end = val_start + val_bars
        test_start = val_end
        test_end = test_start + test_bars
        if test_end > total_rows:
            break

        folds.append(
            {
                "train": PartitionWindow(0, train_end),
                "val": PartitionWindow(val_start, val_end),
                "test": PartitionWindow(test_start, test_end),
            }
        )

        if max_folds is not None and len(folds) >= max_folds:
            break

        train_end += step_bars

    return folds, None, resolved_context


def _apply_single_split_checks(
    *,
    file_report: FileSplitReport,
    timestamps: pd.Series,
    train_window: PartitionWindow | None,
    val_window: PartitionWindow | None,
    test_window: PartitionWindow | None,
    min_train_rows: int,
    min_val_rows: int,
    min_test_rows: int,
    embargo_bars: int | None,
    embargo_seconds: int | None,
    warmup_rows: int,
    coverage_required: bool,
    coverage_rows: int,
) -> None:
    file_report.train_rows = _window_rows(train_window)
    file_report.val_rows = _window_rows(val_window)
    file_report.test_rows = _window_rows(test_window)
    file_report.train_range = _window_to_range_payload(timestamps, train_window)
    file_report.val_range = _window_to_range_payload(timestamps, val_window)
    file_report.test_range = _window_to_range_payload(timestamps, test_window)

    _apply_partition_relationship_checks(
        errors=file_report.errors,
        timestamps=timestamps,
        train_window=train_window,
        val_window=val_window,
        test_window=test_window,
        min_train_rows=min_train_rows,
        min_val_rows=min_val_rows,
        min_test_rows=min_test_rows,
        embargo_bars=embargo_bars,
        embargo_seconds=embargo_seconds,
        warmup_rows=warmup_rows,
        warmup_detail_target=file_report,
    )

    if coverage_required and coverage_rows != len(timestamps):
        file_report.errors.append(
            ValidationIssue(
                code=SPLIT_COVERAGE_INVALID,
                message="Coverage invalid for ratio_chrono; full row coverage is required.",
                context={"rows_total": len(timestamps), "coverage_rows": coverage_rows},
            )
        )
        file_report.split_coverage_ok = False
    elif not coverage_required:
        file_report.split_coverage_ok = True

    _populate_partition_ok_flags(file_report)


def _apply_partition_relationship_checks(
    *,
    errors: list[ValidationIssue],
    timestamps: pd.Series,
    train_window: PartitionWindow | None,
    val_window: PartitionWindow | None,
    test_window: PartitionWindow | None,
    min_train_rows: int,
    min_val_rows: int,
    min_test_rows: int,
    embargo_bars: int | None,
    embargo_seconds: int | None,
    warmup_rows: int,
    warmup_detail_target: FileSplitReport | FoldSplitReport,
) -> None:
    windows: dict[str, PartitionWindow | None] = {
        "train": train_window,
        "val": val_window,
        "test": test_window,
    }

    if train_window is None:
        errors.append(
            ValidationIssue(
                code=SPLIT_EMPTY_PARTITION,
                message="train partition is required.",
                context={"partition": "train"},
            )
        )

    for partition_name, min_rows in (
        ("train", min_train_rows),
        ("val", min_val_rows),
        ("test", min_test_rows),
    ):
        window = windows.get(partition_name)
        if window is None:
            continue
        row_count = _window_rows(window)
        if row_count <= 0:
            errors.append(
                ValidationIssue(
                    code=SPLIT_EMPTY_PARTITION,
                    message="Partition contains zero rows.",
                    context={"partition": partition_name},
                )
            )
        if row_count < min_rows:
            errors.append(
                ValidationIssue(
                    code=SPLIT_MIN_ROWS_NOT_MET,
                    message="Partition does not satisfy min row gate.",
                    context={"partition": partition_name, "rows": row_count, "required": min_rows},
                )
            )

    if _is_overlap(train_window, val_window):
        errors.append(
            ValidationIssue(
                code=SPLIT_OVERLAP_DETECTED,
                message="Overlap detected between train and val partitions.",
                context={"left": "train", "right": "val"},
            )
        )
    if _is_overlap(val_window, test_window):
        errors.append(
            ValidationIssue(
                code=SPLIT_OVERLAP_DETECTED,
                message="Overlap detected between val and test partitions.",
                context={"left": "val", "right": "test"},
            )
        )
    if _is_overlap(train_window, test_window):
        errors.append(
            ValidationIssue(
                code=SPLIT_OVERLAP_DETECTED,
                message="Overlap detected between train and test partitions.",
                context={"left": "train", "right": "test"},
            )
        )

    if not _ordering_ok(timestamps, train_window, val_window):
        errors.append(
            ValidationIssue(
                code=SPLIT_ORDERING_INVALID,
                message="Ordering invalid: train must finish before val starts.",
                context={"left": "train", "right": "val"},
            )
        )
    if not _ordering_ok(timestamps, val_window, test_window):
        errors.append(
            ValidationIssue(
                code=SPLIT_ORDERING_INVALID,
                message="Ordering invalid: val must finish before test starts.",
                context={"left": "val", "right": "test"},
            )
        )
    if not _ordering_ok(timestamps, train_window, test_window):
        errors.append(
            ValidationIssue(
                code=SPLIT_ORDERING_INVALID,
                message="Ordering invalid: train must finish before test starts.",
                context={"left": "train", "right": "test"},
            )
        )

    embargo_pairs = (("train", train_window, "val", val_window), ("val", val_window, "test", test_window))
    for left_name, left_window, right_name, right_window in embargo_pairs:
        if not _embargo_ok(
            timestamps=timestamps,
            left_window=left_window,
            right_window=right_window,
            embargo_bars=embargo_bars,
            embargo_seconds=embargo_seconds,
        ):
            errors.append(
                ValidationIssue(
                    code=SPLIT_EMBARGO_VIOLATION,
                    message="Embargo rule violated between adjacent partitions.",
                    context={
                        "left": left_name,
                        "right": right_name,
                        "embargo_bars": embargo_bars,
                        "embargo_seconds": embargo_seconds,
                    },
                )
            )

    warmup_detail = _evaluate_warmup(
        train_window=train_window,
        val_window=val_window,
        test_window=test_window,
        warmup_rows=warmup_rows,
    )
    warmup_detail_target.warmup_detail = warmup_detail

    for partition in ("val", "test"):
        detail = warmup_detail.get(partition)
        if not isinstance(detail, dict):
            continue
        ok = bool(detail.get("ok", True))
        if not ok:
            errors.append(
                ValidationIssue(
                    code=SPLIT_WARMUP_INSUFFICIENT,
                    message="Warmup history is insufficient before evaluation partition start.",
                    context={
                        "partition": partition,
                        "required_rows": detail.get("required_rows"),
                        "available_rows": detail.get("available_rows"),
                    },
                )
            )


def _evaluate_warmup(
    *,
    train_window: PartitionWindow | None,
    val_window: PartitionWindow | None,
    test_window: PartitionWindow | None,
    warmup_rows: int,
) -> dict[str, Any]:
    del train_window

    detail: dict[str, Any] = {
        "policy": "historical_rows_before_eval_start",
        "warmup_rows": int(warmup_rows),
    }

    for name, window in (("val", val_window), ("test", test_window)):
        if window is None:
            detail[name] = {"checked": False, "ok": True, "required_rows": warmup_rows, "available_rows": None}
            continue
        available = max(0, window.start)
        ok = available >= warmup_rows
        detail[name] = {
            "checked": True,
            "ok": ok,
            "required_rows": warmup_rows,
            "available_rows": available,
        }

    return detail


def _populate_partition_ok_flags(target: FileSplitReport | FoldSplitReport) -> None:
    codes = {issue.code for issue in target.errors}
    target.ordering_ok = SPLIT_ORDERING_INVALID not in codes
    target.non_overlap_ok = SPLIT_OVERLAP_DETECTED not in codes
    target.embargo_ok = SPLIT_EMBARGO_VIOLATION not in codes
    target.warmup_ok = SPLIT_WARMUP_INSUFFICIENT not in codes
    target.min_rows_ok = SPLIT_MIN_ROWS_NOT_MET not in codes and SPLIT_EMPTY_PARTITION not in codes


def _window_rows(window: PartitionWindow | None) -> int:
    if window is None:
        return 0
    return max(0, window.end - window.start)


def _window_to_range_payload(timestamps: pd.Series, window: PartitionWindow | None) -> dict[str, Any] | None:
    if window is None:
        return None

    row_count = _window_rows(window)
    if row_count <= 0:
        return {
            "start_utc": None,
            "end_exclusive_utc": None,
            "end_inclusive_utc": None,
            "row_count": 0,
            "internal_interval": "[start, end)",
        }

    start_ts = timestamps.iloc[window.start]
    end_inclusive = timestamps.iloc[window.end - 1]
    end_exclusive = timestamps.iloc[window.end] if window.end < len(timestamps) else None
    return {
        "start_utc": start_ts.isoformat(),
        "end_exclusive_utc": end_exclusive.isoformat() if end_exclusive is not None else None,
        "end_inclusive_utc": end_inclusive.isoformat(),
        "row_count": row_count,
        "internal_interval": "[start, end)",
    }


def _is_overlap(left: PartitionWindow | None, right: PartitionWindow | None) -> bool:
    if left is None or right is None:
        return False
    return left.start < right.end and right.start < left.end


def _ordering_ok(timestamps: pd.Series, left: PartitionWindow | None, right: PartitionWindow | None) -> bool:
    if left is None or right is None:
        return True
    if _window_rows(left) <= 0 or _window_rows(right) <= 0:
        return False
    left_end_ts = timestamps.iloc[left.end - 1]
    right_start_ts = timestamps.iloc[right.start]
    return bool(left_end_ts < right_start_ts)


def _embargo_ok(
    *,
    timestamps: pd.Series,
    left_window: PartitionWindow | None,
    right_window: PartitionWindow | None,
    embargo_bars: int | None,
    embargo_seconds: int | None,
) -> bool:
    if left_window is None or right_window is None:
        return True
    if _window_rows(left_window) <= 0 or _window_rows(right_window) <= 0:
        return False

    if embargo_bars is not None:
        gap_bars = max(0, right_window.start - left_window.end)
        return gap_bars >= embargo_bars

    if embargo_seconds is not None:
        left_end_ts = timestamps.iloc[left_window.end - 1]
        right_start_ts = timestamps.iloc[right_window.start]
        gap_seconds = float((right_start_ts - left_end_ts).total_seconds())
        return gap_seconds >= float(embargo_seconds)

    return True


def _count_union_rows(windows: Sequence[PartitionWindow | None]) -> int:
    intervals = [(window.start, window.end) for window in windows if window is not None and _window_rows(window) > 0]
    if not intervals:
        return 0
    intervals = sorted(intervals)

    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    total += current_end - current_start
    return total


def _parse_timestamp_series(raw_series: pd.Series | None) -> pd.Series | None:
    if raw_series is None:
        return None
    parsed = pd.to_datetime(raw_series, utc=True, errors="coerce")
    if parsed.isna().any():
        return None
    return parsed.reset_index(drop=True)


def _median_delta_seconds(timestamps: pd.Series) -> float | None:
    deltas = timestamps.diff().dt.total_seconds().dropna()
    if deltas.empty:
        return None
    return float(deltas.median())


def _evaluate_timeframe_consistency(
    *,
    parquet_path: Path,
    observed_median_delta_seconds: float | None,
    expected_timeframe_cli: str | None,
    expected_timeframe_manifest: str | None,
) -> ValidationIssue | None:
    if observed_median_delta_seconds is None:
        return None

    expected_seconds: int | None = None
    source: str | None = None

    if expected_timeframe_cli is not None:
        parsed_cli = _parse_duration_to_seconds(expected_timeframe_cli)
        if parsed_cli is not None:
            expected_seconds = parsed_cli
            source = "cli"

    if expected_seconds is None and expected_timeframe_manifest is not None:
        parsed_manifest = _parse_duration_to_seconds(expected_timeframe_manifest)
        if parsed_manifest is not None:
            expected_seconds = parsed_manifest
            source = "manifest"

    if expected_seconds is None:
        inferred = _infer_timeframe_seconds_from_path(parquet_path)
        if inferred is not None:
            expected_seconds = inferred
            source = "path"

    if expected_seconds is None:
        return None

    tolerance = max(1.0, float(expected_seconds) * TIMEFRAME_TOLERANCE_RATIO)
    delta = abs(float(observed_median_delta_seconds) - float(expected_seconds))
    if delta <= tolerance:
        return None

    return ValidationIssue(
        code=SPLIT_TIMEFRAME_INCONSISTENT,
        message="Observed median delta is inconsistent with expected timeframe.",
        context={
            "expected_timeframe_seconds": expected_seconds,
            "expected_timeframe_source": source,
            "observed_median_delta_seconds": observed_median_delta_seconds,
            "tolerance_seconds": tolerance,
        },
    )


def _resolve_window_to_bars(spec: Any, median_delta_seconds: float | None) -> int | None:
    if not isinstance(spec, dict):
        return None
    unit = spec.get("unit")
    value = spec.get("value")
    if unit == "bars":
        if isinstance(value, int):
            return value
        return None
    if unit == "seconds":
        if not isinstance(value, int):
            return None
        if median_delta_seconds is None or median_delta_seconds <= 0:
            return None
        return max(1, int(math.ceil(float(value) / float(median_delta_seconds))))
    return None


def _allocate_ratio_counts(total_rows: int, ratios: Mapping[str, float]) -> tuple[dict[str, int], dict[str, Any]]:
    raw = {name: float(ratios[name]) * float(total_rows) for name in ("train", "val", "test")}
    floor_counts = {name: int(math.floor(raw[name])) for name in raw}
    remainder = total_rows - sum(floor_counts.values())

    order = ["train", "val", "test"]
    fractional_rank = sorted(
        order,
        key=lambda name: (raw[name] - floor_counts[name], -order.index(name)),
        reverse=True,
    )

    final_counts = dict(floor_counts)
    for idx in range(remainder):
        final_counts[fractional_rank[idx % len(fractional_rank)]] += 1

    context = {
        "total_rows": total_rows,
        "raw_allocations": raw,
        "floor_allocations": floor_counts,
        "remainder": remainder,
        "remainder_assignment_order": fractional_rank,
        "allocated_counts": final_counts,
        "remainder_policy": "largest_fraction_then_train_val_test_tiebreak",
    }
    return final_counts, context


def _parse_duration_to_seconds(value: str) -> int | None:
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


def _parse_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _parse_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if not stripped.isdigit():
            return None
        parsed = int(stripped)
        return parsed if parsed > 0 else None
    return None


def _parse_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="raise")
    except (TypeError, ValueError):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts
    return None


def _parse_optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        out.append(item)
    return out


def _stable_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _finalize_file_report(file_report: FileSplitReport) -> FileSplitReport:
    if file_report.split_coverage_ok is False:
        pass
    elif file_report.split_mode in ("ratio_chrono", "explicit_ranges"):
        file_report.split_coverage_ok = True

    file_report.status = "success" if len(file_report.errors) == 0 else "failed"
    return file_report


def _finalize_report(report: SplitValidationReport) -> None:
    report.succeeded_files = sum(1 for item in report.file_reports if item.status == "success")
    report.failed_files = report.total_files - report.succeeded_files
    report.split_validation_overall = bool(
        report.manifest_valid
        and report.total_files > 0
        and report.failed_files == 0
        and len(report.errors) == 0
    )
