"""Health report models and QC gates for feature engineering outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

from data.features import CONTINUOUS_FEATURE_COLUMNS, EVENT_FLAG_COLUMNS, validate_shift_one


@dataclass
class StructuredError:
    """Structured health error payload."""

    stage: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureHealthReport:
    """Per-file health report for feature outputs."""

    input_file: str
    rows_in: int = 0
    rows_out: int = 0
    schema_ok: bool = False
    monotonic_ok: bool = False
    unique_ts_ok: bool = False
    dtype_ok: bool = False
    leakfree_ok: bool = False
    nan_ratio_ok: bool = False
    status: str = "failed"
    output_file: str | None = None
    nan_ratios: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[StructuredError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report into a JSON-ready dictionary."""

        return asdict(self)


def add_error(report: FeatureHealthReport, stage: str, code: str, message: str, **context: Any) -> None:
    """Append a structured error to a report."""

    report.errors.append(StructuredError(stage=stage, code=code, message=message, context=context))


def add_warning(report: FeatureHealthReport, message: str) -> None:
    """Append a warning message to a report."""

    report.warnings.append(message)


def _expected_columns() -> tuple[str, ...]:
    return (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *CONTINUOUS_FEATURE_COLUMNS,
        *EVENT_FLAG_COLUMNS,
    )


def _continuous_columns() -> tuple[str, ...]:
    return ("open", "high", "low", "close", "volume", *CONTINUOUS_FEATURE_COLUMNS)


def evaluate_feature_health(
    report: FeatureHealthReport,
    feature_df: pd.DataFrame,
    raw_events: pd.DataFrame,
    shifted_events: pd.DataFrame,
    warn_ratio: float,
    critical_warn_ratio: float,
    critical_columns: Sequence[str],
) -> FeatureHealthReport:
    """Run strict QC gates for feature outputs and update report in place."""

    expected = _expected_columns()
    missing = [col for col in expected if col not in feature_df.columns]
    report.schema_ok = len(missing) == 0
    if missing:
        add_error(report, stage="schema", code="MISSING_COLUMNS", message="Required feature columns are missing.", missing=missing)

    if report.schema_ok:
        timestamp = feature_df["timestamp"]
        report.monotonic_ok = bool(timestamp.is_monotonic_increasing)
        report.unique_ts_ok = bool(timestamp.is_unique)

        if not report.monotonic_ok:
            add_error(report, stage="gates", code="TIMESTAMP_NOT_MONOTONIC", message="timestamp is not monotonic increasing")
        if not report.unique_ts_ok:
            add_error(report, stage="gates", code="TIMESTAMP_NOT_UNIQUE", message="timestamp is not unique")

        dtype_ok_continuous = all(str(feature_df[col].dtype) == "float32" for col in _continuous_columns())
        dtype_ok_flags = all(str(feature_df[col].dtype) == "uint8" for col in EVENT_FLAG_COLUMNS)
        report.dtype_ok = bool(dtype_ok_continuous and dtype_ok_flags)
        if not report.dtype_ok:
            add_error(report, stage="gates", code="DTYPE_POLICY_FAILED", message="Dtype policy failed for continuous/event columns")

    report.leakfree_ok = validate_shift_one(raw_events, shifted_events)
    if not report.leakfree_ok:
        add_error(report, stage="gates", code="LEAKFREE_SHIFT_FAILED", message="Event flags are not strict shift(1)")

    report.nan_ratio_ok = True
    critical_set = {col.strip() for col in critical_columns if str(col).strip()}

    for col in _continuous_columns():
        ratio = float(feature_df[col].isna().mean())
        report.nan_ratios[col] = ratio

        if ratio > warn_ratio:
            add_warning(report, f"NaN ratio high | column={col} ratio={ratio:.6f} warn_ratio={warn_ratio:.6f}")

        col_threshold = critical_warn_ratio if col in critical_set else warn_ratio
        if ratio > col_threshold:
            report.nan_ratio_ok = False
            add_error(
                report,
                stage="gates",
                code="NAN_RATIO_TOO_HIGH",
                message="NaN ratio exceeded threshold",
                column=col,
                ratio=ratio,
                threshold=col_threshold,
            )

    report.rows_out = int(len(feature_df))
    if report.rows_out <= 0:
        add_error(report, stage="gates", code="EMPTY_OUTPUT", message="Feature output has no rows")

    report.status = "success" if health_check(report) else "failed"
    return report


def health_check(report: FeatureHealthReport) -> bool:
    """Return True when strict QC gates pass."""

    return (
        report.schema_ok
        and report.monotonic_ok
        and report.unique_ts_ok
        and report.dtype_ok
        and report.leakfree_ok
        and report.nan_ratio_ok
        and report.rows_out > 0
        and len(report.errors) == 0
    )


def summarize_feature_reports(reports: list[FeatureHealthReport]) -> dict[str, Any]:
    """Create global feature summary report."""

    total_files = len(reports)
    succeeded_files = sum(1 for report in reports if report.status == "success")
    failed_files = total_files - succeeded_files

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": total_files,
        "succeeded_files": succeeded_files,
        "failed_files": failed_files,
        "total_rows_in": sum(report.rows_in for report in reports),
        "total_rows_out": sum(report.rows_out for report in reports),
        "failed_inputs": [report.input_file for report in reports if report.status == "failed"],
    }
