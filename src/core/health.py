"""Health reporting and validation gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class StructuredError:
    """Structured error payload for per-file health reports."""

    stage: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileHealthReport:
    """Per-file health report required by milestone contract."""

    input_file: str
    rows_in: int = 0
    rows_out: int = 0
    dropped_invalid_ts: int = 0
    dropped_duplicates: int = 0
    dropped_invalid_numeric: int = 0
    schema_ok: bool = False
    monotonic_ok: bool = False
    unique_ts_ok: bool = False
    dtype_ok: bool = False
    no_nan_ok: bool = False
    status: str = "failed"
    output_file: str | None = None
    timestamp_alias: str | None = None
    alias_confidence: float | None = None
    selected_timestamp_strategy: str | None = None
    parse_valid_ratio: float | None = None
    timestamp_min: str | None = None
    timestamp_max: str | None = None
    unique_days: int = 0
    errors: list[StructuredError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report as a JSON-ready dictionary."""
        return asdict(self)


def add_error(report: FileHealthReport, stage: str, code: str, message: str, **context: Any) -> None:
    """Append a structured error into report."""
    report.errors.append(StructuredError(stage=stage, code=code, message=message, context=context))


def health_check(report: FileHealthReport) -> bool:
    """Return True when all strict QC gates pass."""
    return (
        report.schema_ok
        and report.monotonic_ok
        and report.unique_ts_ok
        and report.dtype_ok
        and report.no_nan_ok
        and report.rows_out > 0
        and len(report.errors) == 0
    )


def finalize_report(report: FileHealthReport) -> FileHealthReport:
    """Finalize status according to gate checks."""
    report.status = "success" if health_check(report) else "failed"
    return report


def summarize_reports(reports: list[FileHealthReport]) -> dict[str, Any]:
    """Build global summary report across all files."""
    total_files = len(reports)
    succeeded_files = sum(1 for item in reports if item.status == "success")
    failed_files = total_files - succeeded_files

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": total_files,
        "succeeded_files": succeeded_files,
        "failed_files": failed_files,
        "total_rows_in": sum(item.rows_in for item in reports),
        "total_rows_out": sum(item.rows_out for item in reports),
        "total_dropped_invalid_ts": sum(item.dropped_invalid_ts for item in reports),
        "total_dropped_duplicates": sum(item.dropped_duplicates for item in reports),
        "total_dropped_invalid_numeric": sum(item.dropped_invalid_numeric for item in reports),
        "failed_inputs": [item.input_file for item in reports if item.status == "failed"],
    }
