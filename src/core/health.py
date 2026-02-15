"""Health reporting and gate checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class HealthIssue:
    """Represents a single health issue."""

    code: str
    severity: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileHealthReport:
    """Represents health status for a single input file."""

    source_file: str
    output_file: str | None
    row_count_in: int
    row_count_out: int
    issues: list[HealthIssue] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary."""
        return asdict(self)


def health_check(report: FileHealthReport, fail_on_critical: bool = True) -> bool:
    """Return True when report satisfies write gate rules."""
    if not fail_on_critical:
        return True
    return all(issue.severity != "critical" for issue in report.issues)


def summarize_reports(reports: list[FileHealthReport]) -> dict[str, Any]:
    """Build a global summary from per-file reports."""
    code_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {"warning": 0, "critical": 0}

    for report in reports:
        for issue in report.issues:
            code_counts[issue.code] = code_counts.get(issue.code, 0) + 1
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    total_files = len(reports)
    succeeded_files = sum(1 for report in reports if report.success)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": total_files,
        "succeeded_files": succeeded_files,
        "failed_files": total_files - succeeded_files,
        "issue_counts_by_code": code_counts,
        "issue_counts_by_severity": severity_counts,
    }


def write_health_json_atomic(report: dict[str, Any], path: Path) -> None:
    """Atomically write health report as JSON."""
    from core.io_atomic import atomic_write_json

    atomic_write_json(report, path)

