"""CLI entrypoint for CSV schema/timestamp inspection."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.logging import setup_logging
from data.csv_inspection import inspect_all_csvs, write_inspection_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Inspect CSV schema and timestamp integrity.")
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT, help="Root directory to scan recursively.")
    parser.add_argument("--run-id", type=str, default="", help="Run id. Default: current UTC timestamp.")
    parser.add_argument("--sample-size", type=int, default=2000, help="Sample row count per file.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _format_failure_summary(report: dict[str, object]) -> str:
    """Format concise per-file failure line."""
    file_name = str(report.get("file", "<unknown>"))
    reasons = report.get("failure_reason_codes", [])
    if isinstance(reasons, list):
        reason_text = ",".join(str(item) for item in reasons) if reasons else "UNKNOWN"
    else:
        reason_text = "UNKNOWN"
    return f"{file_name}: {reason_text}"


def main() -> int:
    """Run CSV inspection and return process exit code."""
    args = parse_args()
    setup_logging(args.log_level)

    if args.sample_size <= 0:
        raise ValueError("sample-size must be > 0")

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = PROJECT_ROOT / "runs" / run_id / "csv_inspection"
    report_path = run_root / "inspection_report.json"

    run_report = inspect_all_csvs(input_root=args.input_root, run_id=run_id, sample_size=args.sample_size)
    write_inspection_report(run_report, report_path)

    if run_report.failed_files == 0:
        sys.stdout.write("çıktı beklediğim gibi,başarılı ✅\n")
        return 0

    for item in run_report.inspections:
        passed = bool(item.get("passed"))
        if not passed:
            sys.stdout.write(_format_failure_summary(item) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
