"""CLI entrypoint for RL train-input contract validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.io_atomic import atomic_write_json
from core.logging import get_logger, setup_logging
from data.train_input_validation import (
    TRAIN_INPUT_RUNTIME_ERROR,
    TrainInputValidationOptions,
    validate_train_inputs,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Validate feature parquet files against RL training input contract.")
    parser.add_argument("--run-id", type=str, required=True, help="Run id under runs/<run_id>/data_features.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional feature parquet root. Default: runs/<run_id>/data_features/parquet",
    )
    parser.add_argument(
        "--strict-extra-columns",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Fail on unexpected columns when true (default).",
    )
    parser.add_argument(
        "--strict-column-order",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Fail on column-order drift when true.",
    )
    parser.add_argument(
        "--expected-timeframe",
        type=str,
        default=None,
        help="Optional expected timeframe (examples: 1m, 5m, 15m, 1h, 3600s).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _default_input_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_features" / "parquet"


def _default_reports_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_features" / "reports"


def _write_report_best_effort(payload: dict[str, Any], report_path: Path) -> None:
    try:
        atomic_write_json(payload, report_path)
    except RuntimeError as exc:
        LOGGER.info("Validation report write failed (best-effort) | path=%s error=%s", report_path, exc)


def _derive_validation_error(payload: dict[str, Any]) -> str | None:
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        first_error = errors[0]
        if isinstance(first_error, dict):
            code = first_error.get("code")
            if isinstance(code, str):
                return code

    file_reports = payload.get("file_reports")
    if not isinstance(file_reports, list):
        return None
    for item in file_reports:
        if not isinstance(item, dict):
            continue
        file_errors = item.get("errors")
        if isinstance(file_errors, list) and file_errors:
            first_file_error = file_errors[0]
            if isinstance(first_file_error, dict):
                code = first_file_error.get("code")
                if isinstance(code, str):
                    return code
    return None


def _update_summary_best_effort(
    *,
    summary_path: Path,
    report_path: Path,
    train_input_validation_overall: bool,
    validation_error: str | None,
) -> None:
    if not summary_path.exists():
        return

    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.info("Summary update skipped: summary read failed | path=%s error=%s", summary_path, exc)
        return

    if not isinstance(summary_payload, dict):
        LOGGER.info("Summary update skipped: summary payload is not an object | path=%s", summary_path)
        return

    summary_payload["train_input_validation_overall"] = bool(train_input_validation_overall)
    summary_payload["train_input_validation_report_path"] = str(report_path)
    summary_payload["train_input_validation_error"] = validation_error

    try:
        atomic_write_json(summary_payload, summary_path)
    except RuntimeError as exc:
        LOGGER.info("Summary update failed (non-blocking) | path=%s error=%s", summary_path, exc)


def _build_runtime_error_payload(run_id: str, reports_root: Path, exc: Exception, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "manifest_path": str(reports_root / "feature_manifest.json"),
        "manifest_loaded": False,
        "manifest_valid": False,
        "manifest_version": None,
        "validator_version": "train_input_validator.v1",
        "invocation_args": {
            "run_id": run_id,
            "input_root": str(args.input_root) if args.input_root is not None else None,
            "strict_extra_columns": _to_bool(args.strict_extra_columns),
            "strict_column_order": _to_bool(args.strict_column_order),
            "expected_timeframe": args.expected_timeframe,
        },
        "strict_extra_columns": _to_bool(args.strict_extra_columns),
        "strict_column_order": _to_bool(args.strict_column_order),
        "total_files": 0,
        "succeeded_files": 0,
        "failed_files": 0,
        "train_input_validation_overall": False,
        "schema_hash_actual": None,
        "dtype_hash_actual": None,
        "column_order_hash_actual": None,
        "file_reports": [],
        "errors": [
            {
                "code": TRAIN_INPUT_RUNTIME_ERROR,
                "message": "Runtime error during train-input validation.",
                "context": {"error": str(exc)},
            }
        ],
        "warnings": [],
    }


def main() -> int:
    """Run train-input validation and return deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run-id must be non-empty")

    input_root = args.input_root.resolve() if args.input_root is not None else _default_input_root(run_id).resolve()
    reports_root = _default_reports_root(run_id).resolve()
    report_path = reports_root / "train_input_validation_report.json"
    summary_path = reports_root / "summary.json"

    try:
        options = TrainInputValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            strict_extra_columns=_to_bool(args.strict_extra_columns),
            strict_column_order=_to_bool(args.strict_column_order),
            expected_timeframe=args.expected_timeframe,
        )
        validation_report = validate_train_inputs(options)
        payload = validation_report.to_dict()

        _write_report_best_effort(payload, report_path)

        validation_error = _derive_validation_error(payload)
        _update_summary_best_effort(
            summary_path=summary_path,
            report_path=report_path,
            train_input_validation_overall=bool(validation_report.train_input_validation_overall),
            validation_error=validation_error,
        )

        exit_code = 0 if validation_report.train_input_validation_overall else 2
        LOGGER.info(
            "Train-input validation summary | run_id=%s total=%d success=%d failed=%d overall=%s exit_code=%d report=%s",
            run_id,
            validation_report.total_files,
            validation_report.succeeded_files,
            validation_report.failed_files,
            validation_report.train_input_validation_overall,
            exit_code,
            report_path,
        )
        return exit_code
    except Exception as exc:
        runtime_payload = _build_runtime_error_payload(run_id=run_id, reports_root=reports_root, exc=exc, args=args)
        _write_report_best_effort(runtime_payload, report_path)
        _update_summary_best_effort(
            summary_path=summary_path,
            report_path=report_path,
            train_input_validation_overall=False,
            validation_error=TRAIN_INPUT_RUNTIME_ERROR,
        )
        LOGGER.info(
            "Train-input validation runtime error | run_id=%s exit_code=3 error=%s report=%s",
            run_id,
            exc,
            report_path,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
