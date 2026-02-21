"""CLI entrypoint for RL split contract validation."""

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
from data.split_validation import (
    SPLIT_RUNTIME_ERROR,
    SplitValidationOptions,
    validate_splits,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Validate RL dataset split contract before training starts.")
    parser.add_argument("--run-id", type=str, required=True, help="Run id under runs/<run_id>/data_features.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional feature parquet root. Default: runs/<run_id>/data_features/parquet",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default=None,
        choices=("ratio_chrono", "explicit_ranges", "walk_forward"),
        help="Split mode. If omitted, split-config must provide mode.",
    )
    parser.add_argument("--split-config", type=Path, default=None, help="Optional split config path (JSON/YAML).")

    parser.add_argument("--train-ratio", type=str, default=None)
    parser.add_argument("--val-ratio", type=str, default=None)
    parser.add_argument("--test-ratio", type=str, default=None)

    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)

    parser.add_argument("--min-train-bars", type=str, default=None)
    parser.add_argument("--min-train-duration", type=str, default=None)
    parser.add_argument("--val-window-bars", type=str, default=None)
    parser.add_argument("--val-window-duration", type=str, default=None)
    parser.add_argument("--test-window-bars", type=str, default=None)
    parser.add_argument("--test-window-duration", type=str, default=None)
    parser.add_argument("--step-bars", type=str, default=None)
    parser.add_argument("--step-duration", type=str, default=None)
    parser.add_argument("--max-folds", type=str, default=None)

    parser.add_argument(
        "--require-train-input-validation",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Fail if prior train_input_validation_report is missing/failed (default: true).",
    )
    parser.add_argument("--min-train-rows", type=int, default=1)
    parser.add_argument("--min-val-rows", type=int, default=1)
    parser.add_argument("--min-test-rows", type=int, default=1)
    parser.add_argument("--embargo-bars", type=int, default=None)
    parser.add_argument("--embargo-seconds", type=int, default=None)
    parser.add_argument("--warmup-rows", type=int, default=0)
    parser.add_argument("--timestamp-column", type=str, default=None)
    parser.add_argument(
        "--expected-timeframe",
        type=str,
        default=None,
        help="Optional expected timeframe (examples: 1m, 5m, 15m, 1h).",
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
        LOGGER.info("Split validation report write failed (best-effort) | path=%s error=%s", report_path, exc)


def _derive_validation_error(payload: dict[str, Any]) -> str | None:
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            code = first.get("code")
            if isinstance(code, str):
                return code

    for level in ("file_reports", "fold_reports"):
        reports = payload.get(level)
        if not isinstance(reports, list):
            continue
        for item in reports:
            if not isinstance(item, dict):
                continue
            report_errors = item.get("errors")
            if isinstance(report_errors, list) and report_errors:
                first_err = report_errors[0]
                if isinstance(first_err, dict):
                    code = first_err.get("code")
                    if isinstance(code, str):
                        return code
    return None


def _update_summary_best_effort(
    *,
    summary_path: Path,
    report_path: Path,
    split_validation_overall: bool,
    split_validation_error: str | None,
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

    summary_payload["split_validation_overall"] = bool(split_validation_overall)
    summary_payload["split_validation_report_path"] = str(report_path)
    summary_payload["split_validation_error"] = split_validation_error

    try:
        atomic_write_json(summary_payload, summary_path)
    except RuntimeError as exc:
        LOGGER.info("Summary update failed (non-blocking) | path=%s error=%s", summary_path, exc)


def _build_split_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "val_start": args.val_start,
        "val_end": args.val_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "min_train_bars": args.min_train_bars,
        "min_train_duration": args.min_train_duration,
        "val_window_bars": args.val_window_bars,
        "val_window_duration": args.val_window_duration,
        "test_window_bars": args.test_window_bars,
        "test_window_duration": args.test_window_duration,
        "step_bars": args.step_bars,
        "step_duration": args.step_duration,
        "max_folds": args.max_folds,
    }


def _build_runtime_error_payload(
    *,
    run_id: str,
    reports_root: Path,
    args: argparse.Namespace,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "validator_version": "split_validator.v1",
        "split_mode": args.split_mode,
        "split_spec": {},
        "manifest_path": str(reports_root / "feature_manifest.json"),
        "manifest_loaded": False,
        "manifest_valid": False,
        "train_input_validation_report_path": str(reports_root / "train_input_validation_report.json"),
        "train_input_validation_checked": False,
        "train_input_validation_required": _to_bool(args.require_train_input_validation),
        "train_input_validation_overall_seen": None,
        "total_files": 0,
        "succeeded_files": 0,
        "failed_files": 0,
        "split_validation_overall": False,
        "embargo_policy": {"mode": "none", "value": 0},
        "warmup_policy": {"warmup_rows": int(args.warmup_rows)},
        "boundary_policy": {
            "input_start": "inclusive",
            "input_end": "inclusive",
            "normalized_internal": "[start, end)",
        },
        "invocation_args": {
            "run_id": run_id,
            "input_root": str(args.input_root) if args.input_root is not None else None,
            "split_mode": args.split_mode,
            "split_config": str(args.split_config) if args.split_config is not None else None,
        },
        "file_reports": [],
        "fold_reports": [],
        "errors": [
            {
                "code": SPLIT_RUNTIME_ERROR,
                "message": "Runtime error during split validation.",
                "context": {"error": str(exc)},
            }
        ],
        "warnings": [],
    }


def main() -> int:
    """Run split validation and return deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run-id must be non-empty")

    if args.embargo_bars is not None and args.embargo_seconds is not None:
        raise ValueError("embargo-bars and embargo-seconds are mutually exclusive")

    input_root = args.input_root.resolve() if args.input_root is not None else _default_input_root(run_id).resolve()
    reports_root = _default_reports_root(run_id).resolve()
    report_path = reports_root / "split_validation_report.json"
    summary_path = reports_root / "summary.json"

    try:
        options = SplitValidationOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            split_mode=args.split_mode,
            split_config_path=args.split_config.resolve() if args.split_config is not None else None,
            split_overrides=_build_split_overrides(args),
            require_train_input_validation=_to_bool(args.require_train_input_validation),
            min_train_rows=int(args.min_train_rows),
            min_val_rows=int(args.min_val_rows),
            min_test_rows=int(args.min_test_rows),
            embargo_bars=args.embargo_bars,
            embargo_seconds=args.embargo_seconds,
            warmup_rows=int(args.warmup_rows),
            timestamp_column_override=args.timestamp_column,
            expected_timeframe=args.expected_timeframe,
        )
        validation_report = validate_splits(options)
        payload = validation_report.to_dict()

        _write_report_best_effort(payload, report_path)

        split_error = _derive_validation_error(payload)
        _update_summary_best_effort(
            summary_path=summary_path,
            report_path=report_path,
            split_validation_overall=bool(validation_report.split_validation_overall),
            split_validation_error=split_error,
        )

        exit_code = 0 if validation_report.split_validation_overall else 2
        LOGGER.info(
            "Split validation summary | run_id=%s total=%d success=%d failed=%d overall=%s exit_code=%d report=%s",
            run_id,
            validation_report.total_files,
            validation_report.succeeded_files,
            validation_report.failed_files,
            validation_report.split_validation_overall,
            exit_code,
            report_path,
        )
        return exit_code
    except Exception as exc:
        runtime_payload = _build_runtime_error_payload(run_id=run_id, reports_root=reports_root, args=args, exc=exc)
        _write_report_best_effort(runtime_payload, report_path)
        _update_summary_best_effort(
            summary_path=summary_path,
            report_path=report_path,
            split_validation_overall=False,
            split_validation_error=SPLIT_RUNTIME_ERROR,
        )
        LOGGER.info(
            "Split validation runtime error | run_id=%s exit_code=3 error=%s report=%s",
            run_id,
            exc,
            report_path,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
