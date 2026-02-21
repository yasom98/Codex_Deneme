"""CLI entrypoint for dataset builder (Milestone 4.3)."""

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
from data.dataset_builder import (
    DATASET_BUILD_RUNTIME_ERROR,
    DatasetBuildOptions,
    build_datasets,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build reproducible RL datasets from validated split/source artifacts.")
    parser.add_argument("--run-id", type=str, required=True, help="Run id under runs/<run_id>/data_features.")
    parser.add_argument(
        "--overwrite",
        type=str,
        default="false",
        choices=("true", "false"),
        help="If true, allow replacing non-empty output root.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional feature parquet root. Default: runs/<run_id>/data_features/parquet",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional dataset output root. Default: runs/<run_id>/data_datasets",
    )
    parser.add_argument("--dataset-config", type=Path, default=None, help="Optional dataset config path (reserved for additive extension).")
    parser.add_argument("--feature-manifest-path", type=Path, default=None)
    parser.add_argument("--train-input-report-path", type=Path, default=None)
    parser.add_argument("--split-report-path", type=Path, default=None)
    parser.add_argument("--include-feature-groups", type=str, default="", help="Comma-separated feature groups to include.")
    parser.add_argument("--exclude-columns", type=str, default="", help="Comma-separated columns to exclude.")
    parser.add_argument("--timestamp-column", type=str, default=None, help="Optional timestamp column override.")
    parser.add_argument(
        "--require-train-input-validation",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Require train_input_validation_report to be present/passed.",
    )
    parser.add_argument(
        "--require-split-validation",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Require split_validation_report to be present/passed.",
    )
    parser.add_argument(
        "--aggregate-walk-forward",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Generate top-level aggregate partitions in addition to fold outputs for walk_forward mode.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _split_csv_arg(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    out = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(out)


def _default_input_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_features" / "parquet"


def _default_reports_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_features" / "reports"


def _default_output_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_datasets"


def _derive_error_code(payload: dict[str, Any]) -> str | None:
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            code = first.get("code")
            if isinstance(code, str):
                return code
    return None


def _write_report_best_effort(payload: dict[str, Any], report_path: Path) -> None:
    try:
        atomic_write_json(payload, report_path)
    except RuntimeError as exc:
        LOGGER.info("Dataset build report write failed (best-effort) | path=%s error=%s", report_path, exc)


def _update_summary_best_effort(
    *,
    summary_path: Path,
    dataset_build_overall: bool,
    report_path: Path,
    manifest_path: Path,
    dataset_build_error: str | None,
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

    summary_payload["dataset_build_overall"] = bool(dataset_build_overall)
    summary_payload["dataset_build_report_path"] = str(report_path)
    summary_payload["dataset_manifest_path"] = str(manifest_path)
    summary_payload["dataset_build_error"] = dataset_build_error

    try:
        atomic_write_json(summary_payload, summary_path)
    except RuntimeError as exc:
        LOGGER.info("Summary update failed (non-blocking) | path=%s error=%s", summary_path, exc)


def _build_runtime_error_payload(
    *,
    run_id: str,
    input_root: Path,
    output_root: Path,
    report_path: Path,
    manifest_path: Path,
    exc: Exception,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "builder_version": "dataset_builder.v1",
        "dataset_build_overall": False,
        "dataset_build_id": None,
        "build_mode": "materialize_only",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dataset_build_report_path": str(report_path),
        "dataset_manifest_path": str(manifest_path),
        "split_mode": None,
        "output_semantics": {},
        "totals": {"files_processed": 0, "files_failed": 0, "rows_read": 0, "rows_written": 0, "artifacts_written": 0},
        "partition_summaries": {},
        "fold_summaries": {},
        "output_completeness_ok": False,
        "invocation_args": {
            "run_id": run_id,
            "input_root": str(input_root),
            "output_root": str(output_root),
            "overwrite": _to_bool(args.overwrite),
            "aggregate_walk_forward": _to_bool(args.aggregate_walk_forward),
        },
        "errors": [
            {
                "code": DATASET_BUILD_RUNTIME_ERROR,
                "message": "Runtime error during dataset build.",
                "context": {"error": str(exc)},
            }
        ],
        "warnings": [],
    }


def main() -> int:
    """Run dataset builder and return deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run-id must be non-empty")

    input_root = args.input_root.resolve() if args.input_root is not None else _default_input_root(run_id).resolve()
    reports_root = _default_reports_root(run_id).resolve()
    output_root = args.output_root.resolve() if args.output_root is not None else _default_output_root(run_id).resolve()

    report_path = output_root / "reports" / "dataset_build_report.json"
    manifest_path = output_root / "reports" / "dataset_manifest.json"
    summary_path = reports_root / "summary.json"

    try:
        options = DatasetBuildOptions(
            run_id=run_id,
            input_root=input_root,
            reports_root=reports_root,
            output_root=output_root,
            feature_manifest_path=args.feature_manifest_path.resolve() if args.feature_manifest_path is not None else None,
            train_input_report_path=args.train_input_report_path.resolve() if args.train_input_report_path is not None else None,
            split_report_path=args.split_report_path.resolve() if args.split_report_path is not None else None,
            dataset_config_path=args.dataset_config.resolve() if args.dataset_config is not None else None,
            include_feature_groups=_split_csv_arg(args.include_feature_groups),
            exclude_columns=_split_csv_arg(args.exclude_columns),
            timestamp_column_override=args.timestamp_column,
            require_train_input_validation=_to_bool(args.require_train_input_validation),
            require_split_validation=_to_bool(args.require_split_validation),
            aggregate_walk_forward=_to_bool(args.aggregate_walk_forward),
            overwrite=_to_bool(args.overwrite),
            strict_column_selection=True,
            build_mode="materialize_only",
        )

        result = build_datasets(options)
        payload = result.report_payload
        dataset_build_overall = bool(payload.get("dataset_build_overall", False))
        dataset_build_error = _derive_error_code(payload)

        if not result.report_path.exists():
            _write_report_best_effort(payload, result.report_path)

        _update_summary_best_effort(
            summary_path=summary_path,
            dataset_build_overall=dataset_build_overall,
            report_path=result.report_path,
            manifest_path=result.manifest_path,
            dataset_build_error=dataset_build_error,
        )

        exit_code = 0 if dataset_build_overall else 2
        LOGGER.info(
            "Dataset build summary | run_id=%s overall=%s exit_code=%d report=%s manifest=%s",
            run_id,
            dataset_build_overall,
            exit_code,
            result.report_path,
            result.manifest_path,
        )
        return exit_code
    except Exception as exc:  # noqa: BLE001
        payload = _build_runtime_error_payload(
            run_id=run_id,
            input_root=input_root,
            output_root=output_root,
            report_path=report_path,
            manifest_path=manifest_path,
            exc=exc,
            args=args,
        )
        _write_report_best_effort(payload, report_path)
        _update_summary_best_effort(
            summary_path=summary_path,
            dataset_build_overall=False,
            report_path=report_path,
            manifest_path=manifest_path,
            dataset_build_error=DATASET_BUILD_RUNTIME_ERROR,
        )
        LOGGER.info("Dataset build runtime error | run_id=%s exit_code=3 error=%s", run_id, exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
