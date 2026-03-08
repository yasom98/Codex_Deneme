"""CLI entrypoint for state builder (Milestone 4.4)."""

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
from data.state_builder import (
    STATE_BUILD_RUNTIME_ERROR,
    STATE_BUILDER_VERSION,
    StateBuildOptions,
    build_states,
)

LOGGER = get_logger(__name__)


def _default_warmup_contract_summary() -> dict[str, Any]:
    """Return a stable warmup summary block for CLI payloads."""

    return {
        "enabled": False,
        "policy": "drop_head_until_all_required_obs_numeric",
        "post_valid_nan_policy": "fail_closed",
        "artifacts_total": 0,
        "artifacts_with_warmup": 0,
        "max_valid_from_row": 0,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build deterministic RL state artifacts from validated datasets.")
    parser.add_argument("--run-id", type=str, required=True, help="Run id under runs/<run_id>/data_datasets.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional dataset input root. Default: runs/<run_id>/data_datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional state output root. Default: runs/<run_id>/data_states",
    )
    parser.add_argument("--dataset-manifest-path", type=Path, default=None)
    parser.add_argument("--dataset-build-report-path", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        type=str,
        default="false",
        choices=("true", "false"),
        help="If true, allow replacing non-empty output root.",
    )
    parser.add_argument(
        "--enable-scaling",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Enable state scaling contract. Default false.",
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        default="none",
        help="Scaler type. Supported in v1: none, standard. Unsupported values fail contract deterministically.",
    )
    parser.add_argument("--timestamp-column", type=str, default=None)
    parser.add_argument("--build-mode", type=str, default="materialize_only")
    parser.add_argument(
        "--strict-column-selection",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Fail on invalid --state-columns members.",
    )
    parser.add_argument("--state-columns", type=str, default="", help="Comma-separated explicit state columns.")
    parser.add_argument(
        "--execution-price-column",
        type=str,
        default=None,
        help="Required runtime execution price column to carry in state artifacts.",
    )
    parser.add_argument(
        "--mark-to-market-column",
        type=str,
        default=None,
        help="Required runtime mark-to-market price column to carry in state artifacts.",
    )
    parser.add_argument(
        "--sequence-mode",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Deferred in v1; true returns deterministic contract failure.",
    )
    parser.add_argument("--lookback", type=int, default=None, help="Reserved for future sequence mode.")
    parser.add_argument(
        "--aggregate-walk-forward",
        type=str,
        default="false",
        choices=("true", "false"),
        help="Deferred in v1; true returns deterministic contract failure.",
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
    return PROJECT_ROOT / "runs" / run_id / "data_datasets"


def _default_output_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_states"


def _default_features_reports_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_features" / "reports"


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
        LOGGER.info("State build report write failed (best-effort) | path=%s error=%s", report_path, exc)


def _update_summary_best_effort(
    *,
    summary_path: Path,
    state_build_overall: bool,
    report_path: Path,
    manifest_path: Path,
    state_build_error: str | None,
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

    summary_payload["state_build_overall"] = bool(state_build_overall)
    summary_payload["state_build_report_path"] = str(report_path)
    summary_payload["state_manifest_path"] = str(manifest_path)
    summary_payload["state_build_error"] = state_build_error

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
    scaler_stats_path: Path,
    exc: Exception,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "builder_version": STATE_BUILDER_VERSION,
        "state_build_overall": False,
        "state_build_id": None,
        "build_mode": args.build_mode,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "state_build_report_path": str(report_path),
        "state_manifest_path": str(manifest_path),
        "scaler_stats_path": str(scaler_stats_path),
        "split_mode": None,
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
        "invocation_args": {
            "run_id": run_id,
            "input_root": str(input_root),
            "output_root": str(output_root),
            "overwrite": _to_bool(args.overwrite),
            "enable_scaling": _to_bool(args.enable_scaling),
            "scaler_type": args.scaler_type,
            "sequence_mode": _to_bool(args.sequence_mode),
            "aggregate_walk_forward": _to_bool(args.aggregate_walk_forward),
            "execution_price_column": args.execution_price_column,
            "mark_to_market_column": args.mark_to_market_column,
        },
        "warmup_contract_summary": _default_warmup_contract_summary(),
        "errors": [
            {
                "code": STATE_BUILD_RUNTIME_ERROR,
                "message": "Runtime error during state build.",
                "context": {"error": str(exc)},
            }
        ],
        "warnings": [],
    }


def main() -> int:
    """Run state builder and return deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run-id must be non-empty")

    input_root = args.input_root.resolve() if args.input_root is not None else _default_input_root(run_id).resolve()
    output_root = args.output_root.resolve() if args.output_root is not None else _default_output_root(run_id).resolve()

    report_path = output_root / "reports" / "state_build_report.json"
    manifest_path = output_root / "reports" / "state_manifest.json"
    scaler_stats_path = output_root / "reports" / "scaler_stats.json"
    summary_path = _default_features_reports_root(run_id).resolve() / "summary.json"

    try:
        options = StateBuildOptions(
            run_id=run_id,
            input_root=input_root,
            output_root=output_root,
            dataset_manifest_path=args.dataset_manifest_path.resolve() if args.dataset_manifest_path is not None else None,
            dataset_build_report_path=args.dataset_build_report_path.resolve() if args.dataset_build_report_path is not None else None,
            overwrite=_to_bool(args.overwrite),
            enable_scaling=_to_bool(args.enable_scaling),
            scaler_type=str(args.scaler_type),
            timestamp_column_override=args.timestamp_column,
            build_mode=str(args.build_mode),
            strict_column_selection=_to_bool(args.strict_column_selection),
            state_columns=_split_csv_arg(args.state_columns),
            sequence_mode=_to_bool(args.sequence_mode),
            lookback=args.lookback,
            aggregate_walk_forward=_to_bool(args.aggregate_walk_forward),
            execution_price_column=args.execution_price_column,
            mark_to_market_column=args.mark_to_market_column,
        )

        result = build_states(options)
        payload = result.report_payload
        state_build_overall = bool(payload.get("state_build_overall", False))
        state_build_error = _derive_error_code(payload)

        if not result.report_path.exists():
            _write_report_best_effort(payload, result.report_path)

        _update_summary_best_effort(
            summary_path=summary_path,
            state_build_overall=state_build_overall,
            report_path=result.report_path,
            manifest_path=result.manifest_path,
            state_build_error=state_build_error,
        )

        exit_code = 0 if state_build_overall else 2
        LOGGER.info(
            "State build summary | run_id=%s overall=%s exit_code=%d report=%s manifest=%s",
            run_id,
            state_build_overall,
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
            scaler_stats_path=scaler_stats_path,
            exc=exc,
            args=args,
        )
        _write_report_best_effort(payload, report_path)
        _update_summary_best_effort(
            summary_path=summary_path,
            state_build_overall=False,
            report_path=report_path,
            manifest_path=manifest_path,
            state_build_error=STATE_BUILD_RUNTIME_ERROR,
        )
        LOGGER.info("State build runtime error | run_id=%s exit_code=3 error=%s", run_id, exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
