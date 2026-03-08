"""CLI entrypoint for Milestone 4.6 training env readiness validation."""

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
from rl.env_readiness import (
    READINESS_CONFIG_INVALID,
    START_POLICY_VALID_FROM_ROW,
    EnvReadinessResult,
    validate_training_env_readiness,
)
from rl.episode_catalog import EPISODE_CATALOG_VERSION
from rl.episode_selector import SELECTION_POLICY_FIXED, SELECTION_POLICY_SEEDED_RANDOM

LOGGER = get_logger(__name__)

READINESS_RUNTIME_ERROR = "READINESS_RUNTIME_ERROR"


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Validate Milestone 4.6 training env readiness.")
    parser.add_argument("--run-id", type=str, required=True, help="Run id under runs/<run_id>/data_states.")
    parser.add_argument("--state-root", type=Path, default=None, help="Optional state root. Default: runs/<run_id>/data_states")
    parser.add_argument("--env-config", type=Path, required=True, help="Strict env config JSON path.")
    parser.add_argument(
        "--selection-policy",
        type=str,
        choices=(SELECTION_POLICY_FIXED, SELECTION_POLICY_SEEDED_RANDOM),
        required=True,
        help="Episode selection policy for readiness validation.",
    )
    parser.add_argument(
        "--start-policy",
        type=str,
        choices=(START_POLICY_VALID_FROM_ROW,),
        required=True,
        help="Episode start policy for readiness validation.",
    )
    parser.add_argument("--min-remaining-steps", type=int, required=True, help="Strict minimum remaining step guard.")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic selector/reset seed.")
    parser.add_argument("--catalog-path", type=Path, default=None, help="Optional episode catalog output path override.")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional readiness report path override.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _default_state_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_states"


def _default_catalog_path(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "env_readiness" / "reports" / "episode_catalog.json"


def _default_report_path(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "env_readiness" / "reports" / "training_env_readiness_report.json"


def _load_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"env-config path does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"env-config is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("env-config payload must be JSON object")
    return payload


def _write_json_best_effort(payload: dict[str, Any], path: Path) -> None:
    try:
        atomic_write_json(payload, path)
    except RuntimeError as exc:
        LOGGER.info("Report write failed (best-effort) | path=%s error=%s", path, exc)


def _base_runtime_error_payload(
    *,
    run_id: str,
    state_root: Path,
    catalog_path: Path,
    report_path: Path,
    invocation_args: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness_version": "training_env_readiness.v1",
        "episode_catalog_version": EPISODE_CATALOG_VERSION,
        "overall": False,
        "readiness_overall": False,
        "episode_catalog_overall": None,
        "run_id": run_id,
        "state_root": str(state_root),
        "selection_policy": invocation_args.get("selection_policy"),
        "start_policy": invocation_args.get("start_policy"),
        "seed": invocation_args.get("seed"),
        "catalog_path": str(catalog_path),
        "report_path": str(report_path),
        "selection_trace": {},
        "reset_trace": {},
        "smoke_rollout_trace_summary": {},
        "warnings": [],
        "errors": [
            {
                "code": READINESS_RUNTIME_ERROR,
                "message": "Runtime error during training env readiness validation.",
                "context": {"error": str(exc)},
            }
        ],
    }


def main() -> int:
    """Run the readiness validator and return deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run-id must be non-empty")

    state_root = args.state_root.resolve() if args.state_root is not None else _default_state_root(run_id).resolve()
    catalog_path = args.catalog_path.resolve() if args.catalog_path is not None else _default_catalog_path(run_id).resolve()
    report_path = args.report_path.resolve() if args.report_path is not None else _default_report_path(run_id).resolve()
    invocation_args = {
        "run_id": run_id,
        "state_root": str(state_root),
        "env_config": str(args.env_config.resolve()),
        "selection_policy": args.selection_policy,
        "start_policy": args.start_policy,
        "min_remaining_steps": int(args.min_remaining_steps),
        "seed": int(args.seed),
        "catalog_path": str(catalog_path),
        "report_path": str(report_path),
    }

    try:
        env_config_payload = _load_json_payload(args.env_config.resolve())
        result: EnvReadinessResult = validate_training_env_readiness(
            run_id=run_id,
            state_root=state_root,
            env_config_payload=env_config_payload,
            selection_policy=args.selection_policy,
            start_policy=args.start_policy,
            min_remaining_steps=int(args.min_remaining_steps),
            seed=int(args.seed),
        )
        catalog_payload = dict(result.catalog_payload)
        catalog_payload["catalog_path"] = str(catalog_path)
        readiness_payload = dict(result.readiness_payload)
        readiness_payload["catalog_path"] = str(catalog_path)
        readiness_payload["report_path"] = str(report_path)
        readiness_payload["invocation_args"] = invocation_args

        _write_json_best_effort(catalog_payload, catalog_path)
        _write_json_best_effort(readiness_payload, report_path)

        overall = bool(readiness_payload.get("readiness_overall", False))
        exit_code = 0 if overall else 2
        LOGGER.info(
            "Training env readiness summary | run_id=%s overall=%s exit_code=%d report=%s",
            run_id,
            overall,
            exit_code,
            report_path,
        )
        return exit_code
    except ValueError as exc:
        payload = _base_runtime_error_payload(
            run_id=run_id,
            state_root=state_root,
            catalog_path=catalog_path,
            report_path=report_path,
            invocation_args=invocation_args,
            exc=exc,
        )
        payload["errors"][0]["code"] = READINESS_CONFIG_INVALID
        payload["errors"][0]["message"] = "Invalid training env readiness input."
        _write_json_best_effort(payload, report_path)
        LOGGER.info("Training env readiness config fail | run_id=%s exit_code=2 error=%s", run_id, exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        payload = _base_runtime_error_payload(
            run_id=run_id,
            state_root=state_root,
            catalog_path=catalog_path,
            report_path=report_path,
            invocation_args=invocation_args,
            exc=exc,
        )
        _write_json_best_effort(payload, report_path)
        LOGGER.info("Training env readiness runtime error | run_id=%s exit_code=3 error=%s", run_id, exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
