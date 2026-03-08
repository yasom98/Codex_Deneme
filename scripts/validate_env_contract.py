"""CLI entrypoint for Milestone 4.5 env contract validation."""

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
from rl.env_contract import (
    ENV_ADAPTER_VERSION,
    ENV_CONTRACT_CONFIG_INVALID,
    ENV_CONTRACT_RUN_ID_MISMATCH,
    ENV_CONTRACT_RUNTIME_ERROR,
    ENV_CONTRACT_VERSION,
    EnvConfig,
    parse_env_config,
    validate_env_contract,
)

LOGGER = get_logger(__name__)


def _default_warmup_contract() -> dict[str, Any]:
    """Return a stable warmup block for CLI contract reports."""

    return {
        "enabled": False,
        "required_observation_columns": [],
        "policy": "drop_head_until_all_required_obs_numeric",
        "valid_from_row": 0,
        "valid_from_timestamp": None,
        "post_valid_nan_policy": "fail_closed",
        "head_nan_profile": {},
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Validate RL env contract against Milestone 4.4 state artifacts.")
    parser.add_argument("--run-id", type=str, required=True, help="Run id under runs/<run_id>/data_states.")
    parser.add_argument("--state-root", type=Path, default=None, help="Optional state root. Default: runs/<run_id>/data_states")
    parser.add_argument("--env-config", type=Path, required=True, help="Strict env config JSON path.")
    parser.add_argument(
        "--smoke-step",
        type=str,
        default="false",
        choices=("true", "false"),
        help="If true, performs one reset/step smoke check after preflight.",
    )
    parser.add_argument("--report-path", type=Path, default=None, help="Optional report path override.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _default_state_root(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_states"


def _default_report_path(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"


def _default_summary_path(run_id: str) -> Path:
    return PROJECT_ROOT / "runs" / run_id / "data_features" / "reports" / "summary.json"


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


def _write_report_best_effort(payload: dict[str, Any], report_path: Path) -> None:
    try:
        atomic_write_json(payload, report_path)
    except RuntimeError as exc:
        LOGGER.info("Env contract report write failed (best-effort) | path=%s error=%s", report_path, exc)


def _derive_error_code(payload: dict[str, Any]) -> str | None:
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            code = first.get("code")
            if isinstance(code, str):
                return code
    return None


def _update_summary_best_effort(
    *,
    summary_path: Path,
    report_path: Path,
    overall: bool,
    error_code: str | None,
) -> None:
    if not summary_path.exists():
        return
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.info("Summary update skipped: read failed | path=%s error=%s", summary_path, exc)
        return

    if not isinstance(payload, dict):
        LOGGER.info("Summary update skipped: payload is not object | path=%s", summary_path)
        return

    payload["env_contract_overall"] = bool(overall)
    payload["env_contract_report_path"] = str(report_path)
    payload["env_contract_error"] = error_code
    try:
        atomic_write_json(payload, summary_path)
    except RuntimeError as exc:
        LOGGER.info("Summary update failed (non-blocking) | path=%s error=%s", summary_path, exc)


def _build_contract_error_payload(
    *,
    run_id: str,
    state_root: Path,
    report_path: Path,
    invocation_args: dict[str, Any],
    code: str,
    message: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "env_contract_version": ENV_CONTRACT_VERSION,
        "env_adapter_version": ENV_ADAPTER_VERSION,
        "env_contract_overall": False,
        "state_root": str(state_root),
        "env_contract_report_path": str(report_path),
        "source_lineage": {},
        "execution_timing_contract": {},
        "position_action_semantics": {},
        "termination_truncation_semantics": {},
        "reward_contract": {},
        "runtime_price_contract": {},
        "warmup_applied": False,
        "warmup_contract": _default_warmup_contract(),
        "episode_valid_start_row": None,
        "effective_episode_start_row": None,
        "seed_reproducibility_contract": {},
        "observation_space_metadata": {"observation_space_type": None, "observation_space_shape": None, "observation_space_dtype": None},
        "action_space_metadata": {"action_space_type": None, "action_space_n": None, "action_mapping": {}},
        "preflight_checks": [],
        "coercions_applied": [],
        "smoke_results": {"executed": False, "success": False},
        "invocation_args": invocation_args,
        "errors": [{"code": code, "message": message, "context": context}],
        "warnings": [],
    }


def _build_runtime_error_payload(
    *,
    run_id: str,
    state_root: Path,
    report_path: Path,
    invocation_args: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return _build_contract_error_payload(
        run_id=run_id,
        state_root=state_root,
        report_path=report_path,
        invocation_args=invocation_args,
        code=ENV_CONTRACT_RUNTIME_ERROR,
        message="Runtime error during env contract validation.",
        context={"error": str(exc)},
    )


def main() -> int:
    """Run env contract validator and return deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run-id must be non-empty")

    state_root = args.state_root.resolve() if args.state_root is not None else _default_state_root(run_id).resolve()
    report_path = args.report_path.resolve() if args.report_path is not None else _default_report_path(run_id).resolve()
    summary_path = _default_summary_path(run_id).resolve()

    invocation_args = {
        "run_id": run_id,
        "state_root": str(state_root),
        "env_config": str(args.env_config.resolve()),
        "smoke_step": _to_bool(args.smoke_step),
        "report_path": str(report_path),
    }

    try:
        payload = _load_json_payload(args.env_config.resolve())

        if "run_id" in payload and payload["run_id"] != run_id:
            contract_payload = _build_contract_error_payload(
                run_id=run_id,
                state_root=state_root,
                report_path=report_path,
                invocation_args=invocation_args,
                code=ENV_CONTRACT_RUN_ID_MISMATCH,
                message="CLI run-id and env-config run_id mismatch.",
                context={"cli_run_id": run_id, "config_run_id": payload.get("run_id")},
            )
            _write_report_best_effort(contract_payload, report_path)
            _update_summary_best_effort(
                summary_path=summary_path,
                report_path=report_path,
                overall=False,
                error_code=ENV_CONTRACT_RUN_ID_MISMATCH,
            )
            return 2

        payload.setdefault("run_id", run_id)
        if "state_root" in payload:
            seen_root = Path(str(payload["state_root"])).resolve()
            if seen_root != state_root:
                contract_payload = _build_contract_error_payload(
                    run_id=run_id,
                    state_root=state_root,
                    report_path=report_path,
                    invocation_args=invocation_args,
                    code=ENV_CONTRACT_CONFIG_INVALID,
                    message="CLI state-root and env-config state_root mismatch.",
                    context={"cli_state_root": str(state_root), "config_state_root": str(seen_root)},
                )
                _write_report_best_effort(contract_payload, report_path)
                _update_summary_best_effort(
                    summary_path=summary_path,
                    report_path=report_path,
                    overall=False,
                    error_code=ENV_CONTRACT_CONFIG_INVALID,
                )
                return 2
        payload.setdefault("state_root", str(state_root))

        try:
            config: EnvConfig = parse_env_config(payload)
        except ValueError as exc:
            contract_payload = _build_contract_error_payload(
                run_id=run_id,
                state_root=state_root,
                report_path=report_path,
                invocation_args=invocation_args,
                code=ENV_CONTRACT_CONFIG_INVALID,
                message="Invalid env config payload.",
                context={"error": str(exc)},
            )
            _write_report_best_effort(contract_payload, report_path)
            _update_summary_best_effort(
                summary_path=summary_path,
                report_path=report_path,
                overall=False,
                error_code=ENV_CONTRACT_CONFIG_INVALID,
            )
            LOGGER.info("Env contract config fail | run_id=%s exit_code=2 error=%s", run_id, exc)
            return 2

        result = validate_env_contract(config=config, smoke_step=_to_bool(args.smoke_step), invocation_args=invocation_args)
        report_payload = result.report_payload
        _write_report_best_effort(report_payload, report_path)

        error_code = _derive_error_code(report_payload)
        overall = bool(report_payload.get("env_contract_overall", False))
        _update_summary_best_effort(
            summary_path=summary_path,
            report_path=report_path,
            overall=overall,
            error_code=error_code,
        )

        exit_code = 0 if overall else 2
        LOGGER.info(
            "Env contract summary | run_id=%s overall=%s exit_code=%d report=%s",
            run_id,
            overall,
            exit_code,
            report_path,
        )
        return exit_code
    except Exception as exc:  # noqa: BLE001
        runtime_payload = _build_runtime_error_payload(
            run_id=run_id,
            state_root=state_root,
            report_path=report_path,
            invocation_args=invocation_args,
            exc=exc,
        )
        _write_report_best_effort(runtime_payload, report_path)
        _update_summary_best_effort(
            summary_path=summary_path,
            report_path=report_path,
            overall=False,
            error_code=ENV_CONTRACT_RUNTIME_ERROR,
        )
        LOGGER.info("Env contract runtime error | run_id=%s exit_code=3 error=%s", run_id, exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
