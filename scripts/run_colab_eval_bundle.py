"""Convenience CLI for staged Colab evaluation-only bundles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.io_atomic import atomic_write_json
from core.logging import get_logger, setup_logging
from rl.colab_runtime import validate_staged_preflight
from rl.evaluation_summary import EVALUATION_SUMMARY_FILENAME, write_evaluation_summary

LOGGER = get_logger(__name__)

CANONICAL_MODEL_FILENAME = "canonical_ppo_model.zip"
DEFAULT_TRAINING_CONFIG_REL = Path("configs") / "training_config.colab_first_real.example.json"
DEFAULT_EVAL_CONFIG_REL = Path("configs") / "eval_config.episodic.example.json"
DEFAULT_STAGE_BASE = Path("/content/codex_stage")
DEFAULT_EVAL_BASE = Path("/content/codex_runs")
DEFAULT_BUNDLE_BASE = Path("/content/codex_eval_bundle")


@dataclass(frozen=True)
class ColabEvalBundleResult:
    """Materialized bundle execution result."""

    exit_code: int
    stage_root: Path
    eval_out_root: Path
    evaluation_summary: dict[str, Any] | None
    evaluation_summary_path: Path | None
    effective_eval_config_path: Path | None
    local_model_artifact_path: Path | None
    stage_performed: bool
    stage_reused: bool


def parse_args() -> argparse.Namespace:
    """Parse Colab evaluation bundle arguments."""

    parser = argparse.ArgumentParser(description="Stage, copy, override, and run one evaluation-only Colab bundle.")
    parser.add_argument("--drive-root", type=Path, required=True, help="Mounted Drive assets root.")
    parser.add_argument("--run-id", type=str, required=True, help="Explicit run lineage id.")
    parser.add_argument("--artifact-attempt-id", type=str, required=True, help="Explicit artifact attempt id.")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT, help="Optional repo root. Default: current project root.")
    parser.add_argument("--stage-root", type=Path, default=None, help="Optional explicit local stage root.")
    parser.add_argument("--eval-out-root", type=Path, default=None, help="Optional explicit fresh local evaluation output root.")
    parser.add_argument("--enable-action-masking", action="store_true", help="Enable local eval_config action masking override.")
    parser.add_argument(
        "--enable-passivity-diagnostics",
        action="store_true",
        help="Enable local eval_config passivity diagnostics override.",
    )
    parser.add_argument("--write-step-trace", action="store_true", help="Enable local eval_config step trace override.")
    parser.add_argument("--print-summary", action="store_true", help="Print a compact JSON evaluation summary to stdout.")
    parser.add_argument(
        "--skip-stage-if-present",
        action="store_true",
        help="Reuse an existing stage root only when staged preflight validation passes.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def execute_colab_eval_bundle(
    *,
    drive_root: Path,
    run_id: str,
    artifact_attempt_id: str,
    repo_root: Path | None = None,
    stage_root: Path | None = None,
    eval_out_root: Path | None = None,
    skip_stage_if_present: bool = False,
    enable_action_masking: bool = False,
    enable_passivity_diagnostics: bool = False,
    write_step_trace: bool = False,
    log_level: str = "INFO",
) -> ColabEvalBundleResult:
    """Execute the staged evaluation-only convenience flow."""

    normalized_run_id = run_id.strip()
    normalized_artifact_attempt_id = artifact_attempt_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must be non-empty")
    if not normalized_artifact_attempt_id:
        raise ValueError("artifact_attempt_id must be non-empty")

    repo_root_resolved = (repo_root or PROJECT_ROOT).resolve()
    drive_root_resolved = drive_root.resolve()
    timestamp = _utcstamp()
    stage_root_resolved = stage_root.resolve() if stage_root is not None else DEFAULT_STAGE_BASE / f"{normalized_run_id}_stage_{timestamp}"
    eval_out_root_resolved = (
        eval_out_root.resolve()
        if eval_out_root is not None
        else DEFAULT_EVAL_BASE / normalized_run_id / f"eval_{timestamp}"
    )
    bundle_root = DEFAULT_BUNDLE_BASE / normalized_run_id / f"bundle_{timestamp}"

    _require_directory(drive_root_resolved, label="drive_root")
    _require_directory(repo_root_resolved, label="repo_root")
    _require_absent(eval_out_root_resolved, label="eval_out_root")
    _require_absent(bundle_root, label="bundle_root")

    stage_performed = False
    stage_reused = False
    if stage_root_resolved.exists():
        if not stage_root_resolved.is_dir():
            raise ValueError(f"stage_root must be a directory when present: {stage_root_resolved}")
        if not skip_stage_if_present:
            raise ValueError(
                f"stage_root already exists and fresh staging is required unless --skip-stage-if-present is set: {stage_root_resolved}"
            )
        stage_validation = validate_staged_preflight(staging_root=stage_root_resolved)
        if not bool(stage_validation.get("overall_valid")):
            raise ValueError(f"Existing stage_root failed staged preflight validation: {stage_root_resolved}")
        stage_reused = True
    else:
        stage_sources = _canonical_stage_source_paths(drive_root=drive_root_resolved, repo_root=repo_root_resolved, run_id=normalized_run_id)
        for label, path in stage_sources.items():
            _require_file(path, label=label)
        stage_command = _build_stage_command(
            repo_root=repo_root_resolved,
            stage_root=stage_root_resolved,
            stage_sources=stage_sources,
            log_level=log_level,
        )
        stage_exit_code = _run_command(stage_command, cwd=repo_root_resolved)
        if stage_exit_code != 0:
            return ColabEvalBundleResult(
                exit_code=stage_exit_code,
                stage_root=stage_root_resolved,
                eval_out_root=eval_out_root_resolved,
                evaluation_summary=None,
                evaluation_summary_path=None,
                effective_eval_config_path=None,
                local_model_artifact_path=None,
                stage_performed=True,
                stage_reused=False,
            )
        stage_performed = True

    local_inputs = _resolve_staged_eval_inputs(stage_root_resolved)
    bundle_root.mkdir(parents=True, exist_ok=False)
    model_source_path = (
        drive_root_resolved
        / "runs"
        / normalized_run_id
        / "ppo_artifact"
        / normalized_artifact_attempt_id
        / CANONICAL_MODEL_FILENAME
    )
    _require_file(model_source_path, label="canonical_model_artifact")
    local_model_path = _copy_model_artifact(model_source_path=model_source_path, bundle_root=bundle_root)

    effective_eval_config_path = local_inputs["eval_config"]
    if enable_action_masking or enable_passivity_diagnostics or write_step_trace:
        effective_eval_config_path = _write_eval_override_config(
            source_eval_config_path=local_inputs["eval_config"],
            output_root=bundle_root / "configs",
            enable_action_masking=enable_action_masking,
            enable_passivity_diagnostics=enable_passivity_diagnostics,
            write_step_trace=write_step_trace,
        )

    eval_command = _build_eval_command(
        repo_root=repo_root_resolved,
        run_id=normalized_run_id,
        model_artifact_path=local_model_path,
        eval_out_root=eval_out_root_resolved,
        eval_config_path=effective_eval_config_path,
        local_inputs=local_inputs,
        log_level=log_level,
    )
    eval_exit_code = _run_command(eval_command, cwd=repo_root_resolved)

    evaluation_summary: dict[str, Any] | None = None
    evaluation_summary_path = eval_out_root_resolved / EVALUATION_SUMMARY_FILENAME
    if eval_exit_code == 0:
        evaluation_summary = write_evaluation_summary(output_dir=eval_out_root_resolved)
    elif eval_out_root_resolved.exists():
        try:
            evaluation_summary = write_evaluation_summary(output_dir=eval_out_root_resolved)
        except ValueError as exc:
            LOGGER.warning(
                "Evaluation summary skipped after nonzero evaluation exit | run_id=%s exit_code=%d error=%s",
                normalized_run_id,
                eval_exit_code,
                exc,
            )

    return ColabEvalBundleResult(
        exit_code=eval_exit_code,
        stage_root=stage_root_resolved,
        eval_out_root=eval_out_root_resolved,
        evaluation_summary=evaluation_summary,
        evaluation_summary_path=evaluation_summary_path if evaluation_summary is not None else None,
        effective_eval_config_path=effective_eval_config_path,
        local_model_artifact_path=local_model_path,
        stage_performed=stage_performed,
        stage_reused=stage_reused,
    )


def main() -> int:
    """Run the convenience bundle and return a bounded exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    try:
        result = execute_colab_eval_bundle(
            drive_root=args.drive_root.resolve(),
            run_id=args.run_id,
            artifact_attempt_id=args.artifact_attempt_id,
            repo_root=args.repo_root.resolve() if args.repo_root is not None else PROJECT_ROOT,
            stage_root=args.stage_root.resolve() if args.stage_root is not None else None,
            eval_out_root=args.eval_out_root.resolve() if args.eval_out_root is not None else None,
            skip_stage_if_present=bool(args.skip_stage_if_present),
            enable_action_masking=bool(args.enable_action_masking),
            enable_passivity_diagnostics=bool(args.enable_passivity_diagnostics),
            write_step_trace=bool(args.write_step_trace),
            log_level=args.log_level,
        )
        if args.print_summary:
            print(
                json.dumps(
                    _build_stdout_summary(result.evaluation_summary, eval_out_root=result.eval_out_root),
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
        LOGGER.info(
            "Colab eval bundle summary | run_id=%s exit_code=%d stage_root=%s eval_out_root=%s stage_performed=%s stage_reused=%s",
            args.run_id,
            result.exit_code,
            result.stage_root,
            result.eval_out_root,
            result.stage_performed,
            result.stage_reused,
        )
        return int(result.exit_code)
    except ValueError as exc:
        LOGGER.error("Colab eval bundle validation failed | error=%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Colab eval bundle runtime error | run_id=%s error=%s", args.run_id, exc)
        return 3


def _canonical_stage_source_paths(*, drive_root: Path, repo_root: Path, run_id: str) -> dict[str, Path]:
    """Return canonical explicit stage inputs for evaluation-only reuse."""

    run_root = drive_root / "runs" / run_id
    return {
        "env_config": run_root / "env_contract" / "tmp" / "bounded_training_preparation_env_config.json",
        "training_config": repo_root / DEFAULT_TRAINING_CONFIG_REL,
        "state_manifest": run_root / "data_states" / "reports" / "state_manifest.json",
        "env_contract_report": run_root / "env_contract" / "reports" / "env_contract_report.json",
        "readiness_report": run_root / "env_readiness" / "reports" / "training_env_readiness_report.json",
        "episode_catalog": run_root / "env_readiness" / "reports" / "episode_catalog.json",
        "split_report": run_root / "data_features" / "reports" / "split_validation_report.json",
        "eval_config": repo_root / DEFAULT_EVAL_CONFIG_REL,
    }


def _resolve_staged_eval_inputs(stage_root: Path) -> dict[str, Path]:
    """Resolve the staged local explicit input set required for evaluation."""

    resolved = {
        "env_config": stage_root / "env_contract" / "tmp" / "bounded_training_preparation_env_config.json",
        "eval_config": stage_root / "configs" / "eval_config.json",
        "state_manifest": stage_root / "data_states" / "reports" / "state_manifest.json",
        "env_contract_report": stage_root / "env_contract" / "reports" / "env_contract_report.json",
        "readiness_report": stage_root / "env_readiness" / "reports" / "training_env_readiness_report.json",
        "episode_catalog": stage_root / "env_readiness" / "reports" / "episode_catalog.json",
        "split_report": stage_root / "data_features" / "reports" / "split_validation_report.json",
    }
    for label, path in resolved.items():
        _require_file(path, label=f"staged_{label}")
    return resolved


def _build_stage_command(
    *,
    repo_root: Path,
    stage_root: Path,
    stage_sources: Mapping[str, Path],
    log_level: str,
) -> list[str]:
    """Build the explicit staging command."""

    return [
        sys.executable,
        str((repo_root / "scripts" / "stage_colab_inputs.py").resolve()),
        "--staging-root",
        str(stage_root),
        "--env-config",
        str(stage_sources["env_config"]),
        "--training-config",
        str(stage_sources["training_config"]),
        "--state-manifest",
        str(stage_sources["state_manifest"]),
        "--env-contract-report",
        str(stage_sources["env_contract_report"]),
        "--readiness-report",
        str(stage_sources["readiness_report"]),
        "--episode-catalog",
        str(stage_sources["episode_catalog"]),
        "--split-report",
        str(stage_sources["split_report"]),
        "--eval-config",
        str(stage_sources["eval_config"]),
        "--log-level",
        log_level,
    ]


def _build_eval_command(
    *,
    repo_root: Path,
    run_id: str,
    model_artifact_path: Path,
    eval_out_root: Path,
    eval_config_path: Path,
    local_inputs: Mapping[str, Path],
    log_level: str,
) -> list[str]:
    """Build the explicit evaluation-only command."""

    return [
        sys.executable,
        str((repo_root / "scripts" / "evaluate_policy.py").resolve()),
        "--run-id",
        run_id,
        "--model-artifact",
        str(model_artifact_path),
        "--env-config",
        str(local_inputs["env_config"]),
        "--eval-config",
        str(eval_config_path),
        "--state-manifest",
        str(local_inputs["state_manifest"]),
        "--env-contract-report",
        str(local_inputs["env_contract_report"]),
        "--readiness-report",
        str(local_inputs["readiness_report"]),
        "--episode-catalog",
        str(local_inputs["episode_catalog"]),
        "--split-report",
        str(local_inputs["split_report"]),
        "--output-dir",
        str(eval_out_root),
        "--progress-mode",
        "auto",
        "--log-level",
        log_level,
    ]


def _copy_model_artifact(*, model_source_path: Path, bundle_root: Path) -> Path:
    """Copy the canonical Drive artifact into the fresh local bundle root."""

    local_model_path = bundle_root / "model" / CANONICAL_MODEL_FILENAME
    local_model_path.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(model_source_path, local_model_path)
    return local_model_path


def _write_eval_override_config(
    *,
    source_eval_config_path: Path,
    output_root: Path,
    enable_action_masking: bool,
    enable_passivity_diagnostics: bool,
    write_step_trace: bool,
) -> Path:
    """Write one local evaluation override config without mutating the staged copy."""

    payload = _load_json_object(source_eval_config_path, label="staged_eval_config")
    if enable_action_masking:
        payload["action_masking"] = True
    if enable_passivity_diagnostics:
        payload["passivity_diagnostics"] = True
    if write_step_trace:
        payload["write_step_trace"] = True

    override_path = output_root / "eval_config.override.json"
    atomic_write_json(payload, override_path)
    return override_path


def _build_stdout_summary(summary_payload: Mapping[str, Any] | None, *, eval_out_root: Path) -> dict[str, Any]:
    """Build the compact stdout summary payload."""

    summary_mapping = summary_payload if isinstance(summary_payload, Mapping) else {}
    return {
        "evaluation_success": summary_mapping.get("evaluation_success"),
        "model_class": summary_mapping.get("model_class"),
        "detected_maskable": summary_mapping.get("detected_maskable"),
        "action_masking_enabled": summary_mapping.get("action_masking_enabled"),
        "passivity_diagnostics_enabled": summary_mapping.get("passivity_diagnostics_enabled"),
        "final_equity": summary_mapping.get("final_equity"),
        "total_return": summary_mapping.get("total_return"),
        "num_trades": summary_mapping.get("num_trades"),
        "deterministic_hold_share": summary_mapping.get("deterministic_hold_share"),
        "deterministic_hold_dominance_margin_band": summary_mapping.get("deterministic_hold_dominance_margin_band"),
        "eval_out": str(eval_out_root),
    }


def _run_command(command: list[str], *, cwd: Path) -> int:
    """Run one explicit subprocess command and return its exit code."""

    LOGGER.info("Subprocess start | cwd=%s command=%s", cwd, command)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    LOGGER.info("Subprocess finish | exit_code=%d command=%s", completed.returncode, command[1] if len(command) > 1 else command)
    return int(completed.returncode)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one JSON object or fail closed."""

    _require_file(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Unreadable JSON for {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must decode to an object for {label}: {path}")
    return payload


def _require_directory(path: Path, *, label: str) -> None:
    """Ensure the given directory exists."""

    if not path.exists():
        raise ValueError(f"Missing required directory for {label}: {path}")
    if not path.is_dir():
        raise ValueError(f"Expected directory for {label}: {path}")


def _require_file(path: Path, *, label: str) -> None:
    """Ensure the given file exists."""

    if not path.exists():
        raise ValueError(f"Missing required file for {label}: {path}")
    if not path.is_file():
        raise ValueError(f"Expected file for {label}: {path}")


def _require_absent(path: Path, *, label: str) -> None:
    """Ensure a fresh path target does not already exist."""

    if path.exists():
        raise ValueError(f"{label} must be fresh and must not already exist: {path}")


def _utcstamp() -> str:
    """Return a stable UTC timestamp string for fresh attempt roots."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
