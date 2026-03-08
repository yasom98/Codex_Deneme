"""Strict training launch gate for Milestone 4.7.

4.7 training configs are intentionally starter/validation configs only.
The smoke-oriented config exists to prove bounded launch correctness.
The baseline-train config exists only as an initial PPO starting point.
Hyperparameter optimization is intentionally deferred to Milestone 4.9.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

from core.io_atomic import atomic_write_json
from rl.env_adapter_gym import TradingEnvGym
from rl.env_contract import EnvConfig, parse_env_config

CANONICAL_JSON_POLICY = "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=True)"

ALGORITHM_PPO = "ppo"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_AUTO = "auto"

SELECTION_MODE_FIXED = "fixed_episode"
SELECTION_MODE_SEEDED_RANDOM = "seeded_random_episode"

STARTUP_POLICY_FRESH_ONLY = "fresh_only"

SMOKE_MODE_PRELAUNCH_ONLY = "prelaunch_only"
SMOKE_MODE_LAUNCH_SMOKE = "launch_smoke"

TRAIN_LAUNCH_INPUT_MISSING = "TRAIN_LAUNCH_INPUT_MISSING"
TRAIN_LAUNCH_PATH_UNREADABLE = "TRAIN_LAUNCH_PATH_UNREADABLE"
TRAIN_LAUNCH_JSON_INVALID = "TRAIN_LAUNCH_JSON_INVALID"
TRAIN_LAUNCH_RUN_ID_MISMATCH = "TRAIN_LAUNCH_RUN_ID_MISMATCH"
TRAIN_LAUNCH_STATE_MANIFEST_INVALID = "TRAIN_LAUNCH_STATE_MANIFEST_INVALID"
TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED = "TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED"
TRAIN_LAUNCH_ENV_CONTRACT_FAILED = "TRAIN_LAUNCH_ENV_CONTRACT_FAILED"
TRAIN_LAUNCH_READINESS_REQUIRED = "TRAIN_LAUNCH_READINESS_REQUIRED"
TRAIN_LAUNCH_READINESS_FAILED = "TRAIN_LAUNCH_READINESS_FAILED"
TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED = "TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED"
TRAIN_LAUNCH_EPISODE_CATALOG_FAILED = "TRAIN_LAUNCH_EPISODE_CATALOG_FAILED"
TRAIN_LAUNCH_NO_ELIGIBLE_TRAINING_EPISODES = "TRAIN_LAUNCH_NO_ELIGIBLE_TRAINING_EPISODES"
TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED = "TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED"
TRAIN_LAUNCH_CONFIG_INVALID = "TRAIN_LAUNCH_CONFIG_INVALID"
TRAIN_LAUNCH_ALGO_UNSUPPORTED = "TRAIN_LAUNCH_ALGO_UNSUPPORTED"
TRAIN_LAUNCH_ALGO_PARAMS_INVALID = "TRAIN_LAUNCH_ALGO_PARAMS_INVALID"
TRAIN_LAUNCH_SEED_REQUIRED = "TRAIN_LAUNCH_SEED_REQUIRED"
TRAIN_LAUNCH_TIMESTEPS_INVALID = "TRAIN_LAUNCH_TIMESTEPS_INVALID"
TRAIN_LAUNCH_DEVICE_INVALID = "TRAIN_LAUNCH_DEVICE_INVALID"
TRAIN_LAUNCH_STARTUP_POLICY_INVALID = "TRAIN_LAUNCH_STARTUP_POLICY_INVALID"
TRAIN_LAUNCH_SMOKE_MODE_INVALID = "TRAIN_LAUNCH_SMOKE_MODE_INVALID"
TRAIN_LAUNCH_SMOKE_TIMESTEPS_INVALID = "TRAIN_LAUNCH_SMOKE_TIMESTEPS_INVALID"
TRAIN_LAUNCH_OUTPUT_CONFLICT = "TRAIN_LAUNCH_OUTPUT_CONFLICT"
TRAIN_LAUNCH_FRESH_ONLY_REQUIRED = "TRAIN_LAUNCH_FRESH_ONLY_REQUIRED"
TRAIN_LAUNCH_ENV_INIT_FAILED = "TRAIN_LAUNCH_ENV_INIT_FAILED"
TRAIN_LAUNCH_ALGO_INIT_FAILED = "TRAIN_LAUNCH_ALGO_INIT_FAILED"
TRAIN_LAUNCH_SMOKE_FAILED = "TRAIN_LAUNCH_SMOKE_FAILED"

PPO_REQUIRED_FIELDS = (
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
)

TRAINING_CONFIG_REQUIRED_FIELDS = (
    "algorithm",
    "seed",
    "total_timesteps",
    "device",
    "episode_selection_mode",
    "startup_policy",
    "smoke_mode",
    "smoke_learn_timesteps",
    "algo_params",
)


@dataclass
class ValidationIssue:
    """Machine-readable training launch issue."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PpoAlgoParams:
    """Strict PPO v1 hyperparameter schema."""

    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float

    def to_sb3_kwargs(self) -> dict[str, Any]:
        """Return SB3-compatible keyword arguments."""

        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Strict launcher training config."""

    algorithm: str
    seed: int
    total_timesteps: int
    device: str
    episode_selection_mode: str
    startup_policy: str
    smoke_mode: str
    smoke_learn_timesteps: int
    algo_params: PpoAlgoParams


@dataclass(frozen=True)
class SelectedEpisode:
    """Deterministic effective episode selection evidence."""

    episode_ref: dict[str, Any]
    eligible_domain_used: str
    selected_index: int
    candidate_count: int
    candidate_refs_sorted_hash: str


@dataclass(frozen=True)
class ReportPaths:
    """Stable 4.7 report output paths."""

    validation_report_path: Path
    manifest_path: Path
    smoke_report_path: Path


@dataclass
class LaunchExecutionResult:
    """Composite launcher execution output."""

    exit_code: int
    validation_payload: dict[str, Any]
    manifest_payload: dict[str, Any] | None
    smoke_payload: dict[str, Any] | None
    report_paths: ReportPaths
    reports_written: bool


def execute_training_launch(
    *,
    run_id: str,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    output_dir: Path,
) -> LaunchExecutionResult:
    """Execute strict prelaunch validation and optional bounded PPO smoke."""

    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must be non-empty")

    output_dir_resolved = output_dir.resolve()
    report_paths = ReportPaths(
        validation_report_path=output_dir_resolved / "training_launch_validation_report.json",
        manifest_path=output_dir_resolved / "training_launch_manifest.json",
        smoke_report_path=output_dir_resolved / "training_smoke_report.json",
    )
    launcher_session_id = _build_launcher_session_id(
        run_id=normalized_run_id,
        output_dir=output_dir_resolved,
        env_config_path=env_config_path.resolve(),
        training_config_path=training_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
    )

    output_guard_issues = _check_output_dir_policy(output_dir_resolved)
    if output_guard_issues:
        validation_payload = _build_validation_payload(
            run_id=normalized_run_id,
            launcher_session_id=launcher_session_id,
            selected_algorithm=None,
            selected_episode_mode=None,
            effective_seed=None,
            requested_device=None,
            resolved_device=None,
            config_hash=None,
            readiness_hash=None,
            env_contract_hash=None,
            state_manifest_hash=None,
            episode_catalog_hash=None,
            prelaunch_checks=[
                _prelaunch_check(
                    check_name="startup_policy_fresh_only",
                    passed=False,
                    reason_code=TRAIN_LAUNCH_FRESH_ONLY_REQUIRED,
                    detail={"output_dir": str(output_dir_resolved)},
                ),
                _prelaunch_check(
                    check_name="output_dir_conflict_free",
                    passed=False,
                    reason_code=TRAIN_LAUNCH_OUTPUT_CONFLICT,
                    detail=output_guard_issues[0].context,
                ),
            ],
            warnings=[],
            errors=output_guard_issues,
        )
        smoke_payload = _build_smoke_payload(
            run_id=normalized_run_id,
            launcher_session_id=launcher_session_id,
            smoke_requested=False,
            smoke_mode=None,
            smoke_success=False,
            selected_algorithm=None,
            selected_episode_mode=None,
            effective_seed=None,
            requested_device=None,
            resolved_device=None,
            startup_phase_trace=_phase_trace(
                prelaunch_status="failed",
                env_init_status="not_started",
                algo_init_status="not_started",
                learn_start_status="not_started",
                learn_finish_status="not_started",
                report_write_status="not_written",
                prelaunch_detail={"failure_codes": _failure_codes(output_guard_issues)},
            ),
            launch_guard_results={
                "output_dir": str(output_dir_resolved),
                "output_dir_policy": "fresh_only",
                "reports_written": False,
            },
            smoke_rollout_summary={
                "learn_invoked": False,
                "smoke_learn_timesteps_used": None,
                "smoke_learn_timesteps_unused": True,
            },
            warnings=[],
            errors=output_guard_issues,
        )
        return LaunchExecutionResult(
            exit_code=2,
            validation_payload=validation_payload,
            manifest_payload=None,
            smoke_payload=smoke_payload,
            report_paths=report_paths,
            reports_written=False,
        )

    output_dir_resolved.mkdir(parents=True, exist_ok=False)

    loaded_inputs, load_issues = _load_launcher_inputs(
        env_config_path=env_config_path.resolve(),
        training_config_path=training_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
    )

    prelaunch_checks: list[dict[str, Any]] = []
    warnings: list[ValidationIssue] = []
    errors: list[ValidationIssue] = list(load_issues)

    state_manifest_hash = _semantic_hash_optional(loaded_inputs.get("state_manifest"))
    env_contract_hash = _semantic_hash_optional(loaded_inputs.get("env_contract_report"))
    readiness_hash = _semantic_hash_optional(loaded_inputs.get("readiness_report"))
    episode_catalog_hash = _semantic_hash_optional(loaded_inputs.get("episode_catalog"))
    config_hash = _semantic_hash_optional(loaded_inputs.get("training_config"))

    training_config_result = _validate_training_config(loaded_inputs.get("training_config"))
    training_config = training_config_result["config"]
    errors.extend(training_config_result["errors"])

    requested_device = training_config.device if training_config is not None else _raw_string(
        loaded_inputs.get("training_config"), "device"
    )
    selected_algorithm = training_config.algorithm if training_config is not None else _raw_string(
        loaded_inputs.get("training_config"), "algorithm"
    )
    selected_episode_mode = training_config.episode_selection_mode if training_config is not None else _raw_string(
        loaded_inputs.get("training_config"), "episode_selection_mode"
    )
    effective_seed = training_config.seed if training_config is not None else _raw_int(
        loaded_inputs.get("training_config"), "seed"
    )

    resolved_device, device_issues, dependency_probe = _resolve_device(requested_device)
    if requested_device is None:
        resolved_device = None
    errors.extend(device_issues)

    env_config_result = _validate_env_config(
        env_config_payload=loaded_inputs.get("env_config"),
        cli_run_id=normalized_run_id,
        state_manifest_path=state_manifest_path.resolve(),
        training_seed=effective_seed,
    )
    env_config = env_config_result["config"]
    errors.extend(env_config_result["errors"])

    errors.extend(_validate_state_manifest(loaded_inputs.get("state_manifest"), normalized_run_id))
    errors.extend(
        _validate_env_contract_report(
            report=loaded_inputs.get("env_contract_report"),
            run_id=normalized_run_id,
        )
    )
    errors.extend(
        _validate_readiness_report(
            report=loaded_inputs.get("readiness_report"),
            run_id=normalized_run_id,
        )
    )
    errors.extend(
        _validate_episode_catalog_report(
            report=loaded_inputs.get("episode_catalog"),
            run_id=normalized_run_id,
        )
    )
    errors.extend(
        _validate_lineage_consistency(
            state_manifest_path=state_manifest_path.resolve(),
            state_manifest_payload=loaded_inputs.get("state_manifest"),
            env_contract_report=loaded_inputs.get("env_contract_report"),
            readiness_report=loaded_inputs.get("readiness_report"),
            episode_catalog=loaded_inputs.get("episode_catalog"),
        )
    )

    selected_episode: SelectedEpisode | None = None
    episode_selection_errors: list[ValidationIssue] = []
    if training_config is not None and env_config is not None:
        selected_episode, episode_selection_errors = _resolve_selected_episode(
            episode_catalog=loaded_inputs.get("episode_catalog"),
            episode_selection_mode=training_config.episode_selection_mode,
            seed=training_config.seed,
            env_config=env_config,
        )
        errors.extend(episode_selection_errors)

    prelaunch_checks.extend(_build_prelaunch_checks(load_issues=load_issues, training_config=training_config_result, requested_device=requested_device))
    prelaunch_checks.extend(_build_input_consistency_checks(errors=errors))

    validation_payload = _build_validation_payload(
        run_id=normalized_run_id,
        launcher_session_id=launcher_session_id,
        selected_algorithm=selected_algorithm,
        selected_episode_mode=selected_episode_mode,
        effective_seed=effective_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        config_hash=config_hash,
        readiness_hash=readiness_hash,
        env_contract_hash=env_contract_hash,
        state_manifest_hash=state_manifest_hash,
        episode_catalog_hash=episode_catalog_hash,
        prelaunch_checks=prelaunch_checks,
        warnings=warnings,
        errors=errors,
    )

    prelaunch_pass = not errors
    manifest_payload: dict[str, Any] | None = None

    if prelaunch_pass:
        assert training_config is not None
        assert env_config is not None
        assert selected_episode is not None
        manifest_payload = _build_manifest_payload(
            run_id=normalized_run_id,
            launcher_session_id=launcher_session_id,
            training_config=training_config,
            env_config_path=env_config_path.resolve(),
            training_config_path=training_config_path.resolve(),
            state_manifest_path=state_manifest_path.resolve(),
            env_contract_report_path=env_contract_report_path.resolve(),
            readiness_report_path=readiness_report_path.resolve(),
            episode_catalog_path=episode_catalog_path.resolve(),
            output_dir=output_dir_resolved,
            selected_episode=selected_episode,
            requested_device=requested_device,
            resolved_device=resolved_device,
            config_hash=config_hash,
            readiness_hash=readiness_hash,
            env_contract_hash=env_contract_hash,
            state_manifest_hash=state_manifest_hash,
            episode_catalog_hash=episode_catalog_hash,
            loaded_inputs=loaded_inputs,
        )

    if not prelaunch_pass:
        smoke_payload = _build_smoke_payload(
            run_id=normalized_run_id,
            launcher_session_id=launcher_session_id,
            smoke_requested=False,
            smoke_mode=_raw_string(loaded_inputs.get("training_config"), "smoke_mode"),
            smoke_success=False,
            selected_algorithm=selected_algorithm,
            selected_episode_mode=selected_episode_mode,
            effective_seed=effective_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            startup_phase_trace=_phase_trace(
                prelaunch_status="failed",
                env_init_status="not_started",
                algo_init_status="not_started",
                learn_start_status="not_started",
                learn_finish_status="not_started",
                report_write_status="completed",
                prelaunch_detail={"failure_codes": validation_payload["failure_codes"]},
            ),
            launch_guard_results={
                "prelaunch_overall_pass": False,
                "dependency_probe": dependency_probe,
                "output_dir": str(output_dir_resolved),
                "selected_episode_ref": selected_episode.episode_ref if selected_episode is not None else None,
            },
            smoke_rollout_summary={
                "learn_invoked": False,
                "smoke_learn_timesteps_used": None,
                "smoke_learn_timesteps_unused": True,
            },
            warnings=warnings,
            errors=errors,
        )
        _write_reports(validation_payload, manifest_payload, smoke_payload, report_paths)
        return LaunchExecutionResult(
            exit_code=2,
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            smoke_payload=smoke_payload,
            report_paths=report_paths,
            reports_written=True,
        )

    assert training_config is not None
    assert env_config is not None
    assert selected_episode is not None

    if training_config.smoke_mode == SMOKE_MODE_PRELAUNCH_ONLY:
        smoke_payload = _build_smoke_payload(
            run_id=normalized_run_id,
            launcher_session_id=launcher_session_id,
            smoke_requested=False,
            smoke_mode=training_config.smoke_mode,
            smoke_success=True,
            selected_algorithm=training_config.algorithm,
            selected_episode_mode=training_config.episode_selection_mode,
            effective_seed=training_config.seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            startup_phase_trace=_phase_trace(
                prelaunch_status="completed",
                env_init_status="skipped",
                algo_init_status="skipped",
                learn_start_status="skipped",
                learn_finish_status="skipped",
                report_write_status="completed",
                prelaunch_detail={"overall_pass": True},
            ),
            launch_guard_results={
                "prelaunch_overall_pass": True,
                "selected_episode_ref": selected_episode.episode_ref,
                "selection_evidence": asdict(selected_episode),
                "dependency_probe": dependency_probe,
                "smoke_learn_timesteps_unused": True,
                "output_dir": str(output_dir_resolved),
            },
            smoke_rollout_summary={
                "learn_invoked": False,
                "smoke_learn_timesteps_requested": training_config.smoke_learn_timesteps,
                "smoke_learn_timesteps_used": None,
                "smoke_learn_timesteps_unused": True,
            },
            warnings=warnings,
            errors=[],
        )
        _write_reports(validation_payload, manifest_payload, smoke_payload, report_paths)
        return LaunchExecutionResult(
            exit_code=0,
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            smoke_payload=smoke_payload,
            report_paths=report_paths,
            reports_written=True,
        )

    smoke_payload = _run_launch_smoke(
        run_id=normalized_run_id,
        launcher_session_id=launcher_session_id,
        training_config=training_config,
        env_config=env_config,
        selected_episode=selected_episode,
        requested_device=requested_device,
        resolved_device=resolved_device,
        dependency_probe=dependency_probe,
        warnings=warnings,
        manifest_payload=manifest_payload,
    )
    _write_reports(validation_payload, manifest_payload, smoke_payload, report_paths)
    return LaunchExecutionResult(
        exit_code=0 if bool(smoke_payload.get("smoke_success")) else 2,
        validation_payload=validation_payload,
        manifest_payload=manifest_payload,
        smoke_payload=smoke_payload,
        report_paths=report_paths,
        reports_written=True,
    )


def _load_launcher_inputs(
    *,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
) -> tuple[dict[str, dict[str, Any] | None], list[ValidationIssue]]:
    """Load all explicit launcher JSON inputs."""

    inputs: dict[str, dict[str, Any] | None] = {}
    issues: list[ValidationIssue] = []
    path_specs = {
        "env_config": env_config_path,
        "training_config": training_config_path,
        "state_manifest": state_manifest_path,
        "env_contract_report": env_contract_report_path,
        "readiness_report": readiness_report_path,
        "episode_catalog": episode_catalog_path,
    }

    for label, path in path_specs.items():
        payload, error = _load_json_object(path=path, label=label)
        inputs[label] = payload
        if error is not None:
            issues.append(error)
    return inputs, issues


def _load_json_object(*, path: Path, label: str) -> tuple[dict[str, Any] | None, ValidationIssue | None]:
    """Load one required JSON object with strict error mapping."""

    if not path.exists():
        return None, ValidationIssue(
            code=TRAIN_LAUNCH_INPUT_MISSING,
            message="Required launcher input is missing.",
            context={"input_label": label, "path": str(path)},
        )
    if not path.is_file():
        return None, ValidationIssue(
            code=TRAIN_LAUNCH_PATH_UNREADABLE,
            message="Required launcher input path is not a readable file.",
            context={"input_label": label, "path": str(path)},
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, ValidationIssue(
            code=TRAIN_LAUNCH_JSON_INVALID,
            message="Required launcher input contains invalid JSON.",
            context={"input_label": label, "path": str(path), "error": str(exc)},
        )
    except OSError as exc:
        return None, ValidationIssue(
            code=TRAIN_LAUNCH_PATH_UNREADABLE,
            message="Required launcher input could not be read.",
            context={"input_label": label, "path": str(path), "error": str(exc)},
        )

    if not isinstance(payload, dict):
        return None, ValidationIssue(
            code=TRAIN_LAUNCH_JSON_INVALID,
            message="Required launcher input JSON must be an object.",
            context={"input_label": label, "path": str(path)},
        )
    return payload, None


def _validate_training_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validate strict training config schema."""

    errors: list[ValidationIssue] = []
    if payload is None:
        return {"config": None, "errors": errors}

    extra_keys = sorted(set(payload.keys()) - set(TRAINING_CONFIG_REQUIRED_FIELDS))
    missing_keys = sorted(set(TRAINING_CONFIG_REQUIRED_FIELDS) - set(payload.keys()))
    if missing_keys or extra_keys:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_CONFIG_INVALID,
                message="training_config top-level fields must match the 4.7 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return {"config": None, "errors": errors}

    algorithm = _raw_string(payload, "algorithm")
    if algorithm != ALGORITHM_PPO:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ALGO_UNSUPPORTED,
                message="Only PPO is supported in 4.7 v1.",
                context={"algorithm": algorithm},
            )
        )

    seed_raw = payload.get("seed")
    if not isinstance(seed_raw, int):
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_SEED_REQUIRED,
                message="seed must be a non-null integer.",
                context={"seed": seed_raw},
            )
        )
    total_timesteps_raw = payload.get("total_timesteps")
    if not isinstance(total_timesteps_raw, int) or int(total_timesteps_raw) <= 0:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_TIMESTEPS_INVALID,
                message="total_timesteps must be a positive integer.",
                context={"total_timesteps": total_timesteps_raw},
            )
        )

    device = _raw_string(payload, "device")
    if device not in {DEVICE_CPU, DEVICE_CUDA, DEVICE_AUTO}:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_DEVICE_INVALID,
                message="device must be one of cpu, cuda, auto.",
                context={"device": device},
            )
        )

    episode_selection_mode = _raw_string(payload, "episode_selection_mode")
    if episode_selection_mode not in {SELECTION_MODE_FIXED, SELECTION_MODE_SEEDED_RANDOM}:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED,
                message="episode_selection_mode is unsupported.",
                context={"episode_selection_mode": episode_selection_mode},
            )
        )

    startup_policy = _raw_string(payload, "startup_policy")
    if startup_policy != STARTUP_POLICY_FRESH_ONLY:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_STARTUP_POLICY_INVALID,
                message="startup_policy must be fresh_only.",
                context={"startup_policy": startup_policy},
            )
        )

    smoke_mode = _raw_string(payload, "smoke_mode")
    if smoke_mode not in {SMOKE_MODE_PRELAUNCH_ONLY, SMOKE_MODE_LAUNCH_SMOKE}:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_SMOKE_MODE_INVALID,
                message="smoke_mode is unsupported.",
                context={"smoke_mode": smoke_mode},
            )
        )

    smoke_learn_timesteps_raw = payload.get("smoke_learn_timesteps")
    if not isinstance(smoke_learn_timesteps_raw, int) or int(smoke_learn_timesteps_raw) <= 0:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_SMOKE_TIMESTEPS_INVALID,
                message="smoke_learn_timesteps must be a positive integer.",
                context={"smoke_learn_timesteps": smoke_learn_timesteps_raw},
            )
        )
    elif isinstance(total_timesteps_raw, int) and int(total_timesteps_raw) > 0 and int(smoke_learn_timesteps_raw) > int(total_timesteps_raw):
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_SMOKE_TIMESTEPS_INVALID,
                message="smoke_learn_timesteps must be <= total_timesteps.",
                context={
                    "smoke_learn_timesteps": int(smoke_learn_timesteps_raw),
                    "total_timesteps": int(total_timesteps_raw),
                },
            )
        )

    algo_params_raw = payload.get("algo_params")
    algo_params, algo_param_errors = _validate_algo_params(algo_params_raw)
    errors.extend(algo_param_errors)

    if errors:
        return {"config": None, "errors": errors}

    assert isinstance(seed_raw, int)
    assert isinstance(total_timesteps_raw, int)
    assert isinstance(smoke_learn_timesteps_raw, int)
    assert algo_params is not None
    return {
        "config": TrainingConfig(
            algorithm=algorithm,
            seed=int(seed_raw),
            total_timesteps=int(total_timesteps_raw),
            device=device,
            episode_selection_mode=episode_selection_mode,
            startup_policy=startup_policy,
            smoke_mode=smoke_mode,
            smoke_learn_timesteps=int(smoke_learn_timesteps_raw),
            algo_params=algo_params,
        ),
        "errors": errors,
    }


def _validate_algo_params(payload: Any) -> tuple[PpoAlgoParams | None, list[ValidationIssue]]:
    """Validate exact PPO v1 algo params."""

    errors: list[ValidationIssue] = []
    if not isinstance(payload, Mapping):
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ALGO_PARAMS_INVALID,
                message="algo_params must be an object.",
                context={"algo_params_type": type(payload).__name__},
            )
        )
        return None, errors

    extra_keys = sorted(set(payload.keys()) - set(PPO_REQUIRED_FIELDS))
    missing_keys = sorted(set(PPO_REQUIRED_FIELDS) - set(payload.keys()))
    if missing_keys or extra_keys:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ALGO_PARAMS_INVALID,
                message="algo_params fields must match the strict PPO v1 schema exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return None, errors

    numeric_values: dict[str, float] = {}
    int_fields = {"n_steps", "batch_size", "n_epochs"}
    positive_fields = {"learning_rate", "n_steps", "batch_size", "n_epochs", "max_grad_norm"}
    non_negative_fields = {"clip_range", "ent_coef", "vf_coef"}

    for field_name in PPO_REQUIRED_FIELDS:
        value = payload.get(field_name)
        if field_name in int_fields:
            if not isinstance(value, int) or int(value) <= 0:
                errors.append(
                    ValidationIssue(
                        code=TRAIN_LAUNCH_ALGO_PARAMS_INVALID,
                        message="PPO integer hyperparameters must be positive integers.",
                        context={"field": field_name, "value": value},
                    )
                )
                continue
            numeric_values[field_name] = float(int(value))
            continue

        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            errors.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_ALGO_PARAMS_INVALID,
                    message="PPO floating hyperparameters must be finite numbers.",
                    context={"field": field_name, "value": value},
                )
            )
            continue

        numeric_values[field_name] = float(value)

    for field_name in positive_fields:
        if field_name in numeric_values and numeric_values[field_name] <= 0.0:
            errors.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_ALGO_PARAMS_INVALID,
                    message="PPO positive hyperparameters must be > 0.",
                    context={"field": field_name, "value": numeric_values[field_name]},
                )
            )
    for field_name in non_negative_fields:
        if field_name in numeric_values and numeric_values[field_name] < 0.0:
            errors.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_ALGO_PARAMS_INVALID,
                    message="PPO bounded hyperparameters must be >= 0.",
                    context={"field": field_name, "value": numeric_values[field_name]},
                )
            )
    for field_name in ("gamma", "gae_lambda"):
        if field_name in numeric_values and not (0.0 < numeric_values[field_name] <= 1.0):
            errors.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_ALGO_PARAMS_INVALID,
                    message="gamma and gae_lambda must be within (0, 1].",
                    context={"field": field_name, "value": numeric_values[field_name]},
                )
            )

    if errors:
        return None, errors

    return (
        PpoAlgoParams(
            learning_rate=numeric_values["learning_rate"],
            n_steps=int(numeric_values["n_steps"]),
            batch_size=int(numeric_values["batch_size"]),
            n_epochs=int(numeric_values["n_epochs"]),
            gamma=numeric_values["gamma"],
            gae_lambda=numeric_values["gae_lambda"],
            clip_range=numeric_values["clip_range"],
            ent_coef=numeric_values["ent_coef"],
            vf_coef=numeric_values["vf_coef"],
            max_grad_norm=numeric_values["max_grad_norm"],
        ),
        errors,
    )


def _validate_env_config(
    *,
    env_config_payload: dict[str, Any] | None,
    cli_run_id: str,
    state_manifest_path: Path,
    training_seed: int | None,
) -> dict[str, Any]:
    """Validate input env config without mutating the source file."""

    errors: list[ValidationIssue] = []
    if env_config_payload is None:
        return {"config": None, "errors": errors}

    config_run_id = _raw_string(env_config_payload, "run_id")
    if config_run_id is not None and config_run_id != cli_run_id:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_RUN_ID_MISMATCH,
                message="env_config.run_id does not match CLI run_id.",
                context={"env_config_run_id": config_run_id, "cli_run_id": cli_run_id},
            )
        )

    state_root_raw = _raw_string(env_config_payload, "state_root")
    if state_root_raw is None:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_CONFIG_INVALID,
                message="env_config.state_root is required.",
                context={},
            )
        )
    else:
        expected_state_root = state_manifest_path.resolve().parents[1]
        try:
            resolved_state_root = Path(state_root_raw).resolve()
        except OSError as exc:
            errors.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_CONFIG_INVALID,
                    message="env_config.state_root could not be resolved.",
                    context={"state_root": state_root_raw, "error": str(exc)},
                )
            )
        else:
            if resolved_state_root != expected_state_root:
                errors.append(
                    ValidationIssue(
                        code=TRAIN_LAUNCH_CONFIG_INVALID,
                        message="env_config.state_root does not match the explicit state_manifest path.",
                        context={"env_config_state_root": str(resolved_state_root), "expected_state_root": str(expected_state_root)},
                    )
                )

    env_seed = env_config_payload.get("seed")
    if env_seed is not None:
        if not isinstance(env_seed, int):
            errors.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_CONFIG_INVALID,
                    message="env_config.seed must be integer or null.",
                    context={"seed": env_seed},
                )
            )
        elif training_seed is not None and env_seed != training_seed:
            errors.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_CONFIG_INVALID,
                    message="env_config.seed must match training_config.seed when provided.",
                    context={"env_config_seed": env_seed, "training_seed": training_seed},
                )
            )

    if errors:
        return {"config": None, "errors": errors}

    try:
        parsed = parse_env_config(env_config_payload)
    except ValueError as exc:
        errors.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_CONFIG_INVALID,
                message="env_config is invalid for runtime launch.",
                context={"error": str(exc)},
            )
        )
        return {"config": None, "errors": errors}
    return {"config": parsed, "errors": errors}


def _validate_state_manifest(payload: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate minimum canonical state manifest evidence."""

    issues: list[ValidationIssue] = []
    if payload is None:
        return issues
    if payload.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_RUN_ID_MISMATCH,
                message="state_manifest.run_id mismatch.",
                context={"state_manifest_run_id": payload.get("run_id"), "cli_run_id": run_id},
            )
        )
    if payload.get("output_completeness_ok") is not True:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_STATE_MANIFEST_INVALID,
                message="state_manifest.output_completeness_ok must be true.",
                context={"output_completeness_ok": payload.get("output_completeness_ok")},
            )
        )
    return issues


def _validate_env_contract_report(report: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate canonical env contract gate evidence."""

    issues: list[ValidationIssue] = []
    if report is None:
        return issues
    if report.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_RUN_ID_MISMATCH,
                message="env_contract_report.run_id mismatch.",
                context={"env_contract_report_run_id": report.get("run_id"), "cli_run_id": run_id},
            )
        )
    if "env_contract_overall" not in report:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED,
                message="env_contract_report must contain env_contract_overall.",
                context={},
            )
        )
    elif report.get("env_contract_overall") is not True:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ENV_CONTRACT_FAILED,
                message="env_contract_report must pass before training launch.",
                context={"env_contract_overall": report.get("env_contract_overall")},
            )
        )
    if not isinstance(report.get("source_lineage"), Mapping):
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED,
                message="env_contract_report.source_lineage is required.",
                context={},
            )
        )
    return issues


def _validate_readiness_report(report: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate canonical readiness gate evidence."""

    issues: list[ValidationIssue] = []
    if report is None:
        return issues
    if report.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_RUN_ID_MISMATCH,
                message="training_env_readiness_report.run_id mismatch.",
                context={"readiness_report_run_id": report.get("run_id"), "cli_run_id": run_id},
            )
        )
    if "readiness_overall" not in report:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_READINESS_REQUIRED,
                message="training_env_readiness_report must contain readiness_overall.",
                context={},
            )
        )
    elif report.get("readiness_overall") is not True:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_READINESS_FAILED,
                message="training_env_readiness_report must pass before training launch.",
                context={"readiness_overall": report.get("readiness_overall")},
            )
        )
    if report.get("episode_catalog_overall") is not True:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_READINESS_FAILED,
                message="training_env_readiness_report must confirm episode_catalog_overall=true.",
                context={"episode_catalog_overall": report.get("episode_catalog_overall")},
            )
        )
    env_contract_reference = report.get("env_contract_reference")
    if not isinstance(env_contract_reference, Mapping) or not isinstance(env_contract_reference.get("source_lineage"), Mapping):
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_READINESS_REQUIRED,
                message="training_env_readiness_report.env_contract_reference.source_lineage is required.",
                context={},
            )
        )
    return issues


def _validate_episode_catalog_report(report: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate canonical episode catalog gate evidence."""

    issues: list[ValidationIssue] = []
    if report is None:
        return issues
    if report.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_RUN_ID_MISMATCH,
                message="episode_catalog.run_id mismatch.",
                context={"episode_catalog_run_id": report.get("run_id"), "cli_run_id": run_id},
            )
        )
    if "episode_catalog_overall" not in report:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED,
                message="episode_catalog must contain episode_catalog_overall.",
                context={},
            )
        )
    elif report.get("episode_catalog_overall") is not True:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_FAILED,
                message="episode_catalog must pass before training launch.",
                context={"episode_catalog_overall": report.get("episode_catalog_overall")},
            )
        )
    if not isinstance(report.get("eligible_episode_count_by_domain"), Mapping):
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED,
                message="episode_catalog.eligible_episode_count_by_domain is required.",
                context={},
            )
        )
    if not isinstance(report.get("eligible_episode_refs_sorted_by_domain"), Mapping):
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED,
                message="episode_catalog.eligible_episode_refs_sorted_by_domain is required.",
                context={},
            )
        )
    if not isinstance(report.get("episodes"), list):
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED,
                message="episode_catalog.episodes is required.",
                context={},
            )
        )
    if not isinstance(report.get("source_lineage"), Mapping):
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED,
                message="episode_catalog.source_lineage is required.",
                context={},
            )
        )
    return issues


def _validate_lineage_consistency(
    *,
    state_manifest_path: Path,
    state_manifest_payload: dict[str, Any] | None,
    env_contract_report: dict[str, Any] | None,
    readiness_report: dict[str, Any] | None,
    episode_catalog: dict[str, Any] | None,
) -> list[ValidationIssue]:
    """Validate only actually-present canonical lineage evidence."""

    issues: list[ValidationIssue] = []
    if state_manifest_payload is None or env_contract_report is None or readiness_report is None or episode_catalog is None:
        return issues

    actual_state_manifest_file_sha = _sha256_file(state_manifest_path)
    env_lineage = env_contract_report.get("source_lineage")
    readiness_env_ref = readiness_report.get("env_contract_reference")
    readiness_lineage = readiness_env_ref.get("source_lineage") if isinstance(readiness_env_ref, Mapping) else None
    catalog_lineage = episode_catalog.get("source_lineage")

    if not isinstance(env_lineage, Mapping):
        return issues
    if not isinstance(readiness_lineage, Mapping):
        return issues
    if not isinstance(catalog_lineage, Mapping):
        return issues

    expected_path = str(state_manifest_path.resolve())
    env_state_path = env_lineage.get("state_manifest_path")
    readiness_state_path = readiness_lineage.get("state_manifest_path")
    catalog_state_path = catalog_lineage.get("state_manifest_path")
    env_state_hash = env_lineage.get("state_manifest_hash")
    readiness_state_hash = readiness_lineage.get("state_manifest_hash")
    catalog_state_hash = catalog_lineage.get("state_manifest_hash")

    if env_state_path != expected_path:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ENV_CONTRACT_FAILED,
                message="env_contract_report.state_manifest_path does not match the explicit launcher input.",
                context={"reported_path": env_state_path, "expected_path": expected_path},
            )
        )
    if readiness_state_path != expected_path:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_READINESS_FAILED,
                message="training_env_readiness_report lineage does not match the explicit state_manifest path.",
                context={"reported_path": readiness_state_path, "expected_path": expected_path},
            )
        )
    if catalog_state_path != expected_path:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_FAILED,
                message="episode_catalog lineage does not match the explicit state_manifest path.",
                context={"reported_path": catalog_state_path, "expected_path": expected_path},
            )
        )
    if env_state_hash != actual_state_manifest_file_sha:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ENV_CONTRACT_FAILED,
                message="env_contract_report lineage hash does not match the explicit state_manifest file.",
                context={"reported_hash": env_state_hash, "actual_hash": actual_state_manifest_file_sha},
            )
        )
    if readiness_state_hash != actual_state_manifest_file_sha:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_READINESS_FAILED,
                message="training_env_readiness_report lineage hash does not match the explicit state_manifest file.",
                context={"reported_hash": readiness_state_hash, "actual_hash": actual_state_manifest_file_sha},
            )
        )
    if catalog_state_hash != actual_state_manifest_file_sha:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_FAILED,
                message="episode_catalog lineage hash does not match the explicit state_manifest file.",
                context={"reported_hash": catalog_state_hash, "actual_hash": actual_state_manifest_file_sha},
            )
        )

    env_build_hash = env_lineage.get("state_build_report_hash")
    readiness_build_hash = readiness_lineage.get("state_build_report_hash")
    catalog_build_hash = catalog_lineage.get("state_build_report_hash")
    if env_build_hash and readiness_build_hash and env_build_hash != readiness_build_hash:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_READINESS_FAILED,
                message="training_env_readiness_report state_build_report hash does not match env_contract_report.",
                context={"env_contract_hash": env_build_hash, "readiness_hash": readiness_build_hash},
            )
        )
    if env_build_hash and catalog_build_hash and env_build_hash != catalog_build_hash:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_FAILED,
                message="episode_catalog state_build_report hash does not match env_contract_report.",
                context={"env_contract_hash": env_build_hash, "episode_catalog_hash": catalog_build_hash},
            )
        )
    return issues


def _resolve_selected_episode(
    *,
    episode_catalog: dict[str, Any] | None,
    episode_selection_mode: str,
    seed: int,
    env_config: EnvConfig,
) -> tuple[SelectedEpisode | None, list[ValidationIssue]]:
    """Resolve effective runtime episode using only canonical catalog evidence."""

    issues: list[ValidationIssue] = []
    if episode_catalog is None:
        return None, issues

    eligible_domain_map = episode_catalog.get("eligible_episode_refs_sorted_by_domain")
    count_map = episode_catalog.get("eligible_episode_count_by_domain")
    episodes = episode_catalog.get("episodes")
    if not isinstance(eligible_domain_map, Mapping) or not isinstance(count_map, Mapping) or not isinstance(episodes, list):
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED,
                message="episode_catalog does not expose the canonical selection fields required by 4.7.",
                context={},
            )
        )
        return None, issues

    entries_by_key = _episode_entries_by_key(episodes)

    if episode_selection_mode == SELECTION_MODE_SEEDED_RANDOM:
        training_candidates = eligible_domain_map.get("training")
        training_count = count_map.get("training")
        if not isinstance(training_candidates, list) or not isinstance(training_count, int):
            issues.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED,
                    message="episode_catalog training-domain selection evidence is required.",
                    context={},
                )
            )
            return None, issues
        if training_count <= 0 or not training_candidates:
            issues.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_NO_ELIGIBLE_TRAINING_EPISODES,
                    message="episode_catalog training domain is empty.",
                    context={"training_count": training_count},
                )
            )
            return None, issues
        normalized_candidates = [_normalize_episode_ref(candidate) for candidate in training_candidates]
        selected_index = int(random.Random(seed).randrange(len(normalized_candidates)))
        selected_ref = normalized_candidates[selected_index]
        return (
            SelectedEpisode(
                episode_ref=selected_ref,
                eligible_domain_used="training",
                selected_index=selected_index,
                candidate_count=len(normalized_candidates),
                candidate_refs_sorted_hash=_hash_canonical_json(normalized_candidates),
            ),
            issues,
        )

    if episode_selection_mode != SELECTION_MODE_FIXED:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED,
                message="episode_selection_mode is unsupported.",
                context={"episode_selection_mode": episode_selection_mode},
            )
        )
        return None, issues

    fixed_ref = {
        "scope": env_config.episode_ref.scope,
        "partition": env_config.episode_ref.partition,
        "source_rel": env_config.episode_ref.source_rel,
        "fold_id": env_config.episode_ref.fold_id,
    }
    entry = entries_by_key.get(_episode_ref_key(fixed_ref))
    if entry is None:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED,
                message="env_config.episode_ref was not found in episode_catalog.",
                context={"episode_ref": fixed_ref},
            )
        )
        return None, issues
    if entry.get("eligible_for_readiness") is not True:
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED,
                message="fixed_episode requires a readiness-eligible env_config.episode_ref.",
                context={
                    "episode_ref": fixed_ref,
                    "eligible_for_readiness": entry.get("eligible_for_readiness"),
                    "readiness_eligibility_reasons": entry.get("readiness_eligibility_reasons"),
                },
            )
        )
        return None, issues

    readiness_candidates_raw = eligible_domain_map.get("readiness")
    readiness_candidates = [_normalize_episode_ref(item) for item in readiness_candidates_raw] if isinstance(readiness_candidates_raw, list) else []
    try:
        selected_index = readiness_candidates.index(fixed_ref)
    except ValueError:
        selected_index = -1
    return (
        SelectedEpisode(
            episode_ref=fixed_ref,
            eligible_domain_used="readiness",
            selected_index=selected_index,
            candidate_count=len(readiness_candidates),
            candidate_refs_sorted_hash=_hash_canonical_json(readiness_candidates),
        ),
        issues,
    )


def _run_launch_smoke(
    *,
    run_id: str,
    launcher_session_id: str,
    training_config: TrainingConfig,
    env_config: EnvConfig,
    selected_episode: SelectedEpisode,
    requested_device: str | None,
    resolved_device: str | None,
    dependency_probe: dict[str, Any],
    warnings: list[ValidationIssue],
    manifest_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run bounded env/algo startup smoke after strict prelaunch validation."""

    phase_status = {
        "prelaunch_validation": "completed",
        "env_init": "not_started",
        "algo_init": "not_started",
        "learn_start": "not_started",
        "learn_finish": "not_started",
        "report_write": "completed",
    }
    phase_detail: dict[str, Any] = {
        "prelaunch_validation": {"overall_pass": True},
        "env_init": {},
        "algo_init": {},
        "learn_start": {},
        "learn_finish": {},
        "report_write": {},
    }
    issues: list[ValidationIssue] = []
    smoke_rollout_summary: dict[str, Any] = {
        "learn_invoked": False,
        "smoke_learn_timesteps_requested": int(training_config.smoke_learn_timesteps),
        "smoke_learn_timesteps_used": int(training_config.smoke_learn_timesteps),
        "selected_episode_ref": dict(selected_episode.episode_ref),
    }
    startup_metadata = _set_global_seed(training_config.seed)
    smoke_rollout_summary["startup_seed_metadata"] = startup_metadata
    smoke_rollout_summary["manifest_hash_snapshot"] = _hash_canonical_json(manifest_payload) if manifest_payload is not None else None

    effective_env_config = _effective_env_config(env_config=env_config, seed=training_config.seed, episode_ref=selected_episode.episode_ref)
    env_client: Any | None = None
    try:
        env_client = TradingEnvGym(config=effective_env_config, validate_on_init=True)
        phase_status["env_init"] = "completed"
        phase_detail["env_init"] = {
            "env_class": type(env_client).__name__,
            "selected_episode_ref": dict(selected_episode.episode_ref),
        }
    except Exception as exc:  # noqa: BLE001
        phase_status["env_init"] = "failed"
        phase_detail["env_init"] = {"error": str(exc)}
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ENV_INIT_FAILED,
                message="Environment initialization failed during launch_smoke.",
                context={"error": str(exc)},
            )
        )
        return _build_smoke_payload(
            run_id=run_id,
            launcher_session_id=launcher_session_id,
            smoke_requested=True,
            smoke_mode=training_config.smoke_mode,
            smoke_success=False,
            selected_algorithm=training_config.algorithm,
            selected_episode_mode=training_config.episode_selection_mode,
            effective_seed=training_config.seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            launch_guard_results={
                "prelaunch_overall_pass": True,
                "selected_episode_ref": selected_episode.episode_ref,
                "selection_evidence": asdict(selected_episode),
                "dependency_probe": dependency_probe,
            },
            smoke_rollout_summary=smoke_rollout_summary,
            warnings=warnings,
            errors=issues,
        )

    try:
        ppo_class = _import_ppo_class()
        model = ppo_class(
            "MlpPolicy",
            env_client,
            seed=training_config.seed,
            device=resolved_device,
            verbose=0,
            **training_config.algo_params.to_sb3_kwargs(),
        )
        phase_status["algo_init"] = "completed"
        phase_detail["algo_init"] = {
            "algo_class": getattr(ppo_class, "__name__", str(ppo_class)),
            "device": resolved_device,
        }
    except Exception as exc:  # noqa: BLE001
        phase_status["algo_init"] = "failed"
        phase_detail["algo_init"] = {"error": str(exc)}
        if env_client is not None:
            env_client.close()
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_ALGO_INIT_FAILED,
                message="PPO initialization failed during launch_smoke.",
                context={"error": str(exc)},
            )
        )
        return _build_smoke_payload(
            run_id=run_id,
            launcher_session_id=launcher_session_id,
            smoke_requested=True,
            smoke_mode=training_config.smoke_mode,
            smoke_success=False,
            selected_algorithm=training_config.algorithm,
            selected_episode_mode=training_config.episode_selection_mode,
            effective_seed=training_config.seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            launch_guard_results={
                "prelaunch_overall_pass": True,
                "selected_episode_ref": selected_episode.episode_ref,
                "selection_evidence": asdict(selected_episode),
                "dependency_probe": dependency_probe,
            },
            smoke_rollout_summary=smoke_rollout_summary,
            warnings=warnings,
            errors=issues,
        )

    try:
        phase_status["learn_start"] = "completed"
        phase_detail["learn_start"] = {"smoke_learn_timesteps": int(training_config.smoke_learn_timesteps)}
        model.learn(total_timesteps=int(training_config.smoke_learn_timesteps))
        phase_status["learn_finish"] = "completed"
        phase_detail["learn_finish"] = {
            "num_timesteps": int(getattr(model, "num_timesteps", training_config.smoke_learn_timesteps)),
        }
        smoke_rollout_summary["learn_invoked"] = True
        smoke_rollout_summary["num_timesteps_after_learn"] = int(
            getattr(model, "num_timesteps", training_config.smoke_learn_timesteps)
        )
        smoke_rollout_summary["algo_class"] = type(model).__name__
        smoke_rollout_summary["launch_metadata_hash"] = _hash_canonical_json(
            {
                "selected_algorithm": training_config.algorithm,
                "selected_episode_mode": training_config.episode_selection_mode,
                "effective_seed": training_config.seed,
                "selected_episode_ref": selected_episode.episode_ref,
                "smoke_learn_timesteps": int(training_config.smoke_learn_timesteps),
                "resolved_device": resolved_device,
            }
        )
    except Exception as exc:  # noqa: BLE001
        phase_status["learn_finish"] = "failed"
        phase_detail["learn_finish"] = {"error": str(exc)}
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_SMOKE_FAILED,
                message="PPO learn() failed during launch_smoke.",
                context={"error": str(exc)},
            )
        )
        smoke_rollout_summary["learn_invoked"] = True
        if env_client is not None:
            env_client.close()
        return _build_smoke_payload(
            run_id=run_id,
            launcher_session_id=launcher_session_id,
            smoke_requested=True,
            smoke_mode=training_config.smoke_mode,
            smoke_success=False,
            selected_algorithm=training_config.algorithm,
            selected_episode_mode=training_config.episode_selection_mode,
            effective_seed=training_config.seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            launch_guard_results={
                "prelaunch_overall_pass": True,
                "selected_episode_ref": selected_episode.episode_ref,
                "selection_evidence": asdict(selected_episode),
                "dependency_probe": dependency_probe,
            },
            smoke_rollout_summary=smoke_rollout_summary,
            warnings=warnings,
            errors=issues,
        )
    finally:
        if env_client is not None:
            env_client.close()

    return _build_smoke_payload(
        run_id=run_id,
        launcher_session_id=launcher_session_id,
        smoke_requested=True,
        smoke_mode=training_config.smoke_mode,
        smoke_success=True,
        selected_algorithm=training_config.algorithm,
        selected_episode_mode=training_config.episode_selection_mode,
        effective_seed=training_config.seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
        launch_guard_results={
            "prelaunch_overall_pass": True,
            "selected_episode_ref": selected_episode.episode_ref,
            "selection_evidence": asdict(selected_episode),
            "dependency_probe": dependency_probe,
        },
        smoke_rollout_summary=smoke_rollout_summary,
        warnings=warnings,
        errors=[],
    )


def _effective_env_config(*, env_config: EnvConfig, seed: int, episode_ref: dict[str, Any]) -> EnvConfig:
    """Build in-memory effective env config without mutating the source file."""

    payload = {
        "run_id": env_config.run_id,
        "state_root": str(env_config.state_root),
        "episode_ref": dict(episode_ref),
        "execution_price_column": env_config.execution_price_column,
        "mark_to_market_column": env_config.mark_to_market_column,
        "include_timestamp_in_observation": env_config.include_timestamp_in_observation,
        "observation_output_dtype": env_config.observation_output_dtype,
        "observation_dtype_policy": env_config.observation_dtype_policy,
        "allowed_safe_casts": list(env_config.allowed_safe_casts),
        "initial_cash": env_config.initial_cash,
        "fee_bps": env_config.fee_bps,
        "slippage_bps": env_config.slippage_bps,
        "max_steps": env_config.max_steps,
        "seed": seed,
        "execution_timing_contract": {
            "observation_timestamp_policy": env_config.execution_timing_contract.observation_timestamp_policy,
            "execution_price_policy": env_config.execution_timing_contract.execution_price_policy,
            "reward_accrual_interval_policy": env_config.execution_timing_contract.reward_accrual_interval_policy,
            "mark_to_market_policy": env_config.execution_timing_contract.mark_to_market_policy,
        },
        "action_semantics_contract": {
            "action_space_type": env_config.action_semantics_contract.action_space_type,
            "action_space_n": env_config.action_semantics_contract.action_space_n,
            "invalid_action_policy": env_config.action_semantics_contract.invalid_action_policy,
            "reversal_policy": env_config.action_semantics_contract.reversal_policy,
            "position_model": env_config.action_semantics_contract.position_model,
        },
        "reward_contract": {
            "reward_version": env_config.reward_contract.reward_version,
            "reward_formula_summary": env_config.reward_contract.reward_formula_summary,
            "included_components": list(env_config.reward_contract.included_components),
            "reward_scale": env_config.reward_contract.reward_scale,
            "reward_clip_min": env_config.reward_contract.reward_clip_min,
            "reward_clip_max": env_config.reward_contract.reward_clip_max,
        },
        "termination_contract": {
            "data_end_terminated": env_config.termination_contract.data_end_terminated,
            "max_steps_truncated": env_config.termination_contract.max_steps_truncated,
        },
    }
    return parse_env_config(payload)


def _import_ppo_class() -> Any:
    """Import PPO lazily so prelaunch_only does not require ML deps."""

    module = importlib.import_module("stable_baselines3")
    ppo_class = getattr(module, "PPO", None)
    if ppo_class is None:
        raise ImportError("stable_baselines3.PPO is unavailable")
    return ppo_class


def _resolve_device(requested_device: str | None) -> tuple[str | None, list[ValidationIssue], dict[str, Any]]:
    """Resolve explicit requested device into an effective runtime device."""

    issues: list[ValidationIssue] = []
    torch_module, torch_error = _optional_import("torch")
    torch_available = torch_module is not None
    cuda_available = bool(torch_available and bool(torch_module.cuda.is_available()))
    dependency_probe = {
        "torch_available": torch_available,
        "torch_error": torch_error,
        "torch_cuda_available": cuda_available,
        "gymnasium_available": _optional_import("gymnasium")[0] is not None,
        "stable_baselines3_available": _optional_import("stable_baselines3")[0] is not None,
    }

    if requested_device is None:
        return None, issues, dependency_probe
    if requested_device == DEVICE_CPU:
        return DEVICE_CPU, issues, dependency_probe
    if requested_device == DEVICE_AUTO:
        return DEVICE_CUDA if cuda_available else DEVICE_CPU, issues, dependency_probe
    if requested_device == DEVICE_CUDA:
        if not torch_available or not cuda_available:
            issues.append(
                ValidationIssue(
                    code=TRAIN_LAUNCH_DEVICE_INVALID,
                    message="Requested cuda device is unavailable.",
                    context={"torch_available": torch_available, "cuda_available": cuda_available},
                )
            )
            return None, issues, dependency_probe
        return DEVICE_CUDA, issues, dependency_probe

    issues.append(
        ValidationIssue(
            code=TRAIN_LAUNCH_DEVICE_INVALID,
            message="Requested device is unsupported.",
            context={"device": requested_device},
        )
    )
    return None, issues, dependency_probe


def _optional_import(module_name: str) -> tuple[Any | None, str | None]:
    """Import an optional dependency without failing prelaunch."""

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return module, None


def _set_global_seed(seed: int) -> dict[str, Any]:
    """Apply deterministic startup seeds across supported libraries."""

    metadata: dict[str, Any] = {
        "seed": int(seed),
        "random_seeded": True,
        "numpy_seeded": False,
        "torch_seeded": False,
        "torch_cuda_seeded": False,
        "torch_cudnn_deterministic": False,
        "torch_cudnn_benchmark": None,
    }
    random.seed(seed)

    numpy_module, _ = _optional_import("numpy")
    if numpy_module is not None:
        numpy_module.random.seed(seed)
        metadata["numpy_seeded"] = True

    torch_module, _ = _optional_import("torch")
    if torch_module is not None:
        torch_module.manual_seed(seed)
        metadata["torch_seeded"] = True
        if bool(torch_module.cuda.is_available()):
            torch_module.cuda.manual_seed_all(seed)
            metadata["torch_cuda_seeded"] = True
        if hasattr(torch_module, "backends") and hasattr(torch_module.backends, "cudnn"):
            torch_module.backends.cudnn.deterministic = True
            torch_module.backends.cudnn.benchmark = False
            metadata["torch_cudnn_deterministic"] = True
            metadata["torch_cudnn_benchmark"] = False
    return metadata


def _check_output_dir_policy(output_dir: Path) -> list[ValidationIssue]:
    """Fail closed when output_dir already exists in any form."""

    issues: list[ValidationIssue] = []
    if not output_dir.exists():
        return issues

    if output_dir.is_file():
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_FRESH_ONLY_REQUIRED,
                message="startup_policy=fresh_only requires a brand-new output_dir path.",
                context={"output_dir": str(output_dir), "path_kind": "file"},
            )
        )
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_OUTPUT_CONFLICT,
                message="output_dir points to an existing file.",
                context={"output_dir": str(output_dir), "path_kind": "file"},
            )
        )
        return issues

    if output_dir.is_dir():
        entries = sorted(path.name for path in output_dir.iterdir())
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_FRESH_ONLY_REQUIRED,
                message="startup_policy=fresh_only requires output_dir to not exist before launch.",
                context={
                    "output_dir": str(output_dir),
                    "path_kind": "directory",
                    "entry_count": len(entries),
                    "entries_preview": entries[:10],
                },
            )
        )
        issues.append(
            ValidationIssue(
                code=TRAIN_LAUNCH_OUTPUT_CONFLICT,
                message="output_dir already exists and fresh_only forbids reuse, including empty or hidden-only directories.",
                context={
                    "output_dir": str(output_dir),
                    "path_kind": "directory",
                    "entry_count": len(entries),
                    "entries_preview": entries[:10],
                    "contains_hidden_only": bool(entries) and all(name.startswith(".") for name in entries),
                    "is_empty_directory": len(entries) == 0,
                },
            )
        )
        return issues

    issues.append(
        ValidationIssue(
            code=TRAIN_LAUNCH_FRESH_ONLY_REQUIRED,
            message="startup_policy=fresh_only requires output_dir to not exist before launch.",
            context={"output_dir": str(output_dir), "path_kind": "other"},
        )
    )
    issues.append(
        ValidationIssue(
            code=TRAIN_LAUNCH_OUTPUT_CONFLICT,
            message="output_dir already exists with an unsupported filesystem type.",
            context={"output_dir": str(output_dir), "path_kind": "other"},
        )
    )
    return issues


def _build_validation_payload(
    *,
    run_id: str,
    launcher_session_id: str,
    selected_algorithm: str | None,
    selected_episode_mode: str | None,
    effective_seed: int | None,
    requested_device: str | None,
    resolved_device: str | None,
    config_hash: str | None,
    readiness_hash: str | None,
    env_contract_hash: str | None,
    state_manifest_hash: str | None,
    episode_catalog_hash: str | None,
    prelaunch_checks: list[dict[str, Any]],
    warnings: list[ValidationIssue],
    errors: list[ValidationIssue],
) -> dict[str, Any]:
    """Build strict machine-readable validation report."""

    return {
        "run_id": run_id,
        "launcher_session_id": launcher_session_id,
        "overall_pass": len(errors) == 0,
        "selected_algorithm": selected_algorithm,
        "selected_episode_mode": selected_episode_mode,
        "effective_seed": effective_seed,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "config_hash": config_hash,
        "readiness_hash": readiness_hash,
        "env_contract_hash": env_contract_hash,
        "state_manifest_hash": state_manifest_hash,
        "episode_catalog_hash": episode_catalog_hash,
        "prelaunch_checks": prelaunch_checks,
        "warnings": [asdict(item) for item in warnings],
        "errors": [asdict(item) for item in errors],
        "failure_codes": _failure_codes(errors),
        "generated_at": _generated_at(),
    }


def _build_manifest_payload(
    *,
    run_id: str,
    launcher_session_id: str,
    training_config: TrainingConfig,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    output_dir: Path,
    selected_episode: SelectedEpisode,
    requested_device: str | None,
    resolved_device: str | None,
    config_hash: str | None,
    readiness_hash: str | None,
    env_contract_hash: str | None,
    state_manifest_hash: str | None,
    episode_catalog_hash: str | None,
    loaded_inputs: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    """Build strict launch manifest for successful prelaunch flows."""

    return {
        "run_id": run_id,
        "launcher_session_id": launcher_session_id,
        "source_artifacts": {
            "state_manifest_path": str(state_manifest_path),
            "env_contract_report_path": str(env_contract_report_path),
            "readiness_report_path": str(readiness_report_path),
            "episode_catalog_path": str(episode_catalog_path),
        },
        "selected_algorithm": training_config.algorithm,
        "selected_episode_mode": training_config.episode_selection_mode,
        "effective_seed": int(training_config.seed),
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "training_config_path": str(training_config_path),
        "env_config_path": str(env_config_path),
        "lineages": {
            "selected_episode_ref": dict(selected_episode.episode_ref),
            "selection_evidence": asdict(selected_episode),
            "hash_policy": {
                "algorithm": "sha256",
                "canonical_json": CANONICAL_JSON_POLICY,
            },
            "source_file_sha256": {
                "state_manifest": _sha256_file(state_manifest_path),
                "env_contract_report": _sha256_file(env_contract_report_path),
                "readiness_report": _sha256_file(readiness_report_path),
                "episode_catalog": _sha256_file(episode_catalog_path),
                "training_config": _sha256_file(training_config_path),
            },
            "env_contract_source_lineage": dict(loaded_inputs["env_contract_report"].get("source_lineage", {}))
            if loaded_inputs.get("env_contract_report") is not None
            else {},
            "readiness_env_contract_reference": dict(loaded_inputs["readiness_report"].get("env_contract_reference", {}))
            if loaded_inputs.get("readiness_report") is not None
            else {},
            "episode_catalog_source_lineage": dict(loaded_inputs["episode_catalog"].get("source_lineage", {}))
            if loaded_inputs.get("episode_catalog") is not None
            else {},
        },
        "config_hash": config_hash,
        "readiness_hash": readiness_hash,
        "env_contract_hash": env_contract_hash,
        "state_manifest_hash": state_manifest_hash,
        "episode_catalog_hash": episode_catalog_hash,
        "startup_policy": training_config.startup_policy,
        "smoke_mode": training_config.smoke_mode,
        "smoke_learn_timesteps": int(training_config.smoke_learn_timesteps),
        "output_dir": str(output_dir),
        "generated_at": _generated_at(),
    }


def _build_smoke_payload(
    *,
    run_id: str,
    launcher_session_id: str,
    smoke_requested: bool,
    smoke_mode: str | None,
    smoke_success: bool,
    selected_algorithm: str | None,
    selected_episode_mode: str | None,
    effective_seed: int | None,
    requested_device: str | None,
    resolved_device: str | None,
    startup_phase_trace: list[dict[str, Any]],
    launch_guard_results: dict[str, Any],
    smoke_rollout_summary: dict[str, Any],
    warnings: list[ValidationIssue],
    errors: list[ValidationIssue],
) -> dict[str, Any]:
    """Build strict smoke report."""

    return {
        "run_id": run_id,
        "launcher_session_id": launcher_session_id,
        "smoke_requested": bool(smoke_requested),
        "smoke_mode": smoke_mode,
        "smoke_success": bool(smoke_success),
        "selected_algorithm": selected_algorithm,
        "selected_episode_mode": selected_episode_mode,
        "effective_seed": effective_seed,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "startup_phase_trace": startup_phase_trace,
        "launch_guard_results": launch_guard_results,
        "smoke_rollout_summary": smoke_rollout_summary,
        "warnings": [asdict(item) for item in warnings],
        "errors": [asdict(item) for item in errors],
        "failure_codes": _failure_codes(errors),
        "generated_at": _generated_at(),
    }


def _write_reports(
    validation_payload: dict[str, Any],
    manifest_payload: dict[str, Any] | None,
    smoke_payload: dict[str, Any] | None,
    report_paths: ReportPaths,
) -> None:
    """Atomically write launcher reports to the fresh output directory."""

    atomic_write_json(validation_payload, report_paths.validation_report_path)
    if manifest_payload is not None:
        atomic_write_json(manifest_payload, report_paths.manifest_path)
    if smoke_payload is not None:
        atomic_write_json(smoke_payload, report_paths.smoke_report_path)


def _build_prelaunch_checks(
    *,
    load_issues: list[ValidationIssue],
    training_config: dict[str, Any],
    requested_device: str | None,
) -> list[dict[str, Any]]:
    """Build stable prelaunch check inventory."""

    checks: list[dict[str, Any]] = []
    missing_inputs = [issue for issue in load_issues if issue.code == TRAIN_LAUNCH_INPUT_MISSING]
    unreadable_inputs = [issue for issue in load_issues if issue.code == TRAIN_LAUNCH_PATH_UNREADABLE]
    invalid_json_inputs = [issue for issue in load_issues if issue.code == TRAIN_LAUNCH_JSON_INVALID]
    config = training_config.get("config")
    checks.append(
        _prelaunch_check(
            check_name="required_inputs_present",
            passed=len(missing_inputs) == 0,
            reason_code=missing_inputs[0].code if missing_inputs else None,
            detail={"missing_inputs": [issue.context for issue in missing_inputs]},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="required_paths_readable",
            passed=len(unreadable_inputs) == 0,
            reason_code=unreadable_inputs[0].code if unreadable_inputs else None,
            detail={"unreadable_inputs": [issue.context for issue in unreadable_inputs]},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="required_json_parseable",
            passed=len(invalid_json_inputs) == 0,
            reason_code=invalid_json_inputs[0].code if invalid_json_inputs else None,
            detail={"invalid_json_inputs": [issue.context for issue in invalid_json_inputs]},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="training_config_valid",
            passed=config is not None,
            reason_code=TRAIN_LAUNCH_CONFIG_INVALID if config is None else None,
            detail={"requested_device": requested_device},
        )
    )
    return checks


def _build_input_consistency_checks(*, errors: list[ValidationIssue]) -> list[dict[str, Any]]:
    """Build machine-readable grouped consistency checks from accumulated issues."""

    checks: list[dict[str, Any]] = []
    checks.append(
        _prelaunch_check(
            check_name="run_id_consistent_across_inputs",
            passed=not any(issue.code == TRAIN_LAUNCH_RUN_ID_MISMATCH for issue in errors),
            reason_code=TRAIN_LAUNCH_RUN_ID_MISMATCH if any(issue.code == TRAIN_LAUNCH_RUN_ID_MISMATCH for issue in errors) else None,
            detail={},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="state_manifest_valid",
            passed=not any(issue.code == TRAIN_LAUNCH_STATE_MANIFEST_INVALID for issue in errors),
            reason_code=TRAIN_LAUNCH_STATE_MANIFEST_INVALID
            if any(issue.code == TRAIN_LAUNCH_STATE_MANIFEST_INVALID for issue in errors)
            else None,
            detail={},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="env_contract_gate_passed",
            passed=not any(issue.code in {TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED, TRAIN_LAUNCH_ENV_CONTRACT_FAILED} for issue in errors),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED, TRAIN_LAUNCH_ENV_CONTRACT_FAILED}),
            detail={},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="readiness_gate_passed",
            passed=not any(issue.code in {TRAIN_LAUNCH_READINESS_REQUIRED, TRAIN_LAUNCH_READINESS_FAILED} for issue in errors),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_READINESS_REQUIRED, TRAIN_LAUNCH_READINESS_FAILED}),
            detail={},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="episode_catalog_gate_passed",
            passed=not any(
                issue.code in {TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED, TRAIN_LAUNCH_EPISODE_CATALOG_FAILED}
                for issue in errors
            ),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED, TRAIN_LAUNCH_EPISODE_CATALOG_FAILED}),
            detail={},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="selected_episode_mode_supported",
            passed=not any(
                issue.code in {TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED, TRAIN_LAUNCH_NO_ELIGIBLE_TRAINING_EPISODES}
                for issue in errors
            ),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED, TRAIN_LAUNCH_NO_ELIGIBLE_TRAINING_EPISODES}),
            detail={},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="launch_device_resolved",
            passed=not any(issue.code == TRAIN_LAUNCH_DEVICE_INVALID for issue in errors),
            reason_code=TRAIN_LAUNCH_DEVICE_INVALID if any(issue.code == TRAIN_LAUNCH_DEVICE_INVALID for issue in errors) else None,
            detail={},
        )
    )
    checks.append(
        _prelaunch_check(
            check_name="smoke_timesteps_valid",
            passed=not any(issue.code == TRAIN_LAUNCH_SMOKE_TIMESTEPS_INVALID for issue in errors),
            reason_code=TRAIN_LAUNCH_SMOKE_TIMESTEPS_INVALID
            if any(issue.code == TRAIN_LAUNCH_SMOKE_TIMESTEPS_INVALID for issue in errors)
            else None,
            detail={},
        )
    )
    return checks


def _prelaunch_check(
    *,
    check_name: str,
    passed: bool,
    reason_code: str | None,
    detail: dict[str, Any],
) -> dict[str, Any]:
    """Build one strict prelaunch check entry."""

    return {
        "check_name": check_name,
        "pass": bool(passed),
        "reason_code": reason_code,
        "detail": detail,
    }


def _phase_trace(
    *,
    prelaunch_status: str,
    env_init_status: str,
    algo_init_status: str,
    learn_start_status: str,
    learn_finish_status: str,
    report_write_status: str,
    prelaunch_detail: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build startup phase trace with the exact phase surface."""

    return [
        {"phase": "prelaunch_validation", "status": prelaunch_status, "detail": prelaunch_detail},
        {"phase": "env_init", "status": env_init_status, "detail": {}},
        {"phase": "algo_init", "status": algo_init_status, "detail": {}},
        {"phase": "learn_start", "status": learn_start_status, "detail": {}},
        {"phase": "learn_finish", "status": learn_finish_status, "detail": {}},
        {"phase": "report_write", "status": report_write_status, "detail": {}},
    ]


def _phase_trace_from_maps(status_map: Mapping[str, str], detail_map: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ordered startup phase trace from phase maps."""

    phases = (
        "prelaunch_validation",
        "env_init",
        "algo_init",
        "learn_start",
        "learn_finish",
        "report_write",
    )
    return [
        {"phase": phase, "status": str(status_map.get(phase, "not_started")), "detail": dict(detail_map.get(phase, {}))}
        for phase in phases
    ]


def _build_launcher_session_id(
    *,
    run_id: str,
    output_dir: Path,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
) -> str:
    """Build deterministic reporting-only launcher session id."""

    payload = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "env_config_path": str(env_config_path),
        "training_config_path": str(training_config_path),
        "state_manifest_path": str(state_manifest_path),
        "env_contract_report_path": str(env_contract_report_path),
        "readiness_report_path": str(readiness_report_path),
        "episode_catalog_path": str(episode_catalog_path),
    }
    return _hash_canonical_json(payload)[:16]


def _generated_at() -> str:
    """Return UTC timestamp for launcher reports."""

    return datetime.now(timezone.utc).isoformat()


def _hash_canonical_json(payload: Any) -> str:
    """Return deterministic semantic hash for JSON-like payloads."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _semantic_hash_optional(payload: Any) -> str | None:
    """Hash JSON payload semantics when available."""

    if payload is None:
        return None
    return _hash_canonical_json(payload)


def _sha256_file(path: Path) -> str:
    """Return file-bytes SHA256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65_536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _failure_codes(issues: list[ValidationIssue]) -> list[str]:
    """Return stable unique failure code list."""

    seen: set[str] = set()
    ordered: list[str] = []
    for issue in issues:
        if issue.code not in seen:
            seen.add(issue.code)
            ordered.append(issue.code)
    return ordered


def _first_code(issues: list[ValidationIssue], candidates: set[str]) -> str | None:
    """Return the first matching failure code from the provided set."""

    for issue in issues:
        if issue.code in candidates:
            return issue.code
    return None


def _raw_string(payload: dict[str, Any] | None, key: str) -> str | None:
    """Return stripped string when present."""

    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _raw_int(payload: dict[str, Any] | None, key: str) -> int | None:
    """Return integer value when present."""

    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, int):
        return int(value)
    return None


def _normalize_episode_ref(payload: Any) -> dict[str, Any]:
    """Normalize canonical episode_ref payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("episode_ref must be object")
    scope = payload.get("scope")
    partition = payload.get("partition")
    source_rel = payload.get("source_rel")
    fold_id = payload.get("fold_id")
    if not isinstance(scope, str) or not isinstance(partition, str) or not isinstance(source_rel, str) or not source_rel.strip():
        raise ValueError("episode_ref fields are invalid")
    if fold_id is not None and not isinstance(fold_id, int):
        raise ValueError("episode_ref.fold_id must be int or null")
    return {
        "scope": scope,
        "partition": partition,
        "source_rel": source_rel,
        "fold_id": fold_id,
    }


def _episode_entries_by_key(entries: list[Any]) -> dict[tuple[str, str, str, int | None], dict[str, Any]]:
    """Index episode_catalog episode entries by episode_ref."""

    index: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        episode_ref = item.get("episode_ref")
        try:
            normalized = _normalize_episode_ref(episode_ref)
        except ValueError:
            continue
        index[_episode_ref_key(normalized)] = dict(item)
    return index


def _episode_ref_key(payload: Mapping[str, Any]) -> tuple[str, str, str, int | None]:
    """Return stable tuple key for episode_ref."""

    return (
        str(payload.get("scope")),
        str(payload.get("partition")),
        str(payload.get("source_rel")),
        payload.get("fold_id") if isinstance(payload.get("fold_id"), int) else None,
    )
