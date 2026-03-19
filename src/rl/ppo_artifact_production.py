"""Canonical PPO artifact production contract for Milestone 4.8 closure prep."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

from core.io_atomic import atomic_write_json
from core.logging import get_logger
from rl.colab_staging_closure import (
    CLOSURE_REPORT_FILENAME,
    RUNTIME_DEPENDENCY_REPORT_FILENAME,
    find_stage_root_for_inputs,
    validate_existing_stage,
)
from rl.colab_runtime import (
    capture_memory_snapshot,
    collect_runtime_environment,
    ensure_model_parameters_finite,
    numpy_fail_fast_context,
)
from rl.env_adapter_gym import TradingEnvGym
from rl.evaluation_backtest import (
    EVAL_RUN_ID_MISMATCH,
    EVAL_SPLIT_REPORT_FAILED,
    EVAL_SPLIT_REPORT_REQUIRED,
    _validate_split_report as _validate_split_report_upstream,
)
from rl.training_launcher import (
    ALGORITHM_PPO,
    CANONICAL_JSON_POLICY,
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    SELECTION_MODE_FIXED,
    SELECTION_MODE_SEEDED_RANDOM,
    STARTUP_POLICY_FRESH_ONLY,
    TRAIN_LAUNCH_DEVICE_INVALID,
    TRAIN_LAUNCH_ENV_CONTRACT_FAILED,
    TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED,
    TRAIN_LAUNCH_EPISODE_CATALOG_FAILED,
    TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED,
    TRAIN_LAUNCH_EPISODE_MODE_UNSUPPORTED,
    TRAIN_LAUNCH_INPUT_MISSING,
    TRAIN_LAUNCH_JSON_INVALID,
    TRAIN_LAUNCH_NO_ELIGIBLE_TRAINING_EPISODES,
    TRAIN_LAUNCH_PATH_UNREADABLE,
    TRAIN_LAUNCH_READINESS_FAILED,
    TRAIN_LAUNCH_READINESS_REQUIRED,
    TRAIN_LAUNCH_RUN_ID_MISMATCH,
    TRAIN_LAUNCH_STATE_MANIFEST_INVALID,
    PpoAlgoParams,
    SelectedEpisode,
    _effective_env_config,
    _failure_codes,
    _hash_canonical_json,
    _import_ppo_class,
    _resolve_device,
    _resolve_selected_episode,
    _semantic_hash_optional,
    _set_global_seed,
    _sha256_file,
    _validate_algo_params,
    _validate_env_config as _validate_env_config_upstream,
    _validate_env_contract_report as _validate_env_contract_report_upstream,
    _validate_episode_catalog_report as _validate_episode_catalog_report_upstream,
    _validate_lineage_consistency as _validate_lineage_consistency_upstream,
    _validate_readiness_report as _validate_readiness_report_upstream,
    _validate_state_manifest as _validate_state_manifest_upstream,
)
from rl.notebook_progress import build_training_progress_callback, log_progress_mode, resolve_progress_mode
LOGGER = get_logger(__name__)

TASK_NAME = "Milestone 4.8 Closure Prep — Canonical PPO Artifact Production Contract"
CONTRACT_VERSION = "ppo_artifact_production.v1"
CANONICAL_ARTIFACT_FILENAME = "canonical_ppo_model.zip"
CHECKPOINT_ARTIFACTS_DIRNAME = "checkpoint_artifacts"
CHECKPOINT_ARTIFACT_FILENAME_TEMPLATE = "ppo_model_step_{step:06d}.zip"
MANIFEST_FILENAME = "artifact_production_manifest.json"
REPORT_FILENAME = "artifact_production_report.json"

POLICY_MLP = "MlpPolicy"

ARTIFACT_PRODUCTION_INPUT_MISSING = "ARTIFACT_PRODUCTION_INPUT_MISSING"
ARTIFACT_PRODUCTION_PATH_UNREADABLE = "ARTIFACT_PRODUCTION_PATH_UNREADABLE"
ARTIFACT_PRODUCTION_JSON_INVALID = "ARTIFACT_PRODUCTION_JSON_INVALID"
ARTIFACT_PRODUCTION_CONFIG_INVALID = "ARTIFACT_PRODUCTION_CONFIG_INVALID"
ARTIFACT_PRODUCTION_POLICY_UNSUPPORTED = "ARTIFACT_PRODUCTION_POLICY_UNSUPPORTED"
ARTIFACT_PRODUCTION_OUTPUT_CONFLICT = "ARTIFACT_PRODUCTION_OUTPUT_CONFLICT"
ARTIFACT_PRODUCTION_FRESH_ONLY_REQUIRED = "ARTIFACT_PRODUCTION_FRESH_ONLY_REQUIRED"
ARTIFACT_PRODUCTION_ENV_INIT_FAILED = "ARTIFACT_PRODUCTION_ENV_INIT_FAILED"
ARTIFACT_PRODUCTION_ALGO_INIT_FAILED = "ARTIFACT_PRODUCTION_ALGO_INIT_FAILED"
ARTIFACT_PRODUCTION_TRAIN_FAILED = "ARTIFACT_PRODUCTION_TRAIN_FAILED"
ARTIFACT_PRODUCTION_SAVE_FAILED = "ARTIFACT_PRODUCTION_SAVE_FAILED"
ARTIFACT_PRODUCTION_ARTIFACT_MISSING = "ARTIFACT_PRODUCTION_ARTIFACT_MISSING"
ARTIFACT_PRODUCTION_ARTIFACT_INVALID = "ARTIFACT_PRODUCTION_ARTIFACT_INVALID"
ARTIFACT_PRODUCTION_LOAD_BACK_FAILED = "ARTIFACT_PRODUCTION_LOAD_BACK_FAILED"
ARTIFACT_PRODUCTION_REPORT_WRITE_FAILED = "ARTIFACT_PRODUCTION_REPORT_WRITE_FAILED"
ARTIFACT_PRODUCTION_AMP_UNSUPPORTED = "ARTIFACT_PRODUCTION_AMP_UNSUPPORTED"
ARTIFACT_PRODUCTION_COMPILE_UNSUPPORTED = "ARTIFACT_PRODUCTION_COMPILE_UNSUPPORTED"
ARTIFACT_PRODUCTION_NUMERIC_INVALID = "ARTIFACT_PRODUCTION_NUMERIC_INVALID"
ARTIFACT_PRODUCTION_COLAB_STAGE_REPORT_REQUIRED = "ARTIFACT_PRODUCTION_COLAB_STAGE_REPORT_REQUIRED"
ARTIFACT_PRODUCTION_COLAB_STAGE_INVALID = "ARTIFACT_PRODUCTION_COLAB_STAGE_INVALID"
ARTIFACT_PRODUCTION_COLAB_RUNTIME_DEPENDENCY_FAILED = "ARTIFACT_PRODUCTION_COLAB_RUNTIME_DEPENDENCY_FAILED"
ARTIFACT_PRODUCTION_CHECKPOINT_EXPORT_FAILED = "ARTIFACT_PRODUCTION_CHECKPOINT_EXPORT_FAILED"

PRODUCTION_CONFIG_REQUIRED_FIELDS = (
    "algorithm",
    "policy",
    "seed",
    "total_timesteps",
    "device",
    "episode_selection_mode",
    "startup_policy",
    "algo_params",
)
PRODUCTION_CONFIG_OPTIONAL_FIELDS = ("checkpoint_export_steps",)


@dataclass
class ValidationIssue:
    """Machine-readable artifact production issue."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactProductionConfig:
    """Strict config for canonical PPO artifact production."""

    algorithm: str
    policy: str
    seed: int
    total_timesteps: int
    device: str
    episode_selection_mode: str
    startup_policy: str
    algo_params: PpoAlgoParams
    checkpoint_export_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReportPaths:
    """Stable artifact production output paths."""

    artifact_path: Path
    checkpoint_artifacts_dir: Path
    manifest_path: Path
    report_path: Path


@dataclass
class ArtifactProductionExecutionResult:
    """Composite execution result for the artifact production lane."""

    exit_code: int
    manifest_payload: dict[str, Any] | None
    report_payload: dict[str, Any]
    report_paths: ReportPaths
    reports_written: bool


def execute_ppo_artifact_production(
    *,
    run_id: str,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
    output_dir: Path,
    progress_mode: str = "off",
    enable_amp: bool = False,
    enable_torch_compile: bool = False,
    memory_log_interval_steps: int = 0,
) -> ArtifactProductionExecutionResult:
    """Produce one canonical, load-validated SB3 PPO artifact from explicit inputs."""

    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must be non-empty")
    if int(memory_log_interval_steps) < 0:
        raise ValueError("memory_log_interval_steps must be >= 0")

    output_dir_resolved = output_dir.resolve()
    report_paths = ReportPaths(
        artifact_path=output_dir_resolved / CANONICAL_ARTIFACT_FILENAME,
        checkpoint_artifacts_dir=output_dir_resolved / CHECKPOINT_ARTIFACTS_DIRNAME,
        manifest_path=output_dir_resolved / MANIFEST_FILENAME,
        report_path=output_dir_resolved / REPORT_FILENAME,
    )
    production_session_id = _build_production_session_id(
        run_id=normalized_run_id,
        output_dir=output_dir_resolved,
        env_config_path=env_config_path.resolve(),
        training_config_path=training_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
        split_report_path=split_report_path.resolve(),
    )

    output_guard_issues = _check_output_dir_policy(output_dir_resolved)
    if output_guard_issues:
        report_payload = _build_report_payload(
            run_id=normalized_run_id,
            production_session_id=production_session_id,
            selected_algorithm=None,
            policy=None,
            effective_seed=None,
            requested_device=None,
            resolved_device=None,
            total_timesteps=None,
            artifact_path=report_paths.artifact_path,
            artifact_sha256=None,
            save_succeeded=False,
            artifact_exists=False,
            artifact_zip_valid=False,
            load_back_succeeded=False,
            load_back_model_class=None,
            canonical_artifact_ready=False,
            validation_checks=[
                _validation_check(
                    check_name="startup_policy_fresh_only",
                    passed=False,
                    reason_code=ARTIFACT_PRODUCTION_FRESH_ONLY_REQUIRED,
                    detail={"output_dir": str(output_dir_resolved)},
                ),
                _validation_check(
                    check_name="output_dir_conflict_free",
                    passed=False,
                    reason_code=ARTIFACT_PRODUCTION_OUTPUT_CONFLICT,
                    detail=output_guard_issues[0].context,
                ),
            ],
            startup_phase_trace=_phase_trace(
                validation_status="failed",
                env_init_status="not_started",
                algo_init_status="not_started",
                learn_start_status="not_started",
                learn_finish_status="not_started",
                artifact_save_status="not_started",
                artifact_load_status="not_started",
                report_write_status="not_written",
                validation_detail={"failure_codes": _failure_codes(output_guard_issues)},
            ),
            production_summary={
                "dependency_probe": None,
                "selected_episode_ref": None,
                "selection_evidence": None,
                "startup_seed_metadata": None,
                "num_timesteps_after_learn": None,
                "canonicality_checks": _canonicality_checks(
                    artifact_path=report_paths.artifact_path,
                    artifact_sha256=None,
                    save_succeeded=False,
                    artifact_exists=False,
                    artifact_zip_valid=False,
                    load_back_succeeded=False,
                    training_config_hash=None,
                    env_contract_hash=None,
                    readiness_hash=None,
                    state_manifest_hash=None,
                    split_report_hash=None,
                ),
            },
            warnings=[],
            errors=output_guard_issues,
        )
        return ArtifactProductionExecutionResult(
            exit_code=2,
            manifest_payload=None,
            report_payload=report_payload,
            report_paths=report_paths,
            reports_written=False,
        )

    output_dir_resolved.mkdir(parents=True, exist_ok=False)

    runtime_environment = collect_runtime_environment()
    progress_resolution = resolve_progress_mode(progress_mode)
    log_progress_mode(logger=LOGGER, resolution=progress_resolution, scope="artifact_production")
    LOGGER.info(
        "Artifact production start | run_id=%s output_dir=%s",
        normalized_run_id,
        output_dir_resolved,
    )
    LOGGER.info(
        "Artifact runtime summary | run_id=%s gpu_name=%s cuda_available=%s gpu_total_memory_gib=%s system_ram_gib=%s torch_version=%s torch_compile_supported=%s",
        normalized_run_id,
        runtime_environment.gpu_name,
        runtime_environment.cuda_available,
        runtime_environment.gpu_total_memory_gib,
        runtime_environment.system_ram_gib,
        runtime_environment.torch_version,
        runtime_environment.torch_compile_supported,
    )
    memory_snapshots: list[dict[str, Any]] = [capture_memory_snapshot(label="startup", step=0)]

    loaded_inputs, load_issues = _load_json_inputs(
        env_config_path=env_config_path.resolve(),
        training_config_path=training_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
        split_report_path=split_report_path.resolve(),
    )

    warnings: list[ValidationIssue] = []
    errors: list[ValidationIssue] = list(load_issues)

    training_config_hash = _semantic_hash_optional(loaded_inputs.get("training_config"))
    env_config_hash = _semantic_hash_optional(loaded_inputs.get("env_config"))
    state_manifest_hash = _semantic_hash_optional(loaded_inputs.get("state_manifest"))
    env_contract_hash = _semantic_hash_optional(loaded_inputs.get("env_contract_report"))
    readiness_hash = _semantic_hash_optional(loaded_inputs.get("readiness_report"))
    episode_catalog_hash = _semantic_hash_optional(loaded_inputs.get("episode_catalog"))
    split_report_hash = _semantic_hash_optional(loaded_inputs.get("split_report"))
    stage_preflight = _resolve_colab_stage_preflight(
        env_config_path=env_config_path.resolve(),
        training_config_path=training_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
        split_report_path=split_report_path.resolve(),
    )
    errors.extend(stage_preflight["issues"])

    config_result = _validate_artifact_production_config(loaded_inputs.get("training_config"))
    production_config = config_result["config"]
    errors.extend(config_result["errors"])

    selected_algorithm = production_config.algorithm if production_config is not None else _raw_string(
        loaded_inputs.get("training_config"), "algorithm"
    )
    policy = production_config.policy if production_config is not None else _raw_string(
        loaded_inputs.get("training_config"), "policy"
    )
    requested_device = production_config.device if production_config is not None else _raw_string(
        loaded_inputs.get("training_config"), "device"
    )
    effective_seed = production_config.seed if production_config is not None else _raw_int(
        loaded_inputs.get("training_config"), "seed"
    )
    total_timesteps = production_config.total_timesteps if production_config is not None else _raw_int(
        loaded_inputs.get("training_config"), "total_timesteps"
    )
    selected_episode_mode = production_config.episode_selection_mode if production_config is not None else _raw_string(
        loaded_inputs.get("training_config"), "episode_selection_mode"
    )

    optimization_runtime, optimization_issues = _validate_optional_optimizations(
        enable_amp=bool(enable_amp),
        enable_torch_compile=bool(enable_torch_compile),
        runtime_environment=runtime_environment,
    )
    errors.extend(optimization_issues)

    resolved_device, device_issues, dependency_probe = _resolve_device(requested_device)
    errors.extend(_normalize_issues(device_issues))

    def _runtime_payload(numeric_safety: dict[str, Any] | None = None) -> dict[str, Any]:
        return _build_runtime_payload(
            runtime_environment=runtime_environment,
            progress_resolution=progress_resolution,
            requested_device=requested_device,
            resolved_device=resolved_device,
            enable_amp=bool(enable_amp),
            enable_torch_compile=bool(enable_torch_compile),
            optimization_runtime=optimization_runtime,
            memory_log_interval_steps=int(memory_log_interval_steps),
            memory_snapshots=memory_snapshots,
            numeric_safety=numeric_safety,
            n_envs=1,
            stage_preflight=stage_preflight,
        )

    env_config_result = _validate_env_config_upstream(
        env_config_payload=loaded_inputs.get("env_config"),
        cli_run_id=normalized_run_id,
        state_manifest_path=state_manifest_path.resolve(),
        training_seed=effective_seed,
    )
    env_config = env_config_result["config"]
    errors.extend(_normalize_issues(env_config_result["errors"]))
    errors.extend(_normalize_issues(_validate_state_manifest_upstream(loaded_inputs.get("state_manifest"), normalized_run_id)))
    errors.extend(
        _normalize_issues(_validate_env_contract_report_upstream(loaded_inputs.get("env_contract_report"), normalized_run_id))
    )
    errors.extend(_normalize_issues(_validate_readiness_report_upstream(loaded_inputs.get("readiness_report"), normalized_run_id)))
    errors.extend(
        _normalize_issues(_validate_episode_catalog_report_upstream(loaded_inputs.get("episode_catalog"), normalized_run_id))
    )
    errors.extend(_normalize_issues(_validate_split_report_upstream(loaded_inputs.get("split_report"), normalized_run_id)))
    errors.extend(
        _normalize_issues(
            _validate_lineage_consistency_upstream(
                state_manifest_path=state_manifest_path.resolve(),
                state_manifest_payload=loaded_inputs.get("state_manifest"),
                env_contract_report=loaded_inputs.get("env_contract_report"),
                readiness_report=loaded_inputs.get("readiness_report"),
                episode_catalog=loaded_inputs.get("episode_catalog"),
            )
        )
    )

    selected_episode: SelectedEpisode | None = None
    if production_config is not None and env_config is not None:
        selected_episode, selected_episode_issues = _resolve_selected_episode(
            episode_catalog=loaded_inputs.get("episode_catalog"),
            episode_selection_mode=production_config.episode_selection_mode,
            seed=production_config.seed,
            env_config=env_config,
        )
        errors.extend(_normalize_issues(selected_episode_issues))

    validation_checks = _build_validation_checks(
        load_issues=load_issues,
        config_present=production_config is not None,
        errors=errors,
        stage_preflight=stage_preflight,
    )

    manifest_payload = _build_manifest_payload(
        run_id=normalized_run_id,
        production_session_id=production_session_id,
        env_config_path=env_config_path.resolve(),
        training_config_path=training_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
        split_report_path=split_report_path.resolve(),
        output_dir=output_dir_resolved,
        selected_algorithm=selected_algorithm,
        policy=policy,
        selected_episode_mode=selected_episode_mode,
        selected_episode=selected_episode,
        effective_seed=effective_seed,
        total_timesteps=total_timesteps,
        requested_device=requested_device,
        resolved_device=resolved_device,
        artifact_sha256=None,
        artifact_exists=False,
        artifact_zip_valid=False,
        load_back_succeeded=False,
        load_back_model_class=None,
        env_config_hash=env_config_hash,
        training_config_hash=training_config_hash,
        state_manifest_hash=state_manifest_hash,
        env_contract_hash=env_contract_hash,
        readiness_hash=readiness_hash,
        episode_catalog_hash=episode_catalog_hash,
        split_report_hash=split_report_hash,
        checkpoint_export_steps=production_config.checkpoint_export_steps if production_config is not None else (),
        checkpoint_exports=(),
        report_paths=report_paths,
    )

    checkpoint_exports: list[dict[str, Any]] = []
    checkpoint_callback: Any | None = None
    if production_config is not None and production_config.checkpoint_export_steps:
        checkpoint_callback = _build_checkpoint_export_callback(
            run_id=normalized_run_id,
            checkpoint_export_steps=production_config.checkpoint_export_steps,
            checkpoint_artifacts_dir=report_paths.checkpoint_artifacts_dir,
            resolved_device=resolved_device,
        )

    if errors:
        report_payload = _build_report_payload(
            run_id=normalized_run_id,
            production_session_id=production_session_id,
            selected_algorithm=selected_algorithm,
            policy=policy,
            effective_seed=effective_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            total_timesteps=total_timesteps,
            artifact_path=report_paths.artifact_path,
            artifact_sha256=None,
            save_succeeded=False,
            artifact_exists=False,
            artifact_zip_valid=False,
            load_back_succeeded=False,
            load_back_model_class=None,
            canonical_artifact_ready=False,
            validation_checks=validation_checks,
            startup_phase_trace=_phase_trace(
                validation_status="failed",
                env_init_status="not_started",
                algo_init_status="not_started",
                learn_start_status="not_started",
                learn_finish_status="not_started",
                artifact_save_status="not_started",
                artifact_load_status="not_started",
                report_write_status="completed",
                validation_detail={"failure_codes": _failure_codes(errors)},
            ),
            production_summary={
                "dependency_probe": dependency_probe,
                "colab_stage_preflight": stage_preflight["validation"],
                "selected_episode_ref": dict(selected_episode.episode_ref) if selected_episode is not None else None,
                "selection_evidence": asdict(selected_episode) if selected_episode is not None else None,
                "startup_seed_metadata": None,
                "num_timesteps_after_learn": None,
                "canonicality_checks": _canonicality_checks(
                    artifact_path=report_paths.artifact_path,
                    artifact_sha256=None,
                    save_succeeded=False,
                    artifact_exists=False,
                    artifact_zip_valid=False,
                    load_back_succeeded=False,
                    training_config_hash=training_config_hash,
                    env_contract_hash=env_contract_hash,
                    readiness_hash=readiness_hash,
                    state_manifest_hash=state_manifest_hash,
                    split_report_hash=split_report_hash,
                ),
            },
            runtime=_runtime_payload(),
            warnings=warnings,
            errors=errors,
            checkpoint_export_steps=production_config.checkpoint_export_steps if production_config is not None else (),
            checkpoint_exports=checkpoint_exports,
            report_paths=report_paths,
        )
        _write_reports(manifest_payload=manifest_payload, report_payload=report_payload, report_paths=report_paths)
        return ArtifactProductionExecutionResult(
            exit_code=2,
            manifest_payload=manifest_payload,
            report_payload=report_payload,
            report_paths=report_paths,
            reports_written=True,
        )

    assert production_config is not None
    assert env_config is not None
    assert selected_episode is not None

    phase_status = {
        "validation": "completed",
        "env_init": "not_started",
        "algo_init": "not_started",
        "learn_start": "not_started",
        "learn_finish": "not_started",
        "artifact_save": "not_started",
        "artifact_load": "not_started",
        "report_write": "not_started",
    }
    phase_detail: dict[str, Any] = {
        "validation": {
            "overall_pass": True,
            "selected_episode_ref": dict(selected_episode.episode_ref),
            "selection_evidence": asdict(selected_episode),
            "dependency_probe": dependency_probe,
            "colab_stage_preflight": stage_preflight["validation"],
        },
        "env_init": {},
        "algo_init": {},
        "learn_start": {},
        "learn_finish": {},
        "artifact_save": {},
        "artifact_load": {},
        "report_write": {},
    }

    startup_seed_metadata = _set_global_seed(production_config.seed)
    env_client: Any | None = None
    model: Any | None = None
    numeric_safety_metadata: dict[str, Any] | None = None
    save_state = {
        "save_succeeded": False,
        "artifact_exists": False,
        "artifact_zip_valid": False,
        "load_back_succeeded": False,
        "load_back_model_class": None,
        "artifact_sha256": None,
    }

    try:
        LOGGER.info("Env init start | run_id=%s episode_ref=%s", normalized_run_id, dict(selected_episode.episode_ref))
        effective_env_config = _effective_env_config(
            env_config=env_config,
            seed=production_config.seed,
            episode_ref=dict(selected_episode.episode_ref),
        )
        env_client = TradingEnvGym(config=effective_env_config, validate_on_init=True)
        memory_snapshots.append(capture_memory_snapshot(label="env_init", step=0))
        phase_status["env_init"] = "completed"
        phase_detail["env_init"] = {
            "env_class": type(env_client).__name__,
            "selected_episode_ref": dict(selected_episode.episode_ref),
            "effective_env_seed": production_config.seed,
        }
        LOGGER.info("Env init finished | run_id=%s env_class=%s", normalized_run_id, type(env_client).__name__)
    except Exception as exc:  # noqa: BLE001
        phase_status["env_init"] = "failed"
        phase_detail["env_init"] = {"error": str(exc)}
        report_payload = _build_report_payload(
            run_id=normalized_run_id,
            production_session_id=production_session_id,
            selected_algorithm=production_config.algorithm,
            policy=production_config.policy,
            effective_seed=production_config.seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            total_timesteps=production_config.total_timesteps,
            artifact_path=report_paths.artifact_path,
            artifact_sha256=None,
            save_succeeded=False,
            artifact_exists=False,
            artifact_zip_valid=False,
            load_back_succeeded=False,
            load_back_model_class=None,
            canonical_artifact_ready=False,
            validation_checks=validation_checks,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            production_summary={
                "dependency_probe": dependency_probe,
                "colab_stage_preflight": stage_preflight["validation"],
                "selected_episode_ref": dict(selected_episode.episode_ref),
                "selection_evidence": asdict(selected_episode),
                "startup_seed_metadata": startup_seed_metadata,
                "num_timesteps_after_learn": None,
                "canonicality_checks": _canonicality_checks(
                    artifact_path=report_paths.artifact_path,
                    artifact_sha256=None,
                    save_succeeded=False,
                    artifact_exists=False,
                    artifact_zip_valid=False,
                    load_back_succeeded=False,
                    training_config_hash=training_config_hash,
                    env_contract_hash=env_contract_hash,
                    readiness_hash=readiness_hash,
                    state_manifest_hash=state_manifest_hash,
                    split_report_hash=split_report_hash,
                ),
            },
            runtime=_runtime_payload(numeric_safety_metadata),
            warnings=warnings,
            errors=[
                ValidationIssue(
                    code=ARTIFACT_PRODUCTION_ENV_INIT_FAILED,
                    message="Environment initialization failed during canonical artifact production.",
                    context={"error": str(exc)},
                )
            ],
        )
        _write_reports(manifest_payload=manifest_payload, report_payload=report_payload, report_paths=report_paths)
        return ArtifactProductionExecutionResult(
            exit_code=2,
            manifest_payload=manifest_payload,
            report_payload=report_payload,
            report_paths=report_paths,
            reports_written=True,
        )

    try:
        LOGGER.info("Algo init start | run_id=%s policy=%s", normalized_run_id, production_config.policy)
        ppo_class = _import_ppo_class()
        model = ppo_class(
            production_config.policy,
            env_client,
            seed=production_config.seed,
            device=resolved_device,
            verbose=0,
            **production_config.algo_params.to_sb3_kwargs(),
        )
        memory_snapshots.append(capture_memory_snapshot(label="algo_init", step=0))
        phase_status["algo_init"] = "completed"
        phase_detail["algo_init"] = {
            "algo_class": getattr(ppo_class, "__name__", str(ppo_class)),
            "policy": production_config.policy,
            "device": resolved_device,
        }
        LOGGER.info("Algo init finished | run_id=%s algo_class=%s", normalized_run_id, getattr(ppo_class, "__name__", str(ppo_class)))
    except Exception as exc:  # noqa: BLE001
        phase_status["algo_init"] = "failed"
        phase_detail["algo_init"] = {"error": str(exc)}
        if env_client is not None:
            env_client.close()
        report_payload = _build_report_payload(
            run_id=normalized_run_id,
            production_session_id=production_session_id,
            selected_algorithm=production_config.algorithm,
            policy=production_config.policy,
            effective_seed=production_config.seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            total_timesteps=production_config.total_timesteps,
            artifact_path=report_paths.artifact_path,
            artifact_sha256=None,
            save_succeeded=False,
            artifact_exists=False,
            artifact_zip_valid=False,
            load_back_succeeded=False,
            load_back_model_class=None,
            canonical_artifact_ready=False,
            validation_checks=validation_checks,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            production_summary={
                "dependency_probe": dependency_probe,
                "colab_stage_preflight": stage_preflight["validation"],
                "selected_episode_ref": dict(selected_episode.episode_ref),
                "selection_evidence": asdict(selected_episode),
                "startup_seed_metadata": startup_seed_metadata,
                "num_timesteps_after_learn": None,
                "canonicality_checks": _canonicality_checks(
                    artifact_path=report_paths.artifact_path,
                    artifact_sha256=None,
                    save_succeeded=False,
                    artifact_exists=False,
                    artifact_zip_valid=False,
                    load_back_succeeded=False,
                    training_config_hash=training_config_hash,
                    env_contract_hash=env_contract_hash,
                    readiness_hash=readiness_hash,
                    state_manifest_hash=state_manifest_hash,
                    split_report_hash=split_report_hash,
                ),
            },
            runtime=_runtime_payload(numeric_safety_metadata),
            warnings=warnings,
            errors=[
                ValidationIssue(
                    code=ARTIFACT_PRODUCTION_ALGO_INIT_FAILED,
                    message="PPO initialization failed during canonical artifact production.",
                    context={"error": str(exc)},
                )
            ],
        )
        _write_reports(manifest_payload=manifest_payload, report_payload=report_payload, report_paths=report_paths)
        return ArtifactProductionExecutionResult(
            exit_code=2,
            manifest_payload=manifest_payload,
            report_payload=report_payload,
            report_paths=report_paths,
            reports_written=True,
        )

    try:
        LOGGER.info(
            "Learn start | run_id=%s total_timesteps=%d resolved_device=%s",
            normalized_run_id,
            int(production_config.total_timesteps),
            resolved_device,
        )
        phase_status["learn_start"] = "completed"
        phase_detail["learn_start"] = {
            "total_timesteps": int(production_config.total_timesteps),
            "progress_mode": progress_resolution.active_mode,
            "checkpoint_export_steps": list(production_config.checkpoint_export_steps),
        }
        assert model is not None
        progress_callback = build_training_progress_callback(
            resolution=progress_resolution,
            total_timesteps=int(production_config.total_timesteps),
            run_id=normalized_run_id,
            resolved_device=resolved_device,
            logger=LOGGER,
            memory_log_interval_steps=int(memory_log_interval_steps),
            memory_snapshots=memory_snapshots,
        )
        with numpy_fail_fast_context() as numeric_safety_metadata:
            learn_kwargs: dict[str, Any] = {"total_timesteps": int(production_config.total_timesteps)}
            callbacks = [
                callback
                for callback in (progress_callback, checkpoint_callback)
                if callback is not None
            ]
            if len(callbacks) == 1:
                learn_kwargs["callback"] = callbacks[0]
            elif len(callbacks) > 1:
                from stable_baselines3.common.callbacks import CallbackList

                learn_kwargs["callback"] = CallbackList(callbacks)
            model.learn(**learn_kwargs)
            ensure_model_parameters_finite(model)
        memory_snapshots.append(
            capture_memory_snapshot(
                label="learn_finish",
                step=int(getattr(model, "num_timesteps", production_config.total_timesteps)),
            )
        )
        phase_status["learn_finish"] = "completed"
        phase_detail["learn_finish"] = {
            "num_timesteps": int(getattr(model, "num_timesteps", production_config.total_timesteps)),
        }
        LOGGER.info(
            "Learn finished | run_id=%s num_timesteps=%d",
            normalized_run_id,
            int(getattr(model, "num_timesteps", production_config.total_timesteps)),
        )
        checkpoint_exports = (
            checkpoint_callback.build_exports_payload() if checkpoint_callback is not None else checkpoint_exports
        )
    except Exception as exc:  # noqa: BLE001
        phase_status["learn_finish"] = "failed"
        phase_detail["learn_finish"] = {"error": str(exc)}
        if env_client is not None:
            env_client.close()
        checkpoint_exports = (
            checkpoint_callback.build_exports_payload() if checkpoint_callback is not None else checkpoint_exports
        )
        if isinstance(exc, ControlledArtifactProductionFailure):
            issue = exc.issue
        elif _looks_like_numeric_failure(exc):
            issue = ValidationIssue(
                code=ARTIFACT_PRODUCTION_NUMERIC_INVALID,
                message="Non-finite numeric state detected during canonical artifact production.",
                context={"error": str(exc)},
            )
        else:
            issue = ValidationIssue(
                code=ARTIFACT_PRODUCTION_TRAIN_FAILED,
                message="PPO learn() failed during canonical artifact production.",
                context={"error": str(exc)},
            )
        manifest_payload = _build_manifest_payload(
            run_id=normalized_run_id,
            production_session_id=production_session_id,
            env_config_path=env_config_path.resolve(),
            training_config_path=training_config_path.resolve(),
            state_manifest_path=state_manifest_path.resolve(),
            env_contract_report_path=env_contract_report_path.resolve(),
            readiness_report_path=readiness_report_path.resolve(),
            episode_catalog_path=episode_catalog_path.resolve(),
            split_report_path=split_report_path.resolve(),
            output_dir=output_dir_resolved,
            selected_algorithm=production_config.algorithm,
            policy=production_config.policy,
            selected_episode_mode=production_config.episode_selection_mode,
            selected_episode=selected_episode,
            effective_seed=production_config.seed,
            total_timesteps=production_config.total_timesteps,
            requested_device=requested_device,
            resolved_device=resolved_device,
            artifact_sha256=None,
            artifact_exists=False,
            artifact_zip_valid=False,
            load_back_succeeded=False,
            load_back_model_class=None,
            env_config_hash=env_config_hash,
            training_config_hash=training_config_hash,
            state_manifest_hash=state_manifest_hash,
            env_contract_hash=env_contract_hash,
            readiness_hash=readiness_hash,
            episode_catalog_hash=episode_catalog_hash,
            split_report_hash=split_report_hash,
            checkpoint_export_steps=production_config.checkpoint_export_steps,
            checkpoint_exports=checkpoint_exports,
            report_paths=report_paths,
        )
        report_payload = _build_report_payload(
            run_id=normalized_run_id,
            production_session_id=production_session_id,
            selected_algorithm=production_config.algorithm,
            policy=production_config.policy,
            effective_seed=production_config.seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            total_timesteps=production_config.total_timesteps,
            artifact_path=report_paths.artifact_path,
            artifact_sha256=None,
            save_succeeded=False,
            artifact_exists=False,
            artifact_zip_valid=False,
            load_back_succeeded=False,
            load_back_model_class=None,
            canonical_artifact_ready=False,
            validation_checks=validation_checks,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            production_summary={
                "dependency_probe": dependency_probe,
                "colab_stage_preflight": stage_preflight["validation"],
                "selected_episode_ref": dict(selected_episode.episode_ref),
                "selection_evidence": asdict(selected_episode),
                "startup_seed_metadata": startup_seed_metadata,
                "num_timesteps_after_learn": int(getattr(model, "num_timesteps", 0)),
                "canonicality_checks": _canonicality_checks(
                    artifact_path=report_paths.artifact_path,
                    artifact_sha256=None,
                    save_succeeded=False,
                    artifact_exists=False,
                    artifact_zip_valid=False,
                    load_back_succeeded=False,
                    training_config_hash=training_config_hash,
                    env_contract_hash=env_contract_hash,
                    readiness_hash=readiness_hash,
                    state_manifest_hash=state_manifest_hash,
                    split_report_hash=split_report_hash,
                ),
            },
            runtime=_runtime_payload(numeric_safety_metadata),
            warnings=warnings,
            errors=[issue],
            checkpoint_export_steps=production_config.checkpoint_export_steps,
            checkpoint_exports=checkpoint_exports,
            report_paths=report_paths,
        )
        _write_reports(manifest_payload=manifest_payload, report_payload=report_payload, report_paths=report_paths)
        return ArtifactProductionExecutionResult(
            exit_code=2,
            manifest_payload=manifest_payload,
            report_payload=report_payload,
            report_paths=report_paths,
            reports_written=True,
        )
    finally:
        if env_client is not None:
            env_client.close()

    artifact_issues: list[ValidationIssue] = []
    try:
        LOGGER.info("Artifact save start | run_id=%s artifact_path=%s", normalized_run_id, report_paths.artifact_path)
        save_state = _save_and_validate_model_artifact(
            model=model,
            artifact_path=report_paths.artifact_path,
            resolved_device=resolved_device,
        )
        memory_snapshots.append(
            capture_memory_snapshot(
                label="artifact_save",
                step=int(getattr(model, "num_timesteps", production_config.total_timesteps)),
            )
        )
        phase_status["artifact_save"] = "completed"
        phase_detail["artifact_save"] = {"artifact_path": str(report_paths.artifact_path)}
        phase_status["artifact_load"] = "completed"
        phase_detail["artifact_load"] = {
            "artifact_path": str(report_paths.artifact_path),
            "load_back_model_class": save_state["load_back_model_class"],
        }
        LOGGER.info(
            "Artifact save finished | run_id=%s artifact_path=%s load_back_model_class=%s",
            normalized_run_id,
            report_paths.artifact_path,
            save_state["load_back_model_class"],
        )
    except ControlledArtifactProductionFailure as exc:
        artifact_issues.append(exc.issue)
        if phase_status["artifact_save"] == "completed":
            phase_status["artifact_load"] = "failed"
            phase_detail["artifact_load"] = {"error": exc.issue.context.get("error")}
        else:
            phase_status["artifact_save"] = "failed"
            phase_detail["artifact_save"] = {"error": exc.issue.context.get("error")}

    canonical_artifact_ready = (
        bool(save_state["save_succeeded"])
        and bool(save_state["artifact_exists"])
        and bool(save_state["artifact_zip_valid"])
        and bool(save_state["load_back_succeeded"])
        and isinstance(save_state["artifact_sha256"], str)
    )
    manifest_payload = _build_manifest_payload(
        run_id=normalized_run_id,
        production_session_id=production_session_id,
        env_config_path=env_config_path.resolve(),
        training_config_path=training_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
        split_report_path=split_report_path.resolve(),
        output_dir=output_dir_resolved,
        selected_algorithm=production_config.algorithm,
        policy=production_config.policy,
        selected_episode_mode=production_config.episode_selection_mode,
        selected_episode=selected_episode,
        effective_seed=production_config.seed,
        total_timesteps=production_config.total_timesteps,
        requested_device=requested_device,
        resolved_device=resolved_device,
        artifact_sha256=save_state["artifact_sha256"],
        artifact_exists=bool(save_state["artifact_exists"]),
        artifact_zip_valid=bool(save_state["artifact_zip_valid"]),
        load_back_succeeded=bool(save_state["load_back_succeeded"]),
        load_back_model_class=save_state["load_back_model_class"],
        env_config_hash=env_config_hash,
        training_config_hash=training_config_hash,
        state_manifest_hash=state_manifest_hash,
        env_contract_hash=env_contract_hash,
        readiness_hash=readiness_hash,
        episode_catalog_hash=episode_catalog_hash,
        split_report_hash=split_report_hash,
        checkpoint_export_steps=production_config.checkpoint_export_steps,
        checkpoint_exports=checkpoint_exports,
        report_paths=report_paths,
    )

    phase_status["report_write"] = "completed"
    phase_detail["report_write"] = {
        "manifest_path": str(report_paths.manifest_path),
        "report_path": str(report_paths.report_path),
        "canonical_artifact_ready": canonical_artifact_ready,
    }
    report_payload = _build_report_payload(
        run_id=normalized_run_id,
        production_session_id=production_session_id,
        selected_algorithm=production_config.algorithm,
        policy=production_config.policy,
        effective_seed=production_config.seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        total_timesteps=production_config.total_timesteps,
        artifact_path=report_paths.artifact_path,
        artifact_sha256=save_state["artifact_sha256"],
        save_succeeded=bool(save_state["save_succeeded"]),
        artifact_exists=bool(save_state["artifact_exists"]),
        artifact_zip_valid=bool(save_state["artifact_zip_valid"]),
        load_back_succeeded=bool(save_state["load_back_succeeded"]),
        load_back_model_class=save_state["load_back_model_class"],
        canonical_artifact_ready=canonical_artifact_ready,
        validation_checks=validation_checks,
        startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
        production_summary={
            "dependency_probe": dependency_probe,
            "colab_stage_preflight": stage_preflight["validation"],
            "selected_episode_ref": dict(selected_episode.episode_ref),
            "selection_evidence": asdict(selected_episode),
            "startup_seed_metadata": startup_seed_metadata,
            "num_timesteps_after_learn": int(getattr(model, "num_timesteps", production_config.total_timesteps)),
            "canonicality_checks": _canonicality_checks(
                artifact_path=report_paths.artifact_path,
                artifact_sha256=save_state["artifact_sha256"],
                save_succeeded=bool(save_state["save_succeeded"]),
                artifact_exists=bool(save_state["artifact_exists"]),
                artifact_zip_valid=bool(save_state["artifact_zip_valid"]),
                load_back_succeeded=bool(save_state["load_back_succeeded"]),
                training_config_hash=training_config_hash,
                env_contract_hash=env_contract_hash,
                readiness_hash=readiness_hash,
                state_manifest_hash=state_manifest_hash,
                split_report_hash=split_report_hash,
            ),
        },
        checkpoint_export_steps=production_config.checkpoint_export_steps,
        checkpoint_exports=checkpoint_exports,
        runtime=_runtime_payload(numeric_safety_metadata),
        warnings=warnings,
        errors=artifact_issues,
        report_paths=report_paths,
    )

    try:
        _write_reports(manifest_payload=manifest_payload, report_payload=report_payload, report_paths=report_paths)
    except Exception as exc:  # noqa: BLE001
        report_issue = ValidationIssue(
            code=ARTIFACT_PRODUCTION_REPORT_WRITE_FAILED,
            message="Artifact production outputs could not be written atomically.",
            context={"error": str(exc), "output_dir": str(output_dir_resolved)},
        )
        phase_status["report_write"] = "failed"
        phase_detail["report_write"] = {"error": str(exc)}
        failed_report = _build_report_payload(
            run_id=normalized_run_id,
            production_session_id=production_session_id,
            selected_algorithm=production_config.algorithm,
            policy=production_config.policy,
            effective_seed=production_config.seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
            total_timesteps=production_config.total_timesteps,
            artifact_path=report_paths.artifact_path,
            artifact_sha256=save_state["artifact_sha256"],
            save_succeeded=bool(save_state["save_succeeded"]),
            artifact_exists=bool(save_state["artifact_exists"]),
            artifact_zip_valid=bool(save_state["artifact_zip_valid"]),
            load_back_succeeded=bool(save_state["load_back_succeeded"]),
            load_back_model_class=save_state["load_back_model_class"],
            canonical_artifact_ready=False,
            validation_checks=validation_checks,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            production_summary={
                "dependency_probe": dependency_probe,
                "colab_stage_preflight": stage_preflight["validation"],
                "selected_episode_ref": dict(selected_episode.episode_ref),
                "selection_evidence": asdict(selected_episode),
                "startup_seed_metadata": startup_seed_metadata,
                "num_timesteps_after_learn": int(getattr(model, "num_timesteps", production_config.total_timesteps)),
                "canonicality_checks": _canonicality_checks(
                    artifact_path=report_paths.artifact_path,
                    artifact_sha256=save_state["artifact_sha256"],
                    save_succeeded=bool(save_state["save_succeeded"]),
                    artifact_exists=bool(save_state["artifact_exists"]),
                    artifact_zip_valid=bool(save_state["artifact_zip_valid"]),
                    load_back_succeeded=bool(save_state["load_back_succeeded"]),
                    training_config_hash=training_config_hash,
                    env_contract_hash=env_contract_hash,
                    readiness_hash=readiness_hash,
                    state_manifest_hash=state_manifest_hash,
                    split_report_hash=split_report_hash,
                ),
            },
            runtime=_runtime_payload(numeric_safety_metadata),
            warnings=warnings,
            errors=[report_issue],
            report_paths=report_paths,
        )
        _best_effort_write_json(manifest_payload, report_paths.manifest_path)
        _best_effort_write_json(failed_report, report_paths.report_path)
        return ArtifactProductionExecutionResult(
            exit_code=3,
            manifest_payload=manifest_payload,
            report_payload=failed_report,
            report_paths=report_paths,
            reports_written=False,
        )

    return ArtifactProductionExecutionResult(
        exit_code=0 if canonical_artifact_ready else 2,
        manifest_payload=manifest_payload,
        report_payload=report_payload,
        report_paths=report_paths,
        reports_written=True,
    )


class ControlledArtifactProductionFailure(Exception):
    """Controlled failure mapped to exit code 2."""

    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue


def _load_json_inputs(
    *,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
) -> tuple[dict[str, dict[str, Any] | None], list[ValidationIssue]]:
    """Load all explicit JSON inputs with strict error mapping."""

    inputs: dict[str, dict[str, Any] | None] = {}
    issues: list[ValidationIssue] = []
    path_specs = {
        "env_config": env_config_path,
        "training_config": training_config_path,
        "state_manifest": state_manifest_path,
        "env_contract_report": env_contract_report_path,
        "readiness_report": readiness_report_path,
        "episode_catalog": episode_catalog_path,
        "split_report": split_report_path,
    }
    for label, path in path_specs.items():
        payload, error = _load_json_object(path=path, label=label)
        inputs[label] = payload
        if error is not None:
            issues.append(error)
    return inputs, issues


def _load_json_object(*, path: Path, label: str) -> tuple[dict[str, Any] | None, ValidationIssue | None]:
    """Load one required JSON object for artifact production."""

    if not path.exists():
        return None, ValidationIssue(
            code=ARTIFACT_PRODUCTION_INPUT_MISSING,
            message="Required artifact production input is missing.",
            context={"input_label": label, "path": str(path)},
        )
    if not path.is_file():
        return None, ValidationIssue(
            code=ARTIFACT_PRODUCTION_PATH_UNREADABLE,
            message="Required artifact production input path is not a readable file.",
            context={"input_label": label, "path": str(path)},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, ValidationIssue(
            code=ARTIFACT_PRODUCTION_JSON_INVALID,
            message="Required artifact production input contains invalid JSON.",
            context={"input_label": label, "path": str(path), "error": str(exc)},
        )
    except OSError as exc:
        return None, ValidationIssue(
            code=ARTIFACT_PRODUCTION_PATH_UNREADABLE,
            message="Required artifact production input could not be read.",
            context={"input_label": label, "path": str(path), "error": str(exc)},
        )
    if not isinstance(payload, dict):
        return None, ValidationIssue(
            code=ARTIFACT_PRODUCTION_JSON_INVALID,
            message="Required artifact production input JSON must be an object.",
            context={"input_label": label, "path": str(path)},
        )
    return payload, None


def _validate_artifact_production_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the strict production config contract."""

    errors: list[ValidationIssue] = []
    if payload is None:
        return {"config": None, "errors": errors}

    extra_keys = sorted(
        set(payload.keys()) - set(PRODUCTION_CONFIG_REQUIRED_FIELDS) - set(PRODUCTION_CONFIG_OPTIONAL_FIELDS)
    )
    missing_keys = sorted(set(PRODUCTION_CONFIG_REQUIRED_FIELDS) - set(payload.keys()))
    if missing_keys or extra_keys:
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="training_config top-level fields must match the artifact production contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return {"config": None, "errors": errors}

    algorithm = _raw_string(payload, "algorithm")
    if algorithm != ALGORITHM_PPO:
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="Only PPO is supported by the canonical artifact production contract.",
                context={"algorithm": algorithm},
            )
        )

    policy = _raw_string(payload, "policy")
    if policy != POLICY_MLP:
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_POLICY_UNSUPPORTED,
                message="Only MlpPolicy is supported by the canonical artifact production contract.",
                context={"policy": policy},
            )
        )

    seed_raw = payload.get("seed")
    if not isinstance(seed_raw, int) or isinstance(seed_raw, bool):
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="seed must be a non-null integer.",
                context={"seed": seed_raw},
            )
        )

    total_timesteps_raw = payload.get("total_timesteps")
    if not isinstance(total_timesteps_raw, int) or isinstance(total_timesteps_raw, bool) or int(total_timesteps_raw) <= 0:
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="total_timesteps must be a positive integer.",
                context={"total_timesteps": total_timesteps_raw},
            )
        )

    device = _raw_string(payload, "device")
    if device not in {DEVICE_CPU, DEVICE_CUDA, DEVICE_AUTO}:
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="device must be one of cpu, cuda, auto.",
                context={"device": device},
            )
        )

    episode_selection_mode = _raw_string(payload, "episode_selection_mode")
    if episode_selection_mode not in {SELECTION_MODE_FIXED, SELECTION_MODE_SEEDED_RANDOM}:
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="episode_selection_mode is unsupported.",
                context={"episode_selection_mode": episode_selection_mode},
            )
        )

    startup_policy = _raw_string(payload, "startup_policy")
    if startup_policy != STARTUP_POLICY_FRESH_ONLY:
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="startup_policy must be fresh_only.",
                context={"startup_policy": startup_policy},
            )
        )

    checkpoint_export_steps: tuple[int, ...] = ()
    if "checkpoint_export_steps" in payload:
        checkpoint_export_steps, checkpoint_errors = _validate_checkpoint_export_steps(
            payload.get("checkpoint_export_steps"),
            total_timesteps=(
                int(total_timesteps_raw)
                if isinstance(total_timesteps_raw, int) and not isinstance(total_timesteps_raw, bool)
                else None
            ),
        )
        errors.extend(checkpoint_errors)

    algo_params, algo_errors = _validate_algo_params(payload.get("algo_params"))
    errors.extend(_normalize_issues(algo_errors))

    if errors:
        return {"config": None, "errors": errors}

    assert isinstance(seed_raw, int)
    assert isinstance(total_timesteps_raw, int)
    assert algo_params is not None
    return {
        "config": ArtifactProductionConfig(
            algorithm=algorithm,
            policy=policy,
            seed=int(seed_raw),
            total_timesteps=int(total_timesteps_raw),
            device=device,
            episode_selection_mode=episode_selection_mode,
            startup_policy=startup_policy,
            algo_params=algo_params,
            checkpoint_export_steps=checkpoint_export_steps,
        ),
        "errors": errors,
    }


def _save_and_validate_model_artifact(
    *,
    model: Any,
    artifact_path: Path,
    resolved_device: str | None,
    artifact_role: str = "canonical",
) -> dict[str, Any]:
    """Save the model to a temp path, validate load-back, then atomically rename."""

    tmp_artifact_path = _tmp_artifact_path(artifact_path)
    save_state = {
        "save_succeeded": False,
        "artifact_exists": False,
        "artifact_zip_valid": False,
        "load_back_succeeded": False,
        "load_back_model_class": None,
        "artifact_sha256": None,
    }

    if tmp_artifact_path.exists():
        tmp_artifact_path.unlink()

    try:
        model.save(str(tmp_artifact_path))
        save_state["save_succeeded"] = True
    except Exception as exc:  # noqa: BLE001
        if tmp_artifact_path.exists():
            tmp_artifact_path.unlink()
        raise ControlledArtifactProductionFailure(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_SAVE_FAILED,
                message=f"{artifact_role.capitalize()} PPO artifact save failed.",
                context={
                    "artifact_role": artifact_role,
                    "error": str(exc),
                    "temp_artifact_path": str(tmp_artifact_path),
                },
            )
        ) from exc

    if not tmp_artifact_path.exists() or not tmp_artifact_path.is_file():
        raise ControlledArtifactProductionFailure(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_ARTIFACT_MISSING,
                message=f"{artifact_role.capitalize()} model save returned without materializing the temp artifact file.",
                context={"artifact_role": artifact_role, "temp_artifact_path": str(tmp_artifact_path)},
            )
        )

    if not zipfile.is_zipfile(tmp_artifact_path):
        tmp_artifact_path.unlink()
        raise ControlledArtifactProductionFailure(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_ARTIFACT_INVALID,
                message=f"Saved temp {artifact_role} artifact is not a readable zip file.",
                context={"artifact_role": artifact_role, "temp_artifact_path": str(tmp_artifact_path)},
            )
        )

    save_state["artifact_zip_valid"] = True
    try:
        loaded_model = _load_ppo_model(model_artifact_path=tmp_artifact_path, device=resolved_device)
        save_state["load_back_succeeded"] = True
        save_state["load_back_model_class"] = type(loaded_model).__name__
    except Exception as exc:  # noqa: BLE001
        tmp_artifact_path.unlink()
        raise ControlledArtifactProductionFailure(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_LOAD_BACK_FAILED,
                message=f"PPO.load(...) failed against the saved temp {artifact_role} artifact.",
                context={
                    "artifact_role": artifact_role,
                    "error": str(exc),
                    "temp_artifact_path": str(tmp_artifact_path),
                },
            )
        ) from exc

    os.replace(tmp_artifact_path, artifact_path)
    save_state["artifact_exists"] = bool(artifact_path.exists() and artifact_path.is_file())
    if not save_state["artifact_exists"]:
        raise ControlledArtifactProductionFailure(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_ARTIFACT_MISSING,
                message=f"{artifact_role.capitalize()} artifact path does not exist after atomic rename.",
                context={"artifact_role": artifact_role, "artifact_path": str(artifact_path)},
            )
        )

    if not zipfile.is_zipfile(artifact_path):
        artifact_path.unlink()
        raise ControlledArtifactProductionFailure(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_ARTIFACT_INVALID,
                message=f"{artifact_role.capitalize()} artifact path is not a readable zip file after rename.",
                context={"artifact_role": artifact_role, "artifact_path": str(artifact_path)},
            )
        )

    save_state["artifact_zip_valid"] = True
    try:
        save_state["artifact_sha256"] = _sha256_file(artifact_path)
    except OSError as exc:
        artifact_path.unlink()
        raise ControlledArtifactProductionFailure(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_ARTIFACT_MISSING,
                message=f"{artifact_role.capitalize()} artifact could not be hashed after rename.",
                context={"artifact_role": artifact_role, "artifact_path": str(artifact_path), "error": str(exc)},
            )
        ) from exc
    return save_state


def _load_ppo_model(*, model_artifact_path: Path, device: str | None) -> Any:
    """Load a PPO model from an explicit artifact path."""

    ppo_class = _import_ppo_class()
    return ppo_class.load(str(model_artifact_path), device=device)


def _write_reports(*, manifest_payload: dict[str, Any], report_payload: dict[str, Any], report_paths: ReportPaths) -> None:
    """Atomically write manifest and report into the fresh output directory."""

    atomic_write_json(manifest_payload, report_paths.manifest_path)
    atomic_write_json(report_payload, report_paths.report_path)


def _best_effort_write_json(payload: dict[str, Any] | None, path: Path) -> None:
    """Best-effort JSON write for runtime-failure reporting."""

    if payload is None:
        return
    try:
        atomic_write_json(payload, path)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Best-effort write failed | path=%s", path)


def _check_output_dir_policy(output_dir: Path) -> list[ValidationIssue]:
    """Fail closed when output_dir already exists in any form."""

    issues: list[ValidationIssue] = []
    if not output_dir.exists():
        return issues
    if output_dir.is_file():
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_FRESH_ONLY_REQUIRED,
                message="startup_policy=fresh_only requires a brand-new output_dir path.",
                context={"output_dir": str(output_dir), "path_kind": "file"},
            )
        )
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_OUTPUT_CONFLICT,
                message="output_dir points to an existing file.",
                context={"output_dir": str(output_dir), "path_kind": "file"},
            )
        )
        return issues
    if output_dir.is_dir():
        entries = sorted(path.name for path in output_dir.iterdir())
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_FRESH_ONLY_REQUIRED,
                message="startup_policy=fresh_only requires output_dir to not exist before artifact production.",
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
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_OUTPUT_CONFLICT,
                message="output_dir already exists and fresh_only forbids reuse.",
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
            code=ARTIFACT_PRODUCTION_OUTPUT_CONFLICT,
            message="output_dir already exists with an unsupported filesystem type.",
            context={"output_dir": str(output_dir), "path_kind": "other"},
        )
    )
    return issues


def _build_validation_checks(
    *,
    load_issues: Sequence[ValidationIssue],
    config_present: bool,
    errors: Sequence[ValidationIssue],
    stage_preflight: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build stable validation checks for the artifact production report."""

    checks: list[dict[str, Any]] = []
    missing_inputs = [issue for issue in load_issues if issue.code == ARTIFACT_PRODUCTION_INPUT_MISSING]
    unreadable_inputs = [issue for issue in load_issues if issue.code == ARTIFACT_PRODUCTION_PATH_UNREADABLE]
    invalid_json_inputs = [issue for issue in load_issues if issue.code == ARTIFACT_PRODUCTION_JSON_INVALID]
    checks.append(
        _validation_check(
            check_name="required_inputs_present",
            passed=len(missing_inputs) == 0,
            reason_code=missing_inputs[0].code if missing_inputs else None,
            detail={"missing_inputs": [issue.context for issue in missing_inputs]},
        )
    )
    checks.append(
        _validation_check(
            check_name="required_paths_readable",
            passed=len(unreadable_inputs) == 0,
            reason_code=unreadable_inputs[0].code if unreadable_inputs else None,
            detail={"unreadable_inputs": [issue.context for issue in unreadable_inputs]},
        )
    )
    checks.append(
        _validation_check(
            check_name="required_json_parseable",
            passed=len(invalid_json_inputs) == 0,
            reason_code=invalid_json_inputs[0].code if invalid_json_inputs else None,
            detail={"invalid_json_inputs": [issue.context for issue in invalid_json_inputs]},
        )
    )
    checks.append(
        _validation_check(
            check_name="training_config_valid",
            passed=config_present,
            reason_code=ARTIFACT_PRODUCTION_CONFIG_INVALID if not config_present else None,
            detail={},
        )
    )
    stage_detected = bool(stage_preflight.get("detected", False))
    stage_validation = stage_preflight.get("validation")
    checks.append(
        _validation_check(
            check_name="colab_stage_preflight_detected",
            passed=True,
            reason_code=None,
            detail={
                "detected": stage_detected,
                "staging_root": stage_preflight.get("staging_root"),
                "closure_report_path": stage_preflight.get("closure_report_path"),
                "runtime_dependency_report_path": stage_preflight.get("runtime_dependency_report_path"),
            },
        )
    )
    checks.append(
        _validation_check(
            check_name="colab_stage_closure_valid",
            passed=not any(
                issue.code in {ARTIFACT_PRODUCTION_COLAB_STAGE_REPORT_REQUIRED, ARTIFACT_PRODUCTION_COLAB_STAGE_INVALID}
                for issue in errors
            ),
            reason_code=_first_code(errors, {ARTIFACT_PRODUCTION_COLAB_STAGE_REPORT_REQUIRED, ARTIFACT_PRODUCTION_COLAB_STAGE_INVALID}),
            detail={"stage_validation": stage_validation},
        )
    )
    checks.append(
        _validation_check(
            check_name="colab_runtime_dependency_valid",
            passed=not any(issue.code == ARTIFACT_PRODUCTION_COLAB_RUNTIME_DEPENDENCY_FAILED for issue in errors),
            reason_code=_first_code(errors, {ARTIFACT_PRODUCTION_COLAB_RUNTIME_DEPENDENCY_FAILED}),
            detail={"stage_validation": stage_validation},
        )
    )
    checks.append(
        _validation_check(
            check_name="run_id_consistent_across_inputs",
            passed=not any(issue.code in {TRAIN_LAUNCH_RUN_ID_MISMATCH, EVAL_RUN_ID_MISMATCH} for issue in errors),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_RUN_ID_MISMATCH, EVAL_RUN_ID_MISMATCH}),
            detail={},
        )
    )
    checks.append(
        _validation_check(
            check_name="state_manifest_valid",
            passed=not any(issue.code == TRAIN_LAUNCH_STATE_MANIFEST_INVALID for issue in errors),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_STATE_MANIFEST_INVALID}),
            detail={},
        )
    )
    checks.append(
        _validation_check(
            check_name="env_contract_gate_passed",
            passed=not any(issue.code in {TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED, TRAIN_LAUNCH_ENV_CONTRACT_FAILED} for issue in errors),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_ENV_CONTRACT_REQUIRED, TRAIN_LAUNCH_ENV_CONTRACT_FAILED}),
            detail={},
        )
    )
    checks.append(
        _validation_check(
            check_name="readiness_gate_passed",
            passed=not any(issue.code in {TRAIN_LAUNCH_READINESS_REQUIRED, TRAIN_LAUNCH_READINESS_FAILED} for issue in errors),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_READINESS_REQUIRED, TRAIN_LAUNCH_READINESS_FAILED}),
            detail={},
        )
    )
    checks.append(
        _validation_check(
            check_name="episode_catalog_gate_passed",
            passed=not any(issue.code in {TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED, TRAIN_LAUNCH_EPISODE_CATALOG_FAILED} for issue in errors),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_EPISODE_CATALOG_REQUIRED, TRAIN_LAUNCH_EPISODE_CATALOG_FAILED}),
            detail={},
        )
    )
    checks.append(
        _validation_check(
            check_name="split_report_gate_passed",
            passed=not any(issue.code in {EVAL_SPLIT_REPORT_REQUIRED, EVAL_SPLIT_REPORT_FAILED} for issue in errors),
            reason_code=_first_code(errors, {EVAL_SPLIT_REPORT_REQUIRED, EVAL_SPLIT_REPORT_FAILED}),
            detail={},
        )
    )
    checks.append(
        _validation_check(
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
        _validation_check(
            check_name="runtime_device_resolved",
            passed=not any(issue.code == TRAIN_LAUNCH_DEVICE_INVALID for issue in errors),
            reason_code=_first_code(errors, {TRAIN_LAUNCH_DEVICE_INVALID}),
            detail={},
        )
    )
    return checks


def _resolve_colab_stage_preflight(
    *,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
) -> dict[str, Any]:
    """Revalidate staged Colab closure/runtime reports when explicit inputs come from a staged root."""

    input_paths = (
        env_config_path,
        training_config_path,
        state_manifest_path,
        env_contract_report_path,
        readiness_report_path,
        episode_catalog_path,
        split_report_path,
    )
    staging_root = find_stage_root_for_inputs(input_paths=input_paths)
    if staging_root is None:
        return {
            "detected": False,
            "staging_root": None,
            "closure_report_path": None,
            "runtime_dependency_report_path": None,
            "validation": None,
            "issues": [],
        }

    closure_report_path = staging_root / CLOSURE_REPORT_FILENAME
    runtime_dependency_report_path = staging_root / RUNTIME_DEPENDENCY_REPORT_FILENAME
    try:
        validation = validate_existing_stage(staging_root=staging_root)
        closure_report = json.loads(closure_report_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "detected": True,
            "staging_root": str(staging_root),
            "closure_report_path": str(closure_report_path),
            "runtime_dependency_report_path": str(runtime_dependency_report_path),
            "validation": None,
            "issues": [
                ValidationIssue(
                    code=ARTIFACT_PRODUCTION_COLAB_STAGE_REPORT_REQUIRED,
                    message="Colab staged inputs require valid closure and runtime dependency reports before training starts.",
                    context={
                        "staging_root": str(staging_root),
                        "closure_report_path": str(closure_report_path),
                        "runtime_dependency_report_path": str(runtime_dependency_report_path),
                        "error": str(exc),
                    },
                )
            ],
        }

    issues: list[ValidationIssue] = []
    expected_seed_paths = _seed_paths_from_closure_report(closure_report)
    explicit_seed_paths = {
        "env_config": str(env_config_path.resolve()),
        "training_config": str(training_config_path.resolve()),
        "state_manifest": str(state_manifest_path.resolve()),
        "env_contract_report": str(env_contract_report_path.resolve()),
        "readiness_report": str(readiness_report_path.resolve()),
        "episode_catalog": str(episode_catalog_path.resolve()),
        "split_report": str(split_report_path.resolve()),
    }
    path_mismatches = [
        {
            "label": label,
            "expected_path": expected_seed_paths.get(label),
            "explicit_path": explicit_seed_paths[label],
        }
        for label in sorted(explicit_seed_paths)
        if expected_seed_paths.get(label) != explicit_seed_paths[label]
    ]
    if path_mismatches:
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_COLAB_STAGE_INVALID,
                message="Explicit artifact-production inputs do not match the validated staged closure seed paths.",
                context={"path_mismatches": path_mismatches},
            )
        )

    check_map = {
        str(item.get("check_name")): item
        for item in validation.get("checks", [])
        if isinstance(item, Mapping)
    }
    closure_checks = (
        "persisted_closure_report_valid",
        "required_artifacts_present",
        "artifact_hashes_unchanged",
        "contract_critical_local_path_leaks_absent",
    )
    if not all(bool(check_map.get(name, {}).get("pass", False)) for name in closure_checks):
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_COLAB_STAGE_INVALID,
                message="Validated Colab staging closure is invalid at canonical training entrypoint start.",
                context={"validation": validation},
            )
        )
    runtime_checks = (
        "persisted_runtime_dependency_report_valid",
        "runtime_dependency_probe_revalidated",
    )
    if not all(bool(check_map.get(name, {}).get("pass", False)) for name in runtime_checks):
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_COLAB_RUNTIME_DEPENDENCY_FAILED,
                message="Validated Colab runtime dependency preflight is invalid at canonical training entrypoint start.",
                context={"validation": validation},
            )
        )

    return {
        "detected": True,
        "staging_root": str(staging_root),
        "closure_report_path": str(closure_report_path),
        "runtime_dependency_report_path": str(runtime_dependency_report_path),
        "validation": validation,
        "issues": issues,
    }


def _seed_paths_from_closure_report(closure_report: Mapping[str, Any]) -> dict[str, str]:
    """Return expected staged seed-input paths from the normalized closure report."""

    dependency_spec = closure_report.get("dependency_spec")
    if not isinstance(dependency_spec, Mapping):
        raise ValueError("closure report dependency_spec is required")
    seed_inputs = dependency_spec.get("seed_inputs")
    if not isinstance(seed_inputs, list):
        raise ValueError("closure report dependency_spec.seed_inputs must be list")

    staging_root_raw = closure_report.get("staging_root")
    if not isinstance(staging_root_raw, str) or not staging_root_raw.strip():
        raise ValueError("closure report staging_root must be non-empty string")
    staging_root = Path(staging_root_raw).resolve()

    expected: dict[str, str] = {}
    for item in seed_inputs:
        if not isinstance(item, Mapping):
            raise ValueError("closure report dependency_spec.seed_inputs entries must be objects")
        label = item.get("label")
        relpath = item.get("staged_relative_path")
        if not isinstance(label, str) or not isinstance(relpath, str):
            raise ValueError("closure report seed_inputs entries must expose label and staged_relative_path")
        expected[label] = str((staging_root / relpath).resolve())
    return expected


def _build_manifest_payload(
    *,
    run_id: str,
    production_session_id: str,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
    output_dir: Path,
    selected_algorithm: str | None,
    policy: str | None,
    selected_episode_mode: str | None,
    selected_episode: SelectedEpisode | None,
    effective_seed: int | None,
    total_timesteps: int | None,
    requested_device: str | None,
    resolved_device: str | None,
    artifact_sha256: str | None,
    artifact_exists: bool,
    artifact_zip_valid: bool,
    load_back_succeeded: bool,
    load_back_model_class: str | None,
    env_config_hash: str | None,
    training_config_hash: str | None,
    state_manifest_hash: str | None,
    env_contract_hash: str | None,
    readiness_hash: str | None,
    episode_catalog_hash: str | None,
    split_report_hash: str | None,
    checkpoint_export_steps: Sequence[int] = (),
    checkpoint_exports: Sequence[Mapping[str, Any]] = (),
    report_paths: ReportPaths | None = None,
) -> dict[str, Any]:
    """Build machine-readable artifact manifest payload."""

    return {
        "task_name": TASK_NAME,
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "production_session_id": production_session_id,
        "artifact": {
            "path": str(output_dir / CANONICAL_ARTIFACT_FILENAME),
            "filename": CANONICAL_ARTIFACT_FILENAME,
            "format": "sb3_ppo_zip",
            "exists": bool(artifact_exists),
            "zip_valid": bool(artifact_zip_valid),
            "sha256": artifact_sha256,
            "load_back_succeeded": bool(load_back_succeeded),
            "load_back_model_class": load_back_model_class,
        },
        "training_contract": {
            "selected_algorithm": selected_algorithm,
            "policy": policy,
            "selected_episode_mode": selected_episode_mode,
            "effective_seed": effective_seed,
            "total_timesteps": total_timesteps,
            "checkpoint_export_steps": [int(step) for step in checkpoint_export_steps],
            "requested_device": requested_device,
            "resolved_device": resolved_device,
            "startup_policy": STARTUP_POLICY_FRESH_ONLY,
        },
        "source_artifacts": {
            "env_config_path": str(env_config_path),
            "training_config_path": str(training_config_path),
            "state_manifest_path": str(state_manifest_path),
            "env_contract_report_path": str(env_contract_report_path),
            "readiness_report_path": str(readiness_report_path),
            "episode_catalog_path": str(episode_catalog_path),
            "split_report_path": str(split_report_path),
        },
        "lineages": {
            "selected_episode_ref": dict(selected_episode.episode_ref) if selected_episode is not None else None,
            "selection_evidence": asdict(selected_episode) if selected_episode is not None else None,
            "hash_policy": {
                "algorithm": "sha256",
                "canonical_json": CANONICAL_JSON_POLICY,
                "binary_file_policy": "raw_file_bytes",
            },
            "semantic_hashes": {
                "env_config_hash": env_config_hash,
                "training_config_hash": training_config_hash,
                "state_manifest_hash": state_manifest_hash,
                "env_contract_hash": env_contract_hash,
                "readiness_hash": readiness_hash,
                "episode_catalog_hash": episode_catalog_hash,
                "split_report_hash": split_report_hash,
            },
            "source_file_sha256": {
                "env_config": _sha256_if_file(env_config_path),
                "training_config": _sha256_if_file(training_config_path),
                "state_manifest": _sha256_if_file(state_manifest_path),
                "env_contract_report": _sha256_if_file(env_contract_report_path),
                "readiness_report": _sha256_if_file(readiness_report_path),
                "episode_catalog": _sha256_if_file(episode_catalog_path),
                "split_report": _sha256_if_file(split_report_path),
            },
        },
        "checkpoint_artifacts": _checkpoint_artifacts_payload(
            checkpoint_export_steps=checkpoint_export_steps,
            checkpoint_exports=checkpoint_exports,
            checkpoint_artifacts_dir=(report_paths.checkpoint_artifacts_dir if report_paths is not None else None),
        ),
        "output_dir": str(output_dir),
        "produced_at_utc": _generated_at(),
    }


def _build_report_payload(
    *,
    run_id: str,
    production_session_id: str,
    selected_algorithm: str | None,
    policy: str | None,
    effective_seed: int | None,
    requested_device: str | None,
    resolved_device: str | None,
    total_timesteps: int | None,
    artifact_path: Path,
    artifact_sha256: str | None,
    save_succeeded: bool,
    artifact_exists: bool,
    artifact_zip_valid: bool,
    load_back_succeeded: bool,
    load_back_model_class: str | None,
    canonical_artifact_ready: bool,
    validation_checks: Sequence[dict[str, Any]],
    startup_phase_trace: Sequence[dict[str, Any]],
    production_summary: dict[str, Any],
    warnings: Sequence[ValidationIssue],
    errors: Sequence[ValidationIssue],
    runtime: dict[str, Any] | None = None,
    checkpoint_export_steps: Sequence[int] = (),
    checkpoint_exports: Sequence[Mapping[str, Any]] = (),
    report_paths: ReportPaths | None = None,
) -> dict[str, Any]:
    """Build machine-readable artifact production report payload."""

    return {
        "task_name": TASK_NAME,
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "production_session_id": production_session_id,
        "status": "success" if canonical_artifact_ready and len(errors) == 0 else "failed",
        "canonical_artifact_ready": bool(canonical_artifact_ready),
        "selected_algorithm": selected_algorithm,
        "policy": policy,
        "effective_seed": effective_seed,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "total_timesteps": total_timesteps,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "save_succeeded": bool(save_succeeded),
        "artifact_exists": bool(artifact_exists),
        "artifact_zip_valid": bool(artifact_zip_valid),
        "load_back_succeeded": bool(load_back_succeeded),
        "load_back_model_class": load_back_model_class,
        "validation_checks": list(validation_checks),
        "startup_phase_trace": list(startup_phase_trace),
        "production_summary": production_summary,
        "checkpoint_artifacts": _checkpoint_artifacts_payload(
            checkpoint_export_steps=checkpoint_export_steps,
            checkpoint_exports=checkpoint_exports,
            checkpoint_artifacts_dir=(report_paths.checkpoint_artifacts_dir if report_paths is not None else None),
        ),
        "runtime": dict(runtime) if runtime is not None else None,
        "warnings": [asdict(item) for item in warnings],
        "errors": [asdict(item) for item in errors],
        "failure_codes": _failure_codes(errors),
        "produced_at_utc": _generated_at(),
    }


def _validation_check(*, check_name: str, passed: bool, reason_code: str | None, detail: dict[str, Any]) -> dict[str, Any]:
    """Build one machine-readable validation check entry."""

    return {
        "check_name": check_name,
        "pass": bool(passed),
        "reason_code": reason_code,
        "detail": detail,
    }


def _phase_trace(
    *,
    validation_status: str,
    env_init_status: str,
    algo_init_status: str,
    learn_start_status: str,
    learn_finish_status: str,
    artifact_save_status: str,
    artifact_load_status: str,
    report_write_status: str,
    validation_detail: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build ordered phase trace for artifact production."""

    return [
        {"phase": "validation", "status": validation_status, "detail": validation_detail},
        {"phase": "env_init", "status": env_init_status, "detail": {}},
        {"phase": "algo_init", "status": algo_init_status, "detail": {}},
        {"phase": "learn_start", "status": learn_start_status, "detail": {}},
        {"phase": "learn_finish", "status": learn_finish_status, "detail": {}},
        {"phase": "artifact_save", "status": artifact_save_status, "detail": {}},
        {"phase": "artifact_load", "status": artifact_load_status, "detail": {}},
        {"phase": "report_write", "status": report_write_status, "detail": {}},
    ]


def _phase_trace_from_maps(status_map: Mapping[str, str], detail_map: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ordered phase trace from phase maps."""

    phases = (
        "validation",
        "env_init",
        "algo_init",
        "learn_start",
        "learn_finish",
        "artifact_save",
        "artifact_load",
        "report_write",
    )
    return [
        {"phase": phase, "status": str(status_map.get(phase, "not_started")), "detail": dict(detail_map.get(phase, {}))}
        for phase in phases
    ]


def _build_production_session_id(
    *,
    run_id: str,
    output_dir: Path,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
) -> str:
    """Build a deterministic reporting-only production session id."""

    payload = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "env_config_path": str(env_config_path),
        "training_config_path": str(training_config_path),
        "state_manifest_path": str(state_manifest_path),
        "env_contract_report_path": str(env_contract_report_path),
        "readiness_report_path": str(readiness_report_path),
        "episode_catalog_path": str(episode_catalog_path),
        "split_report_path": str(split_report_path),
    }
    return _hash_canonical_json(payload)[:16]


def _canonicality_checks(
    *,
    artifact_path: Path,
    artifact_sha256: str | None,
    save_succeeded: bool,
    artifact_exists: bool,
    artifact_zip_valid: bool,
    load_back_succeeded: bool,
    training_config_hash: str | None,
    env_contract_hash: str | None,
    readiness_hash: str | None,
    state_manifest_hash: str | None,
    split_report_hash: str | None,
) -> dict[str, bool]:
    """Build explicit canonicality booleans for closure-grade reporting."""

    return {
        "explicit_output_path_recorded": str(artifact_path) != "",
        "artifact_hash_recorded": isinstance(artifact_sha256, str) and len(artifact_sha256) > 0,
        "training_config_hash_recorded": isinstance(training_config_hash, str) and len(training_config_hash) > 0,
        "env_contract_hash_recorded": isinstance(env_contract_hash, str) and len(env_contract_hash) > 0,
        "readiness_hash_recorded": isinstance(readiness_hash, str) and len(readiness_hash) > 0,
        "state_manifest_hash_recorded": isinstance(state_manifest_hash, str) and len(state_manifest_hash) > 0,
        "split_report_hash_recorded": isinstance(split_report_hash, str) and len(split_report_hash) > 0,
        "save_succeeded": bool(save_succeeded),
        "artifact_exists": bool(artifact_exists),
        "artifact_zip_valid": bool(artifact_zip_valid),
        "load_back_succeeded": bool(load_back_succeeded),
    }


def _checkpoint_artifacts_payload(
    *,
    checkpoint_export_steps: Sequence[int],
    checkpoint_exports: Sequence[Mapping[str, Any]],
    checkpoint_artifacts_dir: Path | None,
) -> dict[str, Any]:
    """Build additive checkpoint export metadata for manifest/report payloads."""

    requested_steps = [int(step) for step in checkpoint_export_steps]
    return {
        "enabled": len(requested_steps) > 0,
        "artifacts_dir": str(checkpoint_artifacts_dir) if checkpoint_artifacts_dir is not None and requested_steps else None,
        "requested_steps": requested_steps,
        "expected_export_count": len(requested_steps),
        "actual_export_count": len(checkpoint_exports),
        "all_requested_exports_succeeded": len(requested_steps) == len(checkpoint_exports),
        "exports": [dict(item) for item in checkpoint_exports],
    }


def _validate_optional_optimizations(
    *,
    enable_amp: bool,
    enable_torch_compile: bool,
    runtime_environment: Any,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    """Validate optional performance flags without weakening canonical semantics."""

    optimization_runtime = {
        "amp": {
            "requested": bool(enable_amp),
            "enabled": False,
            "runtime_supported": bool(runtime_environment.torch_available and runtime_environment.cuda_available),
            "reason": "not_requested",
        },
        "torch_compile": {
            "requested": bool(enable_torch_compile),
            "enabled": False,
            "runtime_supported": bool(runtime_environment.torch_compile_supported),
            "reason": "not_requested",
        },
    }
    issues: list[ValidationIssue] = []

    if enable_amp:
        optimization_runtime["amp"]["reason"] = "unsupported_on_canonical_artifact_path_v1"
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_AMP_UNSUPPORTED,
                message="AMP is not supported on the canonical PPO artifact-production path without broader SB3 changes.",
                context={
                    "requested": True,
                    "cuda_available": bool(runtime_environment.cuda_available),
                    "torch_available": bool(runtime_environment.torch_available),
                    "gpu_name": runtime_environment.gpu_name,
                },
            )
        )

    if enable_torch_compile:
        optimization_runtime["torch_compile"]["reason"] = "unsupported_on_canonical_artifact_path_v1"
        issues.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_COMPILE_UNSUPPORTED,
                message="torch.compile is not supported on the canonical PPO artifact-production path without broader SB3 changes.",
                context={
                    "requested": True,
                    "torch_compile_supported": bool(runtime_environment.torch_compile_supported),
                    "torch_version": runtime_environment.torch_version,
                },
            )
        )

    return optimization_runtime, issues


def _validate_checkpoint_export_steps(
    raw_value: Any,
    *,
    total_timesteps: int | None,
) -> tuple[tuple[int, ...], list[ValidationIssue]]:
    """Validate optional checkpoint export steps."""

    if not isinstance(raw_value, list):
        return (), [
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="checkpoint_export_steps must be an array of strictly increasing positive integers when provided.",
                context={"checkpoint_export_steps": raw_value},
            )
        ]

    errors: list[ValidationIssue] = []
    parsed_steps: list[int] = []
    for index, value in enumerate(raw_value):
        if not isinstance(value, int) or isinstance(value, bool) or int(value) <= 0:
            errors.append(
                ValidationIssue(
                    code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                    message="checkpoint_export_steps entries must be positive integers.",
                    context={"index": index, "value": value},
                )
            )
            continue
        parsed_steps.append(int(value))

    if errors:
        return (), errors

    if parsed_steps != sorted(parsed_steps) or len(parsed_steps) != len(set(parsed_steps)):
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="checkpoint_export_steps must be strictly increasing with no duplicates.",
                context={"checkpoint_export_steps": parsed_steps},
            )
        )
    if total_timesteps is not None and any(step > int(total_timesteps) for step in parsed_steps):
        errors.append(
            ValidationIssue(
                code=ARTIFACT_PRODUCTION_CONFIG_INVALID,
                message="checkpoint_export_steps must be <= total_timesteps.",
                context={"checkpoint_export_steps": parsed_steps, "total_timesteps": total_timesteps},
            )
        )
    return tuple(parsed_steps) if not errors else (), errors


def _build_checkpoint_export_callback(
    *,
    run_id: str,
    checkpoint_export_steps: Sequence[int],
    checkpoint_artifacts_dir: Path,
    resolved_device: str | None,
) -> Any:
    """Build an optional callback that exports load-valid checkpoint artifacts."""

    from stable_baselines3.common.callbacks import BaseCallback

    class _CheckpointExportCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._requested_steps = tuple(int(step) for step in checkpoint_export_steps)
            self._remaining_steps = list(self._requested_steps)
            self._exports: list[dict[str, Any]] = []

        def _on_training_start(self) -> None:
            if not self._requested_steps:
                return
            if checkpoint_artifacts_dir.exists():
                raise ControlledArtifactProductionFailure(
                    ValidationIssue(
                        code=ARTIFACT_PRODUCTION_OUTPUT_CONFLICT,
                        message="checkpoint_artifacts_dir already exists inside the fresh artifact output dir.",
                        context={"checkpoint_artifacts_dir": str(checkpoint_artifacts_dir)},
                    )
                )
            checkpoint_artifacts_dir.mkdir(parents=False, exist_ok=False)

        def _on_step(self) -> bool:
            current_step = int(getattr(self.model, "num_timesteps", 0))
            while self._remaining_steps and current_step >= self._remaining_steps[0]:
                requested_step = int(self._remaining_steps.pop(0))
                artifact_path = checkpoint_artifacts_dir / CHECKPOINT_ARTIFACT_FILENAME_TEMPLATE.format(step=requested_step)
                LOGGER.info(
                    "Checkpoint export start | run_id=%s requested_step=%d artifact_path=%s",
                    run_id,
                    requested_step,
                    artifact_path,
                )
                save_state = _save_and_validate_model_artifact(
                    model=self.model,
                    artifact_path=artifact_path,
                    resolved_device=resolved_device,
                    artifact_role="checkpoint",
                )
                exported_step = int(getattr(self.model, "num_timesteps", requested_step))
                self._exports.append(
                    {
                        "requested_step": requested_step,
                        "exported_num_timesteps": exported_step,
                        "artifact_path": str(artifact_path),
                        "filename": artifact_path.name,
                        "artifact_sha256": save_state["artifact_sha256"],
                        "artifact_exists": bool(save_state["artifact_exists"]),
                        "artifact_zip_valid": bool(save_state["artifact_zip_valid"]),
                        "load_back_succeeded": bool(save_state["load_back_succeeded"]),
                        "load_back_model_class": save_state["load_back_model_class"],
                    }
                )
                LOGGER.info(
                    "Checkpoint export finished | run_id=%s requested_step=%d exported_num_timesteps=%d artifact_path=%s",
                    run_id,
                    requested_step,
                    exported_step,
                    artifact_path,
                )
            return True

        def _on_training_end(self) -> None:
            if not self._remaining_steps:
                return
            raise ControlledArtifactProductionFailure(
                ValidationIssue(
                    code=ARTIFACT_PRODUCTION_CHECKPOINT_EXPORT_FAILED,
                    message="Training finished before all requested checkpoint artifacts were exported.",
                    context={
                        "remaining_requested_steps": list(self._remaining_steps),
                        "num_timesteps_after_learn": int(getattr(self.model, "num_timesteps", 0)),
                    },
                )
            )

        def build_exports_payload(self) -> list[dict[str, Any]]:
            return [dict(item) for item in self._exports]

    return _CheckpointExportCallback()


def _build_runtime_payload(
    *,
    runtime_environment: Any,
    progress_resolution: Any,
    requested_device: str | None,
    resolved_device: str | None,
    enable_amp: bool,
    enable_torch_compile: bool,
    optimization_runtime: dict[str, Any],
    memory_log_interval_steps: int,
    memory_snapshots: Sequence[dict[str, Any]],
    numeric_safety: dict[str, Any] | None,
    n_envs: int,
    stage_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Build additive runtime metadata for Colab/local execution reporting."""

    return {
        "environment": asdict(runtime_environment),
        "progress": asdict(progress_resolution),
        "device": {
            "requested_device": requested_device,
            "resolved_device": resolved_device,
        },
        "execution_bounds": {
            "n_envs": int(n_envs),
            "memory_log_interval_steps": int(memory_log_interval_steps),
        },
        "optimizations": {
            "amp": {
                **dict(optimization_runtime.get("amp", {})),
                "requested": bool(enable_amp),
            },
            "torch_compile": {
                **dict(optimization_runtime.get("torch_compile", {})),
                "requested": bool(enable_torch_compile),
            },
        },
        "numeric_safety": dict(numeric_safety) if numeric_safety is not None else None,
        "memory_snapshots": [dict(item) for item in memory_snapshots],
        "stage_preflight": _serialize_stage_preflight(stage_preflight),
    }


def _looks_like_numeric_failure(exc: Exception) -> bool:
    """Return whether an exception indicates a non-finite training failure."""

    if isinstance(exc, FloatingPointError):
        return True
    message = str(exc).lower()
    return "non-finite" in message or "must be finite" in message


def _normalize_issues(issues: Sequence[Any]) -> list[ValidationIssue]:
    """Normalize imported issue types into this module's issue dataclass."""

    normalized: list[ValidationIssue] = []
    for issue in issues:
        normalized.append(
            ValidationIssue(
                code=str(getattr(issue, "code", "UNKNOWN_ISSUE")),
                message=str(getattr(issue, "message", "")),
                context=dict(getattr(issue, "context", {}) or {}),
            )
        )
    return normalized


def _serialize_stage_preflight(stage_preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable stage-preflight payload."""

    serialized: dict[str, Any] = {}
    for key, value in stage_preflight.items():
        if key == "issues" and isinstance(value, Sequence):
            serialized[key] = [asdict(item) if isinstance(item, ValidationIssue) else dict(item) for item in value]
        elif isinstance(value, Mapping):
            serialized[key] = dict(value)
        else:
            serialized[key] = value
    return serialized


def _first_code(issues: Sequence[ValidationIssue], candidates: set[str]) -> str | None:
    """Return the first matching code from a sequence of issues."""

    for issue in issues:
        if issue.code in candidates:
            return issue.code
    return None


def _raw_string(payload: dict[str, Any] | None, key: str) -> str | None:
    """Return a raw string value when present and well-typed."""

    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, str):
        return value
    return None


def _raw_int(payload: dict[str, Any] | None, key: str) -> int | None:
    """Return a raw integer value when present and well-typed."""

    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _tmp_artifact_path(artifact_path: Path) -> Path:
    """Build the temp path used for atomic artifact production."""

    return artifact_path.with_suffix(f"{artifact_path.suffix}.tmp")


def _sha256_if_file(path: Path) -> str | None:
    """Return raw-file SHA256 when the path is a readable file."""

    if not path.exists() or not path.is_file():
        return None
    try:
        return _sha256_file(path)
    except OSError:
        return None


def _generated_at() -> str:
    """Return the current UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()
