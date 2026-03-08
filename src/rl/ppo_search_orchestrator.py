"""Milestone 4.9 constrained PPO hyperparameter search orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from core.io_atomic import atomic_write_json
from core.logging import get_logger
import pandas as pd
from rl.evaluation_backtest import (
    execute_evaluation_backtest,
    _validate_eval_config,
)
from rl.ppo_artifact_production import (
    execute_ppo_artifact_production,
    _validate_artifact_production_config,
)
from rl.training_launcher import execute_training_launch

LOGGER = get_logger(__name__)

TASK_NAME = "Milestone 4.9 — Constrained PPO Hyperparameter Search Orchestrator Contract"
CONTRACT_VERSION = "ppo_search_orchestrator.v1"
CANONICAL_JSON_POLICY = "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=True)"

STUDY_MODE_PPO = "ppo_hparam_search"
SEARCH_METHOD_GRID = "grid_product_v1"

TRIAL_STATUS_PENDING = "pending"
TRIAL_STATUS_RUNNING = "running"
TRIAL_STATUS_INVALID = "invalid"
TRIAL_STATUS_FAILED = "failed"
TRIAL_STATUS_PRUNED = "pruned"
TRIAL_STATUS_COMPLETED_NONCOMPETITIVE = "completed_noncompetitive"
TRIAL_STATUS_COMPLETED_CANDIDATE = "completed_candidate"
TRIAL_STATUS_PROMOTION_READY_CANDIDATE = "promotion_ready_candidate"

PRIMARY_METRIC_TOTAL_RETURN = "total_return"
PRIMARY_METRIC_EXCESS_TOTAL_RETURN = "excess_total_return"
SUPPORTED_PRIMARY_METRICS = {PRIMARY_METRIC_TOTAL_RETURN, PRIMARY_METRIC_EXCESS_TOTAL_RETURN}

SEARCHABLE_PPO_FIELDS = (
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
FIRST_WAVE_REQUIRED_FIELDS = (
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
)
INT_PPO_FIELDS = {"n_steps", "batch_size", "n_epochs"}

TRIAL_MANIFEST_FILENAME = "trial_manifest.json"
TRIAL_TRAINING_REPORT_FILENAME = "trial_training_report.json"
TRIAL_EVALUATION_REPORT_FILENAME = "trial_evaluation_report.json"
TRIAL_OBJECTIVE_REPORT_FILENAME = "trial_objective_report.json"
TRIAL_GUARDRAIL_REPORT_FILENAME = "trial_guardrail_report.json"
TRIAL_STATUS_FILENAME = "trial_status.json"
STUDY_MANIFEST_FILENAME = "study_manifest.json"
STUDY_PROGRESS_FILENAME = "study_progress.json"
STUDY_SUMMARY_FILENAME = "study_summary.json"

SEARCH_INPUT_MISSING = "SEARCH_INPUT_MISSING"
SEARCH_PATH_UNREADABLE = "SEARCH_PATH_UNREADABLE"
SEARCH_JSON_INVALID = "SEARCH_JSON_INVALID"
SEARCH_STUDY_CONFIG_INVALID = "SEARCH_STUDY_CONFIG_INVALID"
SEARCH_OUTPUT_CONFLICT = "SEARCH_OUTPUT_CONFLICT"
SEARCH_SEARCH_SPACE_INVALID = "SEARCH_SEARCH_SPACE_INVALID"
SEARCH_UPSTREAM_CONTRACT_INVALID = "SEARCH_UPSTREAM_CONTRACT_INVALID"
SEARCH_OBJECTIVE_INVALID = "SEARCH_OBJECTIVE_INVALID"
SEARCH_GUARDRAIL_INVALID = "SEARCH_GUARDRAIL_INVALID"
SEARCH_REPORT_INVALID = "SEARCH_REPORT_INVALID"
SEARCH_ARTIFACT_MISSING = "SEARCH_ARTIFACT_MISSING"
SEARCH_NUMERIC_PATHOLOGY = "SEARCH_NUMERIC_PATHOLOGY"
SEARCH_CATASTROPHIC_RISK_BREACH = "SEARCH_CATASTROPHIC_RISK_BREACH"
SEARCH_PATHOLOGICAL_INACTIVITY = "SEARCH_PATHOLOGICAL_INACTIVITY"
SEARCH_PATHOLOGICAL_TRADE_BEHAVIOR = "SEARCH_PATHOLOGICAL_TRADE_BEHAVIOR"
SEARCH_TRIAL_FAILED = "SEARCH_TRIAL_FAILED"
SEARCH_PRUNED_BY_OBJECTIVE = "SEARCH_PRUNED_BY_OBJECTIVE"

LAUNCHER_RUNTIME_FAILURE_CODES = {
    "TRAIN_LAUNCH_ENV_INIT_FAILED",
    "TRAIN_LAUNCH_ALGO_INIT_FAILED",
    "TRAIN_LAUNCH_SMOKE_FAILED",
}
ARTIFACT_RUNTIME_FAILURE_CODES = {
    "ARTIFACT_PRODUCTION_ENV_INIT_FAILED",
    "ARTIFACT_PRODUCTION_ALGO_INIT_FAILED",
    "ARTIFACT_PRODUCTION_TRAIN_FAILED",
    "ARTIFACT_PRODUCTION_SAVE_FAILED",
    "ARTIFACT_PRODUCTION_LOAD_BACK_FAILED",
    "ARTIFACT_PRODUCTION_REPORT_WRITE_FAILED",
}
EVAL_RUNTIME_FAILURE_CODES = {
    "EVAL_MODEL_LOAD_FAILED",
    "EVAL_ENV_INIT_FAILED",
    "EVAL_EXECUTION_FAILED",
    "EVAL_BENCHMARK_FAILED",
    "EVAL_REPORT_WRITE_FAILED",
}


@dataclass
class ValidationIssue:
    """Machine-readable 4.9 issue payload."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpstreamRefs:
    """Explicit upstream references consumed by a 4.9 study."""

    run_id: str
    env_config_path: Path
    state_manifest_path: Path
    env_contract_report_path: Path
    readiness_report_path: Path
    episode_catalog_path: Path
    split_report_path: Path
    artifact_training_config_template_path: Path
    eval_config_template_path: Path


@dataclass(frozen=True)
class ResourceBudget:
    """Study-level resource budget contract."""

    max_trials: int
    launcher_smoke_learn_timesteps: int
    probe_train_total_timesteps: int | None
    full_train_total_timesteps: int
    max_eval_episodes: int
    max_eval_steps: int


@dataclass(frozen=True)
class ObjectiveSpec:
    """Single-scalar objective configuration."""

    primary_metric: str
    turnover_penalty_weight: float
    instability_penalty_weight: float
    low_trade_count_penalty_weight: float
    soft_trade_rate_target: float


@dataclass(frozen=True)
class GuardrailSpec:
    """Hard constraint configuration."""

    require_step_trace: bool
    max_strategy_max_drawdown: float
    min_num_trades_hard: int
    min_num_trades_soft: int
    max_trade_rate_hard: float


@dataclass(frozen=True)
class PruningSpec:
    """Conservative pruning configuration."""

    enabled: bool
    warmup_trials: int
    min_completed_probe_trials: int
    min_probe_objective_score: float | None
    relative_to_best_completed_margin: float


@dataclass(frozen=True)
class PromotionSpec:
    """Promotion-readiness gates."""

    candidate_top_k: int
    promotion_min_distinct_seeds: int
    require_positive_objective: bool
    max_strategy_max_drawdown: float
    min_num_trades: int


@dataclass(frozen=True)
class StudySpec:
    """Validated 4.9 study contract."""

    study_id: str
    milestone: str
    study_mode: str
    search_method: str
    sampler_seed: int
    trial_seed: int
    upstream_refs: UpstreamRefs
    search_space: dict[str, tuple[float | int, ...]]
    resource_budget: ResourceBudget
    objective_spec: ObjectiveSpec
    guardrail_spec: GuardrailSpec
    pruning_spec: PruningSpec
    promotion_spec: PromotionSpec
    output_root: Path
    artifact_training_template: dict[str, Any]
    eval_template: dict[str, Any]
    study_config_path: Path
    study_config_hash: str


@dataclass(frozen=True)
class TrialSpec:
    """Resolved trial contract."""

    trial_id: str
    trial_index: int
    study_id: str
    seed: int
    ppo_params: dict[str, float | int]
    output_dir: Path
    param_assignment_hash: str
    launcher_config_path: Path
    probe_training_config_path: Path | None
    final_training_config_path: Path
    probe_eval_config_path: Path | None
    final_eval_config_path: Path


@dataclass
class TrialExecutionRecord:
    """Trial execution state carried through study finalization."""

    trial_spec: TrialSpec
    terminal_state: str
    final_status: str
    ranking_eligible: bool
    guardrail_pass: bool
    objective_score: float | None
    probe_objective_score: float | None
    primary_metric_value: float | None
    failure_codes: list[str]
    invalid_reasons: list[str]
    pruned_reason: str | None
    candidate_rank: int | None
    promotion_ready: bool
    promotion_family_hash: str
    artifacts_complete: bool
    started_at_utc: str
    completed_at_utc: str
    manifest_payload: dict[str, Any]
    training_report_payload: dict[str, Any]
    evaluation_report_payload: dict[str, Any]
    objective_report_payload: dict[str, Any]
    guardrail_report_payload: dict[str, Any]
    status_payload: dict[str, Any]


@dataclass(frozen=True)
class StudyReportPaths:
    """Top-level study report paths."""

    manifest_path: Path
    progress_path: Path
    summary_path: Path


@dataclass
class StudyExecutionResult:
    """Composite 4.9 study execution result."""

    exit_code: int
    study_manifest_payload: dict[str, Any] | None
    study_progress_payload: dict[str, Any] | None
    study_summary_payload: dict[str, Any]
    report_paths: StudyReportPaths | None
    reports_written: bool


def execute_ppo_search_study(*, study_config_path: Path) -> StudyExecutionResult:
    """Execute a strict, explicit-path 4.9 PPO search study."""

    loaded_payload, load_issues = _load_json_file(study_config_path.resolve(), label="study_config")
    if load_issues:
        summary_payload = _build_invalid_study_summary(
            study_id=None,
            study_config_path=study_config_path.resolve(),
            failure_codes=_failure_codes(load_issues),
            issues=load_issues,
        )
        return StudyExecutionResult(
            exit_code=2,
            study_manifest_payload=None,
            study_progress_payload=None,
            study_summary_payload=summary_payload,
            report_paths=None,
            reports_written=False,
        )

    assert loaded_payload is not None
    validation = _validate_study_spec(payload=loaded_payload, study_config_path=study_config_path.resolve())
    study_spec = validation.get("study_spec")
    issues = validation["issues"]
    if study_spec is None:
        summary_payload = _build_invalid_study_summary(
            study_id=_safe_string(loaded_payload.get("study_id")),
            study_config_path=study_config_path.resolve(),
            failure_codes=_failure_codes(issues),
            issues=issues,
        )
        return StudyExecutionResult(
            exit_code=2,
            study_manifest_payload=None,
            study_progress_payload=None,
            study_summary_payload=summary_payload,
            report_paths=None,
            reports_written=False,
        )

    report_paths = StudyReportPaths(
        manifest_path=study_spec.output_root / STUDY_MANIFEST_FILENAME,
        progress_path=study_spec.output_root / STUDY_PROGRESS_FILENAME,
        summary_path=study_spec.output_root / STUDY_SUMMARY_FILENAME,
    )
    output_guard_issues = _check_output_dir_policy(study_spec.output_root)
    if output_guard_issues:
        summary_payload = _build_invalid_study_summary(
            study_id=study_spec.study_id,
            study_config_path=study_spec.study_config_path,
            failure_codes=_failure_codes(output_guard_issues),
            issues=output_guard_issues,
        )
        return StudyExecutionResult(
            exit_code=2,
            study_manifest_payload=None,
            study_progress_payload=None,
            study_summary_payload=summary_payload,
            report_paths=report_paths,
            reports_written=False,
        )

    study_spec.output_root.mkdir(parents=True, exist_ok=False)
    trial_specs = _build_trial_specs(study_spec)
    study_manifest_payload = _build_study_manifest_payload(study_spec=study_spec, trial_specs=trial_specs)
    atomic_write_json(study_manifest_payload, report_paths.manifest_path)

    trial_records: list[TrialExecutionRecord] = []
    initial_progress = _build_study_progress_payload(
        study_spec=study_spec,
        trial_records=trial_records,
        study_status="running",
    )
    atomic_write_json(initial_progress, report_paths.progress_path)

    for trial_spec in trial_specs:
        LOGGER.info(
            "4.9 trial start | study_id=%s trial_id=%s trial_index=%d",
            study_spec.study_id,
            trial_spec.trial_id,
            trial_spec.trial_index,
        )
        trial_record = _execute_trial(study_spec=study_spec, trial_spec=trial_spec, prior_records=tuple(trial_records))
        trial_records.append(trial_record)
        _write_trial_reports(trial_record)
        progress_payload = _build_study_progress_payload(
            study_spec=study_spec,
            trial_records=trial_records,
            study_status="running",
        )
        atomic_write_json(progress_payload, report_paths.progress_path)

    _finalize_completed_trial_statuses(study_spec=study_spec, trial_records=trial_records)
    for trial_record in trial_records:
        _write_trial_status(trial_record)

    final_progress = _build_study_progress_payload(
        study_spec=study_spec,
        trial_records=trial_records,
        study_status="completed",
    )
    study_summary_payload = _build_study_summary_payload(study_spec=study_spec, trial_records=trial_records)
    atomic_write_json(final_progress, report_paths.progress_path)
    atomic_write_json(study_summary_payload, report_paths.summary_path)

    exit_code = 0
    if any(record.final_status in {TRIAL_STATUS_FAILED, TRIAL_STATUS_INVALID} for record in trial_records):
        exit_code = 0
    return StudyExecutionResult(
        exit_code=exit_code,
        study_manifest_payload=study_manifest_payload,
        study_progress_payload=final_progress,
        study_summary_payload=study_summary_payload,
        report_paths=report_paths,
        reports_written=True,
    )


def _validate_study_spec(*, payload: dict[str, Any], study_config_path: Path) -> dict[str, Any]:
    """Validate the strict 4.9 study contract."""

    issues: list[ValidationIssue] = []
    required_fields = {
        "study_id",
        "milestone",
        "study_mode",
        "search_method",
        "sampler_seed",
        "trial_seed",
        "upstream_refs",
        "search_space",
        "resource_budget",
        "objective_spec",
        "guardrail_spec",
        "pruning_spec",
        "promotion_spec",
        "output_root",
    }
    extra_keys = sorted(set(payload.keys()) - required_fields)
    missing_keys = sorted(required_fields - set(payload.keys()))
    if missing_keys or extra_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="study_config top-level fields must match the 4.9 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return {"study_spec": None, "issues": issues}

    study_id = _safe_string(payload.get("study_id"))
    if not study_id:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="study_id must be a non-empty string.",
                context={"study_id": payload.get("study_id")},
            )
        )

    milestone = _safe_string(payload.get("milestone"))
    if milestone != TASK_NAME:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="milestone must match the exact frozen 4.9 task name.",
                context={"milestone": milestone, "expected": TASK_NAME},
            )
        )

    study_mode = _safe_string(payload.get("study_mode"))
    if study_mode != STUDY_MODE_PPO:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="study_mode must be ppo_hparam_search.",
                context={"study_mode": study_mode},
            )
        )

    search_method = _safe_string(payload.get("search_method"))
    if search_method != SEARCH_METHOD_GRID:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="search_method must be grid_product_v1 in first-wave 4.9.",
                context={"search_method": search_method},
            )
        )

    sampler_seed = _parse_non_negative_int(
        payload.get("sampler_seed"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "sampler_seed",
    )
    trial_seed = _parse_non_negative_int(
        payload.get("trial_seed"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "trial_seed",
    )

    upstream_refs_result = _validate_upstream_refs(
        payload.get("upstream_refs"),
        base_dir=study_config_path.parent,
        issues=issues,
    )
    artifact_template_payload = upstream_refs_result.get("artifact_template_payload")
    eval_template_payload = upstream_refs_result.get("eval_template_payload")
    upstream_refs = upstream_refs_result.get("upstream_refs")

    search_space = _validate_search_space(payload.get("search_space"), issues)
    resource_budget = _validate_resource_budget(payload.get("resource_budget"), issues)
    objective_spec = _validate_objective_spec(payload.get("objective_spec"), issues)
    guardrail_spec = _validate_guardrail_spec(payload.get("guardrail_spec"), issues)
    pruning_spec = _validate_pruning_spec(payload.get("pruning_spec"), issues)
    promotion_spec = _validate_promotion_spec(payload.get("promotion_spec"), issues)

    output_root_raw = payload.get("output_root")
    output_root = _resolve_path(output_root_raw, study_config_path.parent) if isinstance(output_root_raw, str) else None
    if output_root is None:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="output_root must be a string path.",
                context={"output_root": output_root_raw},
            )
        )

    if artifact_template_payload is not None and objective_spec is not None and guardrail_spec is not None:
        if guardrail_spec.min_num_trades_soft < guardrail_spec.min_num_trades_hard:
            issues.append(
                ValidationIssue(
                    code=SEARCH_GUARDRAIL_INVALID,
                    message="min_num_trades_soft must be >= min_num_trades_hard.",
                    context={
                        "min_num_trades_soft": guardrail_spec.min_num_trades_soft,
                        "min_num_trades_hard": guardrail_spec.min_num_trades_hard,
                    },
                )
            )
        if objective_spec.primary_metric == PRIMARY_METRIC_EXCESS_TOTAL_RETURN:
            benchmark_mode = _safe_string(eval_template_payload.get("benchmark_mode")) if isinstance(eval_template_payload, Mapping) else ""
            if benchmark_mode != "buy_and_hold":
                issues.append(
                    ValidationIssue(
                        code=SEARCH_OBJECTIVE_INVALID,
                        message="excess_total_return requires a buy_and_hold benchmark in the eval template.",
                        context={"benchmark_mode": benchmark_mode},
                    )
                )

    if resource_budget is not None and pruning_spec is not None and pruning_spec.enabled:
        if resource_budget.probe_train_total_timesteps is None:
            issues.append(
                ValidationIssue(
                    code=SEARCH_STUDY_CONFIG_INVALID,
                    message="probe_train_total_timesteps is required when pruning is enabled.",
                    context={"probe_train_total_timesteps": None},
                )
            )
        elif resource_budget.probe_train_total_timesteps >= resource_budget.full_train_total_timesteps:
            issues.append(
                ValidationIssue(
                    code=SEARCH_STUDY_CONFIG_INVALID,
                    message="probe_train_total_timesteps must be < full_train_total_timesteps when pruning is enabled.",
                    context={
                        "probe_train_total_timesteps": resource_budget.probe_train_total_timesteps,
                        "full_train_total_timesteps": resource_budget.full_train_total_timesteps,
                    },
                )
            )

    if search_space is not None and resource_budget is not None:
        search_space_cardinality = _product_cardinality(search_space)
        if search_space_cardinality <= 0:
            issues.append(
                ValidationIssue(
                    code=SEARCH_SEARCH_SPACE_INVALID,
                    message="search_space does not yield any candidate assignments.",
                    context={"search_space_cardinality": search_space_cardinality},
                )
            )
        elif resource_budget.max_trials > search_space_cardinality:
            issues.append(
                ValidationIssue(
                    code=SEARCH_SEARCH_SPACE_INVALID,
                    message="max_trials cannot exceed the explicit first-wave search-space cardinality.",
                    context={
                        "max_trials": resource_budget.max_trials,
                        "search_space_cardinality": search_space_cardinality,
                    },
                )
            )

    if issues:
        return {"study_spec": None, "issues": issues}

    assert study_id
    assert milestone
    assert study_mode
    assert search_method
    assert sampler_seed is not None
    assert trial_seed is not None
    assert upstream_refs is not None
    assert search_space is not None
    assert resource_budget is not None
    assert objective_spec is not None
    assert guardrail_spec is not None
    assert pruning_spec is not None
    assert promotion_spec is not None
    assert output_root is not None
    assert artifact_template_payload is not None
    assert eval_template_payload is not None

    study_spec = StudySpec(
        study_id=study_id,
        milestone=milestone,
        study_mode=study_mode,
        search_method=search_method,
        sampler_seed=sampler_seed,
        trial_seed=trial_seed,
        upstream_refs=upstream_refs,
        search_space=search_space,
        resource_budget=resource_budget,
        objective_spec=objective_spec,
        guardrail_spec=guardrail_spec,
        pruning_spec=pruning_spec,
        promotion_spec=promotion_spec,
        output_root=output_root.resolve(),
        artifact_training_template=dict(artifact_template_payload),
        eval_template=dict(eval_template_payload),
        study_config_path=study_config_path,
        study_config_hash=_hash_canonical_json(payload),
    )
    return {"study_spec": study_spec, "issues": issues}


def _validate_upstream_refs(payload: Any, *, base_dir: Path, issues: list[ValidationIssue]) -> dict[str, Any]:
    """Validate explicit upstream references."""

    if not isinstance(payload, Mapping):
        issues.append(
            ValidationIssue(
                code=SEARCH_UPSTREAM_CONTRACT_INVALID,
                message="upstream_refs must be an object.",
                context={"payload_type": type(payload).__name__},
            )
        )
        return {"upstream_refs": None, "artifact_template_payload": None, "eval_template_payload": None}

    required_fields = {
        "run_id",
        "env_config_path",
        "state_manifest_path",
        "env_contract_report_path",
        "readiness_report_path",
        "episode_catalog_path",
        "split_report_path",
        "artifact_training_config_template_path",
        "eval_config_template_path",
    }
    extra_keys = sorted(set(payload.keys()) - required_fields)
    missing_keys = sorted(required_fields - set(payload.keys()))
    if missing_keys or extra_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_UPSTREAM_CONTRACT_INVALID,
                message="upstream_refs fields must match the strict 4.9 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return {"upstream_refs": None, "artifact_template_payload": None, "eval_template_payload": None}

    run_id = _safe_string(payload.get("run_id"))
    if not run_id:
        issues.append(
            ValidationIssue(
                code=SEARCH_UPSTREAM_CONTRACT_INVALID,
                message="upstream_refs.run_id must be a non-empty string.",
                context={"run_id": payload.get("run_id")},
            )
        )

    resolved_paths: dict[str, Path] = {}
    for key in required_fields - {"run_id"}:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(
                ValidationIssue(
                    code=SEARCH_UPSTREAM_CONTRACT_INVALID,
                    message="All upstream path fields must be non-empty strings.",
                    context={"field": key, "value": value},
                )
            )
            continue
        resolved_path = _resolve_path(value, base_dir)
        resolved_paths[key] = resolved_path
        if not resolved_path.exists():
            issues.append(
                ValidationIssue(
                    code=SEARCH_INPUT_MISSING,
                    message="Required 4.9 upstream input is missing.",
                    context={"field": key, "path": str(resolved_path)},
                )
            )
        elif not resolved_path.is_file():
            issues.append(
                ValidationIssue(
                    code=SEARCH_PATH_UNREADABLE,
                    message="Required 4.9 upstream input is not a readable file.",
                    context={"field": key, "path": str(resolved_path)},
                )
            )

    artifact_template_payload: dict[str, Any] | None = None
    eval_template_payload: dict[str, Any] | None = None
    if "artifact_training_config_template_path" in resolved_paths and resolved_paths["artifact_training_config_template_path"].is_file():
        artifact_template_payload, template_issues = _load_json_file(
            resolved_paths["artifact_training_config_template_path"],
            label="artifact_training_config_template",
        )
        issues.extend(template_issues)
        if artifact_template_payload is not None:
            config_result = _validate_artifact_production_config(artifact_template_payload)
            for item in config_result["errors"]:
                issues.append(
                    ValidationIssue(
                        code=SEARCH_UPSTREAM_CONTRACT_INVALID,
                        message="artifact_training_config_template does not satisfy the canonical artifact contract.",
                        context={"upstream_issue": item.code, "detail": item.context},
                    )
                )
    if "eval_config_template_path" in resolved_paths and resolved_paths["eval_config_template_path"].is_file():
        eval_template_payload, template_issues = _load_json_file(
            resolved_paths["eval_config_template_path"],
            label="eval_config_template",
        )
        issues.extend(template_issues)
        if eval_template_payload is not None:
            eval_result = _validate_eval_config(eval_template_payload)
            for item in eval_result["errors"]:
                issues.append(
                    ValidationIssue(
                        code=SEARCH_UPSTREAM_CONTRACT_INVALID,
                        message="eval_config_template does not satisfy the 4.8 evaluation contract.",
                        context={"upstream_issue": item.code, "detail": item.context},
                    )
                )

    if issues:
        return {"upstream_refs": None, "artifact_template_payload": artifact_template_payload, "eval_template_payload": eval_template_payload}

    assert run_id
    upstream_refs = UpstreamRefs(
        run_id=run_id,
        env_config_path=resolved_paths["env_config_path"].resolve(),
        state_manifest_path=resolved_paths["state_manifest_path"].resolve(),
        env_contract_report_path=resolved_paths["env_contract_report_path"].resolve(),
        readiness_report_path=resolved_paths["readiness_report_path"].resolve(),
        episode_catalog_path=resolved_paths["episode_catalog_path"].resolve(),
        split_report_path=resolved_paths["split_report_path"].resolve(),
        artifact_training_config_template_path=resolved_paths["artifact_training_config_template_path"].resolve(),
        eval_config_template_path=resolved_paths["eval_config_template_path"].resolve(),
    )
    return {"upstream_refs": upstream_refs, "artifact_template_payload": artifact_template_payload, "eval_template_payload": eval_template_payload}


def _validate_search_space(payload: Any, issues: list[ValidationIssue]) -> dict[str, tuple[float | int, ...]] | None:
    """Validate the narrow first-wave PPO search space."""

    if not isinstance(payload, Mapping):
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="search_space must be an object.",
                context={"payload_type": type(payload).__name__},
            )
        )
        return None

    keys = tuple(payload.keys())
    unknown_keys = sorted(set(keys) - set(SEARCHABLE_PPO_FIELDS))
    missing_first_wave = sorted(set(FIRST_WAVE_REQUIRED_FIELDS) - set(keys))
    if unknown_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="search_space contains unsupported keys.",
                context={"unknown_keys": unknown_keys},
            )
        )
    if missing_first_wave:
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="search_space must include the required first-wave PPO keys.",
                context={"missing_keys": missing_first_wave},
            )
        )
    if not (6 <= len(keys) <= 9):
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="first-wave 4.9 search_space must expose between 6 and 9 PPO hyperparameters.",
                context={"field_count": len(keys)},
            )
        )

    validated: dict[str, tuple[float | int, ...]] = {}
    for key, value in payload.items():
        if key not in SEARCHABLE_PPO_FIELDS:
            continue
        if not isinstance(value, list) or not value:
            issues.append(
                ValidationIssue(
                    code=SEARCH_SEARCH_SPACE_INVALID,
                    message="Each search_space field must be a non-empty explicit candidate list.",
                    context={"field": key, "value": value},
                )
            )
            continue
        parsed_values: list[float | int] = []
        for candidate in value:
            if key in INT_PPO_FIELDS:
                if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate <= 0:
                    issues.append(
                        ValidationIssue(
                            code=SEARCH_SEARCH_SPACE_INVALID,
                            message="Integer PPO candidates must be positive integers.",
                            context={"field": key, "candidate": candidate},
                        )
                    )
                    continue
                parsed_values.append(int(candidate))
                continue
            if not isinstance(candidate, (int, float)) or isinstance(candidate, bool) or not math.isfinite(float(candidate)):
                issues.append(
                    ValidationIssue(
                        code=SEARCH_SEARCH_SPACE_INVALID,
                        message="Floating PPO candidates must be finite numbers.",
                        context={"field": key, "candidate": candidate},
                    )
                )
                continue
            parsed_values.append(float(candidate))
        if parsed_values:
            deduped = tuple(_dedupe_sequence(parsed_values))
            validated[key] = deduped
    return validated if validated else None


def _validate_resource_budget(payload: Any, issues: list[ValidationIssue]) -> ResourceBudget | None:
    """Validate resource budget rules."""

    if not isinstance(payload, Mapping):
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="resource_budget must be an object.",
                context={"payload_type": type(payload).__name__},
            )
        )
        return None

    required_fields = {
        "max_trials",
        "launcher_smoke_learn_timesteps",
        "probe_train_total_timesteps",
        "full_train_total_timesteps",
        "max_eval_episodes",
        "max_eval_steps",
    }
    extra_keys = sorted(set(payload.keys()) - required_fields)
    missing_keys = sorted(required_fields - set(payload.keys()))
    if missing_keys or extra_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="resource_budget fields must match the strict 4.9 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return None

    max_trials = _parse_positive_int(payload.get("max_trials"), issues, SEARCH_STUDY_CONFIG_INVALID, "max_trials")
    launcher_smoke = _parse_positive_int(
        payload.get("launcher_smoke_learn_timesteps"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "launcher_smoke_learn_timesteps",
    )
    full_train = _parse_positive_int(
        payload.get("full_train_total_timesteps"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "full_train_total_timesteps",
    )
    max_eval_episodes = _parse_positive_int(
        payload.get("max_eval_episodes"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "max_eval_episodes",
    )
    max_eval_steps = _parse_positive_int(
        payload.get("max_eval_steps"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "max_eval_steps",
    )
    probe_raw = payload.get("probe_train_total_timesteps")
    probe_train: int | None = None
    if probe_raw is not None:
        probe_train = _parse_positive_int(
            probe_raw,
            issues,
            SEARCH_STUDY_CONFIG_INVALID,
            "probe_train_total_timesteps",
        )

    if max_trials is None or launcher_smoke is None or full_train is None or max_eval_episodes is None or max_eval_steps is None:
        return None
    if launcher_smoke > full_train:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="launcher_smoke_learn_timesteps must be <= full_train_total_timesteps.",
                context={
                    "launcher_smoke_learn_timesteps": launcher_smoke,
                    "full_train_total_timesteps": full_train,
                },
            )
        )
        return None

    return ResourceBudget(
        max_trials=max_trials,
        launcher_smoke_learn_timesteps=launcher_smoke,
        probe_train_total_timesteps=probe_train,
        full_train_total_timesteps=full_train,
        max_eval_episodes=max_eval_episodes,
        max_eval_steps=max_eval_steps,
    )


def _validate_objective_spec(payload: Any, issues: list[ValidationIssue]) -> ObjectiveSpec | None:
    """Validate objective contract."""

    if not isinstance(payload, Mapping):
        issues.append(
            ValidationIssue(
                code=SEARCH_OBJECTIVE_INVALID,
                message="objective_spec must be an object.",
                context={"payload_type": type(payload).__name__},
            )
        )
        return None

    required_fields = {
        "primary_metric",
        "turnover_penalty_weight",
        "instability_penalty_weight",
        "low_trade_count_penalty_weight",
        "soft_trade_rate_target",
    }
    extra_keys = sorted(set(payload.keys()) - required_fields)
    missing_keys = sorted(required_fields - set(payload.keys()))
    if missing_keys or extra_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_OBJECTIVE_INVALID,
                message="objective_spec fields must match the strict 4.9 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return None

    primary_metric = _safe_string(payload.get("primary_metric"))
    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        issues.append(
            ValidationIssue(
                code=SEARCH_OBJECTIVE_INVALID,
                message="primary_metric is unsupported.",
                context={"primary_metric": primary_metric, "supported": sorted(SUPPORTED_PRIMARY_METRICS)},
            )
        )

    turnover_penalty_weight = _parse_non_negative_float(
        payload.get("turnover_penalty_weight"),
        issues,
        SEARCH_OBJECTIVE_INVALID,
        "turnover_penalty_weight",
    )
    instability_penalty_weight = _parse_non_negative_float(
        payload.get("instability_penalty_weight"),
        issues,
        SEARCH_OBJECTIVE_INVALID,
        "instability_penalty_weight",
    )
    low_trade_count_penalty_weight = _parse_non_negative_float(
        payload.get("low_trade_count_penalty_weight"),
        issues,
        SEARCH_OBJECTIVE_INVALID,
        "low_trade_count_penalty_weight",
    )
    soft_trade_rate_target = _parse_non_negative_float(
        payload.get("soft_trade_rate_target"),
        issues,
        SEARCH_OBJECTIVE_INVALID,
        "soft_trade_rate_target",
    )
    if soft_trade_rate_target is not None and soft_trade_rate_target > 1.0:
        issues.append(
            ValidationIssue(
                code=SEARCH_OBJECTIVE_INVALID,
                message="soft_trade_rate_target must be <= 1.0.",
                context={"soft_trade_rate_target": soft_trade_rate_target},
            )
        )
        return None

    if (
        primary_metric not in SUPPORTED_PRIMARY_METRICS
        or turnover_penalty_weight is None
        or instability_penalty_weight is None
        or low_trade_count_penalty_weight is None
        or soft_trade_rate_target is None
    ):
        return None

    return ObjectiveSpec(
        primary_metric=primary_metric,
        turnover_penalty_weight=turnover_penalty_weight,
        instability_penalty_weight=instability_penalty_weight,
        low_trade_count_penalty_weight=low_trade_count_penalty_weight,
        soft_trade_rate_target=soft_trade_rate_target,
    )


def _validate_guardrail_spec(payload: Any, issues: list[ValidationIssue]) -> GuardrailSpec | None:
    """Validate hard guardrail rules."""

    if not isinstance(payload, Mapping):
        issues.append(
            ValidationIssue(
                code=SEARCH_GUARDRAIL_INVALID,
                message="guardrail_spec must be an object.",
                context={"payload_type": type(payload).__name__},
            )
        )
        return None

    required_fields = {
        "require_step_trace",
        "max_strategy_max_drawdown",
        "min_num_trades_hard",
        "min_num_trades_soft",
        "max_trade_rate_hard",
    }
    extra_keys = sorted(set(payload.keys()) - required_fields)
    missing_keys = sorted(required_fields - set(payload.keys()))
    if missing_keys or extra_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_GUARDRAIL_INVALID,
                message="guardrail_spec fields must match the strict 4.9 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return None

    require_step_trace = payload.get("require_step_trace")
    if not isinstance(require_step_trace, bool):
        issues.append(
            ValidationIssue(
                code=SEARCH_GUARDRAIL_INVALID,
                message="require_step_trace must be a boolean.",
                context={"require_step_trace": require_step_trace},
            )
        )
    max_drawdown = _parse_non_negative_float(
        payload.get("max_strategy_max_drawdown"),
        issues,
        SEARCH_GUARDRAIL_INVALID,
        "max_strategy_max_drawdown",
    )
    min_num_trades_hard = _parse_non_negative_int(
        payload.get("min_num_trades_hard"),
        issues,
        SEARCH_GUARDRAIL_INVALID,
        "min_num_trades_hard",
    )
    min_num_trades_soft = _parse_non_negative_int(
        payload.get("min_num_trades_soft"),
        issues,
        SEARCH_GUARDRAIL_INVALID,
        "min_num_trades_soft",
    )
    max_trade_rate_hard = _parse_non_negative_float(
        payload.get("max_trade_rate_hard"),
        issues,
        SEARCH_GUARDRAIL_INVALID,
        "max_trade_rate_hard",
    )
    if max_drawdown is not None and max_drawdown > 1.0:
        issues.append(
            ValidationIssue(
                code=SEARCH_GUARDRAIL_INVALID,
                message="max_strategy_max_drawdown must be <= 1.0.",
                context={"max_strategy_max_drawdown": max_drawdown},
            )
        )
        return None
    if max_trade_rate_hard is not None and (max_trade_rate_hard <= 0.0 or max_trade_rate_hard > 1.0):
        issues.append(
            ValidationIssue(
                code=SEARCH_GUARDRAIL_INVALID,
                message="max_trade_rate_hard must be within (0, 1].",
                context={"max_trade_rate_hard": max_trade_rate_hard},
            )
        )
        return None
    if (
        not isinstance(require_step_trace, bool)
        or max_drawdown is None
        or min_num_trades_hard is None
        or min_num_trades_soft is None
        or max_trade_rate_hard is None
    ):
        return None
    return GuardrailSpec(
        require_step_trace=require_step_trace,
        max_strategy_max_drawdown=max_drawdown,
        min_num_trades_hard=min_num_trades_hard,
        min_num_trades_soft=min_num_trades_soft,
        max_trade_rate_hard=max_trade_rate_hard,
    )


def _validate_pruning_spec(payload: Any, issues: list[ValidationIssue]) -> PruningSpec | None:
    """Validate conservative pruning configuration."""

    if not isinstance(payload, Mapping):
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="pruning_spec must be an object.",
                context={"payload_type": type(payload).__name__},
            )
        )
        return None

    required_fields = {
        "enabled",
        "warmup_trials",
        "min_completed_probe_trials",
        "min_probe_objective_score",
        "relative_to_best_completed_margin",
    }
    extra_keys = sorted(set(payload.keys()) - required_fields)
    missing_keys = sorted(required_fields - set(payload.keys()))
    if missing_keys or extra_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="pruning_spec fields must match the strict 4.9 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return None

    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="pruning_spec.enabled must be boolean.",
                context={"enabled": enabled},
            )
        )
    warmup_trials = _parse_non_negative_int(
        payload.get("warmup_trials"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "warmup_trials",
    )
    min_completed_probe_trials = _parse_non_negative_int(
        payload.get("min_completed_probe_trials"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "min_completed_probe_trials",
    )
    min_probe_score_raw = payload.get("min_probe_objective_score")
    min_probe_score: float | None = None
    if min_probe_score_raw is not None:
        min_probe_score = _parse_finite_float(
            min_probe_score_raw,
            issues,
            SEARCH_STUDY_CONFIG_INVALID,
            "min_probe_objective_score",
        )
    relative_margin = _parse_non_negative_float(
        payload.get("relative_to_best_completed_margin"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "relative_to_best_completed_margin",
    )
    if (
        not isinstance(enabled, bool)
        or warmup_trials is None
        or min_completed_probe_trials is None
        or relative_margin is None
    ):
        return None
    return PruningSpec(
        enabled=enabled,
        warmup_trials=warmup_trials,
        min_completed_probe_trials=min_completed_probe_trials,
        min_probe_objective_score=min_probe_score,
        relative_to_best_completed_margin=relative_margin,
    )


def _validate_promotion_spec(payload: Any, issues: list[ValidationIssue]) -> PromotionSpec | None:
    """Validate promotion-readiness gates."""

    if not isinstance(payload, Mapping):
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="promotion_spec must be an object.",
                context={"payload_type": type(payload).__name__},
            )
        )
        return None

    required_fields = {
        "candidate_top_k",
        "promotion_min_distinct_seeds",
        "require_positive_objective",
        "max_strategy_max_drawdown",
        "min_num_trades",
    }
    extra_keys = sorted(set(payload.keys()) - required_fields)
    missing_keys = sorted(required_fields - set(payload.keys()))
    if missing_keys or extra_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="promotion_spec fields must match the strict 4.9 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return None

    candidate_top_k = _parse_positive_int(
        payload.get("candidate_top_k"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "candidate_top_k",
    )
    promotion_min_distinct_seeds = _parse_positive_int(
        payload.get("promotion_min_distinct_seeds"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "promotion_min_distinct_seeds",
    )
    require_positive_objective = payload.get("require_positive_objective")
    if not isinstance(require_positive_objective, bool):
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="promotion_spec.require_positive_objective must be boolean.",
                context={"require_positive_objective": require_positive_objective},
            )
        )
    max_drawdown = _parse_non_negative_float(
        payload.get("max_strategy_max_drawdown"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "promotion.max_strategy_max_drawdown",
    )
    min_num_trades = _parse_non_negative_int(
        payload.get("min_num_trades"),
        issues,
        SEARCH_STUDY_CONFIG_INVALID,
        "promotion.min_num_trades",
    )
    if max_drawdown is not None and max_drawdown > 1.0:
        issues.append(
            ValidationIssue(
                code=SEARCH_STUDY_CONFIG_INVALID,
                message="promotion.max_strategy_max_drawdown must be <= 1.0.",
                context={"max_strategy_max_drawdown": max_drawdown},
            )
        )
        return None
    if (
        candidate_top_k is None
        or promotion_min_distinct_seeds is None
        or not isinstance(require_positive_objective, bool)
        or max_drawdown is None
        or min_num_trades is None
    ):
        return None
    return PromotionSpec(
        candidate_top_k=candidate_top_k,
        promotion_min_distinct_seeds=promotion_min_distinct_seeds,
        require_positive_objective=require_positive_objective,
        max_strategy_max_drawdown=max_drawdown,
        min_num_trades=min_num_trades,
    )


def _build_trial_specs(study_spec: StudySpec) -> list[TrialSpec]:
    """Build deterministic trial specs from the explicit search space."""

    search_keys = tuple(sorted(study_spec.search_space.keys()))
    all_assignments = list(
        itertools.product(*(study_spec.search_space[key] for key in search_keys))
    )
    rng = random.Random(study_spec.sampler_seed)
    rng.shuffle(all_assignments)

    trial_specs: list[TrialSpec] = []
    selected_assignments = all_assignments[: study_spec.resource_budget.max_trials]
    for index, values in enumerate(selected_assignments, start=1):
        override_params = {key: values[position] for position, key in enumerate(search_keys)}
        merged_params = dict(study_spec.artifact_training_template["algo_params"])
        merged_params.update(override_params)
        assignment_hash = _hash_canonical_json(merged_params)
        trial_id = f"trial_{index:03d}_{assignment_hash[:8]}"
        output_dir = study_spec.output_root / trial_id
        config_root = output_dir / "configs"
        trial_specs.append(
            TrialSpec(
                trial_id=trial_id,
                trial_index=index,
                study_id=study_spec.study_id,
                seed=study_spec.trial_seed,
                ppo_params=merged_params,
                output_dir=output_dir,
                param_assignment_hash=assignment_hash,
                launcher_config_path=config_root / "launcher_training_config.json",
                probe_training_config_path=config_root / "probe_training_config.json"
                if study_spec.pruning_spec.enabled and study_spec.resource_budget.probe_train_total_timesteps is not None
                else None,
                final_training_config_path=config_root / "final_training_config.json",
                probe_eval_config_path=config_root / "probe_eval_config.json"
                if study_spec.pruning_spec.enabled and study_spec.resource_budget.probe_train_total_timesteps is not None
                else None,
                final_eval_config_path=config_root / "final_eval_config.json",
            )
        )
    return trial_specs


def _execute_trial(
    *,
    study_spec: StudySpec,
    trial_spec: TrialSpec,
    prior_records: Sequence[TrialExecutionRecord],
) -> TrialExecutionRecord:
    """Execute one trial end-to-end under the 4.9 contract."""

    started_at = _generated_at()
    trial_spec.output_dir.mkdir(parents=True, exist_ok=False)
    (trial_spec.output_dir / "configs").mkdir(parents=True, exist_ok=False)
    trial_spec_payload = _build_trial_manifest_payload(study_spec=study_spec, trial_spec=trial_spec)
    derived_configs = _build_derived_trial_configs(study_spec=study_spec, trial_spec=trial_spec)
    for path, payload in derived_configs.items():
        atomic_write_json(payload, path)

    training_report_payload = _empty_trial_training_report(trial_spec=trial_spec, study_spec=study_spec)
    evaluation_report_payload = _empty_trial_evaluation_report(trial_spec=trial_spec, study_spec=study_spec)
    objective_report_payload = _empty_trial_objective_report(trial_spec=trial_spec, study_spec=study_spec)
    guardrail_report_payload = _empty_trial_guardrail_report(trial_spec=trial_spec, study_spec=study_spec)
    status_payload = _empty_trial_status_payload(trial_spec=trial_spec, study_spec=study_spec)

    invalid_hparam_issues = _validate_trial_hparams(trial_spec.ppo_params)
    if invalid_hparam_issues:
        failure_codes = _failure_codes(invalid_hparam_issues)
        invalid_reasons = [item.message for item in invalid_hparam_issues]
        training_report_payload["prelaunch"] = {
            "stage_status": "skipped_invalid",
            "failure_codes": failure_codes,
            "issues": [item.__dict__ for item in invalid_hparam_issues],
        }
        objective_report_payload["final_stage"] = None
        guardrail_report_payload["selected_stage"] = None
        status_payload.update(
            {
                "status": TRIAL_STATUS_INVALID,
                "ranking_eligible": False,
                "failure_codes": failure_codes,
                "invalid_reasons": invalid_reasons,
                "started_at_utc": started_at,
                "completed_at_utc": _generated_at(),
                "artifacts_complete": False,
            }
        )
        record = TrialExecutionRecord(
            trial_spec=trial_spec,
            terminal_state=TRIAL_STATUS_INVALID,
            final_status=TRIAL_STATUS_INVALID,
            ranking_eligible=False,
            guardrail_pass=False,
            objective_score=None,
            probe_objective_score=None,
            primary_metric_value=None,
            failure_codes=failure_codes,
            invalid_reasons=invalid_reasons,
            pruned_reason=None,
            candidate_rank=None,
            promotion_ready=False,
            promotion_family_hash=trial_spec.param_assignment_hash,
            artifacts_complete=False,
            started_at_utc=started_at,
            completed_at_utc=_generated_at(),
            manifest_payload=trial_spec_payload,
            training_report_payload=training_report_payload,
            evaluation_report_payload=evaluation_report_payload,
            objective_report_payload=objective_report_payload,
            guardrail_report_payload=guardrail_report_payload,
            status_payload=status_payload,
        )
        return record

    prelaunch_dir = trial_spec.output_dir / "launcher_prelaunch"
    prelaunch_result = execute_training_launch(
        run_id=study_spec.upstream_refs.run_id,
        env_config_path=study_spec.upstream_refs.env_config_path,
        training_config_path=trial_spec.launcher_config_path,
        state_manifest_path=study_spec.upstream_refs.state_manifest_path,
        env_contract_report_path=study_spec.upstream_refs.env_contract_report_path,
        readiness_report_path=study_spec.upstream_refs.readiness_report_path,
        episode_catalog_path=study_spec.upstream_refs.episode_catalog_path,
        output_dir=prelaunch_dir,
    )
    training_report_payload["prelaunch"] = _summarize_prelaunch_stage(prelaunch_result, prelaunch_dir)
    prelaunch_failure_codes = list(training_report_payload["prelaunch"]["failure_codes"])
    if prelaunch_result.exit_code != 0:
        return _finalize_unsuccessful_trial(
            study_spec=study_spec,
            trial_spec=trial_spec,
            terminal_state=_classify_nonzero_exit(prelaunch_failure_codes, LAUNCHER_RUNTIME_FAILURE_CODES),
            failure_codes=prelaunch_failure_codes,
            invalid_reasons=list(prelaunch_failure_codes),
            pruned_reason=None,
            started_at=started_at,
            manifest_payload=trial_spec_payload,
            training_report_payload=training_report_payload,
            evaluation_report_payload=evaluation_report_payload,
            objective_report_payload=objective_report_payload,
            guardrail_report_payload=guardrail_report_payload,
        )

    probe_score_payload: dict[str, Any] | None = None
    probe_objective_score: float | None = None
    if trial_spec.probe_training_config_path is not None and trial_spec.probe_eval_config_path is not None:
        probe_training_dir = trial_spec.output_dir / "probe_artifact_production"
        probe_eval_dir = trial_spec.output_dir / "probe_evaluation"
        probe_training_result = execute_ppo_artifact_production(
            run_id=study_spec.upstream_refs.run_id,
            env_config_path=study_spec.upstream_refs.env_config_path,
            training_config_path=trial_spec.probe_training_config_path,
            state_manifest_path=study_spec.upstream_refs.state_manifest_path,
            env_contract_report_path=study_spec.upstream_refs.env_contract_report_path,
            readiness_report_path=study_spec.upstream_refs.readiness_report_path,
            episode_catalog_path=study_spec.upstream_refs.episode_catalog_path,
            split_report_path=study_spec.upstream_refs.split_report_path,
            output_dir=probe_training_dir,
        )
        training_report_payload["probe_training"] = _summarize_artifact_stage(probe_training_result, probe_training_dir)
        probe_training_failure_codes = list(training_report_payload["probe_training"]["failure_codes"])
        if probe_training_result.exit_code != 0:
            return _finalize_unsuccessful_trial(
                study_spec=study_spec,
                trial_spec=trial_spec,
                terminal_state=_classify_nonzero_exit(probe_training_failure_codes, ARTIFACT_RUNTIME_FAILURE_CODES),
                failure_codes=probe_training_failure_codes,
                invalid_reasons=list(probe_training_failure_codes),
                pruned_reason=None,
                started_at=started_at,
                manifest_payload=trial_spec_payload,
                training_report_payload=training_report_payload,
                evaluation_report_payload=evaluation_report_payload,
                objective_report_payload=objective_report_payload,
                guardrail_report_payload=guardrail_report_payload,
            )

        probe_eval_result = execute_evaluation_backtest(
            run_id=study_spec.upstream_refs.run_id,
            model_artifact_path=probe_training_result.report_paths.artifact_path,
            env_config_path=study_spec.upstream_refs.env_config_path,
            eval_config_path=trial_spec.probe_eval_config_path,
            state_manifest_path=study_spec.upstream_refs.state_manifest_path,
            env_contract_report_path=study_spec.upstream_refs.env_contract_report_path,
            readiness_report_path=study_spec.upstream_refs.readiness_report_path,
            episode_catalog_path=study_spec.upstream_refs.episode_catalog_path,
            split_report_path=study_spec.upstream_refs.split_report_path,
            output_dir=probe_eval_dir,
        )
        evaluation_report_payload["probe_evaluation"] = _summarize_eval_stage(probe_eval_result, probe_eval_dir)
        probe_eval_failure_codes = list(evaluation_report_payload["probe_evaluation"]["failure_codes"])
        if probe_eval_result.exit_code != 0:
            return _finalize_unsuccessful_trial(
                study_spec=study_spec,
                trial_spec=trial_spec,
                terminal_state=_classify_nonzero_exit(probe_eval_failure_codes, EVAL_RUNTIME_FAILURE_CODES),
                failure_codes=probe_eval_failure_codes,
                invalid_reasons=list(probe_eval_failure_codes),
                pruned_reason=None,
                started_at=started_at,
                manifest_payload=trial_spec_payload,
                training_report_payload=training_report_payload,
                evaluation_report_payload=evaluation_report_payload,
                objective_report_payload=objective_report_payload,
                guardrail_report_payload=guardrail_report_payload,
            )

        probe_score_payload = _score_evaluation_stage(
            stage_name="probe",
            evaluation_dir=probe_eval_dir,
            objective_spec=study_spec.objective_spec,
            guardrail_spec=study_spec.guardrail_spec,
        )
        objective_report_payload["probe_stage"] = probe_score_payload["objective_report"]
        guardrail_report_payload["probe_stage"] = probe_score_payload["guardrail_report"]
        probe_objective_score = probe_score_payload["objective_score"]
        if not probe_score_payload["guardrail_pass"]:
            return _finalize_unsuccessful_trial(
                study_spec=study_spec,
                trial_spec=trial_spec,
                terminal_state=TRIAL_STATUS_INVALID,
                failure_codes=list(probe_score_payload["failure_codes"]),
                invalid_reasons=list(probe_score_payload["invalid_reasons"]),
                pruned_reason=None,
                started_at=started_at,
                manifest_payload=trial_spec_payload,
                training_report_payload=training_report_payload,
                evaluation_report_payload=evaluation_report_payload,
                objective_report_payload=objective_report_payload,
                guardrail_report_payload=guardrail_report_payload,
            )
        pruning_decision = _should_prune_after_probe(
            study_spec=study_spec,
            trial_spec=trial_spec,
            prior_records=prior_records,
            probe_objective_score=probe_objective_score,
        )
        if pruning_decision["pruned"]:
            objective_report_payload["selected_stage"] = "probe"
            objective_report_payload["selected_objective_score"] = probe_objective_score
            guardrail_report_payload["selected_stage"] = "probe"
            status_payload.update(
                {
                    "status": TRIAL_STATUS_PRUNED,
                    "ranking_eligible": False,
                    "failure_codes": [SEARCH_PRUNED_BY_OBJECTIVE],
                    "pruned_reason": pruning_decision["reason"],
                    "started_at_utc": started_at,
                    "completed_at_utc": _generated_at(),
                    "artifacts_complete": _trial_artifacts_complete(
                        training_report_payload=training_report_payload,
                        evaluation_report_payload=evaluation_report_payload,
                        terminal_state=TRIAL_STATUS_PRUNED,
                    ),
                }
            )
            return TrialExecutionRecord(
                trial_spec=trial_spec,
                terminal_state=TRIAL_STATUS_PRUNED,
                final_status=TRIAL_STATUS_PRUNED,
                ranking_eligible=False,
                guardrail_pass=True,
                objective_score=None,
                probe_objective_score=probe_objective_score,
                primary_metric_value=None,
                failure_codes=[SEARCH_PRUNED_BY_OBJECTIVE],
                invalid_reasons=[],
                pruned_reason=pruning_decision["reason"],
                candidate_rank=None,
                promotion_ready=False,
                promotion_family_hash=trial_spec.param_assignment_hash,
                artifacts_complete=_trial_artifacts_complete(
                    training_report_payload=training_report_payload,
                    evaluation_report_payload=evaluation_report_payload,
                    terminal_state=TRIAL_STATUS_PRUNED,
                ),
                started_at_utc=started_at,
                completed_at_utc=_generated_at(),
                manifest_payload=trial_spec_payload,
                training_report_payload=training_report_payload,
                evaluation_report_payload=evaluation_report_payload,
                objective_report_payload=objective_report_payload,
                guardrail_report_payload=guardrail_report_payload,
                status_payload=status_payload,
            )

    final_training_dir = trial_spec.output_dir / "final_artifact_production"
    final_eval_dir = trial_spec.output_dir / "final_evaluation"
    final_training_result = execute_ppo_artifact_production(
        run_id=study_spec.upstream_refs.run_id,
        env_config_path=study_spec.upstream_refs.env_config_path,
        training_config_path=trial_spec.final_training_config_path,
        state_manifest_path=study_spec.upstream_refs.state_manifest_path,
        env_contract_report_path=study_spec.upstream_refs.env_contract_report_path,
        readiness_report_path=study_spec.upstream_refs.readiness_report_path,
        episode_catalog_path=study_spec.upstream_refs.episode_catalog_path,
        split_report_path=study_spec.upstream_refs.split_report_path,
        output_dir=final_training_dir,
    )
    training_report_payload["final_training"] = _summarize_artifact_stage(final_training_result, final_training_dir)
    final_training_failure_codes = list(training_report_payload["final_training"]["failure_codes"])
    if final_training_result.exit_code != 0:
        return _finalize_unsuccessful_trial(
            study_spec=study_spec,
            trial_spec=trial_spec,
            terminal_state=_classify_nonzero_exit(final_training_failure_codes, ARTIFACT_RUNTIME_FAILURE_CODES),
            failure_codes=final_training_failure_codes,
            invalid_reasons=list(final_training_failure_codes),
            pruned_reason=None,
            started_at=started_at,
            manifest_payload=trial_spec_payload,
            training_report_payload=training_report_payload,
            evaluation_report_payload=evaluation_report_payload,
            objective_report_payload=objective_report_payload,
            guardrail_report_payload=guardrail_report_payload,
        )

    final_eval_result = execute_evaluation_backtest(
        run_id=study_spec.upstream_refs.run_id,
        model_artifact_path=final_training_result.report_paths.artifact_path,
        env_config_path=study_spec.upstream_refs.env_config_path,
        eval_config_path=trial_spec.final_eval_config_path,
        state_manifest_path=study_spec.upstream_refs.state_manifest_path,
        env_contract_report_path=study_spec.upstream_refs.env_contract_report_path,
        readiness_report_path=study_spec.upstream_refs.readiness_report_path,
        episode_catalog_path=study_spec.upstream_refs.episode_catalog_path,
        split_report_path=study_spec.upstream_refs.split_report_path,
        output_dir=final_eval_dir,
    )
    evaluation_report_payload["final_evaluation"] = _summarize_eval_stage(final_eval_result, final_eval_dir)
    final_eval_failure_codes = list(evaluation_report_payload["final_evaluation"]["failure_codes"])
    if final_eval_result.exit_code != 0:
        return _finalize_unsuccessful_trial(
            study_spec=study_spec,
            trial_spec=trial_spec,
            terminal_state=_classify_nonzero_exit(final_eval_failure_codes, EVAL_RUNTIME_FAILURE_CODES),
            failure_codes=final_eval_failure_codes,
            invalid_reasons=list(final_eval_failure_codes),
            pruned_reason=None,
            started_at=started_at,
            manifest_payload=trial_spec_payload,
            training_report_payload=training_report_payload,
            evaluation_report_payload=evaluation_report_payload,
            objective_report_payload=objective_report_payload,
            guardrail_report_payload=guardrail_report_payload,
        )

    final_score_payload = _score_evaluation_stage(
        stage_name="final",
        evaluation_dir=final_eval_dir,
        objective_spec=study_spec.objective_spec,
        guardrail_spec=study_spec.guardrail_spec,
    )
    objective_report_payload["final_stage"] = final_score_payload["objective_report"]
    objective_report_payload["selected_stage"] = "final"
    objective_report_payload["selected_objective_score"] = final_score_payload["objective_score"]
    guardrail_report_payload["final_stage"] = final_score_payload["guardrail_report"]
    guardrail_report_payload["selected_stage"] = "final"

    if not final_score_payload["guardrail_pass"]:
        return _finalize_unsuccessful_trial(
            study_spec=study_spec,
            trial_spec=trial_spec,
            terminal_state=TRIAL_STATUS_INVALID,
            failure_codes=list(final_score_payload["failure_codes"]),
            invalid_reasons=list(final_score_payload["invalid_reasons"]),
            pruned_reason=None,
            started_at=started_at,
            manifest_payload=trial_spec_payload,
            training_report_payload=training_report_payload,
            evaluation_report_payload=evaluation_report_payload,
            objective_report_payload=objective_report_payload,
            guardrail_report_payload=guardrail_report_payload,
        )

    status_payload.update(
        {
            "status": TRIAL_STATUS_COMPLETED_NONCOMPETITIVE,
            "ranking_eligible": True,
            "objective_score": final_score_payload["objective_score"],
            "primary_metric_value": final_score_payload["primary_metric_value"],
            "failure_codes": [],
            "invalid_reasons": [],
            "started_at_utc": started_at,
            "completed_at_utc": _generated_at(),
            "artifacts_complete": _trial_artifacts_complete(
                training_report_payload=training_report_payload,
                evaluation_report_payload=evaluation_report_payload,
                terminal_state="completed",
            ),
        }
    )
    return TrialExecutionRecord(
        trial_spec=trial_spec,
        terminal_state="completed",
        final_status=TRIAL_STATUS_COMPLETED_NONCOMPETITIVE,
        ranking_eligible=True,
        guardrail_pass=True,
        objective_score=final_score_payload["objective_score"],
        probe_objective_score=probe_objective_score,
        primary_metric_value=final_score_payload["primary_metric_value"],
        failure_codes=[],
        invalid_reasons=[],
        pruned_reason=None,
        candidate_rank=None,
        promotion_ready=False,
        promotion_family_hash=trial_spec.param_assignment_hash,
        artifacts_complete=_trial_artifacts_complete(
            training_report_payload=training_report_payload,
            evaluation_report_payload=evaluation_report_payload,
            terminal_state="completed",
        ),
        started_at_utc=started_at,
        completed_at_utc=_generated_at(),
        manifest_payload=trial_spec_payload,
        training_report_payload=training_report_payload,
        evaluation_report_payload=evaluation_report_payload,
        objective_report_payload=objective_report_payload,
        guardrail_report_payload=guardrail_report_payload,
        status_payload=status_payload,
    )


def _build_derived_trial_configs(*, study_spec: StudySpec, trial_spec: TrialSpec) -> dict[Path, dict[str, Any]]:
    """Build and return all explicit per-trial config payloads."""

    base_training = dict(study_spec.artifact_training_template)
    base_training["seed"] = trial_spec.seed
    base_training["algo_params"] = dict(trial_spec.ppo_params)

    launcher_config = {
        "algorithm": "ppo",
        "seed": trial_spec.seed,
        "total_timesteps": study_spec.resource_budget.full_train_total_timesteps,
        "device": base_training["device"],
        "episode_selection_mode": base_training["episode_selection_mode"],
        "startup_policy": "fresh_only",
        "smoke_mode": "prelaunch_only",
        "smoke_learn_timesteps": study_spec.resource_budget.launcher_smoke_learn_timesteps,
        "algo_params": dict(trial_spec.ppo_params),
    }
    final_training_config = dict(base_training)
    final_training_config["total_timesteps"] = study_spec.resource_budget.full_train_total_timesteps

    final_eval_config = dict(study_spec.eval_template)
    final_eval_config["seed"] = trial_spec.seed
    final_eval_config["evaluation_mode"] = "episodic_eval_backtest"
    final_eval_config["target_mode"] = "explicit_partition"
    final_eval_config["target_partition"] = "validation"
    final_eval_config["target_fold_id"] = None
    final_eval_config["target_episode_refs"] = None
    final_eval_config["deterministic"] = True
    final_eval_config["startup_policy"] = "fresh_only"
    final_eval_config["max_eval_episodes"] = study_spec.resource_budget.max_eval_episodes
    final_eval_config["max_eval_steps"] = study_spec.resource_budget.max_eval_steps
    final_eval_config["write_step_trace"] = bool(study_spec.guardrail_spec.require_step_trace)

    configs: dict[Path, dict[str, Any]] = {
        trial_spec.launcher_config_path: launcher_config,
        trial_spec.final_training_config_path: final_training_config,
        trial_spec.final_eval_config_path: final_eval_config,
    }
    if trial_spec.probe_training_config_path is not None:
        probe_training_config = dict(base_training)
        probe_training_config["total_timesteps"] = study_spec.resource_budget.probe_train_total_timesteps
        configs[trial_spec.probe_training_config_path] = probe_training_config
    if trial_spec.probe_eval_config_path is not None:
        probe_eval_config = dict(final_eval_config)
        configs[trial_spec.probe_eval_config_path] = probe_eval_config
    return configs


def _validate_trial_hparams(ppo_params: Mapping[str, Any]) -> list[ValidationIssue]:
    """Validate derived first-wave PPO assignment constraints."""

    issues: list[ValidationIssue] = []
    required_keys = {
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
    }
    extra_keys = sorted(set(ppo_params.keys()) - required_keys)
    missing_keys = sorted(required_keys - set(ppo_params.keys()))
    if extra_keys or missing_keys:
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="Resolved trial PPO params must match the canonical PPO artifact schema exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return issues

    n_steps = ppo_params.get("n_steps")
    batch_size = ppo_params.get("batch_size")
    n_epochs = ppo_params.get("n_epochs")
    learning_rate = ppo_params.get("learning_rate")
    gamma = ppo_params.get("gamma")
    gae_lambda = ppo_params.get("gae_lambda")
    clip_range = ppo_params.get("clip_range")
    ent_coef = ppo_params.get("ent_coef")
    vf_coef = ppo_params.get("vf_coef")
    max_grad_norm = ppo_params.get("max_grad_norm")
    integer_fields = {"n_steps": n_steps, "batch_size": batch_size, "n_epochs": n_epochs}
    for field_name, value in integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            issues.append(
                ValidationIssue(
                    code=SEARCH_SEARCH_SPACE_INVALID,
                    message="Resolved integer PPO params must be positive integers.",
                    context={"field": field_name, "value": value},
                )
            )
    float_fields = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
    }
    for field_name, value in float_fields.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            issues.append(
                ValidationIssue(
                    code=SEARCH_SEARCH_SPACE_INVALID,
                    message="Resolved floating PPO params must be finite numbers.",
                    context={"field": field_name, "value": value},
                )
            )

    if issues:
        return issues

    n_steps_int = int(n_steps)
    batch_size_int = int(batch_size)
    if batch_size_int > n_steps_int:
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="batch_size must be <= n_steps for first-wave single-env PPO search.",
                context={"n_steps": n_steps_int, "batch_size": batch_size_int},
            )
        )
    if n_steps_int % batch_size_int != 0:
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="n_steps must be divisible by batch_size in first-wave 4.9.",
                context={"n_steps": n_steps_int, "batch_size": batch_size_int},
            )
        )
    if float(learning_rate) <= 0.0 or float(max_grad_norm) <= 0.0:
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="learning_rate and max_grad_norm must be > 0.",
                context={"learning_rate": learning_rate, "max_grad_norm": max_grad_norm},
            )
        )
    if not (0.0 < float(gamma) <= 1.0):
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="gamma must be within (0, 1].",
                context={"gamma": gamma},
            )
        )
    if not (0.0 < float(gae_lambda) <= 1.0):
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="gae_lambda must be within (0, 1].",
                context={"gae_lambda": gae_lambda},
            )
        )
    if float(clip_range) < 0.0 or float(ent_coef) < 0.0 or float(vf_coef) < 0.0:
        issues.append(
            ValidationIssue(
                code=SEARCH_SEARCH_SPACE_INVALID,
                message="clip_range, ent_coef, and vf_coef must be >= 0.",
                context={"clip_range": clip_range, "ent_coef": ent_coef, "vf_coef": vf_coef},
            )
        )
    return issues


def _score_evaluation_stage(
    *,
    stage_name: str,
    evaluation_dir: Path,
    objective_spec: ObjectiveSpec,
    guardrail_spec: GuardrailSpec,
) -> dict[str, Any]:
    """Compute objective and guardrail reports for one evaluation stage."""

    required_files = {
        "evaluation_validation_report": evaluation_dir / "evaluation_validation_report.json",
        "evaluation_manifest": evaluation_dir / "evaluation_manifest.json",
        "evaluation_backtest_report": evaluation_dir / "evaluation_backtest_report.json",
    }
    issues: list[ValidationIssue] = []
    loaded_payloads: dict[str, dict[str, Any]] = {}
    for label, path in required_files.items():
        payload, load_issues = _load_json_file(path, label=label)
        issues.extend(load_issues)
        if payload is not None:
            loaded_payloads[label] = payload
    if guardrail_spec.require_step_trace:
        trace_path = evaluation_dir / "evaluation_step_trace.parquet"
        if not trace_path.exists():
            issues.append(
                ValidationIssue(
                    code=SEARCH_ARTIFACT_MISSING,
                    message="Required evaluation step trace is missing.",
                    context={"path": str(trace_path)},
                )
            )
            trace_frame = None
        else:
            try:
                trace_frame = _read_step_trace(trace_path)
            except Exception as exc:  # noqa: BLE001
                trace_frame = None
                issues.append(
                    ValidationIssue(
                        code=SEARCH_REPORT_INVALID,
                        message="Failed to read evaluation step trace parquet.",
                        context={"path": str(trace_path), "error": str(exc)},
                    )
                )
    else:
        trace_frame = None

    if issues:
        failure_codes = _failure_codes(issues)
        return {
            "guardrail_pass": False,
            "objective_score": None,
            "primary_metric_value": None,
            "failure_codes": failure_codes,
            "invalid_reasons": [item.message for item in issues],
            "objective_report": {
                "stage_name": stage_name,
                "objective_score": None,
                "primary_metric_name": objective_spec.primary_metric,
                "primary_metric_value": None,
                "penalties": [],
                "failure_codes": failure_codes,
                "ranking_eligible": False,
            },
            "guardrail_report": {
                "stage_name": stage_name,
                "guardrail_pass": False,
                "hard_constraints": [],
                "soft_penalties": [],
                "failure_codes": failure_codes,
                "invalid_reasons": [item.message for item in issues],
            },
        }

    validation_payload = loaded_payloads["evaluation_validation_report"]
    manifest_payload = loaded_payloads["evaluation_manifest"]
    backtest_payload = loaded_payloads["evaluation_backtest_report"]
    hard_constraints: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    failure_codes: list[str] = []

    if not bool(validation_payload.get("overall_pass")):
        hard_constraints.append(_hard_constraint("validation_overall_pass", False, SEARCH_REPORT_INVALID, validation_payload))
        invalid_reasons.append("evaluation_validation_report.overall_pass must be true.")
        failure_codes.append(SEARCH_REPORT_INVALID)
    else:
        hard_constraints.append(_hard_constraint("validation_overall_pass", True, None, validation_payload))

    if not bool(backtest_payload.get("evaluation_success")):
        hard_constraints.append(_hard_constraint("evaluation_success", False, SEARCH_REPORT_INVALID, backtest_payload))
        invalid_reasons.append("evaluation_backtest_report.evaluation_success must be true.")
        failure_codes.append(SEARCH_REPORT_INVALID)
    else:
        hard_constraints.append(_hard_constraint("evaluation_success", True, None, backtest_payload))

    strategy_metrics = backtest_payload.get("strategy_metrics")
    relative_metrics = backtest_payload.get("relative_metrics")
    metric_status = backtest_payload.get("metric_status")
    if not isinstance(strategy_metrics, Mapping) or not isinstance(metric_status, Mapping):
        hard_constraints.append(_hard_constraint("metrics_surface_present", False, SEARCH_REPORT_INVALID, backtest_payload))
        invalid_reasons.append("evaluation report surface is incomplete.")
        failure_codes.append(SEARCH_REPORT_INVALID)
        primary_metric_value = None
    else:
        hard_constraints.append(_hard_constraint("metrics_surface_present", True, None, {}))
        primary_metric_value = _resolve_primary_metric_value(
            primary_metric=objective_spec.primary_metric,
            strategy_metrics=strategy_metrics,
            relative_metrics=relative_metrics if isinstance(relative_metrics, Mapping) else {},
        )

    if primary_metric_value is None or not math.isfinite(primary_metric_value):
        hard_constraints.append(
            _hard_constraint(
                "primary_metric_finite",
                False,
                SEARCH_NUMERIC_PATHOLOGY,
                {"primary_metric": objective_spec.primary_metric, "primary_metric_value": primary_metric_value},
            )
        )
        invalid_reasons.append("Primary objective metric is missing or non-finite.")
        failure_codes.append(SEARCH_NUMERIC_PATHOLOGY)
    else:
        hard_constraints.append(
            _hard_constraint(
                "primary_metric_finite",
                True,
                None,
                {"primary_metric": objective_spec.primary_metric, "primary_metric_value": primary_metric_value},
            )
        )

    max_drawdown = _mapping_float(strategy_metrics, "max_drawdown") if isinstance(strategy_metrics, Mapping) else None
    if max_drawdown is None or not math.isfinite(max_drawdown):
        hard_constraints.append(
            _hard_constraint("strategy_max_drawdown_finite", False, SEARCH_NUMERIC_PATHOLOGY, {"max_drawdown": max_drawdown})
        )
        invalid_reasons.append("strategy max_drawdown is missing or non-finite.")
        failure_codes.append(SEARCH_NUMERIC_PATHOLOGY)
    elif max_drawdown > guardrail_spec.max_strategy_max_drawdown:
        hard_constraints.append(
            _hard_constraint(
                "strategy_max_drawdown_guard",
                False,
                SEARCH_CATASTROPHIC_RISK_BREACH,
                {
                    "max_drawdown": max_drawdown,
                    "max_strategy_max_drawdown": guardrail_spec.max_strategy_max_drawdown,
                },
            )
        )
        invalid_reasons.append("strategy max_drawdown breached the hard guardrail.")
        failure_codes.append(SEARCH_CATASTROPHIC_RISK_BREACH)
    else:
        hard_constraints.append(
            _hard_constraint(
                "strategy_max_drawdown_guard",
                True,
                None,
                {
                    "max_drawdown": max_drawdown,
                    "max_strategy_max_drawdown": guardrail_spec.max_strategy_max_drawdown,
                },
            )
        )

    num_trades = _mapping_int(strategy_metrics, "num_trades") if isinstance(strategy_metrics, Mapping) else None
    if num_trades is None:
        hard_constraints.append(
            _hard_constraint("num_trades_present", False, SEARCH_REPORT_INVALID, {"num_trades": num_trades})
        )
        invalid_reasons.append("strategy num_trades is missing.")
        failure_codes.append(SEARCH_REPORT_INVALID)
    elif num_trades < guardrail_spec.min_num_trades_hard:
        hard_constraints.append(
            _hard_constraint(
                "min_num_trades_hard",
                False,
                SEARCH_PATHOLOGICAL_INACTIVITY,
                {"num_trades": num_trades, "min_num_trades_hard": guardrail_spec.min_num_trades_hard},
            )
        )
        invalid_reasons.append("Hard inactivity guardrail was breached.")
        failure_codes.append(SEARCH_PATHOLOGICAL_INACTIVITY)
    else:
        hard_constraints.append(
            _hard_constraint(
                "min_num_trades_hard",
                True,
                None,
                {"num_trades": num_trades, "min_num_trades_hard": guardrail_spec.min_num_trades_hard},
            )
        )

    diagnostics = _compute_trace_diagnostics(trace_frame)
    trade_rate = diagnostics.get("trade_rate")
    if trade_rate is None or not math.isfinite(trade_rate):
        hard_constraints.append(
            _hard_constraint("trade_rate_finite", False, SEARCH_NUMERIC_PATHOLOGY, {"trade_rate": trade_rate})
        )
        invalid_reasons.append("trade_rate diagnostic is missing or non-finite.")
        failure_codes.append(SEARCH_NUMERIC_PATHOLOGY)
    elif trade_rate > guardrail_spec.max_trade_rate_hard:
        hard_constraints.append(
            _hard_constraint(
                "max_trade_rate_hard",
                False,
                SEARCH_PATHOLOGICAL_TRADE_BEHAVIOR,
                {"trade_rate": trade_rate, "max_trade_rate_hard": guardrail_spec.max_trade_rate_hard},
            )
        )
        invalid_reasons.append("Trade churn exceeded the hard guardrail.")
        failure_codes.append(SEARCH_PATHOLOGICAL_TRADE_BEHAVIOR)
    else:
        hard_constraints.append(
            _hard_constraint(
                "max_trade_rate_hard",
                True,
                None,
                {"trade_rate": trade_rate, "max_trade_rate_hard": guardrail_spec.max_trade_rate_hard},
            )
        )

    guardrail_pass = len(failure_codes) == 0
    penalties = []
    objective_score: float | None = None
    if guardrail_pass and primary_metric_value is not None:
        turnover_penalty = objective_spec.turnover_penalty_weight * max(0.0, float(trade_rate) - objective_spec.soft_trade_rate_target)
        penalties.append(
            {
                "penalty_name": "turnover_penalty",
                "applied": turnover_penalty > 0.0,
                "penalty_value": turnover_penalty,
                "detail": {
                    "trade_rate": trade_rate,
                    "soft_trade_rate_target": objective_spec.soft_trade_rate_target,
                    "weight": objective_spec.turnover_penalty_weight,
                },
            }
        )

        instability_value = float(diagnostics.get("episode_return_std", 0.0))
        instability_penalty = objective_spec.instability_penalty_weight * max(0.0, instability_value)
        penalties.append(
            {
                "penalty_name": "instability_penalty",
                "applied": instability_penalty > 0.0,
                "penalty_value": instability_penalty,
                "detail": {
                    "episode_return_std": instability_value,
                    "episode_count": diagnostics.get("episode_count"),
                    "weight": objective_spec.instability_penalty_weight,
                },
            }
        )

        trade_gap = max(0, guardrail_spec.min_num_trades_soft - int(num_trades))
        trade_gap_ratio = float(trade_gap) / float(max(1, guardrail_spec.min_num_trades_soft))
        low_trade_penalty = objective_spec.low_trade_count_penalty_weight * trade_gap_ratio
        penalties.append(
            {
                "penalty_name": "low_trade_count_penalty",
                "applied": low_trade_penalty > 0.0,
                "penalty_value": low_trade_penalty,
                "detail": {
                    "num_trades": num_trades,
                    "min_num_trades_soft": guardrail_spec.min_num_trades_soft,
                    "weight": objective_spec.low_trade_count_penalty_weight,
                },
            }
        )

        objective_score = float(primary_metric_value) - sum(float(item["penalty_value"]) for item in penalties)

    return {
        "guardrail_pass": guardrail_pass,
        "objective_score": objective_score,
        "primary_metric_value": primary_metric_value,
        "failure_codes": _dedupe_sequence(failure_codes),
        "invalid_reasons": invalid_reasons,
        "objective_report": {
            "stage_name": stage_name,
            "objective_score": objective_score,
            "primary_metric_name": objective_spec.primary_metric,
            "primary_metric_value": primary_metric_value,
            "penalties": penalties,
            "diagnostics": diagnostics,
            "ranking_eligible": guardrail_pass,
            "failure_codes": _dedupe_sequence(failure_codes),
        },
        "guardrail_report": {
            "stage_name": stage_name,
            "guardrail_pass": guardrail_pass,
            "hard_constraints": hard_constraints,
            "soft_penalties": penalties,
            "diagnostics": diagnostics,
            "failure_codes": _dedupe_sequence(failure_codes),
            "invalid_reasons": invalid_reasons,
            "evaluation_manifest_path": str(manifest_payload.get("output_dir", "")),
        },
    }


def _compute_trace_diagnostics(trace_frame: pd.DataFrame | None) -> dict[str, Any]:
    """Compute small, audit-friendly diagnostics from the step trace."""

    if trace_frame is None or trace_frame.empty:
        return {
            "step_count": 0,
            "trade_event_count": 0,
            "trade_rate": 0.0,
            "episode_count": 0,
            "episode_return_std": 0.0,
        }
    required_columns = {
        "evaluation_episode_index",
        "trade_units",
        "strategy_portfolio_value",
    }
    missing = sorted(required_columns - set(trace_frame.columns))
    if missing:
        return {
            "step_count": int(len(trace_frame)),
            "trade_event_count": None,
            "trade_rate": math.nan,
            "episode_count": None,
            "episode_return_std": math.nan,
            "missing_columns": missing,
        }
    trade_event_count = int((trace_frame["trade_units"].fillna(0).astype(float) > 0.0).sum())
    step_count = int(len(trace_frame))
    trade_rate = float(trade_event_count) / float(step_count) if step_count > 0 else 0.0

    episode_returns: list[float] = []
    for _, group in trace_frame.sort_values(["evaluation_episode_index", "step_ordinal"]).groupby("evaluation_episode_index"):
        values = group["strategy_portfolio_value"].astype(float)
        if values.empty:
            continue
        start_value = float(values.iloc[0])
        end_value = float(values.iloc[-1])
        if start_value > 0.0 and math.isfinite(start_value) and math.isfinite(end_value):
            episode_returns.append((end_value / start_value) - 1.0)

    episode_return_std = float(pd.Series(episode_returns, dtype="float64").std(ddof=0)) if episode_returns else 0.0
    return {
        "step_count": step_count,
        "trade_event_count": trade_event_count,
        "trade_rate": trade_rate,
        "episode_count": len(episode_returns),
        "episode_return_std": episode_return_std,
    }


def _should_prune_after_probe(
    *,
    study_spec: StudySpec,
    trial_spec: TrialSpec,
    prior_records: Sequence[TrialExecutionRecord],
    probe_objective_score: float | None,
) -> dict[str, Any]:
    """Decide whether to prune after a conservative probe stage."""

    if not study_spec.pruning_spec.enabled or probe_objective_score is None:
        return {"pruned": False, "reason": None}
    if trial_spec.trial_index <= study_spec.pruning_spec.warmup_trials:
        return {"pruned": False, "reason": None}

    prior_probe_scores = [
        record.probe_objective_score
        for record in prior_records
        if record.probe_objective_score is not None and record.terminal_state in {"completed", TRIAL_STATUS_PRUNED}
    ]
    if len(prior_probe_scores) < study_spec.pruning_spec.min_completed_probe_trials:
        return {"pruned": False, "reason": None}

    floor_score = study_spec.pruning_spec.min_probe_objective_score
    if floor_score is not None and probe_objective_score < floor_score:
        return {
            "pruned": True,
            "reason": f"probe_objective_score {probe_objective_score:.6f} < min_probe_objective_score {floor_score:.6f}",
        }

    best_probe = max(float(score) for score in prior_probe_scores)
    if probe_objective_score + study_spec.pruning_spec.relative_to_best_completed_margin < best_probe:
        return {
            "pruned": True,
            "reason": (
                "probe_objective_score fell below the conservative best-probe margin gate "
                f"({probe_objective_score:.6f} + margin {study_spec.pruning_spec.relative_to_best_completed_margin:.6f} < {best_probe:.6f})"
            ),
        }
    return {"pruned": False, "reason": None}


def _finalize_unsuccessful_trial(
    *,
    study_spec: StudySpec,
    trial_spec: TrialSpec,
    terminal_state: str,
    failure_codes: list[str],
    invalid_reasons: list[str],
    pruned_reason: str | None,
    started_at: str,
    manifest_payload: dict[str, Any],
    training_report_payload: dict[str, Any],
    evaluation_report_payload: dict[str, Any],
    objective_report_payload: dict[str, Any],
    guardrail_report_payload: dict[str, Any],
) -> TrialExecutionRecord:
    """Return a finalized invalid or failed trial record."""

    status_payload = _empty_trial_status_payload(trial_spec=trial_spec, study_spec=study_spec)
    status_payload.update(
        {
            "status": terminal_state,
            "ranking_eligible": False,
            "failure_codes": _dedupe_sequence(failure_codes),
            "invalid_reasons": invalid_reasons,
            "pruned_reason": pruned_reason,
            "started_at_utc": started_at,
            "completed_at_utc": _generated_at(),
            "artifacts_complete": _trial_artifacts_complete(
                training_report_payload=training_report_payload,
                evaluation_report_payload=evaluation_report_payload,
                terminal_state=terminal_state,
            ),
        }
    )
    return TrialExecutionRecord(
        trial_spec=trial_spec,
        terminal_state=terminal_state,
        final_status=terminal_state,
        ranking_eligible=False,
        guardrail_pass=False,
        objective_score=None,
        probe_objective_score=None,
        primary_metric_value=None,
        failure_codes=_dedupe_sequence(failure_codes),
        invalid_reasons=invalid_reasons,
        pruned_reason=pruned_reason,
        candidate_rank=None,
        promotion_ready=False,
        promotion_family_hash=trial_spec.param_assignment_hash,
        artifacts_complete=_trial_artifacts_complete(
            training_report_payload=training_report_payload,
            evaluation_report_payload=evaluation_report_payload,
            terminal_state=terminal_state,
        ),
        started_at_utc=started_at,
        completed_at_utc=_generated_at(),
        manifest_payload=manifest_payload,
        training_report_payload=training_report_payload,
        evaluation_report_payload=evaluation_report_payload,
        objective_report_payload=objective_report_payload,
        guardrail_report_payload=guardrail_report_payload,
        status_payload=status_payload,
    )


def _summarize_prelaunch_stage(result: Any, output_dir: Path) -> dict[str, Any]:
    """Summarize the 4.7 prelaunch stage."""

    payload = {
        "output_dir": str(output_dir),
        "exit_code": int(result.exit_code),
        "reports_written": bool(result.reports_written),
        "validation_report_path": str(result.report_paths.validation_report_path),
        "manifest_path": str(result.report_paths.manifest_path),
        "smoke_report_path": str(result.report_paths.smoke_report_path),
        "overall_pass": bool(result.validation_payload.get("overall_pass")),
        "failure_codes": list(result.validation_payload.get("failure_codes", [])),
        "stage_status": "completed" if int(result.exit_code) == 0 else "failed",
    }
    return payload


def _summarize_artifact_stage(result: Any, output_dir: Path) -> dict[str, Any]:
    """Summarize the artifact production stage."""

    report_payload = result.report_payload or {}
    manifest_payload = result.manifest_payload or {}
    return {
        "output_dir": str(output_dir),
        "exit_code": int(result.exit_code),
        "reports_written": bool(result.reports_written),
        "artifact_path": str(result.report_paths.artifact_path),
        "manifest_path": str(result.report_paths.manifest_path),
        "report_path": str(result.report_paths.report_path),
        "canonical_artifact_ready": bool(report_payload.get("canonical_artifact_ready")),
        "artifact_exists": bool(report_payload.get("artifact_exists")),
        "load_back_succeeded": bool(report_payload.get("load_back_succeeded")),
        "failure_codes": list(report_payload.get("failure_codes", [])),
        "selected_episode_ref": manifest_payload.get("lineages", {}).get("selected_episode_ref"),
        "stage_status": "completed" if int(result.exit_code) == 0 else "failed",
    }


def _summarize_eval_stage(result: Any, output_dir: Path) -> dict[str, Any]:
    """Summarize the evaluation stage."""

    backtest_payload = result.backtest_payload or {}
    return {
        "output_dir": str(output_dir),
        "exit_code": int(result.exit_code),
        "reports_written": bool(result.reports_written),
        "validation_report_path": str(result.report_paths.validation_report_path),
        "manifest_path": str(result.report_paths.manifest_path),
        "backtest_report_path": str(result.report_paths.backtest_report_path),
        "step_trace_path": str(result.report_paths.step_trace_path),
        "evaluation_success": bool(backtest_payload.get("evaluation_success")),
        "failure_codes": list(backtest_payload.get("failure_codes", [])),
        "stage_status": "completed" if int(result.exit_code) == 0 else "failed",
    }


def _finalize_completed_trial_statuses(*, study_spec: StudySpec, trial_records: Sequence[TrialExecutionRecord]) -> None:
    """Finalize candidate and promotion-ready states after the study completes."""

    completed_records = [
        record
        for record in trial_records
        if record.terminal_state == "completed" and record.objective_score is not None and record.artifacts_complete
    ]
    completed_records.sort(
        key=lambda item: (
            float(item.objective_score),
            -int(item.trial_spec.trial_index),
        ),
        reverse=True,
    )
    candidate_records = completed_records[: study_spec.promotion_spec.candidate_top_k]
    family_seed_counts: dict[str, set[int]] = {}
    for record in completed_records:
        family_seed_counts.setdefault(record.promotion_family_hash, set()).add(int(record.trial_spec.seed))

    for rank, record in enumerate(candidate_records, start=1):
        record.candidate_rank = rank
        record.final_status = TRIAL_STATUS_COMPLETED_CANDIDATE
        distinct_seeds = len(family_seed_counts.get(record.promotion_family_hash, set()))
        max_drawdown = _objective_drawdown(record)
        num_trades = _objective_num_trades(record)
        promotion_ready = (
            distinct_seeds >= study_spec.promotion_spec.promotion_min_distinct_seeds
            and (not study_spec.promotion_spec.require_positive_objective or float(record.objective_score or 0.0) > 0.0)
            and max_drawdown is not None
            and max_drawdown <= study_spec.promotion_spec.max_strategy_max_drawdown
            and num_trades is not None
            and num_trades >= study_spec.promotion_spec.min_num_trades
        )
        if promotion_ready:
            record.final_status = TRIAL_STATUS_PROMOTION_READY_CANDIDATE
            record.promotion_ready = True

    for record in trial_records:
        if record.terminal_state != "completed":
            continue
        if record.final_status not in {TRIAL_STATUS_COMPLETED_CANDIDATE, TRIAL_STATUS_PROMOTION_READY_CANDIDATE}:
            record.final_status = TRIAL_STATUS_COMPLETED_NONCOMPETITIVE
            record.promotion_ready = False
        record.status_payload["status"] = record.final_status
        record.status_payload["candidate_rank"] = record.candidate_rank
        record.status_payload["promotion_ready"] = record.promotion_ready
        record.status_payload["objective_score"] = record.objective_score
        record.status_payload["promotion_family_hash"] = record.promotion_family_hash


def _write_trial_reports(record: TrialExecutionRecord) -> None:
    """Write all top-level 4.9 trial reports."""

    trial_root = record.trial_spec.output_dir
    atomic_write_json(record.manifest_payload, trial_root / TRIAL_MANIFEST_FILENAME)
    atomic_write_json(record.training_report_payload, trial_root / TRIAL_TRAINING_REPORT_FILENAME)
    atomic_write_json(record.evaluation_report_payload, trial_root / TRIAL_EVALUATION_REPORT_FILENAME)
    atomic_write_json(record.objective_report_payload, trial_root / TRIAL_OBJECTIVE_REPORT_FILENAME)
    atomic_write_json(record.guardrail_report_payload, trial_root / TRIAL_GUARDRAIL_REPORT_FILENAME)
    _write_trial_status(record)


def _write_trial_status(record: TrialExecutionRecord) -> None:
    """Write the status file for one trial."""

    atomic_write_json(record.status_payload, record.trial_spec.output_dir / TRIAL_STATUS_FILENAME)


def _build_study_manifest_payload(*, study_spec: StudySpec, trial_specs: Sequence[TrialSpec]) -> dict[str, Any]:
    """Build the top-level study manifest."""

    return {
        "task_name": TASK_NAME,
        "contract_version": CONTRACT_VERSION,
        "study_id": study_spec.study_id,
        "milestone": study_spec.milestone,
        "study_mode": study_spec.study_mode,
        "search_method": study_spec.search_method,
        "sampler_seed": study_spec.sampler_seed,
        "trial_seed": study_spec.trial_seed,
        "study_config_path": str(study_spec.study_config_path),
        "study_config_hash": study_spec.study_config_hash,
        "search_space": {key: list(value) for key, value in study_spec.search_space.items()},
        "resource_budget": {
            "max_trials": study_spec.resource_budget.max_trials,
            "launcher_smoke_learn_timesteps": study_spec.resource_budget.launcher_smoke_learn_timesteps,
            "probe_train_total_timesteps": study_spec.resource_budget.probe_train_total_timesteps,
            "full_train_total_timesteps": study_spec.resource_budget.full_train_total_timesteps,
            "max_eval_episodes": study_spec.resource_budget.max_eval_episodes,
            "max_eval_steps": study_spec.resource_budget.max_eval_steps,
        },
        "objective_spec": study_spec.objective_spec.__dict__,
        "guardrail_spec": study_spec.guardrail_spec.__dict__,
        "pruning_spec": study_spec.pruning_spec.__dict__,
        "promotion_spec": study_spec.promotion_spec.__dict__,
        "upstream_refs": _upstream_refs_payload(study_spec.upstream_refs),
        "trial_manifest_index": [
            {
                "trial_id": trial_spec.trial_id,
                "trial_index": trial_spec.trial_index,
                "param_assignment_hash": trial_spec.param_assignment_hash,
                "output_dir": str(trial_spec.output_dir),
            }
            for trial_spec in trial_specs
        ],
        "generated_at_utc": _generated_at(),
    }


def _build_trial_manifest_payload(*, study_spec: StudySpec, trial_spec: TrialSpec) -> dict[str, Any]:
    """Build the machine-readable trial manifest."""

    return {
        "task_name": TASK_NAME,
        "contract_version": CONTRACT_VERSION,
        "study_id": study_spec.study_id,
        "trial_id": trial_spec.trial_id,
        "trial_index": trial_spec.trial_index,
        "trial_seed": trial_spec.seed,
        "param_assignment_hash": trial_spec.param_assignment_hash,
        "ppo_params": trial_spec.ppo_params,
        "output_dir": str(trial_spec.output_dir),
        "study_lineage": {
            "study_config_path": str(study_spec.study_config_path),
            "study_config_hash": study_spec.study_config_hash,
            "search_method": study_spec.search_method,
        },
        "upstream_refs": _upstream_refs_payload(study_spec.upstream_refs),
        "upstream_file_sha256": {
            "env_config": _sha256_file(study_spec.upstream_refs.env_config_path),
            "state_manifest": _sha256_file(study_spec.upstream_refs.state_manifest_path),
            "env_contract_report": _sha256_file(study_spec.upstream_refs.env_contract_report_path),
            "readiness_report": _sha256_file(study_spec.upstream_refs.readiness_report_path),
            "episode_catalog": _sha256_file(study_spec.upstream_refs.episode_catalog_path),
            "split_report": _sha256_file(study_spec.upstream_refs.split_report_path),
            "artifact_training_config_template": _sha256_file(study_spec.upstream_refs.artifact_training_config_template_path),
            "eval_config_template": _sha256_file(study_spec.upstream_refs.eval_config_template_path),
        },
        "derived_config_paths": {
            "launcher_config_path": str(trial_spec.launcher_config_path),
            "probe_training_config_path": str(trial_spec.probe_training_config_path) if trial_spec.probe_training_config_path else None,
            "final_training_config_path": str(trial_spec.final_training_config_path),
            "probe_eval_config_path": str(trial_spec.probe_eval_config_path) if trial_spec.probe_eval_config_path else None,
            "final_eval_config_path": str(trial_spec.final_eval_config_path),
        },
        "generated_at_utc": _generated_at(),
    }


def _empty_trial_training_report(*, trial_spec: TrialSpec, study_spec: StudySpec) -> dict[str, Any]:
    """Build the base top-level training report."""

    return {
        "study_id": study_spec.study_id,
        "trial_id": trial_spec.trial_id,
        "prelaunch": None,
        "probe_training": None,
        "final_training": None,
        "generated_at_utc": _generated_at(),
    }


def _empty_trial_evaluation_report(*, trial_spec: TrialSpec, study_spec: StudySpec) -> dict[str, Any]:
    """Build the base top-level evaluation report."""

    return {
        "study_id": study_spec.study_id,
        "trial_id": trial_spec.trial_id,
        "probe_evaluation": None,
        "final_evaluation": None,
        "generated_at_utc": _generated_at(),
    }


def _empty_trial_objective_report(*, trial_spec: TrialSpec, study_spec: StudySpec) -> dict[str, Any]:
    """Build the base top-level objective report."""

    return {
        "study_id": study_spec.study_id,
        "trial_id": trial_spec.trial_id,
        "primary_metric": study_spec.objective_spec.primary_metric,
        "probe_stage": None,
        "final_stage": None,
        "selected_stage": None,
        "selected_objective_score": None,
        "generated_at_utc": _generated_at(),
    }


def _empty_trial_guardrail_report(*, trial_spec: TrialSpec, study_spec: StudySpec) -> dict[str, Any]:
    """Build the base top-level guardrail report."""

    return {
        "study_id": study_spec.study_id,
        "trial_id": trial_spec.trial_id,
        "probe_stage": None,
        "final_stage": None,
        "selected_stage": None,
        "generated_at_utc": _generated_at(),
    }


def _empty_trial_status_payload(*, trial_spec: TrialSpec, study_spec: StudySpec) -> dict[str, Any]:
    """Build the base top-level status payload."""

    return {
        "study_id": study_spec.study_id,
        "trial_id": trial_spec.trial_id,
        "trial_index": trial_spec.trial_index,
        "status": TRIAL_STATUS_RUNNING,
        "ranking_eligible": False,
        "objective_score": None,
        "primary_metric_value": None,
        "candidate_rank": None,
        "promotion_ready": False,
        "promotion_family_hash": trial_spec.param_assignment_hash,
        "failure_codes": [],
        "invalid_reasons": [],
        "pruned_reason": None,
        "artifacts_complete": False,
        "started_at_utc": None,
        "completed_at_utc": None,
        "generated_at_utc": _generated_at(),
    }


def _build_study_progress_payload(
    *,
    study_spec: StudySpec,
    trial_records: Sequence[TrialExecutionRecord],
    study_status: str,
) -> dict[str, Any]:
    """Build incremental study progress."""

    counts = _trial_status_counts(trial_records)
    best_completed = [
        record
        for record in trial_records
        if record.objective_score is not None and record.terminal_state == "completed"
    ]
    best_completed.sort(key=lambda item: float(item.objective_score), reverse=True)
    best_trial = best_completed[0] if best_completed else None
    return {
        "study_id": study_spec.study_id,
        "study_status": study_status,
        "trial_counts": counts,
        "best_completed_trial_id": best_trial.trial_spec.trial_id if best_trial is not None else None,
        "best_completed_objective_score": best_trial.objective_score if best_trial is not None else None,
        "trials": [
            {
                "trial_id": record.trial_spec.trial_id,
                "trial_index": record.trial_spec.trial_index,
                "status": record.final_status,
                "objective_score": record.objective_score,
                "probe_objective_score": record.probe_objective_score,
                "failure_codes": record.failure_codes,
                "output_dir": str(record.trial_spec.output_dir),
            }
            for record in trial_records
        ],
        "updated_at_utc": _generated_at(),
    }


def _build_study_summary_payload(*, study_spec: StudySpec, trial_records: Sequence[TrialExecutionRecord]) -> dict[str, Any]:
    """Build the final study summary payload."""

    counts = _trial_status_counts(trial_records)
    ranked_trials = [
        record
        for record in trial_records
        if record.objective_score is not None and record.terminal_state == "completed"
    ]
    ranked_trials.sort(key=lambda item: float(item.objective_score), reverse=True)
    return {
        "task_name": TASK_NAME,
        "contract_version": CONTRACT_VERSION,
        "study_id": study_spec.study_id,
        "study_mode": study_spec.study_mode,
        "status": "completed",
        "trial_counts": counts,
        "top_ranked_trials": [
            {
                "trial_id": record.trial_spec.trial_id,
                "status": record.final_status,
                "objective_score": record.objective_score,
                "candidate_rank": record.candidate_rank,
                "promotion_ready": record.promotion_ready,
            }
            for record in ranked_trials[: study_spec.promotion_spec.candidate_top_k]
        ],
        "promotion_ready_trials": [
            {
                "trial_id": record.trial_spec.trial_id,
                "objective_score": record.objective_score,
                "candidate_rank": record.candidate_rank,
            }
            for record in ranked_trials
            if record.final_status == TRIAL_STATUS_PROMOTION_READY_CANDIDATE
        ],
        "trials": [
            {
                "trial_id": record.trial_spec.trial_id,
                "trial_index": record.trial_spec.trial_index,
                "status": record.final_status,
                "objective_score": record.objective_score,
                "probe_objective_score": record.probe_objective_score,
                "failure_codes": record.failure_codes,
                "invalid_reasons": record.invalid_reasons,
                "pruned_reason": record.pruned_reason,
                "artifacts_complete": record.artifacts_complete,
                "output_dir": str(record.trial_spec.output_dir),
            }
            for record in trial_records
        ],
        "generated_at_utc": _generated_at(),
    }


def _build_invalid_study_summary(
    *,
    study_id: str | None,
    study_config_path: Path,
    failure_codes: list[str],
    issues: Sequence[ValidationIssue],
) -> dict[str, Any]:
    """Build a fail-closed invalid study summary."""

    return {
        "task_name": TASK_NAME,
        "contract_version": CONTRACT_VERSION,
        "study_id": study_id,
        "status": "invalid",
        "study_config_path": str(study_config_path),
        "failure_codes": failure_codes,
        "errors": [item.__dict__ for item in issues],
        "generated_at_utc": _generated_at(),
    }


def _upstream_refs_payload(upstream_refs: UpstreamRefs) -> dict[str, Any]:
    """Serialize upstream refs."""

    return {
        "run_id": upstream_refs.run_id,
        "env_config_path": str(upstream_refs.env_config_path),
        "state_manifest_path": str(upstream_refs.state_manifest_path),
        "env_contract_report_path": str(upstream_refs.env_contract_report_path),
        "readiness_report_path": str(upstream_refs.readiness_report_path),
        "episode_catalog_path": str(upstream_refs.episode_catalog_path),
        "split_report_path": str(upstream_refs.split_report_path),
        "artifact_training_config_template_path": str(upstream_refs.artifact_training_config_template_path),
        "eval_config_template_path": str(upstream_refs.eval_config_template_path),
    }


def _load_json_file(path: Path, *, label: str) -> tuple[dict[str, Any] | None, list[ValidationIssue]]:
    """Load one JSON file with structured error handling."""

    if not path.exists():
        return (
            None,
            [
                ValidationIssue(
                    code=SEARCH_INPUT_MISSING,
                    message="Required 4.9 input file is missing.",
                    context={"label": label, "path": str(path)},
                )
            ],
        )
    if not path.is_file():
        return (
            None,
            [
                ValidationIssue(
                    code=SEARCH_PATH_UNREADABLE,
                    message="Required 4.9 input path is not a file.",
                    context={"label": label, "path": str(path)},
                )
            ],
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            None,
            [
                ValidationIssue(
                    code=SEARCH_JSON_INVALID,
                    message="Required 4.9 JSON input is invalid.",
                    context={"label": label, "path": str(path), "error": str(exc)},
                )
            ],
        )
    if not isinstance(payload, dict):
        return (
            None,
            [
                ValidationIssue(
                    code=SEARCH_JSON_INVALID,
                    message="Required 4.9 JSON input must decode to an object.",
                    context={"label": label, "path": str(path), "payload_type": type(payload).__name__},
                )
            ],
        )
    return payload, []


def _check_output_dir_policy(output_dir: Path) -> list[ValidationIssue]:
    """Enforce fresh-only output behavior for study or trial directories."""

    if output_dir.exists():
        return [
            ValidationIssue(
                code=SEARCH_OUTPUT_CONFLICT,
                message="4.9 output dir must not already exist.",
                context={"output_dir": str(output_dir)},
            )
        ]
    parent = output_dir.parent
    if not parent.exists():
        return []
    if not parent.is_dir():
        return [
            ValidationIssue(
                code=SEARCH_OUTPUT_CONFLICT,
                message="4.9 output parent must be a directory.",
                context={"output_dir_parent": str(parent)},
            )
        ]
    return []


def _resolve_primary_metric_value(
    *,
    primary_metric: str,
    strategy_metrics: Mapping[str, Any],
    relative_metrics: Mapping[str, Any],
) -> float | None:
    """Resolve the selected scalar primary metric."""

    if primary_metric == PRIMARY_METRIC_TOTAL_RETURN:
        return _mapping_float(strategy_metrics, "total_return")
    if primary_metric == PRIMARY_METRIC_EXCESS_TOTAL_RETURN:
        return _mapping_float(relative_metrics, "excess_total_return")
    return None


def _hard_constraint(name: str, passed: bool, reason_code: str | None, detail: Mapping[str, Any]) -> dict[str, Any]:
    """Build a hard-constraint entry."""

    return {
        "constraint_name": name,
        "pass": bool(passed),
        "reason_code": reason_code,
        "detail": dict(detail),
    }


def _classify_nonzero_exit(failure_codes: Sequence[str], runtime_codes: set[str]) -> str:
    """Map controlled upstream failures into invalid vs failed."""

    return TRIAL_STATUS_FAILED if any(code in runtime_codes for code in failure_codes) else TRIAL_STATUS_INVALID


def _trial_artifacts_complete(
    *,
    training_report_payload: Mapping[str, Any],
    evaluation_report_payload: Mapping[str, Any],
    terminal_state: str,
) -> bool:
    """Return True when required trial-level stage artifacts are present."""

    if terminal_state == TRIAL_STATUS_INVALID:
        return False
    if training_report_payload.get("prelaunch") is None:
        return False
    if terminal_state == TRIAL_STATUS_PRUNED:
        return training_report_payload.get("probe_training") is not None and evaluation_report_payload.get("probe_evaluation") is not None
    if terminal_state == "completed":
        return training_report_payload.get("final_training") is not None and evaluation_report_payload.get("final_evaluation") is not None
    return False


def _trial_status_counts(trial_records: Sequence[TrialExecutionRecord]) -> dict[str, int]:
    """Aggregate final status counts."""

    counts = {
        TRIAL_STATUS_INVALID: 0,
        TRIAL_STATUS_FAILED: 0,
        TRIAL_STATUS_PRUNED: 0,
        TRIAL_STATUS_COMPLETED_NONCOMPETITIVE: 0,
        TRIAL_STATUS_COMPLETED_CANDIDATE: 0,
        TRIAL_STATUS_PROMOTION_READY_CANDIDATE: 0,
    }
    for record in trial_records:
        counts.setdefault(record.final_status, 0)
        counts[record.final_status] += 1
    counts["total_trials"] = len(trial_records)
    return counts


def _objective_drawdown(record: TrialExecutionRecord) -> float | None:
    """Extract drawdown from a completed trial."""

    final_stage = record.guardrail_report_payload.get("final_stage")
    if not isinstance(final_stage, Mapping):
        return None
    diagnostics = final_stage.get("diagnostics")
    del diagnostics
    constraints = final_stage.get("hard_constraints")
    if not isinstance(constraints, list):
        return None
    for item in constraints:
        detail = item.get("detail") if isinstance(item, Mapping) else None
        if item.get("constraint_name") == "strategy_max_drawdown_guard" and isinstance(detail, Mapping):
            return _mapping_float(detail, "max_drawdown")
    return None


def _objective_num_trades(record: TrialExecutionRecord) -> int | None:
    """Extract num_trades from the final objective report."""

    final_stage = record.objective_report_payload.get("final_stage")
    if not isinstance(final_stage, Mapping):
        return None
    penalties = final_stage.get("penalties")
    if not isinstance(penalties, list):
        return None
    for item in penalties:
        detail = item.get("detail") if isinstance(item, Mapping) else None
        if item.get("penalty_name") == "low_trade_count_penalty" and isinstance(detail, Mapping):
            value = detail.get("num_trades")
            if isinstance(value, int):
                return int(value)
    return None


def _parse_positive_int(value: Any, issues: list[ValidationIssue], code: str, field_name: str) -> int | None:
    """Parse and validate a positive integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        issues.append(
            ValidationIssue(
                code=code,
                message=f"{field_name} must be a positive integer.",
                context={field_name: value},
            )
        )
        return None
    return int(value)


def _parse_non_negative_int(value: Any, issues: list[ValidationIssue], code: str, field_name: str) -> int | None:
    """Parse and validate a non-negative integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        issues.append(
            ValidationIssue(
                code=code,
                message=f"{field_name} must be a non-negative integer.",
                context={field_name: value},
            )
        )
        return None
    return int(value)


def _parse_non_negative_float(value: Any, issues: list[ValidationIssue], code: str, field_name: str) -> float | None:
    """Parse and validate a non-negative float."""

    parsed = _parse_finite_float(value, issues, code, field_name)
    if parsed is None:
        return None
    if parsed < 0.0:
        issues.append(
            ValidationIssue(
                code=code,
                message=f"{field_name} must be >= 0.",
                context={field_name: value},
            )
        )
        return None
    return parsed


def _parse_finite_float(value: Any, issues: list[ValidationIssue], code: str, field_name: str) -> float | None:
    """Parse and validate a finite float-like value."""

    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        issues.append(
            ValidationIssue(
                code=code,
                message=f"{field_name} must be a finite number.",
                context={field_name: value},
            )
        )
        return None
    return float(value)


def _resolve_path(value: str, base_dir: Path) -> Path:
    """Resolve a relative or absolute path."""

    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _product_cardinality(search_space: Mapping[str, Sequence[Any]]) -> int:
    """Return the cartesian-product cardinality."""

    total = 1
    for values in search_space.values():
        total *= len(values)
    return total


def _mapping_float(mapping: Mapping[str, Any], key: str) -> float | None:
    """Extract one float from a mapping."""

    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return None
    return float(value)


def _mapping_int(mapping: Mapping[str, Any], key: str) -> int | None:
    """Extract one integer from a mapping."""

    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return int(value)


def _safe_string(value: Any) -> str:
    """Return a stripped string or empty string."""

    return value.strip() if isinstance(value, str) else ""


def _dedupe_sequence(values: Sequence[Any]) -> list[Any]:
    """Deduplicate while preserving order."""

    seen: set[Any] = set()
    deduped: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _failure_codes(issues: Sequence[ValidationIssue]) -> list[str]:
    """Collect issue codes in stable order."""

    return _dedupe_sequence([item.code for item in issues])


def _hash_canonical_json(payload: Mapping[str, Any]) -> str:
    """Return a stable semantic hash."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return SHA256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_at() -> str:
    """Return a UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


def _read_step_trace(path: Path) -> pd.DataFrame:
    """Read the evaluation step trace parquet."""

    return pd.read_parquet(path)
