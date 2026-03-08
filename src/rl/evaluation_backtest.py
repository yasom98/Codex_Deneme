"""Strict evaluation/backtest gate for Milestone 4.8."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import zipfile
from typing import Any, Mapping, Sequence

from core.io_atomic import atomic_write_json, atomic_write_parquet
from core.logging import get_logger
import numpy as np
import pandas as pd
from rl.env_adapter_gym import TradingEnvGym
from rl.env_contract import EnvConfig, parse_env_config
from rl.env_core import EpisodeRef

LOGGER = get_logger(__name__)

CANONICAL_JSON_POLICY = "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=True)"
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0

ALGORITHM_PPO = "ppo"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_AUTO = "auto"

EVALUATION_MODE_SINGLE_PATH = "single_path_backtest"
EVALUATION_MODE_EPISODIC = "episodic_eval_backtest"

TARGET_MODE_EXPLICIT_PARTITION = "explicit_partition"
TARGET_MODE_EXPLICIT_EPISODE_REFS = "explicit_episode_refs"

PARTITION_VALIDATION = "validation"
PARTITION_VAL = "val"
PARTITION_TEST = "test"
SUPPORTED_TARGET_PARTITIONS = {PARTITION_VALIDATION, PARTITION_TEST}
PARTITION_ALIAS_RULE = {
    "alias_rule": "validation_to_val_v1_compatibility",
    "requested_partition": PARTITION_VALIDATION,
    "resolved_partition": PARTITION_VAL,
}

BENCHMARK_MODE_NONE = "none"
BENCHMARK_MODE_BUY_AND_HOLD = "buy_and_hold"

STARTUP_POLICY_FRESH_ONLY = "fresh_only"

SUPPORTED_BACKTEST_METRICS = (
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "num_steps",
    "num_trades",
    "win_rate",
    "avg_trade_return",
    "final_equity",
)
RELATIVE_METRICS = (
    "excess_total_return",
    "excess_annualized_return",
    "excess_sharpe_ratio",
    "excess_max_drawdown_delta",
)

ALIAS_WARNING_CODE = "EVAL_PARTITION_ALIAS_APPLIED"

EVAL_INPUT_MISSING = "EVAL_INPUT_MISSING"
EVAL_PATH_UNREADABLE = "EVAL_PATH_UNREADABLE"
EVAL_JSON_INVALID = "EVAL_JSON_INVALID"
EVAL_RUN_ID_MISMATCH = "EVAL_RUN_ID_MISMATCH"
EVAL_MODEL_ARTIFACT_REQUIRED = "EVAL_MODEL_ARTIFACT_REQUIRED"
EVAL_MODEL_ARTIFACT_UNREADABLE = "EVAL_MODEL_ARTIFACT_UNREADABLE"
EVAL_MODEL_ARTIFACT_INVALID = "EVAL_MODEL_ARTIFACT_INVALID"
EVAL_MODEL_LOAD_FAILED = "EVAL_MODEL_LOAD_FAILED"
EVAL_CONFIG_INVALID = "EVAL_CONFIG_INVALID"
EVAL_ALGO_UNSUPPORTED = "EVAL_ALGO_UNSUPPORTED"
EVAL_DEVICE_INVALID = "EVAL_DEVICE_INVALID"
EVAL_SEED_REQUIRED = "EVAL_SEED_REQUIRED"
EVAL_STARTUP_POLICY_INVALID = "EVAL_STARTUP_POLICY_INVALID"
EVAL_MODE_INVALID = "EVAL_MODE_INVALID"
EVAL_TARGET_MODE_INVALID = "EVAL_TARGET_MODE_INVALID"
EVAL_TARGET_INVALID = "EVAL_TARGET_INVALID"
EVAL_BENCHMARK_MODE_INVALID = "EVAL_BENCHMARK_MODE_INVALID"
EVAL_METRICS_INVALID = "EVAL_METRICS_INVALID"
EVAL_ENV_CONTRACT_REQUIRED = "EVAL_ENV_CONTRACT_REQUIRED"
EVAL_ENV_CONTRACT_FAILED = "EVAL_ENV_CONTRACT_FAILED"
EVAL_READINESS_REQUIRED = "EVAL_READINESS_REQUIRED"
EVAL_READINESS_FAILED = "EVAL_READINESS_FAILED"
EVAL_EPISODE_CATALOG_REQUIRED = "EVAL_EPISODE_CATALOG_REQUIRED"
EVAL_EPISODE_CATALOG_FAILED = "EVAL_EPISODE_CATALOG_FAILED"
EVAL_SPLIT_REPORT_REQUIRED = "EVAL_SPLIT_REPORT_REQUIRED"
EVAL_SPLIT_REPORT_FAILED = "EVAL_SPLIT_REPORT_FAILED"
EVAL_TARGET_NOT_ELIGIBLE = "EVAL_TARGET_NOT_ELIGIBLE"
EVAL_OUTPUT_CONFLICT = "EVAL_OUTPUT_CONFLICT"
EVAL_ENV_INIT_FAILED = "EVAL_ENV_INIT_FAILED"
EVAL_EXECUTION_FAILED = "EVAL_EXECUTION_FAILED"
EVAL_BENCHMARK_FAILED = "EVAL_BENCHMARK_FAILED"
EVAL_REPORT_WRITE_FAILED = "EVAL_REPORT_WRITE_FAILED"


@dataclass
class ValidationIssue:
    """Machine-readable evaluation issue."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    """Strict 4.8 evaluation configuration."""

    algorithm: str
    seed: int
    deterministic: bool
    device: str
    evaluation_mode: str
    target_mode: str
    target_partition: str | None
    target_fold_id: int | None
    target_episode_refs: tuple[EpisodeRef, ...]
    benchmark_mode: str
    startup_policy: str
    max_eval_episodes: int
    max_eval_steps: int
    write_step_trace: bool
    backtest_metrics: tuple[str, ...]


@dataclass(frozen=True)
class TargetResolution:
    """Resolved explicit evaluation targets."""

    selected_episode_refs: tuple[dict[str, Any], ...]
    selected_partition: str | None
    selected_fold_id: int | None
    resolved_partition_name: str | None
    alias_events: tuple[dict[str, Any], ...]
    split_targets_checked: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReportPaths:
    """Stable 4.8 report output paths."""

    validation_report_path: Path
    manifest_path: Path
    backtest_report_path: Path
    step_trace_path: Path


@dataclass
class EvaluationExecutionResult:
    """Composite evaluation execution output."""

    exit_code: int
    validation_payload: dict[str, Any]
    manifest_payload: dict[str, Any] | None
    backtest_payload: dict[str, Any] | None
    report_paths: ReportPaths
    reports_written: bool


@dataclass(frozen=True)
class EpisodeRuntime:
    """Per-episode runtime payload used for aggregation."""

    episode_ref: dict[str, Any]
    step_records: tuple[dict[str, Any], ...]
    strategy_metric_values: dict[str, float | int | None]
    strategy_metric_status: dict[str, dict[str, Any]]
    benchmark_metric_values: dict[str, float | int | None] | None
    benchmark_metric_status: dict[str, dict[str, Any]] | None
    closed_trade_proxy_returns: tuple[float, ...]
    closed_trade_pnls: tuple[float, ...]


def execute_evaluation_backtest(
    *,
    run_id: str,
    model_artifact_path: Path,
    env_config_path: Path,
    eval_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
    output_dir: Path,
) -> EvaluationExecutionResult:
    """Execute strict evaluation/backtest over explicit artifacts."""

    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must be non-empty")

    output_dir_resolved = output_dir.resolve()
    report_paths = ReportPaths(
        validation_report_path=output_dir_resolved / "evaluation_validation_report.json",
        manifest_path=output_dir_resolved / "evaluation_manifest.json",
        backtest_report_path=output_dir_resolved / "evaluation_backtest_report.json",
        step_trace_path=output_dir_resolved / "evaluation_step_trace.parquet",
    )
    evaluation_session_id = _build_evaluation_session_id(
        run_id=normalized_run_id,
        output_dir=output_dir_resolved,
        model_artifact_path=model_artifact_path.resolve(),
        env_config_path=env_config_path.resolve(),
        eval_config_path=eval_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
        split_report_path=split_report_path.resolve(),
    )

    output_guard_issues = _check_output_dir_policy(output_dir_resolved)
    if output_guard_issues:
        validation_payload = _build_validation_payload(
            run_id=normalized_run_id,
            evaluation_session_id=evaluation_session_id,
            selected_algorithm=None,
            deterministic=None,
            effective_seed=None,
            requested_device=None,
            resolved_device=None,
            model_artifact_hash=None,
            eval_config_hash=None,
            readiness_hash=None,
            env_contract_hash=None,
            state_manifest_hash=None,
            episode_catalog_hash=None,
            split_report_hash=None,
            validation_checks=[
                _validation_check(
                    check_name="startup_policy_fresh_only",
                    passed=False,
                    reason_code=EVAL_OUTPUT_CONFLICT,
                    detail={"output_dir": str(output_dir_resolved)},
                ),
                _validation_check(
                    check_name="output_dir_conflict_free",
                    passed=False,
                    reason_code=EVAL_OUTPUT_CONFLICT,
                    detail=output_guard_issues[0].context,
                ),
            ],
            warnings=[],
            errors=output_guard_issues,
        )
        backtest_payload = _build_backtest_payload(
            run_id=normalized_run_id,
            evaluation_session_id=evaluation_session_id,
            evaluation_success=False,
            selected_algorithm=None,
            deterministic=None,
            effective_seed=None,
            evaluation_mode=None,
            target_mode=None,
            benchmark_mode=None,
            startup_phase_trace=_phase_trace(
                validation_status="failed",
                model_load_status="not_started",
                env_init_status="not_started",
                eval_start_status="not_started",
                eval_finish_status="not_started",
                report_write_status="not_written",
                validation_detail={"failure_codes": _failure_codes(output_guard_issues)},
            ),
            strategy_metrics=None,
            benchmark_metrics=None,
            relative_metrics=None,
            metric_status={"strategy": None, "benchmark": None, "relative": None},
            trace_artifact_path=None,
            warnings=[],
            errors=output_guard_issues,
        )
        return EvaluationExecutionResult(
            exit_code=2,
            validation_payload=validation_payload,
            manifest_payload=None,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
            reports_written=False,
        )

    output_dir_resolved.mkdir(parents=True, exist_ok=False)

    loaded_inputs, load_issues = _load_json_inputs(
        env_config_path=env_config_path.resolve(),
        eval_config_path=eval_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
        split_report_path=split_report_path.resolve(),
    )
    warnings: list[ValidationIssue] = []
    errors: list[ValidationIssue] = list(load_issues)

    model_artifact_hash, model_issues = _validate_model_artifact(model_artifact_path.resolve())
    errors.extend(model_issues)

    eval_config_hash = _semantic_hash_optional(loaded_inputs.get("eval_config"))
    readiness_hash = _semantic_hash_optional(loaded_inputs.get("readiness_report"))
    env_contract_hash = _semantic_hash_optional(loaded_inputs.get("env_contract_report"))
    state_manifest_hash = _semantic_hash_optional(loaded_inputs.get("state_manifest"))
    episode_catalog_hash = _semantic_hash_optional(loaded_inputs.get("episode_catalog"))
    split_report_hash = _semantic_hash_optional(loaded_inputs.get("split_report"))

    eval_config_result = _validate_eval_config(loaded_inputs.get("eval_config"))
    eval_config = eval_config_result["config"]
    errors.extend(eval_config_result["errors"])

    selected_algorithm = eval_config.algorithm if eval_config is not None else _raw_string(loaded_inputs.get("eval_config"), "algorithm")
    deterministic = eval_config.deterministic if eval_config is not None else _raw_bool(loaded_inputs.get("eval_config"), "deterministic")
    requested_device = eval_config.device if eval_config is not None else _raw_string(loaded_inputs.get("eval_config"), "device")
    effective_seed = eval_config.seed if eval_config is not None else _raw_int(loaded_inputs.get("eval_config"), "seed")
    evaluation_mode = eval_config.evaluation_mode if eval_config is not None else _raw_string(
        loaded_inputs.get("eval_config"), "evaluation_mode"
    )
    target_mode = eval_config.target_mode if eval_config is not None else _raw_string(loaded_inputs.get("eval_config"), "target_mode")
    benchmark_mode = eval_config.benchmark_mode if eval_config is not None else _raw_string(
        loaded_inputs.get("eval_config"), "benchmark_mode"
    )

    resolved_device, device_issues, dependency_probe = _resolve_device(requested_device)
    errors.extend(device_issues)

    env_config_result = _validate_env_config(
        env_config_payload=loaded_inputs.get("env_config"),
        cli_run_id=normalized_run_id,
        state_manifest_path=state_manifest_path.resolve(),
    )
    env_config = env_config_result["config"]
    errors.extend(env_config_result["errors"])

    errors.extend(_validate_state_manifest(loaded_inputs.get("state_manifest"), normalized_run_id))
    errors.extend(_validate_env_contract_report(loaded_inputs.get("env_contract_report"), normalized_run_id))
    errors.extend(_validate_readiness_report(loaded_inputs.get("readiness_report"), normalized_run_id))
    errors.extend(_validate_episode_catalog_report(loaded_inputs.get("episode_catalog"), normalized_run_id))
    errors.extend(_validate_split_report(loaded_inputs.get("split_report"), normalized_run_id))
    errors.extend(
        _validate_lineage_consistency(
            state_manifest_path=state_manifest_path.resolve(),
            state_manifest_payload=loaded_inputs.get("state_manifest"),
            env_contract_report=loaded_inputs.get("env_contract_report"),
            readiness_report=loaded_inputs.get("readiness_report"),
            episode_catalog=loaded_inputs.get("episode_catalog"),
        )
    )

    target_resolution: TargetResolution | None = None
    if eval_config is not None:
        target_resolution_result = _resolve_targets(
            eval_config=eval_config,
            episode_catalog=loaded_inputs.get("episode_catalog"),
            split_report=loaded_inputs.get("split_report"),
        )
        target_resolution = target_resolution_result["resolution"]
        errors.extend(target_resolution_result["errors"])
        warnings.extend(target_resolution_result["warnings"])

    validation_checks = _build_validation_checks(
        load_issues=load_issues,
        model_issues=model_issues,
        all_errors=errors,
        alias_warnings=warnings,
    )
    validation_payload = _build_validation_payload(
        run_id=normalized_run_id,
        evaluation_session_id=evaluation_session_id,
        selected_algorithm=selected_algorithm,
        deterministic=deterministic,
        effective_seed=effective_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        model_artifact_hash=model_artifact_hash,
        eval_config_hash=eval_config_hash,
        readiness_hash=readiness_hash,
        env_contract_hash=env_contract_hash,
        state_manifest_hash=state_manifest_hash,
        episode_catalog_hash=episode_catalog_hash,
        split_report_hash=split_report_hash,
        validation_checks=validation_checks,
        warnings=warnings,
        errors=errors,
    )
    manifest_payload = _build_manifest_payload(
        run_id=normalized_run_id,
        evaluation_session_id=evaluation_session_id,
        model_artifact_path=model_artifact_path.resolve(),
        env_config_path=env_config_path.resolve(),
        eval_config_path=eval_config_path.resolve(),
        state_manifest_path=state_manifest_path.resolve(),
        env_contract_report_path=env_contract_report_path.resolve(),
        readiness_report_path=readiness_report_path.resolve(),
        episode_catalog_path=episode_catalog_path.resolve(),
        split_report_path=split_report_path.resolve(),
        selected_algorithm=selected_algorithm,
        deterministic=deterministic,
        effective_seed=effective_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        evaluation_mode=evaluation_mode,
        target_mode=target_mode,
        benchmark_mode=benchmark_mode,
        target_resolution=target_resolution,
        model_artifact_hash=model_artifact_hash,
        eval_config_hash=eval_config_hash,
        readiness_hash=readiness_hash,
        env_contract_hash=env_contract_hash,
        state_manifest_hash=state_manifest_hash,
        episode_catalog_hash=episode_catalog_hash,
        split_report_hash=split_report_hash,
        output_dir=output_dir_resolved,
        warnings=warnings,
    )

    if errors:
        backtest_payload = _build_backtest_payload(
            run_id=normalized_run_id,
            evaluation_session_id=evaluation_session_id,
            evaluation_success=False,
            selected_algorithm=selected_algorithm,
            deterministic=deterministic,
            effective_seed=effective_seed,
            evaluation_mode=evaluation_mode,
            target_mode=target_mode,
            benchmark_mode=benchmark_mode,
            startup_phase_trace=_phase_trace(
                validation_status="failed",
                model_load_status="not_started",
                env_init_status="not_started",
                eval_start_status="not_started",
                eval_finish_status="not_started",
                report_write_status="completed",
                validation_detail={"failure_codes": validation_payload["failure_codes"]},
            ),
            strategy_metrics=None,
            benchmark_metrics=None,
            relative_metrics=None,
            metric_status={"strategy": None, "benchmark": None, "relative": None},
            trace_artifact_path=None,
            warnings=warnings,
            errors=errors,
        )
        _write_core_reports(
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
        )
        return EvaluationExecutionResult(
            exit_code=2,
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
            reports_written=True,
        )

    assert eval_config is not None
    assert env_config is not None
    assert target_resolution is not None

    phase_status = {
        "validation": "completed",
        "model_load": "not_started",
        "env_init": "not_started",
        "eval_start": "not_started",
        "eval_finish": "not_started",
        "report_write": "not_started",
    }
    phase_detail: dict[str, Any] = {
        "validation": {
            "overall_pass": True,
            "selected_episode_count": len(target_resolution.selected_episode_refs),
            "dependency_probe": dependency_probe,
        },
        "model_load": {},
        "env_init": {},
        "eval_start": {},
        "eval_finish": {},
        "report_write": {},
    }

    _set_global_seed(eval_config.seed)
    try:
        model = _load_ppo_model(model_artifact_path=model_artifact_path.resolve(), device=resolved_device)
        if hasattr(model, "set_random_seed"):
            model.set_random_seed(eval_config.seed)
        phase_status["model_load"] = "completed"
        phase_detail["model_load"] = {
            "model_class": type(model).__name__,
            "device": resolved_device,
        }
    except Exception as exc:  # noqa: BLE001
        model_error = ValidationIssue(
            code=EVAL_MODEL_LOAD_FAILED,
            message="Model load failed for the explicit SB3 PPO artifact.",
            context={"error": str(exc), "model_artifact_path": str(model_artifact_path.resolve())},
        )
        phase_status["model_load"] = "failed"
        phase_detail["model_load"] = {"error": str(exc)}
        backtest_payload = _build_backtest_payload(
            run_id=normalized_run_id,
            evaluation_session_id=evaluation_session_id,
            evaluation_success=False,
            selected_algorithm=eval_config.algorithm,
            deterministic=eval_config.deterministic,
            effective_seed=eval_config.seed,
            evaluation_mode=eval_config.evaluation_mode,
            target_mode=eval_config.target_mode,
            benchmark_mode=eval_config.benchmark_mode,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            strategy_metrics=None,
            benchmark_metrics=None,
            relative_metrics=None,
            metric_status={"strategy": None, "benchmark": None, "relative": None},
            trace_artifact_path=None,
            warnings=warnings,
            errors=[model_error],
        )
        _write_core_reports(
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
        )
        return EvaluationExecutionResult(
            exit_code=2,
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
            reports_written=True,
        )

    env_clients: list[TradingEnvGym] = []
    try:
        for episode_ref in target_resolution.selected_episode_refs:
            effective_env_config = _effective_env_config(
                env_config=env_config,
                seed=eval_config.seed,
                episode_ref=episode_ref,
                max_eval_steps=eval_config.max_eval_steps,
            )
            env_clients.append(TradingEnvGym(config=effective_env_config, validate_on_init=True))
        phase_status["env_init"] = "completed"
        phase_detail["env_init"] = {
            "env_class": "TradingEnvGym",
            "initialized_episode_count": len(env_clients),
            "selected_episode_refs": list(target_resolution.selected_episode_refs),
        }
    except Exception as exc:  # noqa: BLE001
        _close_envs(env_clients)
        env_issue = ValidationIssue(
            code=EVAL_ENV_INIT_FAILED,
            message="Evaluation environment initialization failed.",
            context={"error": str(exc)},
        )
        phase_status["env_init"] = "failed"
        phase_detail["env_init"] = {"error": str(exc)}
        backtest_payload = _build_backtest_payload(
            run_id=normalized_run_id,
            evaluation_session_id=evaluation_session_id,
            evaluation_success=False,
            selected_algorithm=eval_config.algorithm,
            deterministic=eval_config.deterministic,
            effective_seed=eval_config.seed,
            evaluation_mode=eval_config.evaluation_mode,
            target_mode=eval_config.target_mode,
            benchmark_mode=eval_config.benchmark_mode,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            strategy_metrics=None,
            benchmark_metrics=None,
            relative_metrics=None,
            metric_status={"strategy": None, "benchmark": None, "relative": None},
            trace_artifact_path=None,
            warnings=warnings,
            errors=[env_issue],
        )
        _write_core_reports(
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
        )
        return EvaluationExecutionResult(
            exit_code=2,
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
            reports_written=True,
        )

    episode_runtimes: list[EpisodeRuntime] = []
    step_rows: list[dict[str, Any]] = []
    try:
        phase_status["eval_start"] = "completed"
        phase_detail["eval_start"] = {
            "selected_episode_count": len(env_clients),
            "deterministic": eval_config.deterministic,
        }
        for episode_index, (episode_ref, env_client) in enumerate(zip(target_resolution.selected_episode_refs, env_clients, strict=True)):
            runtime = _evaluate_single_episode(
                model=model,
                env_client=env_client,
                episode_ref=episode_ref,
                deterministic=eval_config.deterministic,
                seed=eval_config.seed,
                benchmark_mode=eval_config.benchmark_mode,
                requested_metrics=eval_config.backtest_metrics,
                episode_index=episode_index,
            )
            episode_runtimes.append(runtime)
            step_rows.extend(runtime.step_records)
        phase_status["eval_finish"] = "completed"
        phase_detail["eval_finish"] = {
            "evaluated_episode_count": len(episode_runtimes),
            "step_count_total": sum(len(item.step_records) for item in episode_runtimes),
        }
    except ControlledEvaluationFailure as exc:
        _close_envs(env_clients)
        phase_status["eval_finish"] = "failed"
        phase_detail["eval_finish"] = {"error": str(exc)}
        backtest_payload = _build_backtest_payload(
            run_id=normalized_run_id,
            evaluation_session_id=evaluation_session_id,
            evaluation_success=False,
            selected_algorithm=eval_config.algorithm,
            deterministic=eval_config.deterministic,
            effective_seed=eval_config.seed,
            evaluation_mode=eval_config.evaluation_mode,
            target_mode=eval_config.target_mode,
            benchmark_mode=eval_config.benchmark_mode,
            startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
            strategy_metrics=None,
            benchmark_metrics=None,
            relative_metrics=None,
            metric_status={"strategy": None, "benchmark": None, "relative": None},
            trace_artifact_path=None,
            warnings=warnings,
            errors=[exc.issue],
        )
        _write_core_reports(
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
        )
        return EvaluationExecutionResult(
            exit_code=2,
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
            reports_written=True,
        )
    finally:
        _close_envs(env_clients)

    strategy_metrics, strategy_status = _aggregate_metric_values(
        episode_runtimes=episode_runtimes,
        metric_names=eval_config.backtest_metrics,
        metric_kind="strategy",
    )
    benchmark_metrics: dict[str, Any] | None = None
    benchmark_status: dict[str, Any] | None = None
    relative_metrics: dict[str, Any] | None = None
    relative_status: dict[str, Any] | None = None

    if eval_config.benchmark_mode == BENCHMARK_MODE_BUY_AND_HOLD:
        benchmark_metrics, benchmark_status = _aggregate_metric_values(
            episode_runtimes=episode_runtimes,
            metric_names=eval_config.backtest_metrics,
            metric_kind="benchmark",
        )
        relative_metrics, relative_status = _compute_relative_metrics(
            strategy_metrics=strategy_metrics,
            strategy_status=strategy_status,
            benchmark_metrics=benchmark_metrics,
            benchmark_status=benchmark_status,
        )

    phase_status["report_write"] = "completed"
    phase_detail["report_write"] = {
        "write_step_trace": eval_config.write_step_trace,
        "trace_row_count": len(step_rows),
    }
    backtest_payload = _build_backtest_payload(
        run_id=normalized_run_id,
        evaluation_session_id=evaluation_session_id,
        evaluation_success=True,
        selected_algorithm=eval_config.algorithm,
        deterministic=eval_config.deterministic,
        effective_seed=eval_config.seed,
        evaluation_mode=eval_config.evaluation_mode,
        target_mode=eval_config.target_mode,
        benchmark_mode=eval_config.benchmark_mode,
        startup_phase_trace=_phase_trace_from_maps(phase_status, phase_detail),
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        relative_metrics=relative_metrics,
        metric_status={
            "strategy": strategy_status,
            "benchmark": benchmark_status,
            "relative": relative_status,
        },
        trace_artifact_path=str(report_paths.step_trace_path) if eval_config.write_step_trace else None,
        warnings=warnings,
        errors=[],
    )
    write_ok = _write_reports_with_trace(
        validation_payload=validation_payload,
        manifest_payload=manifest_payload,
        backtest_payload=backtest_payload,
        report_paths=report_paths,
        step_rows=step_rows,
        write_step_trace=eval_config.write_step_trace,
    )
    if not write_ok:
        report_issue = ValidationIssue(
            code=EVAL_REPORT_WRITE_FAILED,
            message="One or more evaluation outputs could not be written atomically.",
            context={"output_dir": str(output_dir_resolved)},
        )
        failed_backtest = _build_backtest_payload(
            run_id=normalized_run_id,
            evaluation_session_id=evaluation_session_id,
            evaluation_success=False,
            selected_algorithm=eval_config.algorithm,
            deterministic=eval_config.deterministic,
            effective_seed=eval_config.seed,
            evaluation_mode=eval_config.evaluation_mode,
            target_mode=eval_config.target_mode,
            benchmark_mode=eval_config.benchmark_mode,
            startup_phase_trace=_phase_trace(
                validation_status="completed",
                model_load_status="completed",
                env_init_status="completed",
                eval_start_status="completed",
                eval_finish_status="completed",
                report_write_status="failed",
                validation_detail={"overall_pass": True},
                report_write_detail={"output_dir": str(output_dir_resolved)},
            ),
            strategy_metrics=strategy_metrics,
            benchmark_metrics=benchmark_metrics,
            relative_metrics=relative_metrics,
            metric_status={
                "strategy": strategy_status,
                "benchmark": benchmark_status,
                "relative": relative_status,
            },
            trace_artifact_path=str(report_paths.step_trace_path) if eval_config.write_step_trace else None,
            warnings=warnings,
            errors=[report_issue],
        )
        _best_effort_write_json(validation_payload, report_paths.validation_report_path)
        _best_effort_write_json(manifest_payload, report_paths.manifest_path)
        _best_effort_write_json(failed_backtest, report_paths.backtest_report_path)
        return EvaluationExecutionResult(
            exit_code=3,
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=failed_backtest,
            report_paths=report_paths,
            reports_written=False,
        )

    return EvaluationExecutionResult(
        exit_code=0,
        validation_payload=validation_payload,
        manifest_payload=manifest_payload,
        backtest_payload=backtest_payload,
        report_paths=report_paths,
        reports_written=True,
    )


class ControlledEvaluationFailure(Exception):
    """Controlled evaluation failure that maps to exit code 2."""

    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue


def _load_json_inputs(
    *,
    env_config_path: Path,
    eval_config_path: Path,
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
        "eval_config": eval_config_path,
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
    """Load one required JSON object."""

    if not path.exists():
        return None, ValidationIssue(
            code=EVAL_INPUT_MISSING,
            message="Required evaluation input is missing.",
            context={"input_label": label, "path": str(path)},
        )
    if not path.is_file():
        return None, ValidationIssue(
            code=EVAL_PATH_UNREADABLE,
            message="Required evaluation input path is not a readable file.",
            context={"input_label": label, "path": str(path)},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, ValidationIssue(
            code=EVAL_JSON_INVALID,
            message="Required evaluation input contains invalid JSON.",
            context={"input_label": label, "path": str(path), "error": str(exc)},
        )
    except OSError as exc:
        return None, ValidationIssue(
            code=EVAL_PATH_UNREADABLE,
            message="Required evaluation input could not be read.",
            context={"input_label": label, "path": str(path), "error": str(exc)},
        )
    if not isinstance(payload, dict):
        return None, ValidationIssue(
            code=EVAL_JSON_INVALID,
            message="Required evaluation input JSON must be an object.",
            context={"input_label": label, "path": str(path)},
        )
    return payload, None


def _validate_model_artifact(model_artifact_path: Path) -> tuple[str | None, list[ValidationIssue]]:
    """Validate the explicit model artifact path and canonical zip stance."""

    issues: list[ValidationIssue] = []
    if not str(model_artifact_path):
        issues.append(
            ValidationIssue(
                code=EVAL_MODEL_ARTIFACT_REQUIRED,
                message="model_artifact path is required.",
                context={},
            )
        )
        return None, issues
    if not model_artifact_path.exists() or not model_artifact_path.is_file():
        issues.append(
            ValidationIssue(
                code=EVAL_MODEL_ARTIFACT_UNREADABLE,
                message="Explicit model artifact path does not point to a readable file.",
                context={"model_artifact_path": str(model_artifact_path)},
            )
        )
        return None, issues
    if model_artifact_path.suffix.lower() != ".zip":
        issues.append(
            ValidationIssue(
                code=EVAL_MODEL_ARTIFACT_INVALID,
                message="Only canonical SB3 PPO .zip artifacts are supported in 4.8 v1.",
                context={"model_artifact_path": str(model_artifact_path), "suffix": model_artifact_path.suffix},
            )
        )
    if not zipfile.is_zipfile(model_artifact_path):
        issues.append(
            ValidationIssue(
                code=EVAL_MODEL_ARTIFACT_INVALID,
                message="model_artifact must be a readable zip file.",
                context={"model_artifact_path": str(model_artifact_path)},
            )
        )
    try:
        artifact_hash = _sha256_file(model_artifact_path)
    except OSError as exc:
        issues.append(
            ValidationIssue(
                code=EVAL_MODEL_ARTIFACT_UNREADABLE,
                message="model_artifact file could not be read.",
                context={"model_artifact_path": str(model_artifact_path), "error": str(exc)},
            )
        )
        return None, issues
    return artifact_hash, issues


def _validate_eval_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the strict evaluation config contract."""

    errors: list[ValidationIssue] = []
    if payload is None:
        return {"config": None, "errors": errors}

    required_fields = {
        "algorithm",
        "seed",
        "deterministic",
        "device",
        "evaluation_mode",
        "target_mode",
        "target_partition",
        "target_fold_id",
        "target_episode_refs",
        "benchmark_mode",
        "startup_policy",
        "max_eval_episodes",
        "max_eval_steps",
        "write_step_trace",
        "backtest_metrics",
    }
    extra_keys = sorted(set(payload.keys()) - required_fields)
    missing_keys = sorted(required_fields - set(payload.keys()))
    if missing_keys or extra_keys:
        errors.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="eval_config top-level fields must match the 4.8 contract exactly.",
                context={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return {"config": None, "errors": errors}

    algorithm = _raw_string(payload, "algorithm")
    if algorithm != ALGORITHM_PPO:
        errors.append(
            ValidationIssue(
                code=EVAL_ALGO_UNSUPPORTED,
                message="Only PPO is supported in 4.8 v1.",
                context={"algorithm": algorithm},
            )
        )

    seed_raw = payload.get("seed")
    if not isinstance(seed_raw, int):
        errors.append(
            ValidationIssue(
                code=EVAL_SEED_REQUIRED,
                message="seed must be a non-null integer.",
                context={"seed": seed_raw},
            )
        )

    deterministic_raw = payload.get("deterministic")
    if not isinstance(deterministic_raw, bool):
        errors.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="deterministic must be a boolean.",
                context={"deterministic": deterministic_raw},
            )
        )

    device = _raw_string(payload, "device")
    if device not in {DEVICE_CPU, DEVICE_CUDA, DEVICE_AUTO}:
        errors.append(
            ValidationIssue(
                code=EVAL_DEVICE_INVALID,
                message="device must be one of cpu, cuda, auto.",
                context={"device": device},
            )
        )

    evaluation_mode = _raw_string(payload, "evaluation_mode")
    if evaluation_mode not in {EVALUATION_MODE_SINGLE_PATH, EVALUATION_MODE_EPISODIC}:
        errors.append(
            ValidationIssue(
                code=EVAL_MODE_INVALID,
                message="evaluation_mode is unsupported.",
                context={"evaluation_mode": evaluation_mode},
            )
        )

    target_mode = _raw_string(payload, "target_mode")
    if target_mode not in {TARGET_MODE_EXPLICIT_PARTITION, TARGET_MODE_EXPLICIT_EPISODE_REFS}:
        errors.append(
            ValidationIssue(
                code=EVAL_TARGET_MODE_INVALID,
                message="target_mode is unsupported.",
                context={"target_mode": target_mode},
            )
        )

    target_partition_raw = payload.get("target_partition")
    if target_mode == TARGET_MODE_EXPLICIT_PARTITION:
        if target_partition_raw not in SUPPORTED_TARGET_PARTITIONS:
            errors.append(
                ValidationIssue(
                    code=EVAL_TARGET_INVALID,
                    message="target_partition must be validation or test when target_mode=explicit_partition.",
                    context={"target_partition": target_partition_raw},
                )
            )
    elif target_partition_raw is not None:
        errors.append(
            ValidationIssue(
                code=EVAL_TARGET_INVALID,
                message="target_partition must be null when target_mode=explicit_episode_refs.",
                context={"target_partition": target_partition_raw},
            )
        )

    target_fold_id_raw = payload.get("target_fold_id")
    if target_fold_id_raw is not None and (not isinstance(target_fold_id_raw, int) or target_fold_id_raw < 0):
        errors.append(
            ValidationIssue(
                code=EVAL_TARGET_INVALID,
                message="target_fold_id must be int >= 0 or null.",
                context={"target_fold_id": target_fold_id_raw},
            )
        )

    target_episode_refs: tuple[EpisodeRef, ...] = ()
    target_episode_refs_raw = payload.get("target_episode_refs")
    if target_mode == TARGET_MODE_EXPLICIT_EPISODE_REFS:
        if not isinstance(target_episode_refs_raw, list) or not target_episode_refs_raw:
            errors.append(
                ValidationIssue(
                    code=EVAL_TARGET_INVALID,
                    message="target_episode_refs must be a non-empty list when target_mode=explicit_episode_refs.",
                    context={"target_episode_refs": target_episode_refs_raw},
                )
            )
        else:
            parsed_refs: list[EpisodeRef] = []
            for index, item in enumerate(target_episode_refs_raw):
                try:
                    parsed_refs.append(_parse_episode_ref(item))
                except ValueError as exc:
                    errors.append(
                        ValidationIssue(
                            code=EVAL_TARGET_INVALID,
                            message="target_episode_refs contains an invalid episode ref.",
                            context={"index": index, "error": str(exc), "episode_ref": item},
                        )
                    )
            target_episode_refs = tuple(parsed_refs)
    elif target_episode_refs_raw is not None:
        errors.append(
            ValidationIssue(
                code=EVAL_TARGET_INVALID,
                message="target_episode_refs must be null when target_mode=explicit_partition.",
                context={"target_episode_refs": target_episode_refs_raw},
            )
        )

    benchmark_mode = _raw_string(payload, "benchmark_mode")
    if benchmark_mode not in {BENCHMARK_MODE_NONE, BENCHMARK_MODE_BUY_AND_HOLD}:
        errors.append(
            ValidationIssue(
                code=EVAL_BENCHMARK_MODE_INVALID,
                message="benchmark_mode is unsupported.",
                context={"benchmark_mode": benchmark_mode},
            )
        )

    startup_policy = _raw_string(payload, "startup_policy")
    if startup_policy != STARTUP_POLICY_FRESH_ONLY:
        errors.append(
            ValidationIssue(
                code=EVAL_STARTUP_POLICY_INVALID,
                message="startup_policy must be fresh_only.",
                context={"startup_policy": startup_policy},
            )
        )

    max_eval_episodes_raw = payload.get("max_eval_episodes")
    if not isinstance(max_eval_episodes_raw, int) or max_eval_episodes_raw <= 0:
        errors.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="max_eval_episodes must be a positive integer.",
                context={"max_eval_episodes": max_eval_episodes_raw},
            )
        )
    elif evaluation_mode == EVALUATION_MODE_SINGLE_PATH and max_eval_episodes_raw != 1:
        errors.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="single_path_backtest requires max_eval_episodes to equal 1.",
                context={"max_eval_episodes": max_eval_episodes_raw},
            )
        )

    max_eval_steps_raw = payload.get("max_eval_steps")
    if not isinstance(max_eval_steps_raw, int) or max_eval_steps_raw <= 0:
        errors.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="max_eval_steps must be a positive integer.",
                context={"max_eval_steps": max_eval_steps_raw},
            )
        )

    write_step_trace_raw = payload.get("write_step_trace")
    if not isinstance(write_step_trace_raw, bool):
        errors.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="write_step_trace must be a boolean.",
                context={"write_step_trace": write_step_trace_raw},
            )
        )

    backtest_metrics_raw = payload.get("backtest_metrics")
    metrics: tuple[str, ...] = ()
    if not isinstance(backtest_metrics_raw, list) or not backtest_metrics_raw:
        errors.append(
            ValidationIssue(
                code=EVAL_METRICS_INVALID,
                message="backtest_metrics must be a non-empty list.",
                context={"backtest_metrics": backtest_metrics_raw},
            )
        )
    else:
        metrics = tuple(str(item) for item in backtest_metrics_raw)
        if len(set(metrics)) != len(metrics):
            errors.append(
                ValidationIssue(
                    code=EVAL_METRICS_INVALID,
                    message="backtest_metrics must not contain duplicates.",
                    context={"backtest_metrics": list(metrics)},
                )
            )
        unsupported_metrics = sorted(set(metrics) - set(SUPPORTED_BACKTEST_METRICS))
        if unsupported_metrics:
            errors.append(
                ValidationIssue(
                    code=EVAL_METRICS_INVALID,
                    message="backtest_metrics contains unsupported metric names.",
                    context={"unsupported_metrics": unsupported_metrics},
                )
            )

    if evaluation_mode == EVALUATION_MODE_SINGLE_PATH and target_mode == TARGET_MODE_EXPLICIT_EPISODE_REFS and len(target_episode_refs) > 1:
        errors.append(
            ValidationIssue(
                code=EVAL_TARGET_INVALID,
                message="single_path_backtest requires exactly one explicit episode ref.",
                context={"target_episode_ref_count": len(target_episode_refs)},
            )
        )

    if errors:
        return {"config": None, "errors": errors}

    assert isinstance(seed_raw, int)
    assert isinstance(deterministic_raw, bool)
    assert isinstance(max_eval_episodes_raw, int)
    assert isinstance(max_eval_steps_raw, int)
    assert isinstance(write_step_trace_raw, bool)
    return {
        "config": EvalConfig(
            algorithm=algorithm,
            seed=seed_raw,
            deterministic=deterministic_raw,
            device=device,
            evaluation_mode=evaluation_mode,
            target_mode=target_mode,
            target_partition=str(target_partition_raw) if isinstance(target_partition_raw, str) else None,
            target_fold_id=target_fold_id_raw if isinstance(target_fold_id_raw, int) else None,
            target_episode_refs=target_episode_refs,
            benchmark_mode=benchmark_mode,
            startup_policy=startup_policy,
            max_eval_episodes=max_eval_episodes_raw,
            max_eval_steps=max_eval_steps_raw,
            write_step_trace=write_step_trace_raw,
            backtest_metrics=metrics,
        ),
        "errors": errors,
    }


def _validate_env_config(
    *,
    env_config_payload: dict[str, Any] | None,
    cli_run_id: str,
    state_manifest_path: Path,
) -> dict[str, Any]:
    """Validate the explicit env config without mutating upstream behavior."""

    errors: list[ValidationIssue] = []
    if env_config_payload is None:
        return {"config": None, "errors": errors}

    if env_config_payload.get("run_id") != cli_run_id:
        errors.append(
            ValidationIssue(
                code=EVAL_RUN_ID_MISMATCH,
                message="env_config.run_id mismatch.",
                context={"env_config_run_id": env_config_payload.get("run_id"), "cli_run_id": cli_run_id},
            )
        )

    expected_state_root = state_manifest_path.resolve().parents[1]
    seen_state_root = env_config_payload.get("state_root")
    if not isinstance(seen_state_root, str) or Path(seen_state_root).resolve() != expected_state_root:
        errors.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="env_config.state_root does not match the explicit state_manifest path.",
                context={
                    "env_config_state_root": seen_state_root,
                    "expected_state_root": str(expected_state_root),
                },
            )
        )

    try:
        config = parse_env_config(env_config_payload)
    except ValueError as exc:
        errors.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="env_config payload is invalid.",
                context={"error": str(exc)},
            )
        )
        return {"config": None, "errors": errors}
    return {"config": config, "errors": errors}


def _validate_state_manifest(payload: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate state manifest prerequisites."""

    if payload is None:
        return []
    issues: list[ValidationIssue] = []
    if payload.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=EVAL_RUN_ID_MISMATCH,
                message="state_manifest.run_id mismatch.",
                context={"state_manifest_run_id": payload.get("run_id"), "cli_run_id": run_id},
            )
        )
    if payload.get("output_completeness_ok") is not True:
        issues.append(
            ValidationIssue(
                code=EVAL_CONFIG_INVALID,
                message="state_manifest.output_completeness_ok must be true.",
                context={"output_completeness_ok": payload.get("output_completeness_ok")},
            )
        )
    return issues


def _validate_env_contract_report(report: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate env contract report prerequisites."""

    if report is None:
        return [
            ValidationIssue(
                code=EVAL_ENV_CONTRACT_REQUIRED,
                message="env_contract_report is required.",
                context={},
            )
        ]
    issues: list[ValidationIssue] = []
    if report.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=EVAL_RUN_ID_MISMATCH,
                message="env_contract_report.run_id mismatch.",
                context={"env_contract_report_run_id": report.get("run_id"), "cli_run_id": run_id},
            )
        )
    if "env_contract_overall" not in report:
        issues.append(
            ValidationIssue(
                code=EVAL_ENV_CONTRACT_REQUIRED,
                message="env_contract_report must contain env_contract_overall.",
                context={},
            )
        )
    elif report.get("env_contract_overall") is not True:
        issues.append(
            ValidationIssue(
                code=EVAL_ENV_CONTRACT_FAILED,
                message="env_contract_report must pass before evaluation.",
                context={"env_contract_overall": report.get("env_contract_overall")},
            )
        )
    if not isinstance(report.get("source_lineage"), dict):
        issues.append(
            ValidationIssue(
                code=EVAL_ENV_CONTRACT_REQUIRED,
                message="env_contract_report.source_lineage is required.",
                context={},
            )
        )
    return issues


def _validate_readiness_report(report: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate readiness report prerequisites."""

    if report is None:
        return [
            ValidationIssue(
                code=EVAL_READINESS_REQUIRED,
                message="training_env_readiness_report is required.",
                context={},
            )
        ]
    issues: list[ValidationIssue] = []
    if report.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=EVAL_RUN_ID_MISMATCH,
                message="training_env_readiness_report.run_id mismatch.",
                context={"readiness_report_run_id": report.get("run_id"), "cli_run_id": run_id},
            )
        )
    if "readiness_overall" not in report:
        issues.append(
            ValidationIssue(
                code=EVAL_READINESS_REQUIRED,
                message="training_env_readiness_report must contain readiness_overall.",
                context={},
            )
        )
    elif report.get("readiness_overall") is not True:
        issues.append(
            ValidationIssue(
                code=EVAL_READINESS_FAILED,
                message="training_env_readiness_report must pass before evaluation.",
                context={"readiness_overall": report.get("readiness_overall")},
            )
        )
    if report.get("episode_catalog_overall") is not True:
        issues.append(
            ValidationIssue(
                code=EVAL_READINESS_FAILED,
                message="training_env_readiness_report must confirm episode_catalog_overall=true.",
                context={"episode_catalog_overall": report.get("episode_catalog_overall")},
            )
        )
    env_contract_reference = report.get("env_contract_reference")
    if not isinstance(env_contract_reference, dict) or not isinstance(env_contract_reference.get("source_lineage"), dict):
        issues.append(
            ValidationIssue(
                code=EVAL_READINESS_REQUIRED,
                message="training_env_readiness_report.env_contract_reference.source_lineage is required.",
                context={},
            )
        )
    return issues


def _validate_episode_catalog_report(report: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate episode catalog prerequisites."""

    if report is None:
        return [
            ValidationIssue(
                code=EVAL_EPISODE_CATALOG_REQUIRED,
                message="episode_catalog is required.",
                context={},
            )
        ]
    issues: list[ValidationIssue] = []
    if report.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=EVAL_RUN_ID_MISMATCH,
                message="episode_catalog.run_id mismatch.",
                context={"episode_catalog_run_id": report.get("run_id"), "cli_run_id": run_id},
            )
        )
    if report.get("episode_catalog_overall") is not True:
        issues.append(
            ValidationIssue(
                code=EVAL_EPISODE_CATALOG_FAILED,
                message="episode_catalog must pass before evaluation.",
                context={"episode_catalog_overall": report.get("episode_catalog_overall")},
            )
        )
    if not isinstance(report.get("source_lineage"), dict):
        issues.append(
            ValidationIssue(
                code=EVAL_EPISODE_CATALOG_REQUIRED,
                message="episode_catalog.source_lineage is required.",
                context={},
            )
        )
    if not isinstance(report.get("episodes"), list):
        issues.append(
            ValidationIssue(
                code=EVAL_EPISODE_CATALOG_FAILED,
                message="episode_catalog.episodes must be present.",
                context={},
            )
        )
    return issues


def _validate_split_report(report: dict[str, Any] | None, run_id: str) -> list[ValidationIssue]:
    """Validate split report prerequisites."""

    if report is None:
        return [
            ValidationIssue(
                code=EVAL_SPLIT_REPORT_REQUIRED,
                message="split_report is required.",
                context={},
            )
        ]
    issues: list[ValidationIssue] = []
    if report.get("run_id") != run_id:
        issues.append(
            ValidationIssue(
                code=EVAL_RUN_ID_MISMATCH,
                message="split_report.run_id mismatch.",
                context={"split_report_run_id": report.get("run_id"), "cli_run_id": run_id},
            )
        )
    if report.get("split_validation_overall") is not True:
        issues.append(
            ValidationIssue(
                code=EVAL_SPLIT_REPORT_FAILED,
                message="split_report must pass before evaluation.",
                context={"split_validation_overall": report.get("split_validation_overall")},
            )
        )
    if not isinstance(report.get("file_reports"), list):
        issues.append(
            ValidationIssue(
                code=EVAL_SPLIT_REPORT_FAILED,
                message="split_report.file_reports must be present.",
                context={},
            )
        )
    if "split_mode" not in report:
        issues.append(
            ValidationIssue(
                code=EVAL_SPLIT_REPORT_REQUIRED,
                message="split_report.split_mode is required.",
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
    """Validate lineage consistency between explicit state manifest and upstream reports."""

    if state_manifest_payload is None or env_contract_report is None or readiness_report is None or episode_catalog is None:
        return []
    actual_state_manifest_sha = _sha256_file(state_manifest_path)
    expected_path = str(state_manifest_path.resolve())
    issues: list[ValidationIssue] = []

    env_lineage = env_contract_report.get("source_lineage", {})
    readiness_lineage = readiness_report.get("env_contract_reference", {}).get("source_lineage", {})
    catalog_lineage = episode_catalog.get("source_lineage", {})

    for label, lineage in (
        ("env_contract_report", env_lineage),
        ("training_env_readiness_report", readiness_lineage),
        ("episode_catalog", catalog_lineage),
    ):
        if lineage.get("state_manifest_path") != expected_path:
            issues.append(
                ValidationIssue(
                    code=EVAL_CONFIG_INVALID,
                    message=f"{label} lineage does not match the explicit state_manifest path.",
                    context={"reported_path": lineage.get("state_manifest_path"), "expected_path": expected_path},
                )
            )
        if lineage.get("state_manifest_hash") != actual_state_manifest_sha:
            issues.append(
                ValidationIssue(
                    code=EVAL_CONFIG_INVALID,
                    message=f"{label} lineage hash does not match the explicit state_manifest file.",
                    context={"reported_hash": lineage.get("state_manifest_hash"), "actual_hash": actual_state_manifest_sha},
                )
            )
    return issues


def _resolve_targets(
    *,
    eval_config: EvalConfig,
    episode_catalog: dict[str, Any] | None,
    split_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve explicit targets against upstream catalog and split evidence."""

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    if episode_catalog is None or split_report is None:
        return {"resolution": None, "errors": errors, "warnings": warnings}

    catalog_episodes = episode_catalog.get("episodes")
    if not isinstance(catalog_episodes, list):
        return {"resolution": None, "errors": errors, "warnings": warnings}

    alias_events: list[dict[str, Any]] = []
    episodes_by_key: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    for item in catalog_episodes:
        if not isinstance(item, Mapping):
            continue
        episode_ref = item.get("episode_ref")
        if not isinstance(episode_ref, Mapping):
            continue
        try:
            ref = _parse_episode_ref(episode_ref)
        except ValueError:
            continue
        episodes_by_key[_episode_ref_key(ref)] = dict(item)

    resolved_refs: list[EpisodeRef] = []
    selected_partition: str | None = None
    resolved_partition_name: str | None = None

    if eval_config.target_mode == TARGET_MODE_EXPLICIT_PARTITION:
        requested_partition = eval_config.target_partition
        resolved_partition_name = _resolve_partition_name(
            requested_partition=requested_partition,
            alias_events=alias_events,
            path="eval_config.target_partition",
        )
        selected_partition = requested_partition
        matching_all = [
            _parse_episode_ref(item["episode_ref"])
            for item in catalog_episodes
            if isinstance(item, Mapping)
            and isinstance(item.get("episode_ref"), Mapping)
            and item["episode_ref"].get("partition") == resolved_partition_name
            and (eval_config.target_fold_id is None or item["episode_ref"].get("fold_id") == eval_config.target_fold_id)
        ]
        eligible = [
            ref
            for ref in matching_all
            if bool(episodes_by_key[_episode_ref_key(ref)].get("eligible_for_readiness", False))
        ]
        if not matching_all:
            errors.append(
                ValidationIssue(
                    code=EVAL_TARGET_INVALID,
                    message="Explicit partition/fold target did not resolve to any upstream episode.",
                    context={"target_partition": requested_partition, "target_fold_id": eval_config.target_fold_id},
                )
            )
        elif not eligible:
            errors.append(
                ValidationIssue(
                    code=EVAL_TARGET_NOT_ELIGIBLE,
                    message="Resolved partition/fold target exists but is not eligible for evaluation.",
                    context={"target_partition": requested_partition, "target_fold_id": eval_config.target_fold_id},
                )
            )
        resolved_refs = eligible
    else:
        for index, ref in enumerate(eval_config.target_episode_refs):
            try:
                normalized_ref = _normalize_eval_episode_ref(ref=ref, alias_events=alias_events, index=index)
            except ValueError as exc:
                errors.append(
                    ValidationIssue(
                        code=EVAL_TARGET_INVALID,
                        message="Explicit episode ref could not be normalized.",
                        context={"index": index, "error": str(exc), "episode_ref": _episode_ref_to_dict(ref)},
                    )
                )
                continue
            catalog_entry = episodes_by_key.get(_episode_ref_key(normalized_ref))
            if catalog_entry is None:
                errors.append(
                    ValidationIssue(
                        code=EVAL_TARGET_INVALID,
                        message="Explicit episode ref was not found in the upstream episode catalog.",
                        context={"episode_ref": _episode_ref_to_dict(ref)},
                    )
                )
                continue
            if not bool(catalog_entry.get("eligible_for_readiness", False)):
                errors.append(
                    ValidationIssue(
                        code=EVAL_TARGET_NOT_ELIGIBLE,
                        message="Explicit episode ref is not eligible for evaluation.",
                        context={"episode_ref": _episode_ref_to_dict(ref)},
                    )
                )
                continue
            if normalized_ref.partition not in {PARTITION_VAL, PARTITION_TEST}:
                errors.append(
                    ValidationIssue(
                        code=EVAL_TARGET_INVALID,
                        message="Evaluation targets must belong to validation/test domain only.",
                        context={"episode_ref": _episode_ref_to_dict(ref)},
                    )
                )
                continue
            if eval_config.target_fold_id is not None and normalized_ref.fold_id != eval_config.target_fold_id:
                errors.append(
                    ValidationIssue(
                        code=EVAL_TARGET_INVALID,
                        message="Explicit episode ref does not match target_fold_id.",
                        context={
                            "episode_ref": _episode_ref_to_dict(ref),
                            "target_fold_id": eval_config.target_fold_id,
                        },
                    )
                )
                continue
            resolved_refs.append(normalized_ref)
        if len(set(_episode_ref_key(ref) for ref in resolved_refs)) != len(resolved_refs):
            errors.append(
                ValidationIssue(
                    code=EVAL_TARGET_INVALID,
                    message="Explicit episode refs must be unique after alias normalization.",
                    context={"target_episode_refs": [_episode_ref_to_dict(ref) for ref in eval_config.target_episode_refs]},
                )
            )

    if eval_config.evaluation_mode == EVALUATION_MODE_SINGLE_PATH and len(resolved_refs) != 1:
        errors.append(
            ValidationIssue(
                code=EVAL_TARGET_INVALID,
                message="single_path_backtest must resolve to exactly one concrete target.",
                context={"resolved_target_count": len(resolved_refs)},
            )
        )
    if eval_config.evaluation_mode == EVALUATION_MODE_EPISODIC:
        if not resolved_refs:
            errors.append(
                ValidationIssue(
                    code=EVAL_TARGET_INVALID,
                    message="episodic_eval_backtest must resolve to a non-empty explicit target set.",
                    context={},
                )
            )
        elif len(resolved_refs) > eval_config.max_eval_episodes:
            errors.append(
                ValidationIssue(
                    code=EVAL_TARGET_INVALID,
                    message="Resolved episodic target set exceeds max_eval_episodes; silent truncation is forbidden.",
                    context={
                        "resolved_target_count": len(resolved_refs),
                        "max_eval_episodes": eval_config.max_eval_episodes,
                    },
                )
            )

    split_targets_checked: list[dict[str, Any]] = []
    for ref in resolved_refs:
        split_check = _validate_split_target(ref=ref, split_report=split_report)
        split_targets_checked.append(split_check["detail"])
        if split_check["issue"] is not None:
            errors.append(split_check["issue"])

    if alias_events:
        warnings.append(
            ValidationIssue(
                code=ALIAS_WARNING_CODE,
                message="The single supported validation->val compatibility alias was applied explicitly.",
                context={"alias_events": alias_events},
            )
        )

    resolution = TargetResolution(
        selected_episode_refs=tuple(_episode_ref_to_dict(ref) for ref in resolved_refs),
        selected_partition=selected_partition,
        selected_fold_id=eval_config.target_fold_id,
        resolved_partition_name=resolved_partition_name,
        alias_events=tuple(alias_events),
        split_targets_checked=tuple(split_targets_checked),
    )
    return {"resolution": resolution, "errors": errors, "warnings": warnings}


def _validate_split_target(*, ref: EpisodeRef, split_report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one resolved target against upstream split evidence."""

    file_reports = split_report.get("file_reports")
    fold_reports = split_report.get("fold_reports")
    split_mode = split_report.get("split_mode")
    detail = {
        "episode_ref": _episode_ref_to_dict(ref),
        "split_mode": split_mode,
        "matched_scope": None,
    }
    if not isinstance(file_reports, list):
        return {
            "detail": detail,
            "issue": ValidationIssue(
                code=EVAL_SPLIT_REPORT_FAILED,
                message="split_report.file_reports is invalid during target resolution.",
                context={"episode_ref": _episode_ref_to_dict(ref)},
            ),
        }
    if ref.scope == "fold":
        if not isinstance(fold_reports, list):
            return {
                "detail": detail,
                "issue": ValidationIssue(
                    code=EVAL_SPLIT_REPORT_FAILED,
                    message="split_report.fold_reports is required for fold-scoped evaluation targets.",
                    context={"episode_ref": _episode_ref_to_dict(ref)},
                ),
            }
        for item in fold_reports:
            if not isinstance(item, Mapping):
                continue
            input_file = str(item.get("input_file", ""))
            if Path(input_file).name != ref.source_rel:
                continue
            if item.get("fold_id") != ref.fold_id:
                continue
            range_key = f"{ref.partition}_range"
            if item.get(range_key) is not None:
                detail["matched_scope"] = "fold_reports"
                return {"detail": detail, "issue": None}
        return {
            "detail": detail,
            "issue": ValidationIssue(
                code=EVAL_TARGET_INVALID,
                message="Resolved fold-scoped target was not found in split_report.fold_reports.",
                context={"episode_ref": _episode_ref_to_dict(ref)},
            ),
        }

    for item in file_reports:
        if not isinstance(item, Mapping):
            continue
        input_file = str(item.get("input_file", ""))
        if Path(input_file).name != ref.source_rel:
            continue
        range_key = f"{ref.partition}_range"
        if item.get(range_key) is not None:
            detail["matched_scope"] = "file_reports"
            return {"detail": detail, "issue": None}
    return {
        "detail": detail,
        "issue": ValidationIssue(
            code=EVAL_TARGET_INVALID,
            message="Resolved target was not found in split_report for the requested evaluation partition.",
            context={"episode_ref": _episode_ref_to_dict(ref)},
        ),
    }


def _evaluate_single_episode(
    *,
    model: Any,
    env_client: TradingEnvGym,
    episode_ref: dict[str, Any],
    deterministic: bool,
    seed: int,
    benchmark_mode: str,
    requested_metrics: Sequence[str],
    episode_index: int,
) -> EpisodeRuntime:
    """Evaluate one isolated environment instance and return machine-readable evidence."""

    try:
        observation, _ = env_client.reset(seed=seed)
    except Exception as exc:  # noqa: BLE001
        raise ControlledEvaluationFailure(
            ValidationIssue(
                code=EVAL_EXECUTION_FAILED,
                message="Evaluation reset failed.",
                context={"episode_ref": dict(episode_ref), "error": str(exc)},
            )
        ) from exc

    episode_data = getattr(getattr(env_client, "_validation", None), "episode_data", None)
    if episode_data is None:
        raise ControlledEvaluationFailure(
            ValidationIssue(
                code=EVAL_EXECUTION_FAILED,
                message="Evaluation env did not expose episode_data required for backtest evidence.",
                context={"episode_ref": dict(episode_ref)},
            )
        )

    step_records: list[dict[str, Any]] = []
    position_open: dict[str, Any] | None = None
    closed_trade_proxy_returns: list[float] = []
    closed_trade_pnls: list[float] = []
    step_counter = 0
    while True:
        try:
            raw_action = model.predict(observation, deterministic=deterministic)
        except Exception as exc:  # noqa: BLE001
            raise ControlledEvaluationFailure(
                ValidationIssue(
                    code=EVAL_EXECUTION_FAILED,
                    message="Model predict() failed during evaluation.",
                    context={"episode_ref": dict(episode_ref), "error": str(exc), "step_counter": step_counter},
                )
            ) from exc
        action = _normalize_action(raw_action)
        try:
            observation, reward, terminated, truncated, info = env_client.step(action)
        except Exception as exc:  # noqa: BLE001
            raise ControlledEvaluationFailure(
                ValidationIssue(
                    code=EVAL_EXECUTION_FAILED,
                    message="Environment step() failed during evaluation.",
                    context={"episode_ref": dict(episode_ref), "error": str(exc), "step_counter": step_counter},
                )
            ) from exc

        info_dict = dict(info)
        current_index = int(info_dict["step_index"])
        next_index = current_index + 1
        current_timestamp = str(episode_data.timestamp_vector[current_index])
        next_timestamp = str(episode_data.timestamp_vector[next_index])
        trade_units = int(info_dict["cost_components"]["trade_units"])
        position_before = int(info_dict["position_before"])
        position_after = int(info_dict["position_after"])
        price_exec = float(info_dict["price_exec"])
        fees = float(info_dict["cost_components"]["fees"])
        slippage_cost = float(info_dict["cost_components"]["slippage_cost"])
        record = {
            "evaluation_episode_index": int(episode_index),
            "episode_scope": episode_ref["scope"],
            "episode_partition": episode_ref["partition"],
            "episode_source_rel": episode_ref["source_rel"],
            "episode_fold_id": episode_ref["fold_id"],
            "step_ordinal": int(step_counter),
            "step_index": current_index,
            "timestamp": current_timestamp,
            "next_timestamp": next_timestamp,
            "action_raw": int(info_dict["action_raw"]),
            "action_semantic": str(info_dict["action_semantic"]),
            "position_before": position_before,
            "position_after": position_after,
            "invalid_action": bool(info_dict["invalid_action"]),
            "invalid_action_reason": info_dict.get("invalid_action_reason"),
            "price_exec": price_exec,
            "next_price": float(episode_data.mark_to_market_price_vector[next_index]),
            "reward_total": float(reward),
            "pnl_delta": float(info_dict["reward_components"]["pnl_delta"]),
            "fees": fees,
            "slippage_cost": slippage_cost,
            "trade_units": trade_units,
            "strategy_portfolio_value": float(info_dict["portfolio_value"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": info_dict.get("termination_reason"),
            "truncation_reason": info_dict.get("truncation_reason"),
        }
        step_records.append(record)

        if position_before == 0 and position_after in {-1, 1} and trade_units > 0:
            position_open = {
                "direction": position_after,
                "entry_price_exec": price_exec,
                "entry_fees": fees,
                "entry_slippage_cost": slippage_cost,
            }
        elif position_before in {-1, 1} and position_after == 0 and trade_units > 0 and position_open is not None:
            entry_price_exec = float(position_open["entry_price_exec"])
            direction = int(position_open["direction"])
            net_pnl = (
                float(direction) * (price_exec - entry_price_exec)
                - float(position_open["entry_fees"])
                - float(position_open["entry_slippage_cost"])
                - fees
                - slippage_cost
            )
            closed_trade_pnls.append(net_pnl)
            closed_trade_proxy_returns.append(net_pnl / entry_price_exec if entry_price_exec > 0.0 else math.nan)
            position_open = None

        step_counter += 1
        if terminated or truncated:
            break

    strategy_values, strategy_status = _compute_metric_values(
        step_records=step_records,
        initial_cash=float(env_client._config.initial_cash),  # type: ignore[attr-defined]
        requested_metrics=requested_metrics,
        proxy_trade_returns=closed_trade_proxy_returns,
        closed_trade_pnls=closed_trade_pnls,
        metric_kind="strategy",
    )

    benchmark_values: dict[str, float | int | None] | None = None
    benchmark_status: dict[str, dict[str, Any]] | None = None
    if benchmark_mode == BENCHMARK_MODE_BUY_AND_HOLD:
        benchmark_trace = _build_buy_and_hold_trace(
            step_records=step_records,
            initial_cash=float(env_client._config.initial_cash),  # type: ignore[attr-defined]
            fee_bps=float(env_client._config.fee_bps),  # type: ignore[attr-defined]
            slippage_bps=float(env_client._config.slippage_bps),  # type: ignore[attr-defined]
        )
        if benchmark_trace["error"] is not None:
            raise ControlledEvaluationFailure(benchmark_trace["error"])
        benchmark_values, benchmark_status = _compute_metric_values(
            step_records=benchmark_trace["step_records"],
            initial_cash=float(env_client._config.initial_cash),  # type: ignore[attr-defined]
            requested_metrics=requested_metrics,
            proxy_trade_returns=[],
            closed_trade_pnls=[],
            metric_kind="benchmark",
        )
        for record, benchmark_record in zip(step_records, benchmark_trace["step_records"], strict=True):
            record["benchmark_equity"] = float(benchmark_record["strategy_portfolio_value"])

    return EpisodeRuntime(
        episode_ref=dict(episode_ref),
        step_records=tuple(step_records),
        strategy_metric_values=strategy_values,
        strategy_metric_status=strategy_status,
        benchmark_metric_values=benchmark_values,
        benchmark_metric_status=benchmark_status,
        closed_trade_proxy_returns=tuple(float(item) for item in closed_trade_proxy_returns if math.isfinite(item)),
        closed_trade_pnls=tuple(float(item) for item in closed_trade_pnls),
    )


def _build_buy_and_hold_trace(
    *,
    step_records: Sequence[Mapping[str, Any]],
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    """Build a capital-normalized buy-and-hold benchmark trace."""

    if not step_records:
        return {
            "step_records": [],
            "error": ValidationIssue(
                code=EVAL_BENCHMARK_FAILED,
                message="Benchmark cannot be computed from an empty step trace.",
                context={},
            ),
        }
    entry_price = float(step_records[0]["price_exec"])
    if entry_price <= 0.0:
        return {
            "step_records": [],
            "error": ValidationIssue(
                code=EVAL_BENCHMARK_FAILED,
                message="Benchmark requires a strictly positive entry price.",
                context={"entry_price": entry_price},
            ),
        }

    entry_cost_multiplier = 1.0 + float(fee_bps) / 10_000.0 + float(slippage_bps) / 10_000.0
    if entry_cost_multiplier <= 0.0:
        return {
            "step_records": [],
            "error": ValidationIssue(
                code=EVAL_BENCHMARK_FAILED,
                message="Benchmark entry cost multiplier must be positive.",
                context={"entry_cost_multiplier": entry_cost_multiplier},
            ),
        }
    units = float(initial_cash) / (entry_price * entry_cost_multiplier)
    benchmark_records: list[dict[str, Any]] = []
    for index, record in enumerate(step_records):
        next_price = float(record["next_price"])
        benchmark_equity = units * next_price
        benchmark_records.append(
            {
                **dict(record),
                "strategy_portfolio_value": benchmark_equity,
                "reward_total": benchmark_equity if index == 0 else benchmark_equity - benchmark_records[index - 1]["strategy_portfolio_value"],
                "position_before": 1,
                "position_after": 1,
                "action_raw": 1 if index == 0 else 0,
                "action_semantic": "BUY_AND_HOLD",
                "invalid_action": False,
                "invalid_action_reason": None,
                "trade_units": 0,
                "fees": 0.0,
                "slippage_cost": 0.0,
                "benchmark_units": units,
                "benchmark_entry_price": entry_price,
                "benchmark_metric_policy": "capital_normalized_buy_and_hold_v1",
            }
        )
    return {"step_records": benchmark_records, "error": None}


def _compute_metric_values(
    *,
    step_records: Sequence[Mapping[str, Any]],
    initial_cash: float,
    requested_metrics: Sequence[str],
    proxy_trade_returns: Sequence[float],
    closed_trade_pnls: Sequence[float],
    metric_kind: str,
) -> tuple[dict[str, float | int | None], dict[str, dict[str, Any]]]:
    """Compute the supported v1 metric surface for one realized path."""

    metric_values: dict[str, float | int | None] = {}
    metric_status: dict[str, dict[str, Any]] = {}

    equity_points = [float(initial_cash)] + [float(item["strategy_portfolio_value"]) for item in step_records]
    timestamps = [str(step_records[0]["timestamp"])] + [str(item["next_timestamp"]) for item in step_records]
    num_steps = int(len(step_records))
    num_trades = int(len(closed_trade_pnls))
    total_return = (float(equity_points[-1]) / float(initial_cash)) - 1.0
    final_equity = float(equity_points[-1])

    total_return_status = _supported_metric_status(
        formula_id="total_return_v1",
        detail={"formula": "(final_equity / initial_cash) - 1"},
    )
    final_equity_status = _supported_metric_status(
        formula_id="final_equity_v1",
        detail={"formula": "last_equity_point"},
    )
    num_steps_status = _supported_metric_status(
        formula_id="num_steps_v1",
        detail={"aggregation_method": "sum"},
    )
    num_trades_status = _supported_metric_status(
        formula_id="num_trades_v1",
        detail={"trade_definition": "closed_round_trip_count", "aggregation_method": "sum"},
    )

    metric_values["total_return"] = float(total_return)
    metric_status["total_return"] = total_return_status
    metric_values["final_equity"] = final_equity
    metric_status["final_equity"] = final_equity_status
    metric_values["num_steps"] = num_steps
    metric_status["num_steps"] = num_steps_status
    metric_values["num_trades"] = num_trades
    metric_status["num_trades"] = num_trades_status

    horizon_seconds, median_step_seconds = _resolve_horizon_seconds(timestamps)
    annualized_return = _unsupported_metric_status(
        reason_code="UNSUPPORTED_HORIZON",
        detail={"horizon_seconds": horizon_seconds},
    )
    annualized_return_value: float | None = None
    if horizon_seconds is not None and horizon_seconds > 0.0 and final_equity > 0.0 and initial_cash > 0.0:
        try:
            annualized_return_value = (final_equity / initial_cash) ** (SECONDS_PER_YEAR / horizon_seconds) - 1.0
            if not math.isfinite(annualized_return_value):
                raise OverflowError("annualized_return produced a non-finite value")
            annualized_return = _supported_metric_status(
                formula_id="annualized_return_v1",
                detail={"formula": "(final_equity / initial_cash) ** (seconds_per_year / horizon_seconds) - 1"},
            )
        except OverflowError:
            annualized_return_value = None
            annualized_return = _unsupported_metric_status(
                reason_code="UNSUPPORTED_ANNUALIZATION_OVERFLOW",
                detail={"horizon_seconds": horizon_seconds, "final_equity": final_equity, "initial_cash": initial_cash},
            )
    metric_values["annualized_return"] = annualized_return_value
    metric_status["annualized_return"] = annualized_return

    returns = _pct_returns(equity_points)
    annualized_volatility_status = _unsupported_metric_status(
        reason_code="UNSUPPORTED_INSUFFICIENT_RETURNS",
        detail={"return_count": len(returns), "median_step_seconds": median_step_seconds},
    )
    annualized_volatility_value: float | None = None
    if len(returns) >= 2 and median_step_seconds is not None and median_step_seconds > 0.0:
        std = float(np.std(np.asarray(returns, dtype=np.float64), ddof=1))
        annualized_volatility_value = std * math.sqrt(SECONDS_PER_YEAR / median_step_seconds)
        annualized_volatility_status = _supported_metric_status(
            formula_id="annualized_volatility_v1",
            detail={"formula": "std(step_returns, ddof=1) * sqrt(seconds_per_year / median_step_seconds)"},
        )
    metric_values["annualized_volatility"] = annualized_volatility_value
    metric_status["annualized_volatility"] = annualized_volatility_status

    sharpe_status = _unsupported_metric_status(
        reason_code="UNSUPPORTED_ZERO_OR_INVALID_VOLATILITY",
        detail={"annualized_volatility": annualized_volatility_value},
    )
    sharpe_value: float | None = None
    if annualized_volatility_value is not None and annualized_volatility_value > 0.0 and median_step_seconds is not None and returns:
        mean_return = float(np.mean(np.asarray(returns, dtype=np.float64)))
        sharpe_value = mean_return / (annualized_volatility_value / math.sqrt(SECONDS_PER_YEAR / median_step_seconds))
        sharpe_value = sharpe_value * math.sqrt(SECONDS_PER_YEAR / median_step_seconds)
        sharpe_status = _supported_metric_status(
            formula_id="sharpe_ratio_v1",
            detail={"formula": "mean(step_returns) / std(step_returns, ddof=1) * sqrt(seconds_per_year / median_step_seconds)"},
        )
    metric_values["sharpe_ratio"] = sharpe_value
    metric_status["sharpe_ratio"] = sharpe_status

    max_drawdown_value = _max_drawdown(equity_points)
    metric_values["max_drawdown"] = max_drawdown_value
    metric_status["max_drawdown"] = _supported_metric_status(
        formula_id="max_drawdown_v1",
        detail={"formula": "abs(min(equity / rolling_peak - 1))"},
    )

    calmar_value: float | None = None
    if annualized_return_value is not None and max_drawdown_value > 0.0:
        calmar_value = annualized_return_value / max_drawdown_value
        calmar_status = _supported_metric_status(
            formula_id="calmar_ratio_v1",
            detail={"formula": "annualized_return / max_drawdown"},
        )
    else:
        calmar_status = _unsupported_metric_status(
            reason_code="UNSUPPORTED_ZERO_DRAWDOWN_OR_ANNUALIZED_RETURN",
            detail={"annualized_return": annualized_return_value, "max_drawdown": max_drawdown_value},
        )
    metric_values["calmar_ratio"] = calmar_value
    metric_status["calmar_ratio"] = calmar_status

    if num_trades == 0:
        metric_values["win_rate"] = None
        metric_status["win_rate"] = _unsupported_metric_status(
            reason_code="UNSUPPORTED_NO_TRADES",
            detail={"num_trades": num_trades},
        )
        metric_values["avg_trade_return"] = None
        metric_status["avg_trade_return"] = _unsupported_metric_status(
            reason_code="UNSUPPORTED_NO_TRADES",
            detail={
                "num_trades": num_trades,
                "metric_policy": "narrow_v1_proxy",
                "formula": "mean(closed_trade_net_pnl / entry_price_exec)",
            },
        )
    else:
        win_rate_value = float(sum(1 for item in closed_trade_pnls if item > 0.0) / num_trades)
        avg_trade_return_value = float(np.mean(np.asarray(proxy_trade_returns, dtype=np.float64)))
        metric_values["win_rate"] = win_rate_value
        metric_status["win_rate"] = _supported_metric_status(
            formula_id="win_rate_v1",
            detail={"formula": "winning_closed_trades / num_trades", "aggregation_method": "pooled"},
        )
        metric_values["avg_trade_return"] = avg_trade_return_value
        metric_status["avg_trade_return"] = _supported_metric_status(
            formula_id="avg_trade_return_v1_proxy",
            detail={
                "metric_policy": "narrow_v1_proxy",
                "formula": "mean(closed_trade_net_pnl / entry_price_exec)",
                "aggregation_method": "pooled",
            },
        )

    filtered_values = {name: metric_values.get(name) for name in requested_metrics}
    filtered_status = {name: metric_status.get(name, _unsupported_metric_status(reason_code="UNREQUESTED", detail={})) for name in requested_metrics}
    return filtered_values, filtered_status


def _aggregate_metric_values(
    *,
    episode_runtimes: Sequence[EpisodeRuntime],
    metric_names: Sequence[str],
    metric_kind: str,
) -> tuple[dict[str, float | int | None], dict[str, dict[str, Any]]]:
    """Aggregate strategy or benchmark metrics across the evaluated set."""

    values: dict[str, float | int | None] = {}
    status: dict[str, dict[str, Any]] = {}

    source_values: list[dict[str, float | int | None]] = []
    source_statuses: list[dict[str, dict[str, Any]]] = []
    if metric_kind == "strategy":
        source_values = [item.strategy_metric_values for item in episode_runtimes]
        source_statuses = [item.strategy_metric_status for item in episode_runtimes]
        pooled_trade_returns = [value for item in episode_runtimes for value in item.closed_trade_proxy_returns]
        pooled_trade_pnls = [value for item in episode_runtimes for value in item.closed_trade_pnls]
    else:
        source_values = [item.benchmark_metric_values or {} for item in episode_runtimes]
        source_statuses = [item.benchmark_metric_status or {} for item in episode_runtimes]
        pooled_trade_returns = []
        pooled_trade_pnls = []

    for metric_name in metric_names:
        if metric_name in {"num_steps", "num_trades"}:
            supported_values = [
                int(item[metric_name])
                for item, item_status in zip(source_values, source_statuses, strict=True)
                if metric_name in item and item_status.get(metric_name, {}).get("supported") is True and item[metric_name] is not None
            ]
            values[metric_name] = int(sum(supported_values))
            status[metric_name] = _supported_metric_status(
                formula_id=f"{metric_name}_aggregate_v1",
                detail={"aggregation_method": "sum"},
            )
            continue

        if metric_name == "win_rate":
            num_trades = len(pooled_trade_pnls)
            if num_trades == 0:
                values[metric_name] = None
                status[metric_name] = _unsupported_metric_status(
                    reason_code="UNSUPPORTED_NO_TRADES",
                    detail={"num_trades": 0, "aggregation_method": "pooled"},
                )
            else:
                values[metric_name] = float(sum(1 for item in pooled_trade_pnls if item > 0.0) / num_trades)
                status[metric_name] = _supported_metric_status(
                    formula_id="win_rate_aggregate_v1",
                    detail={"aggregation_method": "pooled"},
                )
            continue

        if metric_name == "avg_trade_return":
            if not pooled_trade_returns:
                values[metric_name] = None
                status[metric_name] = _unsupported_metric_status(
                    reason_code="UNSUPPORTED_NO_TRADES",
                    detail={
                        "num_trades": 0,
                        "metric_policy": "narrow_v1_proxy",
                        "formula": "mean(closed_trade_net_pnl / entry_price_exec)",
                        "aggregation_method": "pooled",
                    },
                )
            else:
                values[metric_name] = float(np.mean(np.asarray(pooled_trade_returns, dtype=np.float64)))
                status[metric_name] = _supported_metric_status(
                    formula_id="avg_trade_return_aggregate_v1_proxy",
                    detail={
                        "metric_policy": "narrow_v1_proxy",
                        "formula": "mean(closed_trade_net_pnl / entry_price_exec)",
                        "aggregation_method": "pooled",
                    },
                )
            continue

        supported_values = [
            float(item[metric_name])
            for item, item_status in zip(source_values, source_statuses, strict=True)
            if metric_name in item and item_status.get(metric_name, {}).get("supported") is True and item[metric_name] is not None
        ]
        if not supported_values:
            values[metric_name] = None
            status[metric_name] = _unsupported_metric_status(
                reason_code="UNSUPPORTED_NO_SUPPORTED_EPISODES",
                detail={"aggregation_method": "mean"},
            )
            continue
        values[metric_name] = float(np.mean(np.asarray(supported_values, dtype=np.float64)))
        status[metric_name] = _supported_metric_status(
            formula_id=f"{metric_name}_aggregate_v1",
            detail={"aggregation_method": "mean"},
        )

    return values, status


def _compute_relative_metrics(
    *,
    strategy_metrics: Mapping[str, float | int | None],
    strategy_status: Mapping[str, Mapping[str, Any]],
    benchmark_metrics: Mapping[str, float | int | None],
    benchmark_status: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    """Compute relative metrics from already aggregated strategy/benchmark surfaces."""

    metric_values: dict[str, float | None] = {}
    metric_status: dict[str, dict[str, Any]] = {}
    mapping = {
        "excess_total_return": "total_return",
        "excess_annualized_return": "annualized_return",
        "excess_sharpe_ratio": "sharpe_ratio",
        "excess_max_drawdown_delta": "max_drawdown",
    }
    for relative_metric, base_metric in mapping.items():
        strategy_supported = strategy_status.get(base_metric, {}).get("supported") is True
        benchmark_supported = benchmark_status.get(base_metric, {}).get("supported") is True
        if not strategy_supported or not benchmark_supported:
            metric_values[relative_metric] = None
            metric_status[relative_metric] = _unsupported_metric_status(
                reason_code="UNSUPPORTED_BASE_METRIC",
                detail={
                    "base_metric": base_metric,
                    "relative_metric_policy": "aggregate_strategy_minus_aggregate_benchmark",
                },
            )
            continue
        strategy_value = float(strategy_metrics[base_metric]) if strategy_metrics[base_metric] is not None else None
        benchmark_value = float(benchmark_metrics[base_metric]) if benchmark_metrics[base_metric] is not None else None
        if strategy_value is None or benchmark_value is None:
            metric_values[relative_metric] = None
            metric_status[relative_metric] = _unsupported_metric_status(
                reason_code="UNSUPPORTED_BASE_METRIC",
                detail={
                    "base_metric": base_metric,
                    "relative_metric_policy": "aggregate_strategy_minus_aggregate_benchmark",
                },
            )
            continue
        metric_values[relative_metric] = float(strategy_value - benchmark_value)
        metric_status[relative_metric] = _supported_metric_status(
            formula_id=f"{relative_metric}_v1",
            detail={
                "relative_metric_policy": "aggregate_strategy_minus_aggregate_benchmark",
                "strategy_aggregate_field": base_metric,
                "benchmark_aggregate_field": base_metric,
            },
        )
    return metric_values, metric_status


def _write_core_reports(
    *,
    validation_payload: dict[str, Any],
    manifest_payload: dict[str, Any] | None,
    backtest_payload: dict[str, Any] | None,
    report_paths: ReportPaths,
) -> None:
    """Atomically write the core JSON reports."""

    atomic_write_json(validation_payload, report_paths.validation_report_path)
    if manifest_payload is not None:
        atomic_write_json(manifest_payload, report_paths.manifest_path)
    if backtest_payload is not None:
        atomic_write_json(backtest_payload, report_paths.backtest_report_path)


def _write_reports_with_trace(
    *,
    validation_payload: dict[str, Any],
    manifest_payload: dict[str, Any] | None,
    backtest_payload: dict[str, Any] | None,
    report_paths: ReportPaths,
    step_rows: Sequence[Mapping[str, Any]],
    write_step_trace: bool,
) -> bool:
    """Write reports and the optional parquet trace."""

    try:
        if write_step_trace:
            trace_frame = pd.DataFrame(step_rows)
            atomic_write_parquet(trace_frame, report_paths.step_trace_path)
        _write_core_reports(
            validation_payload=validation_payload,
            manifest_payload=manifest_payload,
            backtest_payload=backtest_payload,
            report_paths=report_paths,
        )
        return True
    except RuntimeError as exc:
        LOGGER.error("Evaluation report write failed | error=%s", exc)
        return False


def _best_effort_write_json(payload: dict[str, Any] | None, path: Path) -> None:
    """Best-effort JSON write used only during report-write recovery."""

    if payload is None:
        return
    try:
        atomic_write_json(payload, path)
    except RuntimeError:
        LOGGER.error("Best-effort evaluation report write also failed | path=%s", path)


def _validation_check(*, check_name: str, passed: bool, reason_code: str | None, detail: dict[str, Any]) -> dict[str, Any]:
    """Build a stable validation check payload."""

    return {
        "check_name": check_name,
        "pass": bool(passed),
        "reason_code": reason_code,
        "detail": detail,
    }


def _build_validation_checks(
    *,
    load_issues: Sequence[ValidationIssue],
    model_issues: Sequence[ValidationIssue],
    all_errors: Sequence[ValidationIssue],
    alias_warnings: Sequence[ValidationIssue],
) -> list[dict[str, Any]]:
    """Build stable validation checks for the validation report."""

    checks: list[dict[str, Any]] = []
    missing_inputs = [item for item in load_issues if item.code == EVAL_INPUT_MISSING]
    unreadable_inputs = [item for item in load_issues if item.code == EVAL_PATH_UNREADABLE]
    invalid_json = [item for item in load_issues if item.code == EVAL_JSON_INVALID]
    checks.append(
        _validation_check(
            check_name="required_inputs_present",
            passed=not missing_inputs,
            reason_code=missing_inputs[0].code if missing_inputs else None,
            detail={"missing_inputs": [item.context for item in missing_inputs]},
        )
    )
    checks.append(
        _validation_check(
            check_name="required_paths_readable",
            passed=not unreadable_inputs,
            reason_code=unreadable_inputs[0].code if unreadable_inputs else None,
            detail={"unreadable_inputs": [item.context for item in unreadable_inputs]},
        )
    )
    checks.append(
        _validation_check(
            check_name="required_json_valid",
            passed=not invalid_json,
            reason_code=invalid_json[0].code if invalid_json else None,
            detail={"invalid_json_inputs": [item.context for item in invalid_json]},
        )
    )
    checks.append(
        _validation_check(
            check_name="model_artifact_canonical_zip",
            passed=not model_issues,
            reason_code=model_issues[0].code if model_issues else None,
            detail={"model_artifact_issues": [asdict(item) for item in model_issues]},
        )
    )
    checks.append(
        _validation_check(
            check_name="contract_validation_passed",
            passed=not all_errors,
            reason_code=all_errors[0].code if all_errors else None,
            detail={"failure_codes": _failure_codes(all_errors)},
        )
    )
    alias_warning = next((item for item in alias_warnings if item.code == ALIAS_WARNING_CODE), None)
    checks.append(
        _validation_check(
            check_name="partition_alias_compatibility_rule",
            passed=True,
            reason_code=alias_warning.code if alias_warning is not None else None,
            detail=alias_warning.context if alias_warning is not None else {"alias_events": []},
        )
    )
    return checks


def _build_validation_payload(
    *,
    run_id: str,
    evaluation_session_id: str,
    selected_algorithm: str | None,
    deterministic: bool | None,
    effective_seed: int | None,
    requested_device: str | None,
    resolved_device: str | None,
    model_artifact_hash: str | None,
    eval_config_hash: str | None,
    readiness_hash: str | None,
    env_contract_hash: str | None,
    state_manifest_hash: str | None,
    episode_catalog_hash: str | None,
    split_report_hash: str | None,
    validation_checks: Sequence[dict[str, Any]],
    warnings: Sequence[ValidationIssue],
    errors: Sequence[ValidationIssue],
) -> dict[str, Any]:
    """Build evaluation validation report."""

    return {
        "run_id": run_id,
        "evaluation_session_id": evaluation_session_id,
        "overall_pass": len(errors) == 0,
        "selected_algorithm": selected_algorithm,
        "deterministic": deterministic,
        "effective_seed": effective_seed,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "model_artifact_hash": model_artifact_hash,
        "eval_config_hash": eval_config_hash,
        "readiness_hash": readiness_hash,
        "env_contract_hash": env_contract_hash,
        "state_manifest_hash": state_manifest_hash,
        "episode_catalog_hash": episode_catalog_hash,
        "split_report_hash": split_report_hash,
        "validation_checks": list(validation_checks),
        "warnings": [asdict(item) for item in warnings],
        "errors": [asdict(item) for item in errors],
        "failure_codes": _failure_codes(errors),
        "generated_at": _generated_at(),
    }


def _build_manifest_payload(
    *,
    run_id: str,
    evaluation_session_id: str,
    model_artifact_path: Path,
    env_config_path: Path,
    eval_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
    selected_algorithm: str | None,
    deterministic: bool | None,
    effective_seed: int | None,
    requested_device: str | None,
    resolved_device: str | None,
    evaluation_mode: str | None,
    target_mode: str | None,
    benchmark_mode: str | None,
    target_resolution: TargetResolution | None,
    model_artifact_hash: str | None,
    eval_config_hash: str | None,
    readiness_hash: str | None,
    env_contract_hash: str | None,
    state_manifest_hash: str | None,
    episode_catalog_hash: str | None,
    split_report_hash: str | None,
    output_dir: Path,
    warnings: Sequence[ValidationIssue],
) -> dict[str, Any]:
    """Build evaluation manifest payload."""

    alias_warning = next((item for item in warnings if item.code == ALIAS_WARNING_CODE), None)
    return {
        "run_id": run_id,
        "evaluation_session_id": evaluation_session_id,
        "model_artifact_path": str(model_artifact_path),
        "env_config_path": str(env_config_path),
        "eval_config_path": str(eval_config_path),
        "source_artifacts": {
            "state_manifest_path": str(state_manifest_path),
            "env_contract_report_path": str(env_contract_report_path),
            "readiness_report_path": str(readiness_report_path),
            "episode_catalog_path": str(episode_catalog_path),
            "split_report_path": str(split_report_path),
        },
        "selected_algorithm": selected_algorithm,
        "deterministic": deterministic,
        "effective_seed": effective_seed,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "evaluation_mode": evaluation_mode,
        "target_mode": target_mode,
        "selected_partition": target_resolution.selected_partition if target_resolution is not None else None,
        "selected_fold_id": target_resolution.selected_fold_id if target_resolution is not None else None,
        "selected_episode_refs": list(target_resolution.selected_episode_refs) if target_resolution is not None else [],
        "benchmark_mode": benchmark_mode,
        "lineages": {
            "hash_policy": {
                "algorithm": "sha256",
                "canonical_json": CANONICAL_JSON_POLICY,
                "binary_file_policy": "raw_file_bytes",
            },
            "partition_alias_resolution": alias_warning.context.get("alias_events", []) if alias_warning is not None else [],
            "target_resolution": {
                "resolved_partition_name": target_resolution.resolved_partition_name if target_resolution is not None else None,
                "split_targets_checked": list(target_resolution.split_targets_checked) if target_resolution is not None else [],
                "relative_metrics_policy": "aggregate_strategy_minus_aggregate_benchmark",
                "avg_trade_return_policy": {
                    "metric_policy": "narrow_v1_proxy",
                    "formula": "mean(closed_trade_net_pnl / entry_price_exec)",
                },
                "benchmark_policy": "capital_normalized_buy_and_hold_v1",
                "aggregation_policy": {
                    "episode_metric_aggregation": "mean_for_non_count_metrics",
                    "count_metric_aggregation": "sum",
                    "trade_metric_aggregation": "pooled_closed_trades",
                    "relative_metric_aggregation": "aggregate_strategy_minus_aggregate_benchmark",
                },
            },
        },
        "model_artifact_hash": model_artifact_hash,
        "eval_config_hash": eval_config_hash,
        "readiness_hash": readiness_hash,
        "env_contract_hash": env_contract_hash,
        "state_manifest_hash": state_manifest_hash,
        "episode_catalog_hash": episode_catalog_hash,
        "split_report_hash": split_report_hash,
        "output_dir": str(output_dir),
        "generated_at": _generated_at(),
    }


def _build_backtest_payload(
    *,
    run_id: str,
    evaluation_session_id: str,
    evaluation_success: bool,
    selected_algorithm: str | None,
    deterministic: bool | None,
    effective_seed: int | None,
    evaluation_mode: str | None,
    target_mode: str | None,
    benchmark_mode: str | None,
    startup_phase_trace: Sequence[dict[str, Any]],
    strategy_metrics: dict[str, Any] | None,
    benchmark_metrics: dict[str, Any] | None,
    relative_metrics: dict[str, Any] | None,
    metric_status: dict[str, Any],
    trace_artifact_path: str | None,
    warnings: Sequence[ValidationIssue],
    errors: Sequence[ValidationIssue],
) -> dict[str, Any]:
    """Build evaluation backtest report payload."""

    return {
        "run_id": run_id,
        "evaluation_session_id": evaluation_session_id,
        "evaluation_success": bool(evaluation_success),
        "selected_algorithm": selected_algorithm,
        "deterministic": deterministic,
        "effective_seed": effective_seed,
        "evaluation_mode": evaluation_mode,
        "target_mode": target_mode,
        "benchmark_mode": benchmark_mode,
        "startup_phase_trace": list(startup_phase_trace),
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "relative_metrics": relative_metrics,
        "metric_status": metric_status,
        "trace_artifact_path": trace_artifact_path,
        "warnings": [asdict(item) for item in warnings],
        "errors": [asdict(item) for item in errors],
        "failure_codes": _failure_codes(errors),
        "generated_at": _generated_at(),
    }


def _phase_trace(
    *,
    validation_status: str,
    model_load_status: str,
    env_init_status: str,
    eval_start_status: str,
    eval_finish_status: str,
    report_write_status: str,
    validation_detail: dict[str, Any] | None = None,
    model_load_detail: dict[str, Any] | None = None,
    env_init_detail: dict[str, Any] | None = None,
    eval_start_detail: dict[str, Any] | None = None,
    eval_finish_detail: dict[str, Any] | None = None,
    report_write_detail: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build stable startup phase trace."""

    return [
        {"phase": "validation", "status": validation_status, "detail": validation_detail or {}},
        {"phase": "model_load", "status": model_load_status, "detail": model_load_detail or {}},
        {"phase": "env_init", "status": env_init_status, "detail": env_init_detail or {}},
        {"phase": "eval_start", "status": eval_start_status, "detail": eval_start_detail or {}},
        {"phase": "eval_finish", "status": eval_finish_status, "detail": eval_finish_detail or {}},
        {"phase": "report_write", "status": report_write_status, "detail": report_write_detail or {}},
    ]


def _phase_trace_from_maps(phase_status: Mapping[str, str], phase_detail: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a phase trace from status/detail maps."""

    return _phase_trace(
        validation_status=phase_status["validation"],
        model_load_status=phase_status["model_load"],
        env_init_status=phase_status["env_init"],
        eval_start_status=phase_status["eval_start"],
        eval_finish_status=phase_status["eval_finish"],
        report_write_status=phase_status["report_write"],
        validation_detail=phase_detail.get("validation"),
        model_load_detail=phase_detail.get("model_load"),
        env_init_detail=phase_detail.get("env_init"),
        eval_start_detail=phase_detail.get("eval_start"),
        eval_finish_detail=phase_detail.get("eval_finish"),
        report_write_detail=phase_detail.get("report_write"),
    )


def _effective_env_config(*, env_config: EnvConfig, seed: int, episode_ref: dict[str, Any], max_eval_steps: int) -> EnvConfig:
    """Build an in-memory evaluation env config without mutating the source file."""

    effective_max_steps = max_eval_steps
    if env_config.max_steps is not None:
        effective_max_steps = min(int(env_config.max_steps), int(max_eval_steps))

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
        "max_steps": effective_max_steps,
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


def _load_ppo_model(*, model_artifact_path: Path, device: str | None) -> Any:
    """Load the explicit SB3 PPO model artifact."""

    module = importlib.import_module("stable_baselines3")
    ppo_class = getattr(module, "PPO", None)
    if ppo_class is None:
        raise ImportError("stable_baselines3.PPO is unavailable")
    return ppo_class.load(str(model_artifact_path), device=device)


def _check_output_dir_policy(output_dir: Path) -> list[ValidationIssue]:
    """Fail closed when output_dir already exists in any form."""

    issues: list[ValidationIssue] = []
    if not output_dir.exists():
        return issues
    if output_dir.is_file():
        issues.append(
            ValidationIssue(
                code=EVAL_OUTPUT_CONFLICT,
                message="startup_policy=fresh_only requires a brand-new output_dir path.",
                context={"output_dir": str(output_dir), "path_kind": "file"},
            )
        )
        return issues
    if output_dir.is_dir():
        entries = sorted(path.name for path in output_dir.iterdir())
        issues.append(
            ValidationIssue(
                code=EVAL_OUTPUT_CONFLICT,
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
            code=EVAL_OUTPUT_CONFLICT,
            message="output_dir already exists with an unsupported filesystem type.",
            context={"output_dir": str(output_dir), "path_kind": "other"},
        )
    )
    return issues


def _resolve_partition_name(
    *,
    requested_partition: str | None,
    alias_events: list[dict[str, Any]],
    path: str,
) -> str | None:
    """Resolve the single supported compatibility alias."""

    if requested_partition is None:
        return None
    if requested_partition == PARTITION_VALIDATION:
        alias_event = {
            **PARTITION_ALIAS_RULE,
            "path": path,
        }
        alias_events.append(alias_event)
        return PARTITION_VAL
    return requested_partition


def _normalize_eval_episode_ref(*, ref: EpisodeRef, alias_events: list[dict[str, Any]], index: int) -> EpisodeRef:
    """Normalize explicit evaluation episode refs with the single alias rule."""

    resolved_partition = _resolve_partition_name(
        requested_partition=ref.partition,
        alias_events=alias_events,
        path=f"eval_config.target_episode_refs[{index}].partition",
    )
    if resolved_partition not in {PARTITION_VAL, PARTITION_TEST}:
        raise ValueError("episode_ref.partition must resolve to val or test for evaluation")
    return EpisodeRef(scope=ref.scope, partition=resolved_partition, source_rel=ref.source_rel, fold_id=ref.fold_id)


def _parse_episode_ref(payload: Any) -> EpisodeRef:
    """Parse a strict episode ref payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("episode_ref must be an object")
    expected_keys = {"scope", "partition", "source_rel", "fold_id"}
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise ValueError(f"episode_ref keys must be exactly {sorted(expected_keys)}")
    scope = payload.get("scope")
    partition = payload.get("partition")
    source_rel = payload.get("source_rel")
    fold_id = payload.get("fold_id")
    if not isinstance(scope, str):
        raise ValueError("episode_ref.scope must be string")
    if not isinstance(partition, str):
        raise ValueError("episode_ref.partition must be string")
    if not isinstance(source_rel, str):
        raise ValueError("episode_ref.source_rel must be string")
    if fold_id is not None and (not isinstance(fold_id, int) or fold_id < 0):
        raise ValueError("episode_ref.fold_id must be int >= 0 or null")
    return EpisodeRef(scope=scope, partition=partition, source_rel=source_rel, fold_id=fold_id if isinstance(fold_id, int) else None)


def _normalize_action(raw_action: Any) -> int:
    """Normalize model predict output into one discrete action."""

    action = raw_action[0] if isinstance(raw_action, tuple) else raw_action
    if isinstance(action, np.ndarray):
        if action.size != 1:
            raise ControlledEvaluationFailure(
                ValidationIssue(
                    code=EVAL_EXECUTION_FAILED,
                    message="Model predict returned a non-scalar action array.",
                    context={"shape": list(action.shape)},
                )
            )
        return int(action.reshape(-1)[0])
    if isinstance(action, (list, tuple)):
        if len(action) != 1:
            raise ControlledEvaluationFailure(
                ValidationIssue(
                    code=EVAL_EXECUTION_FAILED,
                    message="Model predict returned a non-scalar action sequence.",
                    context={"length": len(action)},
                )
            )
        return int(action[0])
    return int(action)


def _pct_returns(equity_points: Sequence[float]) -> list[float]:
    """Return finite percentage returns for the supplied equity points."""

    returns: list[float] = []
    for previous, current in zip(equity_points[:-1], equity_points[1:], strict=True):
        if previous <= 0.0:
            return []
        value = (current / previous) - 1.0
        if not math.isfinite(value):
            return []
        returns.append(float(value))
    return returns


def _resolve_horizon_seconds(timestamps: Sequence[str]) -> tuple[float | None, float | None]:
    """Resolve exact horizon and median step size from timestamp evidence."""

    if len(timestamps) < 2:
        return None, None
    parsed = pd.to_datetime(list(timestamps), utc=True, errors="coerce")
    if parsed.isna().any():
        return None, None
    deltas = np.diff(parsed.view("int64")) / 1_000_000_000.0
    finite_deltas = [float(delta) for delta in deltas if float(delta) > 0.0 and math.isfinite(float(delta))]
    if not finite_deltas:
        return None, None
    horizon = float((parsed[-1] - parsed[0]).total_seconds())
    if horizon <= 0.0:
        return None, float(np.median(np.asarray(finite_deltas, dtype=np.float64)))
    return horizon, float(np.median(np.asarray(finite_deltas, dtype=np.float64)))


def _max_drawdown(equity_points: Sequence[float]) -> float:
    """Return drawdown magnitude in positive form."""

    peak = -math.inf
    max_drawdown = 0.0
    for equity in equity_points:
        peak = max(peak, float(equity))
        if peak <= 0.0:
            continue
        drawdown = abs((float(equity) / peak) - 1.0)
        max_drawdown = max(max_drawdown, drawdown)
    return float(max_drawdown)


def _supported_metric_status(*, formula_id: str, detail: dict[str, Any]) -> dict[str, Any]:
    """Build a supported metric status payload."""

    return {
        "supported": True,
        "reason_code": None,
        "detail": {"formula_id": formula_id, **detail},
    }


def _unsupported_metric_status(*, reason_code: str, detail: dict[str, Any]) -> dict[str, Any]:
    """Build an unsupported metric status payload."""

    return {
        "supported": False,
        "reason_code": reason_code,
        "detail": detail,
    }


def _build_evaluation_session_id(
    *,
    run_id: str,
    output_dir: Path,
    model_artifact_path: Path,
    env_config_path: Path,
    eval_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
) -> str:
    """Build a deterministic session id for this invocation."""

    payload = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "model_artifact_path": str(model_artifact_path),
        "env_config_path": str(env_config_path),
        "eval_config_path": str(eval_config_path),
        "state_manifest_path": str(state_manifest_path),
        "env_contract_report_path": str(env_contract_report_path),
        "readiness_report_path": str(readiness_report_path),
        "episode_catalog_path": str(episode_catalog_path),
        "split_report_path": str(split_report_path),
    }
    return _hash_canonical_json(payload)[:16]


def _optional_import(module_name: str) -> tuple[Any | None, str | None]:
    """Import a module optionally and surface the error as text."""

    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _resolve_device(requested_device: str | None) -> tuple[str | None, list[ValidationIssue], dict[str, Any]]:
    """Resolve explicit requested device into an effective runtime device."""

    issues: list[ValidationIssue] = []
    dependency_probe = {
        "torch_available": None,
        "torch_error": None,
        "torch_cuda_available": None,
        "gymnasium_available": None,
        "stable_baselines3_available": None,
    }
    if requested_device is None:
        return None, issues, dependency_probe
    if requested_device == DEVICE_CPU:
        return DEVICE_CPU, issues, dependency_probe
    torch_module, torch_error = _optional_import("torch")
    torch_available = torch_module is not None
    cuda_available = bool(torch_available and bool(torch_module.cuda.is_available()))
    dependency_probe.update(
        {
            "torch_available": torch_available,
            "torch_error": torch_error,
            "torch_cuda_available": cuda_available,
        }
    )
    if requested_device == DEVICE_AUTO:
        return DEVICE_CUDA if cuda_available else DEVICE_CPU, issues, dependency_probe
    if requested_device == DEVICE_CUDA:
        if not torch_available or not cuda_available:
            issues.append(
                ValidationIssue(
                    code=EVAL_DEVICE_INVALID,
                    message="Requested cuda device is unavailable.",
                    context={"torch_available": torch_available, "cuda_available": cuda_available},
                )
            )
            return None, issues, dependency_probe
        return DEVICE_CUDA, issues, dependency_probe
    issues.append(
        ValidationIssue(
            code=EVAL_DEVICE_INVALID,
            message="Requested device is unsupported.",
            context={"requested_device": requested_device},
        )
    )
    return None, issues, dependency_probe


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
    np.random.seed(seed)
    metadata["numpy_seeded"] = True

    torch_module = sys.modules.get("torch")
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


def _close_envs(env_clients: Sequence[TradingEnvGym]) -> None:
    """Close all environment clients best-effort."""

    for env_client in env_clients:
        try:
            env_client.close()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Evaluation env close failed")


def _failure_codes(issues: Sequence[ValidationIssue]) -> list[str]:
    """Return stable unique failure codes in first-seen order."""

    seen: set[str] = set()
    codes: list[str] = []
    for issue in issues:
        if issue.code not in seen:
            seen.add(issue.code)
            codes.append(issue.code)
    return codes


def _hash_canonical_json(payload: Any) -> str:
    """Hash canonical JSON content."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _semantic_hash_optional(payload: dict[str, Any] | None) -> str | None:
    """Hash JSON payloads using canonical JSON semantics."""

    if payload is None:
        return None
    return _hash_canonical_json(payload)


def _sha256_file(path: Path) -> str:
    """Hash raw file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _episode_ref_to_dict(ref: EpisodeRef) -> dict[str, Any]:
    """Serialize an episode ref into a JSON-ready mapping."""

    return {
        "scope": ref.scope,
        "partition": ref.partition,
        "source_rel": ref.source_rel,
        "fold_id": ref.fold_id,
    }


def _episode_ref_key(ref: EpisodeRef) -> tuple[str, str, str, int | None]:
    """Return a stable episode ref key."""

    return (ref.scope, ref.partition, ref.source_rel, ref.fold_id)


def _generated_at() -> str:
    """Return a stable UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


def _raw_string(payload: dict[str, Any] | None, key: str) -> str | None:
    """Best-effort string extraction for partially invalid payloads."""

    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _raw_int(payload: dict[str, Any] | None, key: str) -> int | None:
    """Best-effort integer extraction for partially invalid payloads."""

    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, int) else None


def _raw_bool(payload: dict[str, Any] | None, key: str) -> bool | None:
    """Best-effort bool extraction for partially invalid payloads."""

    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, bool) else None
