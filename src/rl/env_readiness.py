"""Training env readiness orchestration for Milestone 4.6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rl.env_adapter_gym import TradingEnvGym
from rl.env_contract import EnvConfig, parse_env_config, validate_env_contract
from rl.env_core import (
    ACTION_CLOSE_POSITION,
    ACTION_HOLD,
    ACTION_MAPPING,
    ACTION_OPEN_LONG,
    EpisodeRunnerConfig,
    EpisodeRunnerCore,
)
from rl.episode_catalog import (
    EPISODE_CATALOG_VERSION,
    EpisodeCatalogResult,
    build_episode_catalog,
    _episode_ref_to_dict,
    _hash_canonical_json,
)
from rl.episode_selector import (
    SELECTION_POLICY_FIXED,
    select_episode,
)

ENV_READINESS_VERSION = "training_env_readiness.v1"
ROLLOUT_HASH_VERSION = "readiness_rollout_hash.v1"
ROLLOUT_CANONICALIZATION_POLICY = "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=True)"
SMOKE_ACTION_SCRIPT_VERSION = "readiness_smoke_actions.v1"

READINESS_CONFIG_INVALID = "READINESS_CONFIG_INVALID"
READINESS_CATALOG_NOT_READY = "READINESS_CATALOG_NOT_READY"
READINESS_SELECTED_EPISODE_COUNT_INVALID = "READINESS_SELECTED_EPISODE_COUNT_INVALID"
READINESS_SELECTION_FAILED = "READINESS_SELECTION_FAILED"
READINESS_MIN_REMAINING_STEPS_FAILED = "READINESS_MIN_REMAINING_STEPS_FAILED"
READINESS_ENV_CONTRACT_FAILED = "READINESS_ENV_CONTRACT_FAILED"
READINESS_RESET_REPLAY_FAILED = "READINESS_RESET_REPLAY_FAILED"
READINESS_ROLLOUT_REPLAY_FAILED = "READINESS_ROLLOUT_REPLAY_FAILED"
READINESS_START_POLICY_UNSUPPORTED = "READINESS_START_POLICY_UNSUPPORTED"

START_POLICY_VALID_FROM_ROW = "start_at_valid_from_row"

SMOKE_ACTION_SCRIPT = {
    "version": SMOKE_ACTION_SCRIPT_VERSION,
    "name": "fixed_readiness_smoke_v1",
    "actions": [ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_POSITION],
    "action_names": [ACTION_MAPPING[ACTION_HOLD], ACTION_MAPPING[ACTION_OPEN_LONG], ACTION_MAPPING[ACTION_CLOSE_POSITION]],
}


@dataclass
class ValidationIssue:
    """Machine-readable readiness issue."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvReadinessResult:
    """Composite result containing catalog and readiness payloads."""

    catalog_payload: dict[str, Any]
    readiness_payload: dict[str, Any]


def validate_training_env_readiness(
    *,
    run_id: str,
    state_root: Path,
    env_config_payload: Mapping[str, Any],
    selection_policy: str,
    start_policy: str,
    min_remaining_steps: int,
    seed: int,
) -> EnvReadinessResult:
    """Validate training env readiness over deterministic episode orchestration."""

    state_root = state_root.resolve()
    catalog_result = build_episode_catalog(run_id=run_id, state_root=state_root)

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if start_policy != START_POLICY_VALID_FROM_ROW:
        errors.append(
            ValidationIssue(
                code=READINESS_START_POLICY_UNSUPPORTED,
                message="Unsupported start policy.",
                context={"start_policy": start_policy},
            )
        )
    if min_remaining_steps <= 0:
        errors.append(
            ValidationIssue(
                code=READINESS_CONFIG_INVALID,
                message="min_remaining_steps must be > 0.",
                context={"min_remaining_steps": min_remaining_steps},
            )
        )

    env_config: EnvConfig | None = None
    try:
        payload_copy = dict(env_config_payload)
        payload_copy.setdefault("run_id", run_id)
        payload_copy.setdefault("state_root", str(state_root))
        env_config = parse_env_config(payload_copy)
    except ValueError as exc:
        errors.append(
            ValidationIssue(
                code=READINESS_CONFIG_INVALID,
                message="Invalid env config payload.",
                context={"error": str(exc)},
            )
        )

    if not bool(catalog_result.payload.get("episode_catalog_overall", False)):
        errors.append(
            ValidationIssue(
                code=READINESS_CATALOG_NOT_READY,
                message="Episode catalog must succeed before readiness validation.",
                context={},
            )
        )

    fixed_episode_ref = env_config.episode_ref if env_config is not None else None
    selection_result = select_episode(
        catalog=catalog_result,
        selection_policy=selection_policy,
        seed=seed,
        fixed_episode_ref=fixed_episode_ref,
    )
    for item in selection_result.errors:
        errors.append(ValidationIssue(code=item.code, message=item.message, context=dict(item.context)))

    selected_entry = selection_result.selected_entry
    selected_episode_ref = selection_result.selected_episode_ref
    selected_episode_refs = [_episode_ref_to_dict(selected_episode_ref)] if selected_episode_ref is not None else []
    selected_episode_domain_validity = _selected_episode_domain_validity(selected_entry, selection_result.eligible_domain_used)

    if selected_episode_ref is not None and selection_result.eligible_domain_used == "readiness" and not bool(selected_entry and selected_entry.eligible_for_readiness):
        errors.append(
            ValidationIssue(
                code=READINESS_SELECTION_FAILED,
                message="Selected fixed episode is not readiness-eligible.",
                context={"episode_ref": _episode_ref_to_dict(selected_episode_ref)},
            )
        )
    if selected_episode_ref is not None and selection_result.eligible_domain_used == "training" and not bool(selected_entry and selected_entry.eligible_for_training):
        errors.append(
            ValidationIssue(
                code=READINESS_SELECTION_FAILED,
                message="Selected seeded-random episode is not training-eligible.",
                context={"episode_ref": _episode_ref_to_dict(selected_episode_ref)},
            )
        )

    if selected_episode_ref is not None and len(selected_episode_refs) != 1:
        errors.append(
            ValidationIssue(
                code=READINESS_SELECTED_EPISODE_COUNT_INVALID,
                message="Successful readiness flow must carry exactly one selected episode ref.",
                context={"selected_episode_refs": selected_episode_refs},
            )
        )

    selection_replay_match = False
    if not selection_result.errors:
        replay = select_episode(
            catalog=catalog_result,
            selection_policy=selection_policy,
            seed=seed,
            fixed_episode_ref=fixed_episode_ref,
        )
        selection_replay_match = (
            replay.trace == selection_result.trace
            and replay.candidate_refs_sorted == selection_result.candidate_refs_sorted
        )
        if not selection_replay_match:
            errors.append(
                ValidationIssue(
                    code=READINESS_SELECTION_FAILED,
                    message="Selection replay did not match the initial deterministic selection trace.",
                    context={"initial_trace": selection_result.trace, "replay_trace": replay.trace},
                )
            )

    min_remaining_steps_requested = int(min_remaining_steps)
    min_remaining_steps_effective = max(int(min_remaining_steps_requested), int(len(SMOKE_ACTION_SCRIPT["actions"])))
    usable_step_count_after_warmup = int(selected_entry.usable_step_count_after_warmup) if selected_entry is not None else 0
    max_steps_configured = int(env_config.max_steps) if env_config is not None and env_config.max_steps is not None else None
    min_remaining_steps_guard_passed = False
    if selected_entry is not None:
        min_remaining_steps_guard_passed = usable_step_count_after_warmup >= min_remaining_steps_effective
        if max_steps_configured is not None and max_steps_configured < min_remaining_steps_effective:
            min_remaining_steps_guard_passed = False

    if selected_entry is not None and not min_remaining_steps_guard_passed:
        errors.append(
            ValidationIssue(
                code=READINESS_MIN_REMAINING_STEPS_FAILED,
                message="Selected episode does not satisfy the required minimum remaining steps.",
                context={
                    "usable_step_count_after_warmup": usable_step_count_after_warmup,
                    "min_remaining_steps_requested": min_remaining_steps_requested,
                    "min_remaining_steps_effective": min_remaining_steps_effective,
                    "max_steps_configured": max_steps_configured,
                },
            )
        )

    env_validation_payload: dict[str, Any] | None = None
    selected_config: EnvConfig | None = None
    if env_config is not None and selected_episode_ref is not None and not errors:
        selected_config = replace(env_config, episode_ref=selected_episode_ref)
        env_validation = validate_env_contract(
            config=selected_config,
            smoke_step=False,
            invocation_args={
                "selection_policy": selection_policy,
                "start_policy": start_policy,
                "seed": seed,
            },
        )
        env_validation_payload = env_validation.report_payload
        if not bool(env_validation.report_payload.get("env_contract_overall", False)):
            errors.append(
                ValidationIssue(
                    code=READINESS_ENV_CONTRACT_FAILED,
                    message="Selected episode failed the existing 4.5 env contract validation path.",
                    context={"env_contract_errors": env_validation.report_payload.get("errors", [])},
                )
            )

    reset_trace: dict[str, Any] = {
        "requested_seed": int(seed),
        "effective_seed": None,
        "episode_ref": _episode_ref_to_dict(selected_episode_ref) if selected_episode_ref is not None else None,
        "usable_start_row": int(selected_entry.usable_start_row) if selected_entry is not None else None,
        "usable_start_timestamp": selected_entry.usable_start_timestamp if selected_entry is not None else None,
    }
    smoke_rollout_trace_summary = {
        "executed_step_count": 0,
        "first_observation_shape": None,
        "first_observation_dtype": None,
        "reward_finite_all": False,
        "termination_seen": False,
        "truncation_seen": False,
        "rollout_trace_hash": None,
        "rollout_hash_version": ROLLOUT_HASH_VERSION,
        "rollout_hash_inputs": [
            "action_script.version",
            "action_script.actions",
            "per_step.step_index",
            "per_step.action_raw",
            "per_step.reward",
            "per_step.terminated",
            "per_step.truncated",
            "per_step.position_after",
            "per_step.portfolio_value",
            "per_step.invalid_action",
            "per_step.invalid_action_reason",
            "per_step.timestamp",
            "per_step.observation_hash",
        ],
        "canonicalization_policy": ROLLOUT_CANONICALIZATION_POLICY,
    }

    reset_replay_match = False
    rollout_replay_match = False
    deterministic_reset_ok = False
    observation_shape_ok = False
    observation_dtype_ok = False
    reward_finite_ok = False
    termination_semantics_ok = False

    if selected_config is not None and not errors:
        run_a = _run_reset_and_smoke_rollout(
            config=selected_config,
            seed=seed,
            usable_start_timestamp=selected_entry.usable_start_timestamp if selected_entry is not None else None,
        )
        run_b = _run_reset_and_smoke_rollout(
            config=selected_config,
            seed=seed,
            usable_start_timestamp=selected_entry.usable_start_timestamp if selected_entry is not None else None,
        )

        reset_trace = dict(run_a["reset_trace"])
        smoke_rollout_trace_summary = dict(run_a["smoke_rollout_trace_summary"])
        selection_replay_match = bool(selection_replay_match)
        reset_replay_match = run_a["reset_trace"] == run_b["reset_trace"]
        rollout_replay_match = run_a["smoke_rollout_trace_summary"]["rollout_trace_hash"] == run_b["smoke_rollout_trace_summary"]["rollout_trace_hash"]
        deterministic_reset_ok = bool(reset_replay_match)
        observation_shape_ok = bool(run_a["observation_shape_ok"])
        observation_dtype_ok = bool(run_a["observation_dtype_ok"])
        reward_finite_ok = bool(run_a["reward_finite_ok"])
        termination_semantics_ok = bool(run_a["termination_semantics_ok"])

        if not reset_replay_match:
            errors.append(
                ValidationIssue(
                    code=READINESS_RESET_REPLAY_FAILED,
                    message="Reset replay did not produce identical deterministic reset traces.",
                    context={"initial_reset_trace": run_a["reset_trace"], "replay_reset_trace": run_b["reset_trace"]},
                )
            )
        if not rollout_replay_match:
            errors.append(
                ValidationIssue(
                    code=READINESS_ROLLOUT_REPLAY_FAILED,
                    message="Rollout replay did not produce identical deterministic rollout hashes.",
                    context={
                        "initial_rollout_trace_hash": run_a["smoke_rollout_trace_summary"]["rollout_trace_hash"],
                        "replay_rollout_trace_hash": run_b["smoke_rollout_trace_summary"]["rollout_trace_hash"],
                    },
                )
            )

    deterministic_replay_match = bool(selection_replay_match and reset_replay_match and rollout_replay_match)
    readiness_overall = len(errors) == 0 and bool(catalog_result.payload.get("episode_catalog_overall", False))

    readiness_payload = {
        "generated_at_utc": _generated_at_utc(),
        "readiness_version": ENV_READINESS_VERSION,
        "episode_catalog_version": EPISODE_CATALOG_VERSION,
        "overall": readiness_overall,
        "readiness_overall": readiness_overall,
        "episode_catalog_overall": bool(catalog_result.payload.get("episode_catalog_overall", False)),
        "run_id": run_id,
        "state_root": str(state_root),
        "selection_policy": selection_policy,
        "start_policy": start_policy,
        "selection_order_policy": dict(catalog_result.payload.get("selection_order_policy", {})),
        "seed": int(seed),
        "eligibility_domain_used": selection_result.eligible_domain_used,
        "candidate_count_under_domain": int(len(selection_result.candidate_refs_sorted)),
        "selected_episode_domain_validity": selected_episode_domain_validity,
        "episodes_considered": int(len(catalog_result.entries)),
        "eligible_episode_count": int(len(selection_result.candidate_refs_sorted)),
        "eligible_episode_refs_sorted_under_domain": list(selection_result.candidate_refs_sorted),
        "eligible_episode_refs_sorted_hash_under_domain": selection_result.trace.get("eligible_episode_refs_sorted_hash"),
        "selected_episode_refs": selected_episode_refs,
        "selection_trace": dict(selection_result.trace),
        "fixed_episode_input_source": selection_result.fixed_episode_input_source,
        "fixed_episode_input_value": selection_result.fixed_episode_input_value,
        "orchestration_input_consumed": bool(selection_policy == SELECTION_POLICY_FIXED and selection_result.fixed_episode_input_source is not None),
        "min_remaining_steps_requested": min_remaining_steps_requested,
        "min_remaining_steps_effective": min_remaining_steps_effective,
        "usable_step_count_after_warmup": usable_step_count_after_warmup,
        "min_remaining_steps_guard_passed": bool(min_remaining_steps_guard_passed),
        "max_steps_configured": max_steps_configured,
        "smoke_action_script": dict(SMOKE_ACTION_SCRIPT),
        "reset_trace": reset_trace,
        "smoke_rollout_trace_summary": smoke_rollout_trace_summary,
        "deterministic_reset_ok": deterministic_reset_ok,
        "selection_replay_match": bool(selection_replay_match),
        "reset_replay_match": bool(reset_replay_match),
        "rollout_replay_match": bool(rollout_replay_match),
        "deterministic_replay_match": deterministic_replay_match,
        "observation_shape_ok": observation_shape_ok,
        "observation_dtype_ok": observation_dtype_ok,
        "reward_finite_ok": reward_finite_ok,
        "termination_semantics_ok": termination_semantics_ok,
        "warnings": [asdict(item) for item in warnings],
        "errors": [asdict(item) for item in errors],
        "env_contract_reference": {
            "env_contract_overall": bool(env_validation_payload.get("env_contract_overall", False)) if env_validation_payload is not None else None,
            "source_lineage": dict(env_validation_payload.get("source_lineage", {})) if env_validation_payload is not None else {},
        },
    }
    if readiness_overall and len(selected_episode_refs) != 1:
        readiness_payload["overall"] = False
        readiness_payload["readiness_overall"] = False
        readiness_payload["errors"].append(
            asdict(
                ValidationIssue(
                    code=READINESS_SELECTED_EPISODE_COUNT_INVALID,
                    message="Successful readiness payload must include exactly one selected episode ref.",
                    context={"selected_episode_refs": selected_episode_refs},
                )
            )
        )

    return EnvReadinessResult(catalog_payload=catalog_result.payload, readiness_payload=readiness_payload)


def _run_reset_and_smoke_rollout(
    *,
    config: EnvConfig,
    seed: int,
    usable_start_timestamp: str | None,
) -> dict[str, Any]:
    """Run deterministic reset and fixed smoke rollout against the Gym adapter."""

    env = _build_env_client(config)
    obs, info = env.reset(seed=seed)
    reset_trace = {
        "requested_seed": int(seed),
        "effective_seed": info.get("seed"),
        "episode_ref": dict(info.get("episode_ref", {})),
        "usable_start_row": int(info.get("effective_episode_start_row")),
        "usable_start_timestamp": usable_start_timestamp,
        "first_observation_hash": _array_hash(obs),
        "first_observation_shape": list(obs.shape),
        "first_observation_dtype": str(obs.dtype),
    }

    per_step: list[dict[str, Any]] = []
    reward_finite_all = True
    termination_seen = False
    truncation_seen = False
    termination_semantics_ok = True
    observation_shape_ok = bool(obs.ndim == 1 and obs.shape == env.observation_space.shape)
    observation_dtype_ok = bool(str(obs.dtype) == str(env.observation_space.dtype))

    for action in SMOKE_ACTION_SCRIPT["actions"]:
        next_obs, reward, terminated, truncated, step_info = env.step(int(action))
        reward_is_finite = bool(np.isfinite(float(reward)))
        reward_finite_all = bool(reward_finite_all and reward_is_finite)
        if bool(terminated and truncated):
            termination_semantics_ok = False
        termination_seen = bool(termination_seen or terminated)
        truncation_seen = bool(truncation_seen or truncated)
        per_step.append(
            {
                "step_index": int(step_info.get("step_index")),
                "action_raw": int(action),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "position_after": int(step_info.get("position_after")),
                "portfolio_value": float(step_info.get("portfolio_value")),
                "invalid_action": bool(step_info.get("invalid_action")),
                "invalid_action_reason": step_info.get("invalid_action_reason"),
                "timestamp": step_info.get("timestamp"),
                "observation_hash": _array_hash(next_obs),
            }
        )
    env.close()

    rollout_trace_hash = _hash_canonical_json(
        {
            "action_script": {"version": SMOKE_ACTION_SCRIPT["version"], "actions": list(SMOKE_ACTION_SCRIPT["actions"])},
            "per_step": per_step,
        }
    )
    smoke_rollout_trace_summary = {
        "executed_step_count": int(len(per_step)),
        "first_observation_shape": list(obs.shape),
        "first_observation_dtype": str(obs.dtype),
        "reward_finite_all": reward_finite_all,
        "termination_seen": termination_seen,
        "truncation_seen": truncation_seen,
        "rollout_trace_hash": rollout_trace_hash,
        "rollout_hash_version": ROLLOUT_HASH_VERSION,
        "rollout_hash_inputs": [
            "action_script.version",
            "action_script.actions",
            "per_step.step_index",
            "per_step.action_raw",
            "per_step.reward",
            "per_step.terminated",
            "per_step.truncated",
            "per_step.position_after",
            "per_step.portfolio_value",
            "per_step.invalid_action",
            "per_step.invalid_action_reason",
            "per_step.timestamp",
            "per_step.observation_hash",
        ],
        "canonicalization_policy": ROLLOUT_CANONICALIZATION_POLICY,
    }
    return {
        "reset_trace": reset_trace,
        "smoke_rollout_trace_summary": smoke_rollout_trace_summary,
        "observation_shape_ok": observation_shape_ok,
        "observation_dtype_ok": observation_dtype_ok,
        "reward_finite_ok": reward_finite_all,
        "termination_semantics_ok": termination_semantics_ok,
    }


def _build_env_client(config: EnvConfig) -> Any:
    """Build a resettable env client, falling back to env core when gymnasium is unavailable."""

    try:
        return TradingEnvGym(config=config, validate_on_init=True)
    except RuntimeError as exc:
        if "gymnasium is required" not in str(exc):
            raise
        validation = validate_env_contract(config=config, smoke_step=False, invocation_args={"fallback_runner": True})
        if not bool(validation.report_payload.get("env_contract_overall", False)) or validation.episode_data is None:
            raise
        return _RunnerEnvAdapter(config=config, episode_data=validation.episode_data)


class _RunnerEnvAdapter:
    """Small adapter that mirrors the Gym reset/step surface over EpisodeRunnerCore."""

    def __init__(self, *, config: EnvConfig, episode_data: Any) -> None:
        self._runner = EpisodeRunnerCore(
            episode_ref=config.episode_ref,
            episode_data=episode_data,
            config=EpisodeRunnerConfig(
                initial_cash=config.initial_cash,
                fee_bps=config.fee_bps,
                slippage_bps=config.slippage_bps,
                max_steps=config.max_steps,
                reward_scale=config.reward_contract.reward_scale,
                reward_clip_min=config.reward_contract.reward_clip_min,
                reward_clip_max=config.reward_contract.reward_clip_max,
                seed=config.seed,
            ),
        )
        self.observation_space = _ObservationSpaceStub(shape=(self._runner.observation_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset runner core with Gym-like signature."""

        obs, info = self._runner.reset(seed=seed)
        return obs.astype(np.float32, copy=False), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step runner core with Gym-like signature."""

        obs, reward, terminated, truncated, info = self._runner.step(int(action))
        return obs.astype(np.float32, copy=False), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """No-op close for readiness fallback."""

        return None


@dataclass(frozen=True)
class _ObservationSpaceStub:
    """Minimal observation-space stub for fallback execution."""

    shape: tuple[int, ...]
    dtype: Any


def _selected_episode_domain_validity(entry: Any, domain: str) -> dict[str, Any]:
    """Build selected episode validity summary for the used domain."""

    if entry is None:
        return {
            "eligible_for_readiness": False,
            "eligible_for_training": False,
            "eligible_under_used_domain": False,
        }
    eligible_under_domain = entry.eligible_for_readiness if domain == "readiness" else entry.eligible_for_training
    return {
        "eligible_for_readiness": bool(entry.eligible_for_readiness),
        "eligible_for_training": bool(entry.eligible_for_training),
        "eligible_under_used_domain": bool(eligible_under_domain),
    }


def _array_hash(values: np.ndarray) -> str:
    """Hash an ndarray deterministically by value."""

    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _generated_at_utc() -> str:
    """Return deterministic UTC timestamp string."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
