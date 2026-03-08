"""Contract-first validation and config schema for RL env Milestone 4.5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl.env_core import (
    ACTION_MAPPING,
    ACTION_HOLD,
    EpisodeCatalog,
    EpisodeData,
    EpisodeRef,
    EpisodeRunnerConfig,
    EpisodeRunnerCore,
    EpisodeSource,
    EpisodeSpec,
)

ENV_ADAPTER_VERSION = "env_adapter_gym.v1"
ENV_CONTRACT_VERSION = "env.contract.v1"

ENV_CONTRACT_PRECONDITION_FAILED = "ENV_CONTRACT_PRECONDITION_FAILED"
ENV_CONTRACT_CONFIG_INVALID = "ENV_CONTRACT_CONFIG_INVALID"
ENV_CONTRACT_STATE_MANIFEST_MISSING = "ENV_CONTRACT_STATE_MANIFEST_MISSING"
ENV_CONTRACT_STATE_MANIFEST_INVALID = "ENV_CONTRACT_STATE_MANIFEST_INVALID"
ENV_CONTRACT_STATE_BUILD_REPORT_MISSING = "ENV_CONTRACT_STATE_BUILD_REPORT_MISSING"
ENV_CONTRACT_STATE_BUILD_REPORT_INVALID = "ENV_CONTRACT_STATE_BUILD_REPORT_INVALID"
ENV_CONTRACT_RUN_ID_MISMATCH = "ENV_CONTRACT_RUN_ID_MISMATCH"
ENV_CONTRACT_STATE_BUILD_NOT_PASSED = "ENV_CONTRACT_STATE_BUILD_NOT_PASSED"
ENV_CONTRACT_OUTPUT_COMPLETENESS_FAILED_UPSTREAM = "ENV_CONTRACT_OUTPUT_COMPLETENESS_FAILED_UPSTREAM"
ENV_CONTRACT_OUTPUT_SEMANTICS_UNSUPPORTED = "ENV_CONTRACT_OUTPUT_SEMANTICS_UNSUPPORTED"
ENV_CONTRACT_LINEAGE_MISMATCH = "ENV_CONTRACT_LINEAGE_MISMATCH"
ENV_CONTRACT_LINEAGE_HASH_MISSING = "ENV_CONTRACT_LINEAGE_HASH_MISSING"
ENV_CONTRACT_EPISODE_REF_INVALID = "ENV_CONTRACT_EPISODE_REF_INVALID"
ENV_CONTRACT_EPISODE_NOT_FOUND = "ENV_CONTRACT_EPISODE_NOT_FOUND"
ENV_CONTRACT_STATE_ARTIFACT_MISSING = "ENV_CONTRACT_STATE_ARTIFACT_MISSING"
ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_MISSING = "ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_MISSING"
ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID = "ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID"
ENV_CONTRACT_RUNTIME_PRICE_CONFIG_MISMATCH = "ENV_CONTRACT_RUNTIME_PRICE_CONFIG_MISMATCH"
ENV_CONTRACT_EXECUTION_PRICE_COLUMN_MISSING = "ENV_CONTRACT_EXECUTION_PRICE_COLUMN_MISSING"
ENV_CONTRACT_MARK_TO_MARKET_COLUMN_MISSING = "ENV_CONTRACT_MARK_TO_MARKET_COLUMN_MISSING"
ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH = "ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH"
ENV_CONTRACT_OBSERVATION_DTYPE_MISMATCH = "ENV_CONTRACT_OBSERVATION_DTYPE_MISMATCH"
ENV_CONTRACT_WARMUP_CONTRACT_MISSING = "ENV_CONTRACT_WARMUP_CONTRACT_MISSING"
ENV_CONTRACT_WARMUP_CONTRACT_INVALID = "ENV_CONTRACT_WARMUP_CONTRACT_INVALID"
ENV_CONTRACT_POST_VALID_OBSERVATION_NAN = "ENV_CONTRACT_POST_VALID_OBSERVATION_NAN"
ENV_CONTRACT_EPISODE_TOO_SHORT_AFTER_WARMUP = "ENV_CONTRACT_EPISODE_TOO_SHORT_AFTER_WARMUP"
ENV_CONTRACT_EFFECTIVE_START_INVALID = "ENV_CONTRACT_EFFECTIVE_START_INVALID"
ENV_CONTRACT_ORDERING_VIOLATION = "ENV_CONTRACT_ORDERING_VIOLATION"
ENV_CONTRACT_TIMESTAMP_DUPLICATES = "ENV_CONTRACT_TIMESTAMP_DUPLICATES"
ENV_CONTRACT_ACTION_CONFIG_INVALID = "ENV_CONTRACT_ACTION_CONFIG_INVALID"
ENV_CONTRACT_TIMING_POLICY_UNSUPPORTED = "ENV_CONTRACT_TIMING_POLICY_UNSUPPORTED"
ENV_CONTRACT_REWARD_POLICY_UNSUPPORTED = "ENV_CONTRACT_REWARD_POLICY_UNSUPPORTED"
ENV_CONTRACT_TERMINATION_POLICY_UNSUPPORTED = "ENV_CONTRACT_TERMINATION_POLICY_UNSUPPORTED"
ENV_CONTRACT_SMOKE_STEP_FAILED = "ENV_CONTRACT_SMOKE_STEP_FAILED"
ENV_CONTRACT_RUNTIME_ERROR = "ENV_CONTRACT_RUNTIME_ERROR"

WARMUP_POLICY_DROP_HEAD = "drop_head_until_all_required_obs_numeric"
WARMUP_POST_VALID_NAN_POLICY = "fail_closed"
CONDITIONAL_COLUMN_POLICY_EXCLUDE_AND_REPLACE = "exclude_from_core_and_replace_with_geometry"


@dataclass
class ValidationIssue:
    """Machine-readable validation issue."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTimingContract:
    """Explicit execution timing policy bundle."""

    observation_timestamp_policy: str
    execution_price_policy: str
    reward_accrual_interval_policy: str
    mark_to_market_policy: str

    def validate_supported(self) -> None:
        """Validate supported v1 timing semantics."""

        if self.observation_timestamp_policy != "row_t":
            raise ValueError("execution_timing_contract.observation_timestamp_policy must be row_t")
        if self.execution_price_policy != "close_t":
            raise ValueError("execution_timing_contract.execution_price_policy must be close_t")
        if self.reward_accrual_interval_policy != "post_action_t_to_t_plus_1":
            raise ValueError("execution_timing_contract.reward_accrual_interval_policy must be post_action_t_to_t_plus_1")
        if self.mark_to_market_policy != "next_row_close":
            raise ValueError("execution_timing_contract.mark_to_market_policy must be next_row_close")


@dataclass(frozen=True)
class ActionSemanticsContract:
    """Explicit action-space and invalid-action policy."""

    action_space_type: str
    action_space_n: int
    invalid_action_policy: str
    reversal_policy: str
    position_model: str

    def validate_supported(self) -> None:
        """Validate v1 action semantics lock."""

        if self.action_space_type != "Discrete" or self.action_space_n != 4:
            raise ValueError("action_semantics_contract must lock Discrete(4)")
        if self.invalid_action_policy != "noop_with_info_flag":
            raise ValueError("action_semantics_contract.invalid_action_policy must be noop_with_info_flag")
        if self.reversal_policy != "disallow_same_step":
            raise ValueError("action_semantics_contract.reversal_policy must be disallow_same_step")
        if self.position_model != "single_position_unit":
            raise ValueError("action_semantics_contract.position_model must be single_position_unit")


@dataclass(frozen=True)
class RewardContract:
    """Reward decomposition policy."""

    reward_version: str
    reward_formula_summary: str
    included_components: tuple[str, ...]
    reward_scale: float
    reward_clip_min: float | None
    reward_clip_max: float | None

    def validate_supported(self) -> None:
        """Validate v1 reward contract."""

        if self.reward_version != "reward.v1":
            raise ValueError("reward_contract.reward_version must be reward.v1")
        expected_formula = "pnl_delta - fees - slippage_cost"
        if self.reward_formula_summary != expected_formula:
            raise ValueError(f"reward_contract.reward_formula_summary must be: {expected_formula}")
        expected_components = ("pnl_delta", "fees", "slippage_cost")
        if self.included_components != expected_components:
            raise ValueError("reward_contract.included_components must be [pnl_delta, fees, slippage_cost]")
        if float(self.reward_scale) <= 0.0:
            raise ValueError("reward_contract.reward_scale must be > 0")
        if self.reward_clip_min is not None and self.reward_clip_max is not None:
            if float(self.reward_clip_min) > float(self.reward_clip_max):
                raise ValueError("reward_contract.reward_clip_min cannot exceed reward_clip_max")


@dataclass(frozen=True)
class TerminationContract:
    """Termination/truncation policy lock."""

    data_end_terminated: bool
    max_steps_truncated: bool

    def validate_supported(self) -> None:
        """Validate v1 termination semantics."""

        if self.data_end_terminated is not True:
            raise ValueError("termination_contract.data_end_terminated must be true")
        if self.max_steps_truncated is not True:
            raise ValueError("termination_contract.max_steps_truncated must be true")


@dataclass(frozen=True)
class EnvConfig:
    """Strict env configuration schema for Milestone 4.5."""

    run_id: str
    state_root: Path
    episode_ref: EpisodeRef
    execution_price_column: str
    mark_to_market_column: str
    include_timestamp_in_observation: bool
    observation_output_dtype: str
    observation_dtype_policy: str
    allowed_safe_casts: tuple[str, ...]
    initial_cash: float
    fee_bps: float
    slippage_bps: float
    max_steps: int | None
    seed: int | None
    execution_timing_contract: ExecutionTimingContract
    action_semantics_contract: ActionSemanticsContract
    reward_contract: RewardContract
    termination_contract: TerminationContract

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.execution_price_column.strip():
            raise ValueError("execution_price_column must be non-empty")
        if not self.mark_to_market_column.strip():
            raise ValueError("mark_to_market_column must be non-empty")
        if self.include_timestamp_in_observation:
            raise ValueError("include_timestamp_in_observation=true is unsupported in v1")
        if self.observation_output_dtype != "float32":
            raise ValueError("observation_output_dtype must be float32 in v1")
        if self.observation_dtype_policy != "strict":
            raise ValueError("observation_dtype_policy must be strict in v1")
        if float(self.initial_cash) <= 0.0:
            raise ValueError("initial_cash must be > 0")
        if float(self.fee_bps) < 0.0:
            raise ValueError("fee_bps must be >= 0")
        if float(self.slippage_bps) < 0.0:
            raise ValueError("slippage_bps must be >= 0")
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError("max_steps must be > 0 when provided")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be int or null")
        self.execution_timing_contract.validate_supported()
        self.action_semantics_contract.validate_supported()
        self.reward_contract.validate_supported()
        self.termination_contract.validate_supported()


@dataclass
class EnvContractValidationResult:
    """Validation output with optional in-memory episode payload."""

    report_payload: dict[str, Any]
    episode_data: EpisodeData | None
    selected_episode: EpisodeSpec | None
    catalog: EpisodeCatalog | None


@dataclass(frozen=True)
class ResolvedEpisodeContract:
    """Resolved per-episode contract payload for loader execution."""

    timestamp_column: str
    observation_columns: tuple[str, ...]
    strict_post_valid_numeric_columns: tuple[str, ...]
    artifact_columns: tuple[str, ...]
    artifact_dtypes: dict[str, str]
    runtime_price_contract: dict[str, Any]
    warmup_contract: dict[str, Any]


def parse_env_config(payload: Mapping[str, Any]) -> EnvConfig:
    """Parse and validate strict env config payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("ENV_CONTRACT_CONFIG_INVALID: env config payload must be object")

    run_id = _require_string(payload, "run_id")
    state_root_raw = _require_string(payload, "state_root")
    episode_ref_payload = _require_mapping(payload, "episode_ref")
    execution_price_column = _require_string(payload, "execution_price_column")
    mark_to_market_column = _require_string(payload, "mark_to_market_column")

    include_timestamp = _require_bool(payload, "include_timestamp_in_observation")
    observation_output_dtype = _require_string(payload, "observation_output_dtype")
    observation_dtype_policy = _require_string(payload, "observation_dtype_policy")
    allowed_safe_casts = tuple(_require_string_list(payload, "allowed_safe_casts"))

    initial_cash = _require_float(payload, "initial_cash")
    fee_bps = _require_float(payload, "fee_bps")
    slippage_bps = _require_float(payload, "slippage_bps")
    max_steps = _require_optional_int(payload, "max_steps")
    seed = _require_optional_int(payload, "seed")

    timing_payload = _require_mapping(payload, "execution_timing_contract")
    action_payload = _require_mapping(payload, "action_semantics_contract")
    reward_payload = _require_mapping(payload, "reward_contract")
    termination_payload = _require_mapping(payload, "termination_contract")

    return EnvConfig(
        run_id=run_id,
        state_root=Path(state_root_raw).resolve(),
        episode_ref=EpisodeRef(
            scope=_require_string(episode_ref_payload, "scope"),
            partition=_require_string(episode_ref_payload, "partition"),
            source_rel=_require_string(episode_ref_payload, "source_rel"),
            fold_id=_require_optional_int(episode_ref_payload, "fold_id"),
        ),
        execution_price_column=execution_price_column,
        mark_to_market_column=mark_to_market_column,
        include_timestamp_in_observation=include_timestamp,
        observation_output_dtype=observation_output_dtype,
        observation_dtype_policy=observation_dtype_policy,
        allowed_safe_casts=allowed_safe_casts,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        max_steps=max_steps,
        seed=seed,
        execution_timing_contract=ExecutionTimingContract(
            observation_timestamp_policy=_require_string(timing_payload, "observation_timestamp_policy"),
            execution_price_policy=_require_string(timing_payload, "execution_price_policy"),
            reward_accrual_interval_policy=_require_string(timing_payload, "reward_accrual_interval_policy"),
            mark_to_market_policy=_require_string(timing_payload, "mark_to_market_policy"),
        ),
        action_semantics_contract=ActionSemanticsContract(
            action_space_type=_require_string(action_payload, "action_space_type"),
            action_space_n=_require_int(action_payload, "action_space_n"),
            invalid_action_policy=_require_string(action_payload, "invalid_action_policy"),
            reversal_policy=_require_string(action_payload, "reversal_policy"),
            position_model=_require_string(action_payload, "position_model"),
        ),
        reward_contract=RewardContract(
            reward_version=_require_string(reward_payload, "reward_version"),
            reward_formula_summary=_require_string(reward_payload, "reward_formula_summary"),
            included_components=tuple(_require_string_list(reward_payload, "included_components")),
            reward_scale=_require_float(reward_payload, "reward_scale"),
            reward_clip_min=_require_optional_float(reward_payload, "reward_clip_min"),
            reward_clip_max=_require_optional_float(reward_payload, "reward_clip_max"),
        ),
        termination_contract=TerminationContract(
            data_end_terminated=_require_bool(termination_payload, "data_end_terminated"),
            max_steps_truncated=_require_bool(termination_payload, "max_steps_truncated"),
        ),
    )


class EnvContractValidator:
    """Contract validator for state lineage and env compatibility."""

    def validate(
        self,
        *,
        config: EnvConfig,
        smoke_step: bool = False,
        invocation_args: Mapping[str, Any] | None = None,
    ) -> EnvContractValidationResult:
        """Validate env contract and optionally run a minimal smoke step."""

        state_manifest_path = config.state_root / "reports" / "state_manifest.json"
        state_build_report_path = config.state_root / "reports" / "state_build_report.json"

        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        preflight_checks: list[dict[str, Any]] = []

        manifest = _load_json_object(
            path=state_manifest_path,
            missing_code=ENV_CONTRACT_STATE_MANIFEST_MISSING,
            invalid_code=ENV_CONTRACT_STATE_MANIFEST_INVALID,
            errors=errors,
        )
        report = _load_json_object(
            path=state_build_report_path,
            missing_code=ENV_CONTRACT_STATE_BUILD_REPORT_MISSING,
            invalid_code=ENV_CONTRACT_STATE_BUILD_REPORT_INVALID,
            errors=errors,
        )

        _check(preflight_checks, "state_manifest_exists", manifest is not None)
        _check(preflight_checks, "state_build_report_exists", report is not None)

        catalog: EpisodeCatalog | None = None
        selected_episode: EpisodeSpec | None = None
        episode_data: EpisodeData | None = None
        runtime_price_contract_payload: dict[str, Any] = {}
        warmup_contract_payload = _default_warmup_contract_payload()
        episode_valid_start_row: int | None = None
        effective_episode_start_row: int | None = None
        warmup_applied = False

        if manifest is not None:
            _validate_run_id("state_manifest.run_id", manifest, config.run_id, errors)
        if report is not None:
            _validate_run_id("state_build_report.run_id", report, config.run_id, errors)

        if report is not None:
            overall = report.get("state_build_overall")
            completeness = report.get("output_completeness_ok")
            if overall is not True:
                errors.append(
                    ValidationIssue(
                        code=ENV_CONTRACT_STATE_BUILD_NOT_PASSED,
                        message="state_build_overall must be true before env validation.",
                        context={"state_build_overall": overall},
                    )
                )
            if completeness is not True:
                errors.append(
                    ValidationIssue(
                        code=ENV_CONTRACT_OUTPUT_COMPLETENESS_FAILED_UPSTREAM,
                        message="state_build_report.output_completeness_ok must be true.",
                        context={"output_completeness_ok": completeness},
                    )
                )

        if manifest is not None:
            manifest_completeness = manifest.get("output_completeness_ok")
            if manifest_completeness is not True:
                errors.append(
                    ValidationIssue(
                        code=ENV_CONTRACT_OUTPUT_COMPLETENESS_FAILED_UPSTREAM,
                        message="state_manifest.output_completeness_ok must be true.",
                        context={"output_completeness_ok": manifest_completeness},
                    )
                )

        if manifest is not None and report is not None:
            _validate_output_semantics(manifest, report, errors)
            _validate_hash_lineage(manifest, report, warnings, errors)

        if manifest is not None:
            try:
                catalog = EpisodeCatalog.from_manifest(manifest)
            except ValueError as exc:
                errors.append(
                    ValidationIssue(
                        code=ENV_CONTRACT_STATE_MANIFEST_INVALID,
                        message="state manifest episode catalog is invalid.",
                        context={"error": str(exc)},
                    )
                )

        if catalog is not None:
            for spec in catalog.entries:
                if not spec.output_path.exists():
                    errors.append(
                        ValidationIssue(
                            code=ENV_CONTRACT_STATE_ARTIFACT_MISSING,
                            message="Referenced state parquet artifact is missing.",
                            context={"output_path": str(spec.output_path), "episode_key": list(spec.key())},
                        )
                    )

            selected_episode = catalog.find_episode(config.episode_ref)
            if selected_episode is None:
                errors.append(
                    ValidationIssue(
                        code=ENV_CONTRACT_EPISODE_NOT_FOUND,
                        message="episode_ref does not match any state manifest entry.",
                        context={"episode_ref": asdict(config.episode_ref)},
                    )
                )

        if manifest is not None and selected_episode is not None:
            episode_entry = _find_episode_manifest_entry(manifest=manifest, selected_episode=selected_episode)
            episode_contract = _resolve_episode_contract(
                manifest=manifest,
                config=config,
                selected_episode=selected_episode,
                episode_entry=episode_entry,
                errors=errors,
            )
            if episode_contract is not None and not errors:
                runtime_price_contract_payload = dict(episode_contract.runtime_price_contract)
                warmup_contract_payload = dict(episode_contract.warmup_contract)
                episode_valid_start_row = int(warmup_contract_payload["valid_from_row"])
                effective_episode_start_row = int(warmup_contract_payload["valid_from_row"])
                warmup_applied = bool(warmup_contract_payload["enabled"])
                source = EpisodeSource()
                try:
                    episode_data = source.load_episode(
                        spec=selected_episode,
                        expected_columns=episode_contract.artifact_columns,
                        observation_columns=episode_contract.observation_columns,
                        strict_post_valid_numeric_columns=episode_contract.strict_post_valid_numeric_columns,
                        expected_dtypes=episode_contract.artifact_dtypes,
                        timestamp_column=episode_contract.timestamp_column,
                        execution_price_column=config.execution_price_column,
                        mark_to_market_column=config.mark_to_market_column,
                        include_timestamp_in_observation=config.include_timestamp_in_observation,
                        observation_output_dtype=config.observation_output_dtype,
                        allowed_safe_casts=set(config.allowed_safe_casts),
                        valid_observation_start_row=episode_valid_start_row,
                        valid_observation_start_timestamp=warmup_contract_payload["valid_from_timestamp"],
                        warmup_head_nan_profile=warmup_contract_payload["head_nan_profile"],
                    )
                except ValueError as exc:
                    msg = str(exc)
                    code = _map_episode_source_error_to_code(msg)
                    errors.append(ValidationIssue(code=code, message="Episode load validation failed.", context={"error": msg}))

        _check(preflight_checks, "warmup_contract_resolved", episode_valid_start_row is not None or selected_episode is None)
        _check(
            preflight_checks,
            "usable_rows_after_warmup",
            episode_data is not None or not any(
                item.code == ENV_CONTRACT_EPISODE_TOO_SHORT_AFTER_WARMUP for item in errors
            ),
        )

        smoke_results: dict[str, Any] = {"executed": bool(smoke_step), "success": False}
        if smoke_step and episode_data is not None and not errors:
            try:
                runner = EpisodeRunnerCore(
                    episode_ref=config.episode_ref,
                    episode_data=episode_data,
                    config=_runner_config_from_env(config),
                )
                runner.reset(seed=config.seed)
                _, reward, terminated, truncated, info = runner.step(ACTION_HOLD)
                smoke_results.update(
                    {
                        "success": True,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "step_info_keys": sorted(info.keys()),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                smoke_results.update({"success": False, "error": str(exc)})
                errors.append(
                    ValidationIssue(
                        code=ENV_CONTRACT_SMOKE_STEP_FAILED,
                        message="Smoke reset/step failed.",
                        context={"error": str(exc)},
                    )
                )

        overall = len(errors) == 0
        report_payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": config.run_id,
            "env_contract_version": ENV_CONTRACT_VERSION,
            "env_adapter_version": ENV_ADAPTER_VERSION,
            "env_contract_overall": overall,
            "state_root": str(config.state_root),
            "source_lineage": {
                "state_manifest_path": str(state_manifest_path),
                "state_build_report_path": str(state_build_report_path),
                "state_manifest_hash": _sha256_file_optional(state_manifest_path),
                "state_build_report_hash": _sha256_file_optional(state_build_report_path),
            },
            "execution_timing_contract": asdict(config.execution_timing_contract),
            "position_action_semantics": {
                **asdict(config.action_semantics_contract),
                "action_mapping": {str(k): v for k, v in ACTION_MAPPING.items()},
            },
            "termination_truncation_semantics": {
                **asdict(config.termination_contract),
                "termination_reason_enum": ["data_exhausted"],
                "truncation_reason_enum": ["max_steps"],
            },
            "reward_contract": asdict(config.reward_contract),
            "runtime_price_contract": runtime_price_contract_payload,
            "warmup_applied": bool(warmup_applied),
            "warmup_contract": warmup_contract_payload,
            "episode_valid_start_row": episode_valid_start_row,
            "effective_episode_start_row": effective_episode_start_row,
            "seed_reproducibility_contract": {
                "seed_reset_supported": True,
                "stochastic_components_present": False,
                "determinism_statement": "same seed + config + episode_ref + actions => identical rollout",
            },
            "observation_space_metadata": _observation_space_metadata(episode_data),
            "action_space_metadata": {
                "action_space_type": "Discrete",
                "action_space_n": 4,
                "action_mapping": {str(k): v for k, v in ACTION_MAPPING.items()},
            },
            "preflight_checks": preflight_checks,
            "coercions_applied": list(episode_data.coercions_applied) if episode_data is not None else [],
            "smoke_results": smoke_results,
            "invocation_args": dict(invocation_args or {}),
            "errors": [asdict(item) for item in errors],
            "warnings": [asdict(item) for item in warnings],
        }
        return EnvContractValidationResult(
            report_payload=report_payload,
            episode_data=episode_data if overall else None,
            selected_episode=selected_episode if overall else None,
            catalog=catalog if overall else None,
        )


def validate_env_contract(
    *,
    config: EnvConfig,
    smoke_step: bool = False,
    invocation_args: Mapping[str, Any] | None = None,
) -> EnvContractValidationResult:
    """Convenience entrypoint to validate env contract."""

    validator = EnvContractValidator()
    return validator.validate(config=config, smoke_step=smoke_step, invocation_args=invocation_args)


def _load_json_object(
    *,
    path: Path,
    missing_code: str,
    invalid_code: str,
    errors: list[ValidationIssue],
) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(ValidationIssue(code=missing_code, message="JSON artifact not found.", context={"path": str(path)}))
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(ValidationIssue(code=invalid_code, message="JSON artifact is invalid.", context={"path": str(path), "error": str(exc)}))
        return None
    if not isinstance(payload, dict):
        errors.append(ValidationIssue(code=invalid_code, message="JSON payload must be object.", context={"path": str(path)}))
        return None
    return payload


def _validate_run_id(field: str, payload: Mapping[str, Any], expected_run_id: str, errors: list[ValidationIssue]) -> None:
    seen = payload.get("run_id")
    if seen != expected_run_id:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUN_ID_MISMATCH,
                message="run_id mismatch across env contract lineage.",
                context={"field": field, "expected_run_id": expected_run_id, "seen_run_id": seen},
            )
        )


def _validate_output_semantics(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    errors: list[ValidationIssue],
) -> None:
    manifest_sem = manifest.get("output_semantics")
    report_sem = report.get("output_semantics")
    if not isinstance(manifest_sem, Mapping) or not isinstance(report_sem, Mapping):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OUTPUT_SEMANTICS_UNSUPPORTED,
                message="output_semantics block missing in state artifacts.",
                context={},
            )
        )
        return

    mode_manifest = manifest_sem.get("mode")
    mode_report = report_sem.get("mode")
    if mode_manifest != mode_report:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_LINEAGE_MISMATCH,
                message="output_semantics.mode mismatch between manifest and report.",
                context={"manifest_mode": mode_manifest, "report_mode": mode_report},
            )
        )
        return

    supported_modes = {"standard_partitions", "walk_forward_fold_only"}
    if mode_manifest not in supported_modes:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OUTPUT_SEMANTICS_UNSUPPORTED,
                message="state output semantics are unsupported for env adapter v1.",
                context={"mode": mode_manifest, "supported_modes": sorted(supported_modes)},
            )
        )


def _validate_hash_lineage(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    warnings: list[ValidationIssue],
    errors: list[ValidationIssue],
) -> None:
    manifest_hashes = manifest.get("source_hashes")
    report_hashes = report.get("source_hashes")
    keys = ("dataset_manifest_hash", "dataset_build_report_hash", "source_file_inventory_hash")

    if not isinstance(manifest_hashes, Mapping) or not isinstance(report_hashes, Mapping):
        warnings.append(
            ValidationIssue(
                code=ENV_CONTRACT_LINEAGE_HASH_MISSING,
                message="source_hashes block missing; lineage hash checks are partial.",
                context={},
            )
        )
        return

    for key in keys:
        m_value = manifest_hashes.get(key)
        r_value = report_hashes.get(key)
        if isinstance(m_value, str) and isinstance(r_value, str):
            if m_value != r_value:
                errors.append(
                    ValidationIssue(
                        code=ENV_CONTRACT_LINEAGE_MISMATCH,
                        message="Lineage hash mismatch between state manifest and report.",
                        context={"key": key, "manifest_value": m_value, "report_value": r_value},
                    )
                )
        else:
            warnings.append(
                ValidationIssue(
                    code=ENV_CONTRACT_LINEAGE_HASH_MISSING,
                    message="Optional lineage hash missing; check is partial.",
                    context={"key": key},
                )
            )


def _map_episode_source_error_to_code(error_message: str) -> str:
    if "EXECUTION_PRICE_COLUMN_MISSING" in error_message:
        return ENV_CONTRACT_EXECUTION_PRICE_COLUMN_MISSING
    if "MARK_TO_MARKET_COLUMN_MISSING" in error_message:
        return ENV_CONTRACT_MARK_TO_MARKET_COLUMN_MISSING
    if "OBSERVATION_COLUMN_ORDER_MISMATCH" in error_message or "OBSERVATION_STRICT_COLUMNS_MISMATCH" in error_message:
        return ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH
    if "OBSERVATION_DTYPE_MISMATCH" in error_message or "OBS_CAST_NOT_ALLOWED" in error_message:
        return ENV_CONTRACT_OBSERVATION_DTYPE_MISMATCH
    if "MARK_TO_MARKET_PRICE_PARSE_FAILED" in error_message or "EXECUTION_PRICE_PARSE_FAILED" in error_message:
        return ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID
    if "POST_VALID_OBSERVATION_NAN" in error_message:
        return ENV_CONTRACT_POST_VALID_OBSERVATION_NAN
    if "EPISODE_TOO_SHORT_AFTER_WARMUP" in error_message:
        return ENV_CONTRACT_EPISODE_TOO_SHORT_AFTER_WARMUP
    if "EFFECTIVE_START_INVALID" in error_message:
        return ENV_CONTRACT_EFFECTIVE_START_INVALID
    if "TIMESTAMP_DUPLICATES" in error_message:
        return ENV_CONTRACT_TIMESTAMP_DUPLICATES
    if "TIMESTAMP_ORDERING_VIOLATION" in error_message:
        return ENV_CONTRACT_ORDERING_VIOLATION
    if "TIMESTAMP_PARSE_FAILED" in error_message:
        return ENV_CONTRACT_ORDERING_VIOLATION
    return ENV_CONTRACT_PRECONDITION_FAILED


def _default_warmup_contract_payload(*, valid_from_row: int = 0, valid_from_timestamp: str | None = None) -> dict[str, Any]:
    """Return a stable warmup contract payload for env reports."""

    return {
        "enabled": False,
        "required_observation_columns": [],
        "policy": WARMUP_POLICY_DROP_HEAD,
        "valid_from_row": int(valid_from_row),
        "valid_from_timestamp": valid_from_timestamp,
        "post_valid_nan_policy": WARMUP_POST_VALID_NAN_POLICY,
        "head_nan_profile": {},
    }


def _find_episode_manifest_entry(
    *,
    manifest: Mapping[str, Any],
    selected_episode: EpisodeSpec,
) -> Mapping[str, Any] | None:
    """Return the raw manifest entry for the selected episode."""

    entries = manifest.get("partition_metadata")
    if not isinstance(entries, list):
        return None
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        if (
            item.get("scope") == selected_episode.scope
            and item.get("partition") == selected_episode.partition
            and item.get("source_rel") == selected_episode.source_rel
            and item.get("fold_id") == selected_episode.fold_id
        ):
            return item
    return None


def _resolve_episode_contract(
    *,
    manifest: Mapping[str, Any],
    config: EnvConfig,
    selected_episode: EpisodeSpec,
    episode_entry: Mapping[str, Any] | None,
    errors: list[ValidationIssue],
) -> ResolvedEpisodeContract | None:
    observation_contract = manifest.get("observation_contract")
    if not isinstance(observation_contract, Mapping):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract missing or invalid in state manifest.",
                context={},
            )
        )
        return None

    if episode_entry is None:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_STATE_MANIFEST_INVALID,
                message="Selected episode entry is missing from state manifest.",
                context={"episode_key": list(selected_episode.key())},
            )
        )
        return None

    runtime_price_contract = manifest.get("runtime_price_contract")
    if not isinstance(runtime_price_contract, Mapping):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_MISSING,
                message="runtime_price_contract missing or invalid in state manifest.",
                context={},
            )
        )
        return None

    timestamp_policy = observation_contract.get("timestamp_policy")
    dtype_policy = observation_contract.get("dtype_policy")
    selected_input_columns = observation_contract.get("selected_input_columns")
    state_feature_columns = observation_contract.get("state_feature_columns")
    event_columns = observation_contract.get("event_columns")
    regime_columns = observation_contract.get("regime_columns")
    geometry_columns = observation_contract.get("geometry_columns")
    strict_post_valid_numeric_columns = observation_contract.get("strict_post_valid_numeric_columns")
    conditional_raw_columns = observation_contract.get("conditional_raw_columns")
    conditional_column_policy = observation_contract.get("conditional_column_policy")
    conditional_column_replacements = observation_contract.get("conditional_column_replacements")
    geometry_feature_version = observation_contract.get("geometry_feature_version")
    geometry_feature_formulas = observation_contract.get("geometry_feature_formulas")
    future_feature_hooks = observation_contract.get("future_feature_hooks")
    if not isinstance(timestamp_policy, Mapping):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.timestamp_policy must be object.",
                context={},
            )
        )
        return None
    timestamp_column = timestamp_policy.get("timestamp_column")
    if not isinstance(timestamp_column, str) or not timestamp_column.strip():
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.timestamp_policy.timestamp_column is required.",
                context={},
            )
        )
        return None
    if not isinstance(selected_input_columns, list) or not all(isinstance(item, str) for item in selected_input_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.selected_input_columns must be list[str].",
                context={},
            )
        )
        return None
    if not isinstance(state_feature_columns, list) or not all(isinstance(item, str) for item in state_feature_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.state_feature_columns must be list[str].",
                context={},
            )
        )
        return None
    if not isinstance(event_columns, list) or not all(isinstance(item, str) for item in event_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.event_columns must be list[str].",
                context={},
            )
        )
        return None
    if not isinstance(regime_columns, list) or not all(isinstance(item, str) for item in regime_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.regime_columns must be list[str].",
                context={},
            )
        )
        return None
    if not isinstance(geometry_columns, list) or not all(isinstance(item, str) for item in geometry_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.geometry_columns must be list[str].",
                context={},
            )
        )
        return None
    if not isinstance(strict_post_valid_numeric_columns, list) or not all(
        isinstance(item, str) for item in strict_post_valid_numeric_columns
    ):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.strict_post_valid_numeric_columns must be list[str].",
                context={},
            )
        )
        return None
    if not isinstance(conditional_raw_columns, list) or not all(isinstance(item, str) for item in conditional_raw_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.conditional_raw_columns must be list[str].",
                context={},
            )
        )
        return None
    if not isinstance(geometry_feature_version, str) or not geometry_feature_version.strip():
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.geometry_feature_version must be non-empty string.",
                context={},
            )
        )
        return None
    if not isinstance(geometry_feature_formulas, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in geometry_feature_formulas.items()
    ):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.geometry_feature_formulas must be dict[str, str].",
                context={},
            )
        )
        return None
    if not isinstance(future_feature_hooks, Mapping):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.future_feature_hooks must be object.",
                context={},
            )
        )
        return None
    if not isinstance(conditional_column_replacements, Mapping) or not all(
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        for key, value in conditional_column_replacements.items()
    ):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.conditional_column_replacements must be dict[str, list[str]].",
                context={},
            )
        )
        return None
    if conditional_column_policy != CONDITIONAL_COLUMN_POLICY_EXCLUDE_AND_REPLACE:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.conditional_column_policy is unsupported.",
                context={
                    "conditional_column_policy": conditional_column_policy,
                    "supported": CONDITIONAL_COLUMN_POLICY_EXCLUDE_AND_REPLACE,
                },
            )
        )
        return None
    if not isinstance(dtype_policy, Mapping):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_DTYPE_MISMATCH,
                message="observation_contract.dtype_policy must be object.",
                context={},
            )
        )
        return None
    selected_dtypes = dtype_policy.get("selected_dtypes")
    if not isinstance(selected_dtypes, Mapping):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_DTYPE_MISMATCH,
                message="observation_contract.dtype_policy.selected_dtypes must be object.",
                context={},
            )
        )
        return None

    expected_selected_input = [timestamp_column, *state_feature_columns]
    if selected_input_columns != expected_selected_input:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.selected_input_columns must equal timestamp + state_feature_columns.",
                context={"expected": expected_selected_input, "actual": selected_input_columns},
            )
        )
        return None

    missing_dtype_columns = [col for col in expected_selected_input if col not in selected_dtypes]
    if missing_dtype_columns:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_DTYPE_MISMATCH,
                message="Observation contract dtypes are incomplete.",
                context={"missing_columns": missing_dtype_columns},
            )
        )
        return None
    if strict_post_valid_numeric_columns != list(state_feature_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.strict_post_valid_numeric_columns must align with final observation columns.",
                context={
                    "state_feature_columns": state_feature_columns,
                    "strict_post_valid_numeric_columns": strict_post_valid_numeric_columns,
                },
            )
        )
        return None
    role_columns = {
        "event_columns": event_columns,
        "regime_columns": regime_columns,
        "geometry_columns": geometry_columns,
    }
    for role_name, role_items in role_columns.items():
        if role_items != list(dict.fromkeys(role_items)):
            errors.append(
                ValidationIssue(
                    code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                    message=f"observation_contract.{role_name} must be unique and ordered.",
                    context={role_name: role_items},
                )
            )
            return None
        unknown_role_columns = [column for column in role_items if column not in state_feature_columns]
        if unknown_role_columns:
            errors.append(
                ValidationIssue(
                    code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                    message=f"observation_contract.{role_name} must be subset of state_feature_columns.",
                    context={"unknown_columns": unknown_role_columns, "state_feature_columns": state_feature_columns},
                )
            )
            return None
    if conditional_raw_columns != list(dict.fromkeys(conditional_raw_columns)):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="observation_contract.conditional_raw_columns must be unique and ordered.",
                context={"conditional_raw_columns": conditional_raw_columns},
            )
        )
        return None
    if any(column in state_feature_columns for column in conditional_raw_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="conditional_raw_columns must be excluded from final observation core.",
                context={
                    "conditional_raw_columns": conditional_raw_columns,
                    "state_feature_columns": state_feature_columns,
                },
            )
        )
        return None
    replacement_keys = list(conditional_column_replacements.keys())
    if replacement_keys != conditional_raw_columns:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                message="conditional_column_replacements must enumerate conditional_raw_columns in order.",
                context={
                    "conditional_raw_columns": conditional_raw_columns,
                    "conditional_column_replacements_keys": replacement_keys,
                },
            )
        )
        return None
    for raw_column, replacements in conditional_column_replacements.items():
        if not replacements:
            errors.append(
                ValidationIssue(
                    code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                    message="Each conditional raw column must declare replacement observation features.",
                    context={"raw_column": raw_column},
                )
            )
            return None
        unknown_replacements = [column for column in replacements if column not in state_feature_columns]
        if unknown_replacements:
            errors.append(
                ValidationIssue(
                    code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                    message="Conditional raw column replacements must exist in state_feature_columns.",
                    context={"raw_column": raw_column, "unknown_replacements": unknown_replacements},
                )
            )
            return None
    if conditional_raw_columns:
        required_formula_keys = {"ST_active_line_formula", "ST_distance_to_active_line_formula"}
        if not required_formula_keys.issubset(set(geometry_feature_formulas.keys())):
            errors.append(
                ValidationIssue(
                    code=ENV_CONTRACT_OBSERVATION_COLUMN_MISMATCH,
                    message="SuperTrend geometry formula metadata is incomplete.",
                    context={"required_formula_keys": sorted(required_formula_keys)},
                )
            )
            return None

    runtime_timestamp_column = runtime_price_contract.get("timestamp_column")
    execution_price_column = runtime_price_contract.get("execution_price_column")
    mark_to_market_column = runtime_price_contract.get("mark_to_market_column")
    required_runtime_columns = runtime_price_contract.get("required_runtime_columns")
    runtime_price_dtypes = runtime_price_contract.get("runtime_price_dtypes")
    artifact_columns = runtime_price_contract.get("artifact_columns")

    if runtime_timestamp_column != timestamp_column:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract.timestamp_column must match observation contract timestamp column.",
                context={"observation_timestamp_column": timestamp_column, "runtime_timestamp_column": runtime_timestamp_column},
            )
        )
        return None
    if not isinstance(execution_price_column, str) or not execution_price_column.strip():
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract.execution_price_column is required.",
                context={},
            )
        )
        return None
    if not isinstance(mark_to_market_column, str) or not mark_to_market_column.strip():
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract.mark_to_market_column is required.",
                context={},
            )
        )
        return None
    if not isinstance(required_runtime_columns, list) or not all(isinstance(item, str) for item in required_runtime_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract.required_runtime_columns must be list[str].",
                context={},
            )
        )
        return None
    if required_runtime_columns != list(dict.fromkeys(required_runtime_columns)):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract.required_runtime_columns must be unique and ordered.",
                context={"required_runtime_columns": required_runtime_columns},
            )
        )
        return None
    if execution_price_column not in required_runtime_columns or mark_to_market_column not in required_runtime_columns:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime price columns must be included in required_runtime_columns.",
                context={
                    "execution_price_column": execution_price_column,
                    "mark_to_market_column": mark_to_market_column,
                    "required_runtime_columns": required_runtime_columns,
                },
            )
        )
        return None
    if any(col == timestamp_column or col in state_feature_columns for col in required_runtime_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="required_runtime_columns must be runtime-only and separate from observation features.",
                context={
                    "timestamp_column": timestamp_column,
                    "state_feature_columns": state_feature_columns,
                    "required_runtime_columns": required_runtime_columns,
                },
            )
        )
        return None
    if config.execution_price_column != execution_price_column or config.mark_to_market_column != mark_to_market_column:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONFIG_MISMATCH,
                message="Env config runtime price columns must exactly match runtime_price_contract.",
                context={
                    "config_execution_price_column": config.execution_price_column,
                    "contract_execution_price_column": execution_price_column,
                    "config_mark_to_market_column": config.mark_to_market_column,
                    "contract_mark_to_market_column": mark_to_market_column,
                },
            )
        )
        return None
    if not isinstance(runtime_price_dtypes, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in runtime_price_dtypes.items()):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract.runtime_price_dtypes must be dict[str, str].",
                context={},
            )
        )
        return None
    missing_runtime_dtype_columns = [col for col in required_runtime_columns if col not in runtime_price_dtypes]
    if missing_runtime_dtype_columns:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="Runtime price dtype declarations are incomplete.",
                context={"missing_columns": missing_runtime_dtype_columns},
            )
        )
        return None
    if not isinstance(artifact_columns, list) or not all(isinstance(item, str) for item in artifact_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract.artifact_columns must be list[str].",
                context={},
            )
        )
        return None
    expected_artifact_columns = [timestamp_column, *state_feature_columns, *required_runtime_columns]
    if artifact_columns != expected_artifact_columns:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract.artifact_columns must be ordered as timestamp + observation features + runtime-only price columns.",
                context={"expected": expected_artifact_columns, "actual": artifact_columns},
            )
        )
        return None

    artifact_dtypes = {str(key): str(value) for key, value in selected_dtypes.items() if str(key) in expected_selected_input}
    for key in required_runtime_columns:
        artifact_dtypes[key] = str(runtime_price_dtypes[key])

    if "warmup_contract" not in episode_entry:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_MISSING,
                message="warmup_contract is required on every artifact entry.",
                context={"episode_key": list(selected_episode.key())},
            )
        )
        return None
    warmup_contract_raw = episode_entry.get("warmup_contract")
    if not isinstance(warmup_contract_raw, Mapping):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract must be an object.",
                context={"episode_key": list(selected_episode.key())},
            )
        )
        return None

    enabled = warmup_contract_raw.get("enabled")
    required_observation_columns = warmup_contract_raw.get("required_observation_columns")
    policy = warmup_contract_raw.get("policy")
    valid_from_row = warmup_contract_raw.get("valid_from_row")
    valid_from_timestamp = warmup_contract_raw.get("valid_from_timestamp")
    post_valid_nan_policy = warmup_contract_raw.get("post_valid_nan_policy")
    head_nan_profile_raw = warmup_contract_raw.get("head_nan_profile")

    if not isinstance(enabled, bool):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.enabled must be boolean.",
                context={"episode_key": list(selected_episode.key())},
            )
        )
        return None
    if not isinstance(required_observation_columns, list) or not all(isinstance(item, str) for item in required_observation_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.required_observation_columns must be list[str].",
                context={"episode_key": list(selected_episode.key())},
            )
        )
        return None
    if policy != WARMUP_POLICY_DROP_HEAD:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.policy is unsupported.",
                context={"policy": policy, "supported": WARMUP_POLICY_DROP_HEAD},
            )
        )
        return None
    if not isinstance(valid_from_row, int) or valid_from_row < 0:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.valid_from_row must be a non-negative integer.",
                context={"valid_from_row": valid_from_row},
            )
        )
        return None
    if valid_from_timestamp is not None and (not isinstance(valid_from_timestamp, str) or not valid_from_timestamp.strip()):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.valid_from_timestamp must be iso8601 string or null.",
                context={"valid_from_timestamp": valid_from_timestamp},
            )
        )
        return None
    if post_valid_nan_policy != WARMUP_POST_VALID_NAN_POLICY:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.post_valid_nan_policy is unsupported.",
                context={"post_valid_nan_policy": post_valid_nan_policy, "supported": WARMUP_POST_VALID_NAN_POLICY},
            )
        )
        return None
    if not isinstance(head_nan_profile_raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, int) and value > 0 for key, value in head_nan_profile_raw.items()
    ):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.head_nan_profile must be dict[str, int>0].",
                context={"episode_key": list(selected_episode.key())},
            )
        )
        return None

    head_nan_profile = {str(key): int(value) for key, value in head_nan_profile_raw.items()}
    if any(column not in state_feature_columns for column in head_nan_profile):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.head_nan_profile keys must be observation columns only.",
                context={"head_nan_profile_keys": sorted(head_nan_profile), "state_feature_columns": state_feature_columns},
            )
        )
        return None

    expected_required_observation_columns = [col for col in state_feature_columns if head_nan_profile.get(col, 0) > 0]
    if required_observation_columns != expected_required_observation_columns:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.required_observation_columns must match leading-warmup observation columns.",
                context={
                    "expected": expected_required_observation_columns,
                    "actual": required_observation_columns,
                },
            )
        )
        return None

    if enabled != bool(expected_required_observation_columns):
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.enabled must match required_observation_columns.",
                context={"enabled": enabled, "required_observation_columns": required_observation_columns},
            )
        )
        return None

    expected_valid_from_row = max(head_nan_profile.values(), default=0)
    if valid_from_row != expected_valid_from_row:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.valid_from_row must equal max(head_nan_profile.values()).",
                context={"expected": expected_valid_from_row, "actual": valid_from_row},
            )
        )
        return None

    if valid_from_row > selected_episode.row_count:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.valid_from_row exceeds artifact row_count.",
                context={"valid_from_row": valid_from_row, "row_count": selected_episode.row_count},
            )
        )
        return None

    if valid_from_row < selected_episode.row_count:
        if not isinstance(valid_from_timestamp, str) or not valid_from_timestamp.strip():
            errors.append(
                ValidationIssue(
                    code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                    message="warmup_contract.valid_from_timestamp is required when valid_from_row points to an in-range row.",
                    context={"valid_from_row": valid_from_row, "row_count": selected_episode.row_count},
                )
            )
            return None
    elif valid_from_timestamp is not None:
        errors.append(
            ValidationIssue(
                code=ENV_CONTRACT_WARMUP_CONTRACT_INVALID,
                message="warmup_contract.valid_from_timestamp must be null when valid_from_row is out of range.",
                context={"valid_from_row": valid_from_row, "row_count": selected_episode.row_count},
            )
        )
        return None

    warmup_contract = _default_warmup_contract_payload(
        valid_from_row=valid_from_row,
        valid_from_timestamp=str(valid_from_timestamp).strip() if isinstance(valid_from_timestamp, str) else None,
    )
    warmup_contract["enabled"] = bool(enabled)
    warmup_contract["required_observation_columns"] = list(required_observation_columns)
    warmup_contract["head_nan_profile"] = dict(head_nan_profile)

    return ResolvedEpisodeContract(
        timestamp_column=timestamp_column,
        observation_columns=tuple(state_feature_columns),
        strict_post_valid_numeric_columns=tuple(strict_post_valid_numeric_columns),
        artifact_columns=tuple(artifact_columns),
        artifact_dtypes=artifact_dtypes,
        runtime_price_contract={
            "timestamp_column": timestamp_column,
            "execution_price_column": execution_price_column,
            "mark_to_market_column": mark_to_market_column,
            "required_runtime_columns": list(required_runtime_columns),
            "runtime_price_dtypes": {str(key): str(runtime_price_dtypes[key]) for key in required_runtime_columns},
            "artifact_columns": list(artifact_columns),
        },
        warmup_contract=warmup_contract,
    )


def _observation_space_metadata(episode_data: EpisodeData | None) -> dict[str, Any]:
    if episode_data is None:
        return {
            "observation_space_type": None,
            "observation_space_shape": None,
            "observation_space_dtype": None,
        }
    shape = [int(item) for item in episode_data.observation_matrix.shape[1:]]
    return {
        "observation_space_type": "Box",
        "observation_space_shape": shape,
        "observation_space_dtype": str(episode_data.observation_matrix.dtype),
    }


def _check(checks: list[dict[str, Any]], name: str, ok: bool) -> None:
    checks.append({"check": name, "ok": bool(ok)})


def _runner_config_from_env(config: EnvConfig) -> EpisodeRunnerConfig:
    return EpisodeRunnerConfig(
        initial_cash=config.initial_cash,
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
        max_steps=config.max_steps,
        reward_scale=config.reward_contract.reward_scale,
        reward_clip_min=config.reward_contract.reward_clip_min,
        reward_clip_max=config.reward_contract.reward_clip_max,
        seed=config.seed,
    )


def _sha256_file_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65_536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be non-empty string")
    return value.strip()


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be object")
    return value


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be boolean")
    return bool(value)


def _require_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be integer")
    return int(value)


def _require_optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be integer or null")
    return int(value)


def _require_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be numeric")
    return float(value)


def _require_optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be numeric or null")
    return float(value)


def _require_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be list[str]")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"ENV_CONTRACT_CONFIG_INVALID: {key} must be list[str]")
        out.append(item.strip())
    return out
