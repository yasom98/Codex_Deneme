"""Deterministic portfolio-risk overlay gateway for Milestone 4.10."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl.env_core import ACTION_CLOSE_POSITION, ACTION_HOLD, ACTION_MAPPING, ExecutionEngine

RISK_OVERLAY_CONTRACT_VERSION = "risk_overlay_gateway.v1"

RISK_DECISION_ALLOW = "ALLOW"
RISK_DECISION_CLAMP = "CLAMP"
RISK_DECISION_VETO = "VETO"
RISK_DECISION_KILL = "KILL"
RISK_DECISION_TYPES = {
    RISK_DECISION_ALLOW,
    RISK_DECISION_CLAMP,
    RISK_DECISION_VETO,
    RISK_DECISION_KILL,
}

RISK_MODE_NORMAL = "NORMAL"
RISK_MODE_DEFENSIVE = "DEFENSIVE"
RISK_MODE_FREEZE_ENTRIES = "FREEZE_ENTRIES"
RISK_MODE_KILL_HALTED = "KILL_HALTED"
RISK_MODES = {
    RISK_MODE_NORMAL,
    RISK_MODE_DEFENSIVE,
    RISK_MODE_FREEZE_ENTRIES,
    RISK_MODE_KILL_HALTED,
}

RISK_STATUS_APPROVED = "approved"
RISK_STATUS_CLAMPED = "clamped"
RISK_STATUS_VETOED = "vetoed"
RISK_STATUS_KILLED = "killed"

RISK_REASON_MISSING_REQUIRED_INPUT = "RISK_MISSING_REQUIRED_INPUT"
RISK_REASON_STALE_MARKET_SNAPSHOT = "RISK_STALE_MARKET_SNAPSHOT"
RISK_REASON_STALE_PORTFOLIO_STATE = "RISK_STALE_PORTFOLIO_STATE"
RISK_REASON_STALE_ACTION_PROPOSAL = "RISK_STALE_ACTION_PROPOSAL"
RISK_REASON_INVALID_NUMERIC_INPUT = "RISK_INVALID_NUMERIC_INPUT"
RISK_REASON_INVALID_TIMESTAMP = "RISK_INVALID_TIMESTAMP"
RISK_REASON_INSTRUMENT_NOT_ALLOWED = "RISK_INSTRUMENT_NOT_ALLOWED"
RISK_REASON_HARD_EXPOSURE_CAP_CLAMP = "RISK_HARD_EXPOSURE_CAP_CLAMP"
RISK_REASON_HARD_LEVERAGE_CAP_CLAMP = "RISK_HARD_LEVERAGE_CAP_CLAMP"
RISK_REASON_DEFENSIVE_SCALE_DOWN = "RISK_DEFENSIVE_SCALE_DOWN"
RISK_REASON_FREEZE_ENTRIES_BLOCK = "RISK_FREEZE_ENTRIES_BLOCK"
RISK_REASON_FREEZE_ENTRIES_CLOSE_ONLY = "RISK_FREEZE_ENTRIES_CLOSE_ONLY"
RISK_REASON_DRAWDOWN_TO_DEFENSIVE = "RISK_DRAWDOWN_TO_DEFENSIVE"
RISK_REASON_DRAWDOWN_TO_FREEZE = "RISK_DRAWDOWN_TO_FREEZE"
RISK_REASON_DRAWDOWN_RECOVER_TO_DEFENSIVE = "RISK_DRAWDOWN_RECOVER_TO_DEFENSIVE"
RISK_REASON_DRAWDOWN_RECOVER_TO_NORMAL = "RISK_DRAWDOWN_RECOVER_TO_NORMAL"
RISK_REASON_CATASTROPHIC_DRAWDOWN = "RISK_CATASTROPHIC_DRAWDOWN"
RISK_REASON_KILL_HALTED_ACTIVE = "RISK_KILL_HALTED_ACTIVE"
RISK_REASON_SYSTEMIC_INPUT_FAILURE = "RISK_SYSTEMIC_INPUT_FAILURE"
RISK_REASON_RECOVERY_TOKEN_REQUIRED = "RISK_RECOVERY_TOKEN_REQUIRED"

_TOP_LEVEL_CONFIG_KEYS = {
    "config_version",
    "allowed_instruments",
    "freshness_limits",
    "exposure_limits",
    "leverage_limits",
    "drawdown_thresholds",
    "hysteresis_bands",
    "recovery_policy",
}


@dataclass(frozen=True)
class FreshnessLimits:
    """Freshness requirements for required overlay inputs."""

    max_market_data_age_seconds: int
    max_portfolio_state_age_seconds: int
    max_proposal_age_seconds: int


@dataclass(frozen=True)
class ExposureLimits:
    """Portfolio exposure ceilings and defensive scale-down."""

    max_abs_target_exposure: float
    max_gross_exposure: float
    max_net_exposure: float
    max_instrument_exposure: float
    defensive_scale_down: float


@dataclass(frozen=True)
class LeverageLimits:
    """Hard leverage ceiling and defensive scale-down."""

    max_leverage: float
    defensive_scale_down: float


@dataclass(frozen=True)
class DrawdownThresholds:
    """Drawdown entry and exit thresholds for the fixed first-wave state machine."""

    defensive_enter_pct: float
    defensive_exit_pct: float
    freeze_enter_pct: float
    freeze_exit_pct: float
    kill_pct: float


@dataclass(frozen=True)
class HysteresisBands:
    """Anti-flap controls for state transitions."""

    min_steps_in_state: int


@dataclass(frozen=True)
class RecoveryPolicy:
    """Recovery requirements for kill/freeze states."""

    freeze_cooldown_steps: int
    systemic_failure_kill_threshold: int
    kill_requires_recovery_token: bool


@dataclass(frozen=True)
class RiskOverlayConfig:
    """Strict 4.10 JSON configuration."""

    config_version: str
    allowed_instruments: tuple[str, ...]
    freshness_limits: FreshnessLimits
    exposure_limits: ExposureLimits
    leverage_limits: LeverageLimits
    drawdown_thresholds: DrawdownThresholds
    hysteresis_bands: HysteresisBands
    recovery_policy: RecoveryPolicy


@dataclass(frozen=True)
class AgentActionProposal:
    """Explicit agent action proposal surface."""

    instrument: str
    proposal_timestamp_utc: str
    target_exposure: float
    requested_side: str
    intent_type: str
    requested_notional: float
    action_raw: int


@dataclass(frozen=True)
class PortfolioRiskState:
    """Portfolio state snapshot consumed by the gateway."""

    state_timestamp_utc: str
    equity: float
    gross_exposure: float
    net_exposure: float
    current_leverage: float
    instrument_exposure: float
    drawdown_pct: float


@dataclass(frozen=True)
class MarketSnapshot:
    """Minimal market snapshot used by first-wave risk checks."""

    snapshot_timestamp_utc: str
    instrument: str
    mid_price: float
    tradable: bool


@dataclass(frozen=True)
class RiskState:
    """Explicit deterministic overlay state."""

    mode: str
    kill_active: bool
    freeze_entries_active: bool
    drawdown_regime: str
    last_transition_utc: str | None
    consecutive_input_failures: int
    recovery_token_present: bool
    mode_step_count: int
    cooldown_remaining_steps: int


@dataclass(frozen=True)
class RiskOverlayInput:
    """Top-level input contract for the gateway."""

    decision_timestamp_utc: str
    contract_version: str
    agent_action_proposal: AgentActionProposal
    portfolio_state: PortfolioRiskState
    market_snapshot: MarketSnapshot
    risk_state: RiskState
    risk_config: RiskOverlayConfig


@dataclass(frozen=True)
class RiskDecision:
    """Machine-readable approval decision."""

    decision_type: str
    status: str
    approved_action: dict[str, Any]
    approved_target_exposure: float
    approved_leverage: float
    applied_limits: dict[str, Any]
    reason_codes: tuple[str, ...]
    rule_hits: tuple[str, ...]
    state_transition: dict[str, Any] | None
    counters: dict[str, Any]
    only_close_allowed: bool
    kill_active: bool
    decision_timestamp_utc: str
    contract_version: str
    risk_state_after: RiskState

    def to_report_dict(self) -> dict[str, Any]:
        """Serialize the decision for logging/reporting."""

        payload = asdict(self)
        payload["risk_state_after"] = asdict(self.risk_state_after)
        payload["reason_codes"] = list(self.reason_codes)
        payload["rule_hits"] = list(self.rule_hits)
        return payload


@dataclass(frozen=True)
class RiskOverlayReportBundle:
    """Runtime artifact payloads for one evaluation session."""

    decision_log_rows: tuple[dict[str, Any], ...]
    summary_payload: dict[str, Any]
    transition_log_rows: tuple[dict[str, Any], ...]


@dataclass
class RiskOverlaySession:
    """Stateful risk overlay session with machine-readable audit accumulation."""

    config: RiskOverlayConfig
    config_path: Path
    config_hash: str
    state: RiskState = field(default_factory=lambda: build_initial_risk_state())
    decision_log_rows: list[dict[str, Any]] = field(default_factory=list)
    transition_log_rows: list[dict[str, Any]] = field(default_factory=list)
    decision_counts: Counter[str] = field(default_factory=Counter)
    reason_counts: Counter[str] = field(default_factory=Counter)
    state_transition_counts: Counter[str] = field(default_factory=Counter)
    state_visit_counts: Counter[str] = field(default_factory=Counter)
    episode_count: int = 0

    def start_episode(self) -> None:
        """Reset the mutable state for a fresh isolated evaluation episode."""

        self.state = build_initial_risk_state()
        self.state_visit_counts[self.state.mode] += 1
        self.episode_count += 1

    def evaluate(self, input_payload: RiskOverlayInput, *, record_context: Mapping[str, Any] | None = None) -> RiskDecision:
        """Evaluate one proposal, update session state, and append logs."""

        decision = decide_risk_overlay(input_payload)
        self.state = decision.risk_state_after
        self.decision_counts[decision.decision_type] += 1
        self.state_visit_counts[decision.risk_state_after.mode] += 1
        for code in decision.reason_codes:
            self.reason_counts[code] += 1

        decision_row = decision.to_report_dict()
        if record_context is not None:
            decision_row.update(dict(record_context))
        self.decision_log_rows.append(decision_row)

        if decision.state_transition is not None:
            transition_row = dict(decision.state_transition)
            if record_context is not None:
                transition_row.update(dict(record_context))
            self.transition_log_rows.append(transition_row)
            transition_key = f"{transition_row['from_mode']}->{transition_row['to_mode']}"
            self.state_transition_counts[transition_key] += 1

        return decision

    def build_report_bundle(self) -> RiskOverlayReportBundle:
        """Build the minimal 4.10 runtime artifact family."""

        summary_payload = {
            "contract_version": RISK_OVERLAY_CONTRACT_VERSION,
            "config_version": self.config.config_version,
            "config_path": str(self.config_path),
            "config_hash": self.config_hash,
            "overlay_enabled": True,
            "episode_count": int(self.episode_count),
            "decision_counts": {key: int(self.decision_counts.get(key, 0)) for key in sorted(RISK_DECISION_TYPES)},
            "reason_code_counts": {key: int(value) for key, value in sorted(self.reason_counts.items())},
            "state_transition_counts": {key: int(value) for key, value in sorted(self.state_transition_counts.items())},
            "state_visit_counts": {key: int(value) for key, value in sorted(self.state_visit_counts.items())},
            "final_state": asdict(self.state),
            "generated_at_utc": _generated_at(),
        }
        return RiskOverlayReportBundle(
            decision_log_rows=tuple(self.decision_log_rows),
            summary_payload=summary_payload,
            transition_log_rows=tuple(self.transition_log_rows),
        )


def build_initial_risk_state() -> RiskState:
    """Return the fixed initial overlay state."""

    return RiskState(
        mode=RISK_MODE_NORMAL,
        kill_active=False,
        freeze_entries_active=False,
        drawdown_regime=RISK_MODE_NORMAL,
        last_transition_utc=None,
        consecutive_input_failures=0,
        recovery_token_present=False,
        mode_step_count=0,
        cooldown_remaining_steps=0,
    )


def load_risk_overlay_config(path: Path) -> tuple[RiskOverlayConfig, str]:
    """Load and validate the strict 4.10 JSON config."""

    if not path.exists():
        raise FileNotFoundError(f"Risk overlay config not found: {path}")
    if not path.is_file():
        raise ValueError(f"Risk overlay config path is not a file: {path}")
    try:
        raw_text = path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Risk overlay config is not valid JSON: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Risk overlay config could not be read: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Risk overlay config JSON must be an object.")
    _require_exact_keys(payload, _TOP_LEVEL_CONFIG_KEYS, location="risk_overlay_config")

    config_version = _require_non_empty_str(payload.get("config_version"), "config_version")
    allowed_instruments_raw = payload.get("allowed_instruments")
    if not isinstance(allowed_instruments_raw, list) or not allowed_instruments_raw:
        raise ValueError("allowed_instruments must be a non-empty list.")
    allowed_instruments = tuple(_require_non_empty_str(item, "allowed_instruments[]") for item in allowed_instruments_raw)

    freshness_limits_raw = _require_object(payload.get("freshness_limits"), "freshness_limits")
    _require_exact_keys(
        freshness_limits_raw,
        {"max_market_data_age_seconds", "max_portfolio_state_age_seconds", "max_proposal_age_seconds"},
        location="freshness_limits",
    )
    freshness_limits = FreshnessLimits(
        max_market_data_age_seconds=_require_non_negative_int(
            freshness_limits_raw.get("max_market_data_age_seconds"), "freshness_limits.max_market_data_age_seconds"
        ),
        max_portfolio_state_age_seconds=_require_non_negative_int(
            freshness_limits_raw.get("max_portfolio_state_age_seconds"), "freshness_limits.max_portfolio_state_age_seconds"
        ),
        max_proposal_age_seconds=_require_non_negative_int(
            freshness_limits_raw.get("max_proposal_age_seconds"), "freshness_limits.max_proposal_age_seconds"
        ),
    )

    exposure_limits_raw = _require_object(payload.get("exposure_limits"), "exposure_limits")
    _require_exact_keys(
        exposure_limits_raw,
        {
            "max_abs_target_exposure",
            "max_gross_exposure",
            "max_net_exposure",
            "max_instrument_exposure",
            "defensive_scale_down",
        },
        location="exposure_limits",
    )
    exposure_limits = ExposureLimits(
        max_abs_target_exposure=_require_non_negative_float(
            exposure_limits_raw.get("max_abs_target_exposure"), "exposure_limits.max_abs_target_exposure"
        ),
        max_gross_exposure=_require_non_negative_float(
            exposure_limits_raw.get("max_gross_exposure"), "exposure_limits.max_gross_exposure"
        ),
        max_net_exposure=_require_non_negative_float(
            exposure_limits_raw.get("max_net_exposure"), "exposure_limits.max_net_exposure"
        ),
        max_instrument_exposure=_require_non_negative_float(
            exposure_limits_raw.get("max_instrument_exposure"), "exposure_limits.max_instrument_exposure"
        ),
        defensive_scale_down=_require_fraction(
            exposure_limits_raw.get("defensive_scale_down"), "exposure_limits.defensive_scale_down"
        ),
    )

    leverage_limits_raw = _require_object(payload.get("leverage_limits"), "leverage_limits")
    _require_exact_keys(
        leverage_limits_raw,
        {"max_leverage", "defensive_scale_down"},
        location="leverage_limits",
    )
    leverage_limits = LeverageLimits(
        max_leverage=_require_non_negative_float(leverage_limits_raw.get("max_leverage"), "leverage_limits.max_leverage"),
        defensive_scale_down=_require_fraction(
            leverage_limits_raw.get("defensive_scale_down"), "leverage_limits.defensive_scale_down"
        ),
    )

    drawdown_raw = _require_object(payload.get("drawdown_thresholds"), "drawdown_thresholds")
    _require_exact_keys(
        drawdown_raw,
        {
            "defensive_enter_pct",
            "defensive_exit_pct",
            "freeze_enter_pct",
            "freeze_exit_pct",
            "kill_pct",
        },
        location="drawdown_thresholds",
    )
    drawdown_thresholds = DrawdownThresholds(
        defensive_enter_pct=_require_percentage(drawdown_raw.get("defensive_enter_pct"), "drawdown_thresholds.defensive_enter_pct"),
        defensive_exit_pct=_require_percentage(drawdown_raw.get("defensive_exit_pct"), "drawdown_thresholds.defensive_exit_pct"),
        freeze_enter_pct=_require_percentage(drawdown_raw.get("freeze_enter_pct"), "drawdown_thresholds.freeze_enter_pct"),
        freeze_exit_pct=_require_percentage(drawdown_raw.get("freeze_exit_pct"), "drawdown_thresholds.freeze_exit_pct"),
        kill_pct=_require_percentage(drawdown_raw.get("kill_pct"), "drawdown_thresholds.kill_pct"),
    )
    if not (
        drawdown_thresholds.defensive_enter_pct > drawdown_thresholds.defensive_exit_pct
        and drawdown_thresholds.freeze_enter_pct > drawdown_thresholds.freeze_exit_pct
        and drawdown_thresholds.freeze_enter_pct > drawdown_thresholds.defensive_enter_pct
        and drawdown_thresholds.kill_pct > drawdown_thresholds.freeze_enter_pct
    ):
        raise ValueError("drawdown_thresholds must satisfy defensive_exit < defensive_enter < freeze_enter < kill.")

    hysteresis_raw = _require_object(payload.get("hysteresis_bands"), "hysteresis_bands")
    _require_exact_keys(hysteresis_raw, {"min_steps_in_state"}, location="hysteresis_bands")
    hysteresis_bands = HysteresisBands(
        min_steps_in_state=_require_non_negative_int(hysteresis_raw.get("min_steps_in_state"), "hysteresis_bands.min_steps_in_state")
    )

    recovery_raw = _require_object(payload.get("recovery_policy"), "recovery_policy")
    _require_exact_keys(
        recovery_raw,
        {"freeze_cooldown_steps", "systemic_failure_kill_threshold", "kill_requires_recovery_token"},
        location="recovery_policy",
    )
    kill_requires_token = recovery_raw.get("kill_requires_recovery_token")
    if not isinstance(kill_requires_token, bool):
        raise ValueError("recovery_policy.kill_requires_recovery_token must be a boolean.")
    recovery_policy = RecoveryPolicy(
        freeze_cooldown_steps=_require_non_negative_int(
            recovery_raw.get("freeze_cooldown_steps"), "recovery_policy.freeze_cooldown_steps"
        ),
        systemic_failure_kill_threshold=_require_positive_int(
            recovery_raw.get("systemic_failure_kill_threshold"), "recovery_policy.systemic_failure_kill_threshold"
        ),
        kill_requires_recovery_token=kill_requires_token,
    )

    config = RiskOverlayConfig(
        config_version=config_version,
        allowed_instruments=allowed_instruments,
        freshness_limits=freshness_limits,
        exposure_limits=exposure_limits,
        leverage_limits=leverage_limits,
        drawdown_thresholds=drawdown_thresholds,
        hysteresis_bands=hysteresis_bands,
        recovery_policy=recovery_policy,
    )
    config_hash = _hash_canonical_json(payload)
    return config, config_hash


def build_risk_overlay_session(*, config: RiskOverlayConfig, config_path: Path, config_hash: str) -> RiskOverlaySession:
    """Create a mutable overlay session."""

    return RiskOverlaySession(config=config, config_path=config_path.resolve(), config_hash=config_hash)


def derive_action_proposal(
    *,
    instrument: str,
    timestamp_utc: str,
    current_exposure: int,
    action_raw: int,
    mid_price: float,
) -> AgentActionProposal:
    """Translate the discrete env action into a 4.10 proposal surface."""

    exec_result = ExecutionEngine.apply_action(position_before=int(current_exposure), action_raw=int(action_raw))
    delta_exposure = int(exec_result.position_after) - int(current_exposure)
    if exec_result.position_after > 0:
        requested_side = "LONG"
    elif exec_result.position_after < 0:
        requested_side = "SHORT"
    else:
        requested_side = "FLAT"

    if exec_result.position_after == current_exposure:
        intent_type = "hold"
    elif abs(exec_result.position_after) < abs(current_exposure):
        intent_type = "reduce"
    elif exec_result.position_after == 0:
        intent_type = "close"
    else:
        intent_type = "increase"

    return AgentActionProposal(
        instrument=str(instrument),
        proposal_timestamp_utc=str(timestamp_utc),
        target_exposure=float(exec_result.position_after),
        requested_side=requested_side,
        intent_type=intent_type,
        requested_notional=float(abs(delta_exposure) * float(mid_price)),
        action_raw=int(action_raw),
    )


def approved_target_to_action(*, current_exposure: int, approved_target_exposure: float) -> int:
    """Map an approved target exposure back into the fixed env action space."""

    normalized_target = _discretize_target_exposure(approved_target_exposure)
    if normalized_target == int(current_exposure):
        return ACTION_HOLD
    if normalized_target == 0:
        return ACTION_CLOSE_POSITION if int(current_exposure) != 0 else ACTION_HOLD
    if normalized_target == 1 and int(current_exposure) == 0:
        return 1
    if normalized_target == -1 and int(current_exposure) == 0:
        return 2
    raise ValueError(
        f"Cannot map current_exposure={current_exposure} and approved_target_exposure={approved_target_exposure} into a valid action."
    )


def decide_risk_overlay(input_payload: RiskOverlayInput) -> RiskDecision:
    """Execute the strict first-wave 4.10 decision contract."""

    if input_payload.contract_version != RISK_OVERLAY_CONTRACT_VERSION:
        raise ValueError(
            f"risk overlay contract version mismatch: {input_payload.contract_version} != {RISK_OVERLAY_CONTRACT_VERSION}"
        )

    current_state = input_payload.risk_state
    proposal = input_payload.agent_action_proposal
    portfolio = input_payload.portfolio_state
    market = input_payload.market_snapshot
    config = input_payload.risk_config

    invalid_reasons = _validate_input_contract(input_payload)
    next_input_failure_count = current_state.consecutive_input_failures + (1 if invalid_reasons else 0)
    if invalid_reasons:
        if next_input_failure_count >= config.recovery_policy.systemic_failure_kill_threshold:
            next_state = _transition_to_kill(
                current_state=current_state,
                timestamp_utc=input_payload.decision_timestamp_utc,
                keep_cooldown=False,
            )
            return _build_decision(
                decision_type=RISK_DECISION_KILL,
                status=RISK_STATUS_KILLED,
                approved_target_exposure=portfolio.instrument_exposure,
                approved_action_raw=ACTION_HOLD,
                approved_leverage=max(float(portfolio.current_leverage), 0.0),
                applied_limits=_build_applied_limits(config=config, state=current_state, effective_state=current_state),
                reason_codes=tuple([*invalid_reasons, RISK_REASON_SYSTEMIC_INPUT_FAILURE]),
                rule_hits=tuple([*invalid_reasons, RISK_REASON_SYSTEMIC_INPUT_FAILURE]),
                state_transition=_build_transition_record(
                    from_mode=current_state.mode,
                    to_mode=next_state.mode,
                    timestamp_utc=input_payload.decision_timestamp_utc,
                    reason=RISK_REASON_SYSTEMIC_INPUT_FAILURE,
                ),
                counters={"consecutive_input_failures": next_input_failure_count},
                only_close_allowed=False,
                kill_active=True,
                decision_timestamp_utc=input_payload.decision_timestamp_utc,
                risk_state_after=next_state,
                current_exposure=int(round(portfolio.instrument_exposure)),
            )
        next_state = RiskState(
            mode=current_state.mode,
            kill_active=current_state.kill_active,
            freeze_entries_active=current_state.freeze_entries_active,
            drawdown_regime=current_state.drawdown_regime,
            last_transition_utc=current_state.last_transition_utc,
            consecutive_input_failures=next_input_failure_count,
            recovery_token_present=current_state.recovery_token_present,
            mode_step_count=current_state.mode_step_count + 1,
            cooldown_remaining_steps=max(current_state.cooldown_remaining_steps - 1, 0),
        )
        return _build_decision(
            decision_type=RISK_DECISION_VETO,
            status=RISK_STATUS_VETOED,
            approved_target_exposure=portfolio.instrument_exposure,
            approved_action_raw=ACTION_HOLD,
            approved_leverage=max(float(portfolio.current_leverage), 0.0),
            applied_limits=_build_applied_limits(config=config, state=current_state, effective_state=current_state),
            reason_codes=tuple(invalid_reasons),
            rule_hits=tuple(invalid_reasons),
            state_transition=None,
            counters={"consecutive_input_failures": next_input_failure_count},
            only_close_allowed=False,
            kill_active=False,
            decision_timestamp_utc=input_payload.decision_timestamp_utc,
            risk_state_after=next_state,
            current_exposure=int(round(portfolio.instrument_exposure)),
        )

    if current_state.kill_active or current_state.mode == RISK_MODE_KILL_HALTED:
        if config.recovery_policy.kill_requires_recovery_token and not current_state.recovery_token_present:
            next_state = RiskState(
                mode=RISK_MODE_KILL_HALTED,
                kill_active=True,
                freeze_entries_active=False,
                drawdown_regime=RISK_MODE_KILL_HALTED,
                last_transition_utc=current_state.last_transition_utc,
                consecutive_input_failures=0,
                recovery_token_present=current_state.recovery_token_present,
                mode_step_count=current_state.mode_step_count + 1,
                cooldown_remaining_steps=0,
            )
            return _build_decision(
                decision_type=RISK_DECISION_KILL,
                status=RISK_STATUS_KILLED,
                approved_target_exposure=portfolio.instrument_exposure,
                approved_action_raw=ACTION_HOLD,
                approved_leverage=max(float(portfolio.current_leverage), 0.0),
                applied_limits=_build_applied_limits(config=config, state=current_state, effective_state=next_state),
                reason_codes=(RISK_REASON_KILL_HALTED_ACTIVE, RISK_REASON_RECOVERY_TOKEN_REQUIRED),
                rule_hits=(RISK_REASON_KILL_HALTED_ACTIVE, RISK_REASON_RECOVERY_TOKEN_REQUIRED),
                state_transition=None,
                counters={"consecutive_input_failures": 0},
                only_close_allowed=False,
                kill_active=True,
                decision_timestamp_utc=input_payload.decision_timestamp_utc,
                risk_state_after=next_state,
                current_exposure=int(round(portfolio.instrument_exposure)),
            )

    current_exposure = int(round(portfolio.instrument_exposure))
    proposal_target = _discretize_target_exposure(proposal.target_exposure)

    state_after_transition, transition_reason, transition_record = _apply_state_machine(input_payload)
    effective_state = state_after_transition
    if proposal.instrument not in set(config.allowed_instruments) or market.instrument not in set(config.allowed_instruments):
        return _build_decision(
            decision_type=RISK_DECISION_VETO,
            status=RISK_STATUS_VETOED,
            approved_target_exposure=portfolio.instrument_exposure,
            approved_action_raw=ACTION_HOLD,
            approved_leverage=max(float(portfolio.current_leverage), 0.0),
            applied_limits=_build_applied_limits(config=config, state=current_state, effective_state=effective_state),
            reason_codes=_with_transition_reason((RISK_REASON_INSTRUMENT_NOT_ALLOWED,), transition_reason),
            rule_hits=_with_transition_reason((RISK_REASON_INSTRUMENT_NOT_ALLOWED,), transition_reason),
            state_transition=transition_record,
            counters={"consecutive_input_failures": 0},
            only_close_allowed=effective_state.mode == RISK_MODE_FREEZE_ENTRIES,
            kill_active=effective_state.kill_active,
            decision_timestamp_utc=input_payload.decision_timestamp_utc,
            risk_state_after=effective_state,
            current_exposure=current_exposure,
        )

    if effective_state.mode == RISK_MODE_KILL_HALTED or portfolio.drawdown_pct >= config.drawdown_thresholds.kill_pct:
        approved_target = 0.0 if current_exposure != 0 else 0.0
        approved_action_raw = approved_target_to_action(current_exposure=current_exposure, approved_target_exposure=approved_target)
        return _build_decision(
            decision_type=RISK_DECISION_KILL,
            status=RISK_STATUS_KILLED,
            approved_target_exposure=approved_target,
            approved_action_raw=approved_action_raw,
            approved_leverage=0.0,
            applied_limits=_build_applied_limits(config=config, state=current_state, effective_state=effective_state),
            reason_codes=_with_transition_reason((RISK_REASON_CATASTROPHIC_DRAWDOWN,), transition_reason),
            rule_hits=_with_transition_reason((RISK_REASON_CATASTROPHIC_DRAWDOWN,), transition_reason),
            state_transition=transition_record,
            counters={"consecutive_input_failures": 0},
            only_close_allowed=current_exposure != 0,
            kill_active=True,
            decision_timestamp_utc=input_payload.decision_timestamp_utc,
            risk_state_after=effective_state,
            current_exposure=current_exposure,
        )

    applied_limits = _build_applied_limits(config=config, state=current_state, effective_state=effective_state)
    only_close_allowed = effective_state.mode == RISK_MODE_FREEZE_ENTRIES
    if only_close_allowed:
        if abs(proposal_target) > abs(current_exposure):
            return _build_decision(
                decision_type=RISK_DECISION_VETO,
                status=RISK_STATUS_VETOED,
                approved_target_exposure=portfolio.instrument_exposure,
                approved_action_raw=ACTION_HOLD,
                approved_leverage=max(float(portfolio.current_leverage), 0.0),
                applied_limits=applied_limits,
                reason_codes=_with_transition_reason((RISK_REASON_FREEZE_ENTRIES_BLOCK,), transition_reason),
                rule_hits=_with_transition_reason((RISK_REASON_FREEZE_ENTRIES_BLOCK,), transition_reason),
                state_transition=transition_record,
                counters={"consecutive_input_failures": 0},
                only_close_allowed=True,
                kill_active=False,
                decision_timestamp_utc=input_payload.decision_timestamp_utc,
                risk_state_after=effective_state,
                current_exposure=current_exposure,
            )
        if abs(current_exposure) > 0 and proposal_target == current_exposure:
            approved_target = 0.0
            approved_action_raw = approved_target_to_action(
                current_exposure=current_exposure,
                approved_target_exposure=approved_target,
            )
            return _build_decision(
                decision_type=RISK_DECISION_CLAMP,
                status=RISK_STATUS_CLAMPED,
                approved_target_exposure=approved_target,
                approved_action_raw=approved_action_raw,
                approved_leverage=0.0,
                applied_limits=applied_limits,
                reason_codes=_with_transition_reason((RISK_REASON_FREEZE_ENTRIES_CLOSE_ONLY,), transition_reason),
                rule_hits=_with_transition_reason((RISK_REASON_FREEZE_ENTRIES_CLOSE_ONLY,), transition_reason),
                state_transition=transition_record,
                counters={"consecutive_input_failures": 0},
                only_close_allowed=True,
                kill_active=False,
                decision_timestamp_utc=input_payload.decision_timestamp_utc,
                risk_state_after=effective_state,
                current_exposure=current_exposure,
            )

    hard_exposure_cap = float(
        min(
            config.exposure_limits.max_abs_target_exposure,
            config.exposure_limits.max_gross_exposure,
            config.exposure_limits.max_net_exposure,
            config.exposure_limits.max_instrument_exposure,
        )
    )
    effective_exposure_cap = hard_exposure_cap
    effective_leverage_cap = float(config.leverage_limits.max_leverage)
    if effective_state.mode == RISK_MODE_DEFENSIVE:
        effective_exposure_cap = hard_exposure_cap * float(config.exposure_limits.defensive_scale_down)
        effective_leverage_cap = float(config.leverage_limits.max_leverage) * float(config.leverage_limits.defensive_scale_down)

    proposed_abs = abs(float(proposal_target))
    hard_breach_reasons: list[str] = []
    clamp_reasons: list[str] = []
    approved_target = float(proposal_target)

    if proposed_abs > hard_exposure_cap:
        approved_target = 0.0
        hard_breach_reasons.append(RISK_REASON_HARD_EXPOSURE_CAP_CLAMP)
    if proposed_abs > float(config.leverage_limits.max_leverage):
        approved_target = 0.0
        hard_breach_reasons.append(RISK_REASON_HARD_LEVERAGE_CAP_CLAMP)

    if not hard_breach_reasons:
        if proposed_abs > effective_exposure_cap or proposed_abs > effective_leverage_cap:
            approved_target = 0.0
            clamp_reasons.append(RISK_REASON_DEFENSIVE_SCALE_DOWN)

    approved_target = float(_discretize_target_exposure(approved_target))
    approved_action_raw = approved_target_to_action(
        current_exposure=current_exposure,
        approved_target_exposure=approved_target,
    )
    approved_leverage = abs(float(approved_target))

    if hard_breach_reasons:
        return _build_decision(
            decision_type=RISK_DECISION_CLAMP,
            status=RISK_STATUS_CLAMPED,
            approved_target_exposure=approved_target,
            approved_action_raw=approved_action_raw,
            approved_leverage=approved_leverage,
            applied_limits=applied_limits,
            reason_codes=_with_transition_reason(tuple(hard_breach_reasons), transition_reason),
            rule_hits=_with_transition_reason(tuple(hard_breach_reasons), transition_reason),
            state_transition=transition_record,
            counters={"consecutive_input_failures": 0},
            only_close_allowed=only_close_allowed,
            kill_active=False,
            decision_timestamp_utc=input_payload.decision_timestamp_utc,
            risk_state_after=effective_state,
            current_exposure=current_exposure,
        )

    if clamp_reasons:
        return _build_decision(
            decision_type=RISK_DECISION_CLAMP,
            status=RISK_STATUS_CLAMPED,
            approved_target_exposure=approved_target,
            approved_action_raw=approved_action_raw,
            approved_leverage=approved_leverage,
            applied_limits=applied_limits,
            reason_codes=_with_transition_reason(tuple(clamp_reasons), transition_reason),
            rule_hits=_with_transition_reason(tuple(clamp_reasons), transition_reason),
            state_transition=transition_record,
            counters={"consecutive_input_failures": 0},
            only_close_allowed=only_close_allowed,
            kill_active=False,
            decision_timestamp_utc=input_payload.decision_timestamp_utc,
            risk_state_after=effective_state,
            current_exposure=current_exposure,
        )

    return _build_decision(
        decision_type=RISK_DECISION_ALLOW,
        status=RISK_STATUS_APPROVED,
        approved_target_exposure=float(proposal_target),
        approved_action_raw=int(proposal.action_raw),
        approved_leverage=abs(float(proposal_target)),
        applied_limits=applied_limits,
        reason_codes=_with_transition_reason(tuple(), transition_reason),
        rule_hits=_with_transition_reason(tuple(), transition_reason),
        state_transition=transition_record,
        counters={"consecutive_input_failures": 0},
        only_close_allowed=only_close_allowed,
        kill_active=False,
        decision_timestamp_utc=input_payload.decision_timestamp_utc,
        risk_state_after=effective_state,
        current_exposure=current_exposure,
    )


def write_risk_overlay_artifacts(bundle: RiskOverlayReportBundle, *, output_dir: Path) -> dict[str, str]:
    """Write the fixed minimal 4.10 artifact family atomically."""

    output_dir = output_dir.resolve()
    decision_log_path = output_dir / "risk_decision_log.jsonl"
    summary_path = output_dir / "risk_overlay_summary.json"
    transition_path = output_dir / "risk_state_transition_log.jsonl"
    _atomic_write_jsonl(bundle.decision_log_rows, decision_log_path)
    _atomic_write_json(bundle.summary_payload, summary_path)
    _atomic_write_jsonl(bundle.transition_log_rows, transition_path)
    return {
        "risk_decision_log_path": str(decision_log_path),
        "risk_overlay_summary_path": str(summary_path),
        "risk_state_transition_log_path": str(transition_path),
    }


def _apply_state_machine(input_payload: RiskOverlayInput) -> tuple[RiskState, str | None, dict[str, Any] | None]:
    """Apply the fixed 4-state drawdown state machine with hysteresis."""

    current_state = input_payload.risk_state
    drawdown = float(input_payload.portfolio_state.drawdown_pct)
    thresholds = input_payload.risk_config.drawdown_thresholds
    min_steps = int(input_payload.risk_config.hysteresis_bands.min_steps_in_state)
    freeze_cooldown_steps = int(input_payload.risk_config.recovery_policy.freeze_cooldown_steps)
    cooldown_remaining = max(int(current_state.cooldown_remaining_steps) - 1, 0)

    next_mode = current_state.mode
    transition_reason: str | None = None

    if drawdown >= thresholds.kill_pct:
        next_mode = RISK_MODE_KILL_HALTED
        transition_reason = RISK_REASON_CATASTROPHIC_DRAWDOWN
    elif current_state.mode == RISK_MODE_NORMAL:
        if drawdown >= thresholds.defensive_enter_pct:
            next_mode = RISK_MODE_DEFENSIVE
            transition_reason = RISK_REASON_DRAWDOWN_TO_DEFENSIVE
    elif current_state.mode == RISK_MODE_DEFENSIVE:
        if drawdown >= thresholds.freeze_enter_pct:
            next_mode = RISK_MODE_FREEZE_ENTRIES
            transition_reason = RISK_REASON_DRAWDOWN_TO_FREEZE
        elif drawdown <= thresholds.defensive_exit_pct and current_state.mode_step_count >= min_steps:
            next_mode = RISK_MODE_NORMAL
            transition_reason = RISK_REASON_DRAWDOWN_RECOVER_TO_NORMAL
    elif current_state.mode == RISK_MODE_FREEZE_ENTRIES:
        if drawdown >= thresholds.kill_pct:
            next_mode = RISK_MODE_KILL_HALTED
            transition_reason = RISK_REASON_CATASTROPHIC_DRAWDOWN
        elif (
            drawdown <= thresholds.freeze_exit_pct
            and current_state.mode_step_count >= min_steps
            and cooldown_remaining == 0
        ):
            next_mode = RISK_MODE_DEFENSIVE
            transition_reason = RISK_REASON_DRAWDOWN_RECOVER_TO_DEFENSIVE
    elif current_state.mode == RISK_MODE_KILL_HALTED:
        next_mode = RISK_MODE_KILL_HALTED

    if next_mode == current_state.mode:
        next_state = RiskState(
            mode=current_state.mode,
            kill_active=current_state.kill_active or current_state.mode == RISK_MODE_KILL_HALTED,
            freeze_entries_active=current_state.mode == RISK_MODE_FREEZE_ENTRIES,
            drawdown_regime=current_state.mode,
            last_transition_utc=current_state.last_transition_utc,
            consecutive_input_failures=0,
            recovery_token_present=current_state.recovery_token_present,
            mode_step_count=current_state.mode_step_count + 1,
            cooldown_remaining_steps=cooldown_remaining,
        )
        return next_state, None, None

    next_state = RiskState(
        mode=next_mode,
        kill_active=next_mode == RISK_MODE_KILL_HALTED,
        freeze_entries_active=next_mode == RISK_MODE_FREEZE_ENTRIES,
        drawdown_regime=next_mode,
        last_transition_utc=input_payload.decision_timestamp_utc,
        consecutive_input_failures=0,
        recovery_token_present=current_state.recovery_token_present,
        mode_step_count=0,
        cooldown_remaining_steps=freeze_cooldown_steps if next_mode == RISK_MODE_FREEZE_ENTRIES else 0,
    )
    transition_record = _build_transition_record(
        from_mode=current_state.mode,
        to_mode=next_mode,
        timestamp_utc=input_payload.decision_timestamp_utc,
        reason=transition_reason,
    )
    return next_state, transition_reason, transition_record


def _build_transition_record(*, from_mode: str, to_mode: str, timestamp_utc: str, reason: str | None) -> dict[str, Any] | None:
    if from_mode == to_mode:
        return None
    return {
        "from_mode": from_mode,
        "to_mode": to_mode,
        "transition_timestamp_utc": timestamp_utc,
        "reason_code": reason,
    }


def _transition_to_kill(*, current_state: RiskState, timestamp_utc: str, keep_cooldown: bool) -> RiskState:
    """Return a kill-halted state."""

    return RiskState(
        mode=RISK_MODE_KILL_HALTED,
        kill_active=True,
        freeze_entries_active=False,
        drawdown_regime=RISK_MODE_KILL_HALTED,
        last_transition_utc=timestamp_utc,
        consecutive_input_failures=0,
        recovery_token_present=current_state.recovery_token_present,
        mode_step_count=0,
        cooldown_remaining_steps=current_state.cooldown_remaining_steps if keep_cooldown else 0,
    )


def _validate_input_contract(input_payload: RiskOverlayInput) -> list[str]:
    """Validate the required first-wave input surface."""

    reasons: list[str] = []
    proposal = input_payload.agent_action_proposal
    portfolio = input_payload.portfolio_state
    market = input_payload.market_snapshot
    risk_state = input_payload.risk_state
    config = input_payload.risk_config

    if proposal.instrument == "" or market.instrument == "":
        reasons.append(RISK_REASON_MISSING_REQUIRED_INPUT)
    if not market.tradable:
        reasons.append(RISK_REASON_MISSING_REQUIRED_INPUT)

    decision_ts = _parse_timestamp(input_payload.decision_timestamp_utc)
    market_ts = _parse_timestamp(market.snapshot_timestamp_utc)
    portfolio_ts = _parse_timestamp(portfolio.state_timestamp_utc)
    proposal_ts = _parse_timestamp(proposal.proposal_timestamp_utc)
    if decision_ts is None or market_ts is None or portfolio_ts is None or proposal_ts is None:
        reasons.append(RISK_REASON_INVALID_TIMESTAMP)
        return reasons

    if _age_seconds(decision_ts, market_ts) > config.freshness_limits.max_market_data_age_seconds:
        reasons.append(RISK_REASON_STALE_MARKET_SNAPSHOT)
    if _age_seconds(decision_ts, portfolio_ts) > config.freshness_limits.max_portfolio_state_age_seconds:
        reasons.append(RISK_REASON_STALE_PORTFOLIO_STATE)
    if _age_seconds(decision_ts, proposal_ts) > config.freshness_limits.max_proposal_age_seconds:
        reasons.append(RISK_REASON_STALE_ACTION_PROPOSAL)

    numeric_values = (
        portfolio.equity,
        portfolio.gross_exposure,
        portfolio.net_exposure,
        portfolio.current_leverage,
        portfolio.instrument_exposure,
        portfolio.drawdown_pct,
        market.mid_price,
        proposal.target_exposure,
        proposal.requested_notional,
    )
    if not all(_is_finite_number(item) for item in numeric_values):
        reasons.append(RISK_REASON_INVALID_NUMERIC_INPUT)

    if risk_state.mode not in RISK_MODES:
        reasons.append(RISK_REASON_MISSING_REQUIRED_INPUT)
    if float(portfolio.equity) <= 0.0:
        reasons.append(RISK_REASON_INVALID_NUMERIC_INPUT)
    if float(portfolio.drawdown_pct) < 0.0:
        reasons.append(RISK_REASON_INVALID_NUMERIC_INPUT)

    deduped: list[str] = []
    for code in reasons:
        if code not in deduped:
            deduped.append(code)
    return deduped


def _build_applied_limits(*, config: RiskOverlayConfig, state: RiskState, effective_state: RiskState) -> dict[str, Any]:
    """Build a stable applied-limits payload for reporting."""

    hard_exposure_cap = min(
        config.exposure_limits.max_abs_target_exposure,
        config.exposure_limits.max_gross_exposure,
        config.exposure_limits.max_net_exposure,
        config.exposure_limits.max_instrument_exposure,
    )
    effective_exposure_cap = hard_exposure_cap
    effective_leverage_cap = config.leverage_limits.max_leverage
    if effective_state.mode == RISK_MODE_DEFENSIVE:
        effective_exposure_cap = hard_exposure_cap * config.exposure_limits.defensive_scale_down
        effective_leverage_cap = config.leverage_limits.max_leverage * config.leverage_limits.defensive_scale_down
    if effective_state.mode == RISK_MODE_FREEZE_ENTRIES:
        effective_exposure_cap = 0.0
        effective_leverage_cap = 0.0

    return {
        "input_mode": state.mode,
        "effective_mode": effective_state.mode,
        "hard_abs_target_exposure_cap": float(hard_exposure_cap),
        "effective_abs_target_exposure_cap": float(effective_exposure_cap),
        "hard_leverage_cap": float(config.leverage_limits.max_leverage),
        "effective_leverage_cap": float(effective_leverage_cap),
        "kill_drawdown_pct": float(config.drawdown_thresholds.kill_pct),
        "freeze_drawdown_pct": float(config.drawdown_thresholds.freeze_enter_pct),
        "defensive_drawdown_pct": float(config.drawdown_thresholds.defensive_enter_pct),
    }


def _build_decision(
    *,
    decision_type: str,
    status: str,
    approved_target_exposure: float,
    approved_action_raw: int,
    approved_leverage: float,
    applied_limits: dict[str, Any],
    reason_codes: Sequence[str],
    rule_hits: Sequence[str],
    state_transition: dict[str, Any] | None,
    counters: dict[str, Any],
    only_close_allowed: bool,
    kill_active: bool,
    decision_timestamp_utc: str,
    risk_state_after: RiskState,
    current_exposure: int,
) -> RiskDecision:
    """Build a stable decision payload."""

    approved_action = {
        "action_raw": int(approved_action_raw),
        "action_semantic": ACTION_MAPPING[int(approved_action_raw)],
        "approved_target_exposure": float(approved_target_exposure),
        "current_exposure": int(current_exposure),
    }
    return RiskDecision(
        decision_type=decision_type,
        status=status,
        approved_action=approved_action,
        approved_target_exposure=float(approved_target_exposure),
        approved_leverage=float(approved_leverage),
        applied_limits=dict(applied_limits),
        reason_codes=tuple(reason_codes),
        rule_hits=tuple(rule_hits),
        state_transition=state_transition,
        counters=dict(counters),
        only_close_allowed=bool(only_close_allowed),
        kill_active=bool(kill_active),
        decision_timestamp_utc=str(decision_timestamp_utc),
        contract_version=RISK_OVERLAY_CONTRACT_VERSION,
        risk_state_after=risk_state_after,
    )


def _with_transition_reason(reason_codes: Sequence[str], transition_reason: str | None) -> tuple[str, ...]:
    codes = list(reason_codes)
    if transition_reason is not None and transition_reason not in codes:
        codes.append(transition_reason)
    return tuple(codes)


def _discretize_target_exposure(value: float) -> int:
    """Project a target exposure into the env's fixed {-1, 0, +1} model."""

    numeric = float(value)
    if numeric >= 0.5:
        return 1
    if numeric <= -0.5:
        return -1
    return 0


def _age_seconds(reference_ts: datetime, candidate_ts: datetime) -> int:
    delta = reference_ts - candidate_ts
    return int(max(delta.total_seconds(), 0.0))


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_canonical_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _require_exact_keys(payload: Mapping[str, Any], expected_keys: set[str], *, location: str) -> None:
    actual_keys = set(payload.keys())
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys or extra_keys:
        raise ValueError(f"{location} keys must match exactly. missing={missing_keys} extra={extra_keys}")


def _require_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return dict(value)


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0.")
    return int(value)


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be an integer > 0.")
    return int(value)


def _require_non_negative_float(value: Any, field_name: str) -> float:
    if not _is_finite_number(value) or float(value) < 0.0:
        raise ValueError(f"{field_name} must be a finite number >= 0.")
    return float(value)


def _require_percentage(value: Any, field_name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be a finite percentage.")
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return numeric


def _require_fraction(value: Any, field_name: str) -> float:
    numeric = _require_percentage(value, field_name)
    if numeric > 1.0:
        raise ValueError(f"{field_name} must be <= 1.")
    return numeric


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _atomic_write_json(payload: Mapping[str, Any], dest: Path) -> None:
    tmp = dest.with_suffix(f"{dest.suffix}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to atomically write risk overlay json: {dest}") from exc


def _atomic_write_jsonl(rows: Sequence[Mapping[str, Any]], dest: Path) -> None:
    tmp = dest.with_suffix(f"{dest.suffix}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = "\n".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) for row in rows)
        if content:
            content = f"{content}\n"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to atomically write risk overlay jsonl: {dest}") from exc
