"""Unit tests for Milestone 4.10 risk overlay gateway."""

from __future__ import annotations

import json
from pathlib import Path

from rl.risk_overlay import (
    RISK_DECISION_CLAMP,
    RISK_DECISION_KILL,
    RISK_DECISION_VETO,
    RISK_MODE_DEFENSIVE,
    RISK_MODE_KILL_HALTED,
    RISK_MODE_NORMAL,
    AgentActionProposal,
    MarketSnapshot,
    PortfolioRiskState,
    RiskOverlayInput,
    RiskState,
    build_initial_risk_state,
    build_risk_overlay_session,
    decide_risk_overlay,
    load_risk_overlay_config,
)


def _write_config(tmp_path: Path, *, overrides: dict | None = None) -> Path:
    payload = {
        "config_version": "risk_overlay.v1",
        "allowed_instruments": ["BTC_USDT"],
        "freshness_limits": {
            "max_market_data_age_seconds": 0,
            "max_portfolio_state_age_seconds": 0,
            "max_proposal_age_seconds": 0,
        },
        "exposure_limits": {
            "max_abs_target_exposure": 1.0,
            "max_gross_exposure": 1.0,
            "max_net_exposure": 1.0,
            "max_instrument_exposure": 1.0,
            "defensive_scale_down": 0.5,
        },
        "leverage_limits": {
            "max_leverage": 1.0,
            "defensive_scale_down": 0.5,
        },
        "drawdown_thresholds": {
            "defensive_enter_pct": 0.02,
            "defensive_exit_pct": 0.01,
            "freeze_enter_pct": 0.04,
            "freeze_exit_pct": 0.025,
            "kill_pct": 0.08,
        },
        "hysteresis_bands": {
            "min_steps_in_state": 1,
        },
        "recovery_policy": {
            "freeze_cooldown_steps": 1,
            "systemic_failure_kill_threshold": 2,
            "kill_requires_recovery_token": True,
        },
    }
    if overrides:
        payload.update(overrides)
    path = tmp_path / "risk_overlay_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_input(
    *,
    config_path: Path,
    risk_state: RiskState | None = None,
    decision_timestamp_utc: str = "2026-03-08T00:00:00+00:00",
    proposal_timestamp_utc: str = "2026-03-08T00:00:00+00:00",
    state_timestamp_utc: str = "2026-03-08T00:00:00+00:00",
    snapshot_timestamp_utc: str = "2026-03-08T00:00:00+00:00",
    target_exposure: float = 1.0,
    current_exposure: float = 0.0,
    drawdown_pct: float = 0.0,
    action_raw: int = 1,
    instrument: str = "BTC_USDT",
) -> RiskOverlayInput:
    config, _ = load_risk_overlay_config(config_path)
    return RiskOverlayInput(
        decision_timestamp_utc=decision_timestamp_utc,
        contract_version="risk_overlay_gateway.v1",
        agent_action_proposal=AgentActionProposal(
            instrument=instrument,
            proposal_timestamp_utc=proposal_timestamp_utc,
            target_exposure=target_exposure,
            requested_side="LONG" if target_exposure > 0 else "SHORT" if target_exposure < 0 else "FLAT",
            intent_type="increase" if abs(target_exposure) > abs(current_exposure) else "hold",
            requested_notional=100.0,
            action_raw=action_raw,
        ),
        portfolio_state=PortfolioRiskState(
            state_timestamp_utc=state_timestamp_utc,
            equity=1000.0,
            gross_exposure=abs(current_exposure),
            net_exposure=current_exposure,
            current_leverage=abs(current_exposure),
            instrument_exposure=current_exposure,
            drawdown_pct=drawdown_pct,
        ),
        market_snapshot=MarketSnapshot(
            snapshot_timestamp_utc=snapshot_timestamp_utc,
            instrument=instrument,
            mid_price=100.0,
            tradable=True,
        ),
        risk_state=risk_state or build_initial_risk_state(),
        risk_config=config,
    )


def test_local_invalid_input_veto_then_systemic_failure_kill(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config, config_hash = load_risk_overlay_config(config_path)
    session = build_risk_overlay_session(config=config, config_path=config_path, config_hash=config_hash)
    session.start_episode()

    first = session.evaluate(
        _make_input(
            config_path=config_path,
            risk_state=session.state,
            snapshot_timestamp_utc="2026-03-07T23:59:58+00:00",
        )
    )
    assert first.decision_type == RISK_DECISION_VETO
    assert first.risk_state_after.mode == RISK_MODE_NORMAL
    assert first.counters["consecutive_input_failures"] == 1

    second = session.evaluate(
        _make_input(
            config_path=config_path,
            risk_state=session.state,
            snapshot_timestamp_utc="2026-03-07T23:59:58+00:00",
        )
    )
    assert second.decision_type == RISK_DECISION_KILL
    assert second.risk_state_after.mode == RISK_MODE_KILL_HALTED


def test_hard_exposure_cap_clamps_and_never_allows(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        overrides={
            "exposure_limits": {
                "max_abs_target_exposure": 0.0,
                "max_gross_exposure": 0.0,
                "max_net_exposure": 0.0,
                "max_instrument_exposure": 0.0,
                "defensive_scale_down": 0.5,
            }
        },
    )
    decision = decide_risk_overlay(_make_input(config_path=config_path))

    assert decision.decision_type == RISK_DECISION_CLAMP
    assert decision.approved_action["action_semantic"] == "HOLD"
    assert "RISK_HARD_EXPOSURE_CAP_CLAMP" in decision.reason_codes


def test_drawdown_hysteresis_enters_and_exits_defensive_without_thrashing(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    first = decide_risk_overlay(_make_input(config_path=config_path, current_exposure=1.0, drawdown_pct=0.03, action_raw=0))
    assert first.risk_state_after.mode == RISK_MODE_DEFENSIVE
    assert first.decision_type == RISK_DECISION_CLAMP

    second = decide_risk_overlay(
        _make_input(
            config_path=config_path,
            risk_state=first.risk_state_after,
            current_exposure=0.0,
            target_exposure=0.0,
            action_raw=0,
            drawdown_pct=0.015,
        )
    )
    assert second.risk_state_after.mode == RISK_MODE_DEFENSIVE

    third = decide_risk_overlay(
        _make_input(
            config_path=config_path,
            risk_state=second.risk_state_after,
            current_exposure=0.0,
            target_exposure=0.0,
            action_raw=0,
            drawdown_pct=0.005,
        )
    )
    assert third.risk_state_after.mode == RISK_MODE_NORMAL


def test_catastrophic_drawdown_kill_closes_existing_position(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    decision = decide_risk_overlay(
        _make_input(
            config_path=config_path,
            current_exposure=1.0,
            target_exposure=1.0,
            action_raw=0,
            drawdown_pct=0.09,
        )
    )

    assert decision.decision_type == RISK_DECISION_KILL
    assert decision.risk_state_after.mode == RISK_MODE_KILL_HALTED
    assert decision.approved_action["action_semantic"] == "CLOSE_POSITION"
