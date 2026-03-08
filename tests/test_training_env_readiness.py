"""Unit tests for Milestone 4.6 env readiness service."""

from __future__ import annotations

import json

from rl.env_readiness import START_POLICY_VALID_FROM_ROW, validate_training_env_readiness
from rl.episode_selector import SELECTION_POLICY_FIXED, SELECTION_POLICY_SEEDED_RANDOM
from tests.rl_readiness_fixtures import patch_read_parquet, seed_state_run


def test_readiness_passes_for_fixed_val_episode_on_readiness_domain(monkeypatch, tmp_path) -> None:
    run_id = "readiness_fixed_val"
    state_root, config_path, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=[
            {"partition": "train", "source_rel": "train_ready.parquet", "row_count": 6},
            {"partition": "val", "source_rel": "val_ready.parquet", "row_count": 6},
        ],
        selected_episode_ref={
            "scope": "partition",
            "partition": "val",
            "source_rel": "val_ready.parquet",
            "fold_id": None,
        },
    )
    patch_read_parquet(monkeypatch, frame_map)

    result = validate_training_env_readiness(
        run_id=run_id,
        state_root=state_root,
        env_config_payload=json.loads(config_path.read_text(encoding="utf-8")),
        selection_policy=SELECTION_POLICY_FIXED,
        start_policy=START_POLICY_VALID_FROM_ROW,
        min_remaining_steps=2,
        seed=42,
    )

    payload = result.readiness_payload
    assert payload["readiness_overall"] is True
    assert payload["eligibility_domain_used"] == "readiness"
    assert payload["selected_episode_domain_validity"]["eligible_for_readiness"] is True
    assert payload["selected_episode_domain_validity"]["eligible_for_training"] is False
    assert payload["selected_episode_refs"] == [
        {"scope": "partition", "partition": "val", "source_rel": "val_ready.parquet", "fold_id": None}
    ]
    assert payload["fixed_episode_input_source"] == "env_config.episode_ref"
    assert payload["orchestration_input_consumed"] is True
    assert payload["smoke_action_script"]["actions"] == [0, 1, 3]
    assert payload["selection_replay_match"] is True
    assert payload["reset_replay_match"] is True
    assert payload["rollout_replay_match"] is True
    assert payload["deterministic_replay_match"] is True
    assert payload["smoke_rollout_trace_summary"]["rollout_hash_version"] == "readiness_rollout_hash.v1"
    assert payload["smoke_rollout_trace_summary"]["rollout_hash_inputs"][-1] == "per_step.observation_hash"


def test_readiness_guard_fails_closed_when_usable_steps_are_insufficient(monkeypatch, tmp_path) -> None:
    run_id = "readiness_guard_fail"
    state_root, config_path, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=[{"partition": "train", "source_rel": "train_short.parquet", "row_count": 4, "warmup_rows": 1}],
    )
    patch_read_parquet(monkeypatch, frame_map)

    result = validate_training_env_readiness(
        run_id=run_id,
        state_root=state_root,
        env_config_payload=json.loads(config_path.read_text(encoding="utf-8")),
        selection_policy=SELECTION_POLICY_SEEDED_RANDOM,
        start_policy=START_POLICY_VALID_FROM_ROW,
        min_remaining_steps=4,
        seed=42,
    )

    payload = result.readiness_payload
    assert payload["episode_catalog_overall"] is True
    assert payload["readiness_overall"] is False
    assert payload["min_remaining_steps_guard_passed"] is False
    assert payload["usable_step_count_after_warmup"] == 2
    assert payload["min_remaining_steps_requested"] == 4
    assert payload["min_remaining_steps_effective"] == 4
    assert payload["errors"][0]["code"] == "READINESS_MIN_REMAINING_STEPS_FAILED"
