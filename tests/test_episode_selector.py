"""Unit tests for Milestone 4.6 episode selector."""

from __future__ import annotations

from rl.episode_catalog import build_episode_catalog
from rl.episode_selector import SELECTION_POLICY_FIXED, SELECTION_POLICY_SEEDED_RANDOM, select_episode
from rl.env_core import EpisodeRef
from tests.rl_readiness_fixtures import patch_read_parquet, seed_state_run


def test_seeded_random_selection_uses_training_domain_and_replays_stably(monkeypatch, tmp_path) -> None:
    run_id = "selector_seeded_random"
    state_root, _, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=[
            {"partition": "train", "source_rel": "b_train.parquet", "row_count": 6},
            {"partition": "train", "source_rel": "a_train.parquet", "row_count": 6},
            {"partition": "val", "source_rel": "val_ready.parquet", "row_count": 6},
        ],
    )
    patch_read_parquet(monkeypatch, frame_map)
    catalog = build_episode_catalog(run_id=run_id, state_root=state_root)

    first = select_episode(catalog=catalog, selection_policy=SELECTION_POLICY_SEEDED_RANDOM, seed=42)
    second = select_episode(catalog=catalog, selection_policy=SELECTION_POLICY_SEEDED_RANDOM, seed=42)

    assert first.errors == ()
    assert first.eligible_domain_used == "training"
    assert first.candidate_refs_sorted == (
        {"scope": "partition", "partition": "train", "source_rel": "a_train.parquet", "fold_id": None},
        {"scope": "partition", "partition": "train", "source_rel": "b_train.parquet", "fold_id": None},
    )
    assert first.trace == second.trace


def test_fixed_episode_uses_readiness_domain_and_persists_input_provenance(monkeypatch, tmp_path) -> None:
    run_id = "selector_fixed_readiness"
    state_root, _, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=[
            {"partition": "train", "source_rel": "train_ready.parquet", "row_count": 6},
            {"partition": "val", "source_rel": "val_ready.parquet", "row_count": 6},
        ],
    )
    patch_read_parquet(monkeypatch, frame_map)
    catalog = build_episode_catalog(run_id=run_id, state_root=state_root)
    episode_ref = EpisodeRef(scope="partition", partition="val", source_rel="val_ready.parquet", fold_id=None)

    result = select_episode(catalog=catalog, selection_policy=SELECTION_POLICY_FIXED, seed=42, fixed_episode_ref=episode_ref)

    assert result.errors == ()
    assert result.eligible_domain_used == "readiness"
    assert result.fixed_episode_input_source == "env_config.episode_ref"
    assert result.fixed_episode_input_value == {
        "scope": "partition",
        "partition": "val",
        "source_rel": "val_ready.parquet",
        "fold_id": None,
    }
    assert result.trace["selected_episode_ref"] == result.fixed_episode_input_value
