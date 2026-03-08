"""Unit tests for Milestone 4.6 episode catalog."""

from __future__ import annotations

from rl.episode_catalog import build_episode_catalog
from tests.rl_readiness_fixtures import patch_read_parquet, seed_state_run


def test_episode_catalog_persists_dual_eligibility_and_sort_policy(monkeypatch, tmp_path) -> None:
    run_id = "episode_catalog_dual_domain"
    state_root, _, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=[
            {"partition": "val", "source_rel": "z_val.parquet", "row_count": 6, "warmup_rows": 0},
            {"partition": "train", "source_rel": "b_train.parquet", "row_count": 6, "warmup_rows": 1},
            {"partition": "train", "source_rel": "a_train.parquet", "row_count": 6, "warmup_rows": 0},
        ],
    )
    patch_read_parquet(monkeypatch, frame_map)

    result = build_episode_catalog(run_id=run_id, state_root=state_root)

    assert result.payload["episode_catalog_overall"] is True
    assert result.payload["selection_order_policy"] == {
        "partition_order": {"train": 0, "val": 1, "test": 2},
        "scope_order": {"partition": 0, "fold": 1, "aggregate": 2},
        "fold_order_policy": "null_as_minus_one_then_ascending",
        "source_rel_order_policy": "lexicographic_ascending",
        "final_episode_sort_key_schema": [
            "partition_order",
            "scope_order",
            "fold_id_normalized",
            "source_rel",
        ],
    }
    assert result.payload["eligible_episode_refs_sorted_by_domain"]["training"] == [
        {"scope": "partition", "partition": "train", "source_rel": "a_train.parquet", "fold_id": None},
        {"scope": "partition", "partition": "train", "source_rel": "b_train.parquet", "fold_id": None},
    ]
    assert result.payload["eligible_episode_count_by_domain"] == {"readiness": 3, "training": 2}

    entry = result.entries_by_key[("partition", "val", "z_val.parquet", None)]
    assert entry.eligible_for_readiness is True
    assert entry.eligible_for_training is False
    assert entry.training_eligibility_reasons == ("partition_not_train",)
    assert entry.usable_row_count_after_warmup == 6
    assert entry.usable_step_count_after_warmup == 5


def test_episode_catalog_excludes_post_valid_nan_episode_from_training(monkeypatch, tmp_path) -> None:
    run_id = "episode_catalog_post_valid_nan"
    state_root, _, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=[
            {"partition": "train", "source_rel": "good_train.parquet", "row_count": 6},
            {"partition": "train", "source_rel": "bad_train.parquet", "row_count": 6, "post_valid_nan_row": 4},
        ],
    )
    patch_read_parquet(monkeypatch, frame_map)

    result = build_episode_catalog(run_id=run_id, state_root=state_root)

    bad_entry = result.entries_by_key[("partition", "train", "bad_train.parquet", None)]
    assert bad_entry.eligible_for_readiness is False
    assert bad_entry.eligible_for_training is False
    assert "post_valid_observation_non_finite" in bad_entry.readiness_eligibility_reasons
    assert result.payload["eligible_episode_refs_sorted_by_domain"]["training"] == [
        {"scope": "partition", "partition": "train", "source_rel": "good_train.parquet", "fold_id": None}
    ]
