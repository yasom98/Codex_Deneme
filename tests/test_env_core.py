"""Unit tests for RL env core contract (Milestone 4.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl.env_core import (
    ACTION_CLOSE_POSITION,
    ACTION_HOLD,
    ACTION_OPEN_LONG,
    ACTION_OPEN_SHORT,
    EpisodeData,
    EpisodeRef,
    EpisodeRunnerConfig,
    EpisodeRunnerCore,
    EpisodeSource,
    EpisodeSpec,
)


def _episode_data() -> EpisodeData:
    return EpisodeData(
        observation_matrix=np.asarray(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        execution_price_vector=np.asarray([100.0, 110.0, 90.0], dtype=np.float64),
        mark_to_market_price_vector=np.asarray([100.0, 110.0, 90.0], dtype=np.float64),
        timestamp_vector=(
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T00:01:00+00:00",
            "2024-01-01T00:02:00+00:00",
        ),
        observation_columns=("feat_a", "evt_flag"),
        execution_price_column="close",
        mark_to_market_column="close",
        coercions_applied=(),
        episode_valid_start_row=0,
        warmup_applied=False,
    )


def _runner(data: EpisodeData, *, max_steps: int | None = None) -> EpisodeRunnerCore:
    return EpisodeRunnerCore(
        episode_ref=EpisodeRef(scope="partition", partition="train", source_rel="sample.parquet", fold_id=None),
        episode_data=data,
        config=EpisodeRunnerConfig(
            initial_cash=1000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            max_steps=max_steps,
            reward_scale=1.0,
            reward_clip_min=None,
            reward_clip_max=None,
            seed=42,
        ),
    )


def test_reset_step_progression_and_reward_timing() -> None:
    runner = _runner(_episode_data())
    obs, info = runner.reset(seed=42)
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert info["seed"] == 42

    _, reward_1, term_1, trunc_1, info_1 = runner.step(ACTION_OPEN_LONG)
    assert reward_1 == 10.0
    assert term_1 is False
    assert trunc_1 is False
    assert info_1["position_before"] == 0
    assert info_1["position_after"] == 1
    assert info_1["action_semantic"] == "OPEN_LONG"

    _, reward_2, term_2, trunc_2, info_2 = runner.step(ACTION_CLOSE_POSITION)
    assert reward_2 == 0.0
    assert term_2 is True
    assert trunc_2 is False
    assert info_2["position_before"] == 1
    assert info_2["position_after"] == 0
    assert info_2["termination_reason"] == "data_exhausted"


def test_invalid_actions_are_noop_with_info_flags() -> None:
    runner = _runner(
        EpisodeData(
            observation_matrix=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            execution_price_vector=np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
            mark_to_market_price_vector=np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
            timestamp_vector=(
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T00:01:00+00:00",
                "2024-01-01T00:02:00+00:00",
                "2024-01-01T00:03:00+00:00",
            ),
            observation_columns=("feat_a",),
            execution_price_column="close",
            mark_to_market_column="close",
            coercions_applied=(),
            episode_valid_start_row=0,
            warmup_applied=False,
        )
    )

    runner.reset(seed=42)
    _, _, _, _, info_1 = runner.step(ACTION_CLOSE_POSITION)
    assert info_1["invalid_action"] is True
    assert info_1["invalid_action_reason"] == "already_flat"

    _, _, _, _, info_2 = runner.step(ACTION_OPEN_LONG)
    assert info_2["invalid_action"] is False
    assert info_2["position_after"] == 1

    _, _, term_3, _, info_3 = runner.step(ACTION_OPEN_SHORT)
    assert info_3["invalid_action"] is True
    assert info_3["invalid_action_reason"] == "reversal_disallowed"
    assert info_3["position_after"] == 1
    assert term_3 is True


def test_max_steps_sets_truncation_only() -> None:
    runner = _runner(
        EpisodeData(
            observation_matrix=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            execution_price_vector=np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
            mark_to_market_price_vector=np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
            timestamp_vector=(
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T00:01:00+00:00",
                "2024-01-01T00:02:00+00:00",
                "2024-01-01T00:03:00+00:00",
            ),
            observation_columns=("feat_a",),
            execution_price_column="close",
            mark_to_market_column="close",
            coercions_applied=(),
            episode_valid_start_row=0,
            warmup_applied=False,
        ),
        max_steps=1,
    )

    runner.reset(seed=42)
    _, _, terminated, truncated, info = runner.step(ACTION_HOLD)
    assert terminated is False
    assert truncated is True
    assert info["truncation_reason"] == "max_steps"


def test_same_seed_and_actions_produce_identical_rollout() -> None:
    actions = [ACTION_OPEN_LONG, ACTION_HOLD, ACTION_CLOSE_POSITION]
    data = EpisodeData(
        observation_matrix=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        execution_price_vector=np.asarray([100.0, 101.0, 102.0, 100.0], dtype=np.float64),
        mark_to_market_price_vector=np.asarray([100.0, 101.0, 102.0, 100.0], dtype=np.float64),
        timestamp_vector=(
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T00:01:00+00:00",
            "2024-01-01T00:02:00+00:00",
            "2024-01-01T00:03:00+00:00",
        ),
        observation_columns=("feat_a",),
        execution_price_column="close",
        mark_to_market_column="close",
        coercions_applied=(),
        episode_valid_start_row=0,
        warmup_applied=False,
    )

    runner_a = _runner(data)
    runner_b = _runner(data)

    runner_a.reset(seed=42)
    runner_b.reset(seed=42)

    out_a: list[tuple[float, bool, bool, float]] = []
    out_b: list[tuple[float, bool, bool, float]] = []

    for action in actions:
        _, r_a, t_a, tr_a, i_a = runner_a.step(action)
        _, r_b, t_b, tr_b, i_b = runner_b.step(action)
        out_a.append((r_a, t_a, tr_a, i_a["portfolio_value"]))
        out_b.append((r_b, t_b, tr_b, i_b["portfolio_value"]))

    assert out_a == out_b


def test_episode_source_reads_once_and_runner_has_no_step_io() -> None:
    call_count = {"read_parquet": 0}
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC"),
            "feat_a": pd.Series([1.0, 2.0, 3.0, 4.0], dtype="float32"),
            "evt_flag": pd.Series([0, 1, 0, 1], dtype="uint8"),
            "close": pd.Series([100.0, 101.0, 102.0, 103.0], dtype="float32"),
        }
    )

    def _fake_read(_: Path) -> pd.DataFrame:
        call_count["read_parquet"] += 1
        return frame.copy()

    source = EpisodeSource(read_parquet_fn=_fake_read)
    spec = EpisodeSpec(
        scope="partition",
        partition="train",
        source_rel="sample.parquet",
        fold_id=None,
        output_path=Path("/tmp/sample.parquet"),
        row_count=4,
    )
    episode_data = source.load_episode(
        spec=spec,
        expected_columns=["timestamp", "feat_a", "evt_flag", "close"],
        observation_columns=["feat_a", "evt_flag"],
        strict_post_valid_numeric_columns=["feat_a", "evt_flag"],
        expected_dtypes={
            "timestamp": "datetime64[ns, UTC]",
            "feat_a": "float32",
            "evt_flag": "uint8",
            "close": "float32",
        },
        timestamp_column="timestamp",
        execution_price_column="close",
        mark_to_market_column="close",
        include_timestamp_in_observation=False,
        observation_output_dtype="float32",
        allowed_safe_casts={"uint8->float32"},
        valid_observation_start_row=0,
        valid_observation_start_timestamp="2024-01-01T00:00:00+00:00",
        warmup_head_nan_profile={},
    )

    assert call_count["read_parquet"] == 1
    assert episode_data.observation_matrix.shape == (4, 2)
    assert episode_data.execution_price_column == "close"

    runner = _runner(episode_data)
    runner.reset(seed=42)
    runner.step(ACTION_HOLD)
    runner.step(ACTION_HOLD)
    runner.step(ACTION_HOLD)

    assert call_count["read_parquet"] == 1


def test_episode_source_allows_leading_warmup_and_reset_starts_at_valid_row() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
            "EMA_200": pd.Series([np.nan, np.nan, 3.0, 4.0, 5.0], dtype="float32"),
            "evt_flag": pd.Series([0, 1, 0, 1, 0], dtype="uint8"),
            "close": pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], dtype="float32"),
        }
    )

    source = EpisodeSource(read_parquet_fn=lambda _: frame.copy())
    spec = EpisodeSpec(
        scope="partition",
        partition="train",
        source_rel="warmup.parquet",
        fold_id=None,
        output_path=Path("/tmp/warmup.parquet"),
        row_count=5,
    )
    episode_data = source.load_episode(
        spec=spec,
        expected_columns=["timestamp", "EMA_200", "evt_flag", "close"],
        observation_columns=["EMA_200", "evt_flag"],
        strict_post_valid_numeric_columns=["EMA_200", "evt_flag"],
        expected_dtypes={
            "timestamp": "datetime64[ns, UTC]",
            "EMA_200": "float32",
            "evt_flag": "uint8",
            "close": "float32",
        },
        timestamp_column="timestamp",
        execution_price_column="close",
        mark_to_market_column="close",
        include_timestamp_in_observation=False,
        observation_output_dtype="float32",
        allowed_safe_casts={"uint8->float32"},
        valid_observation_start_row=2,
        valid_observation_start_timestamp="2024-01-01T00:02:00+00:00",
        warmup_head_nan_profile={"EMA_200": 2},
    )

    assert episode_data.episode_valid_start_row == 2
    assert episode_data.warmup_applied is True

    runner = _runner(episode_data)
    obs, info = runner.reset(seed=42)
    assert np.allclose(obs, np.asarray([3.0, 0.0], dtype=np.float32))
    assert info["episode_valid_start_row"] == 2
    assert info["effective_episode_start_row"] == 2
    assert info["warmup_applied"] is True
    assert info["episode_transitions"] == 2


def test_episode_source_fails_closed_on_post_valid_non_finite() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
            "EMA_200": pd.Series([np.nan, np.nan, 3.0, np.nan, 5.0], dtype="float32"),
            "evt_flag": pd.Series([0, 1, 0, 1, 0], dtype="uint8"),
            "close": pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], dtype="float32"),
        }
    )

    source = EpisodeSource(read_parquet_fn=lambda _: frame.copy())
    spec = EpisodeSpec(
        scope="partition",
        partition="train",
        source_rel="warmup_fail.parquet",
        fold_id=None,
        output_path=Path("/tmp/warmup_fail.parquet"),
        row_count=5,
    )

    try:
        source.load_episode(
            spec=spec,
            expected_columns=["timestamp", "EMA_200", "evt_flag", "close"],
            observation_columns=["EMA_200", "evt_flag"],
            strict_post_valid_numeric_columns=["EMA_200", "evt_flag"],
            expected_dtypes={
                "timestamp": "datetime64[ns, UTC]",
                "EMA_200": "float32",
                "evt_flag": "uint8",
                "close": "float32",
            },
            timestamp_column="timestamp",
            execution_price_column="close",
            mark_to_market_column="close",
            include_timestamp_in_observation=False,
            observation_output_dtype="float32",
            allowed_safe_casts={"uint8->float32"},
            valid_observation_start_row=2,
            valid_observation_start_timestamp="2024-01-01T00:02:00+00:00",
            warmup_head_nan_profile={"EMA_200": 2},
        )
    except ValueError as exc:
        assert str(exc) == "POST_VALID_OBSERVATION_NAN:EMA_200:3"
    else:  # pragma: no cover
        raise AssertionError("Expected post-valid observation NaN failure.")


def test_episode_source_accepts_explicit_supertrend_geometry_contract() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
            "ST_trend": pd.Series([1.0, 1.0, -1.0, -1.0, -1.0], dtype="float32"),
            "ST_active_line": pd.Series([np.nan, 100.0, 101.0, 102.0, 103.0], dtype="float32"),
            "ST_distance_to_active_line": pd.Series([np.nan, 1.0, -1.0, -1.0, 1.0], dtype="float32"),
            "evt_st_buy": pd.Series([0, 1, 0, 0, 0], dtype="uint8"),
            "close": pd.Series([99.0, 101.0, 100.0, 101.0, 104.0], dtype="float32"),
        }
    )

    source = EpisodeSource(read_parquet_fn=lambda _: frame.copy())
    spec = EpisodeSpec(
        scope="partition",
        partition="train",
        source_rel="st_geometry.parquet",
        fold_id=None,
        output_path=Path("/tmp/st_geometry.parquet"),
        row_count=5,
    )
    episode_data = source.load_episode(
        spec=spec,
        expected_columns=["timestamp", "ST_trend", "ST_active_line", "ST_distance_to_active_line", "evt_st_buy", "close"],
        observation_columns=["ST_trend", "ST_active_line", "ST_distance_to_active_line", "evt_st_buy"],
        strict_post_valid_numeric_columns=["ST_trend", "ST_active_line", "ST_distance_to_active_line", "evt_st_buy"],
        expected_dtypes={
            "timestamp": "datetime64[ns, UTC]",
            "ST_trend": "float32",
            "ST_active_line": "float32",
            "ST_distance_to_active_line": "float32",
            "evt_st_buy": "uint8",
            "close": "float32",
        },
        timestamp_column="timestamp",
        execution_price_column="close",
        mark_to_market_column="close",
        include_timestamp_in_observation=False,
        observation_output_dtype="float32",
        allowed_safe_casts={"uint8->float32"},
        valid_observation_start_row=1,
        valid_observation_start_timestamp="2024-01-01T00:01:00+00:00",
        warmup_head_nan_profile={"ST_active_line": 1, "ST_distance_to_active_line": 1},
    )

    assert episode_data.warmup_applied is True
    runner = _runner(episode_data)
    obs, info = runner.reset(seed=42)
    assert np.allclose(obs, np.asarray([1.0, 100.0, 1.0, 1.0], dtype=np.float32))
    assert info["effective_episode_start_row"] == 1


def test_episode_source_fails_closed_on_post_valid_geometry_non_finite() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
            "ST_trend": pd.Series([1.0, 1.0, -1.0, -1.0, -1.0], dtype="float32"),
            "ST_active_line": pd.Series([np.nan, 100.0, 101.0, np.nan, 103.0], dtype="float32"),
            "ST_distance_to_active_line": pd.Series([np.nan, 1.0, -1.0, np.nan, 1.0], dtype="float32"),
            "evt_st_buy": pd.Series([0, 1, 0, 0, 0], dtype="uint8"),
            "close": pd.Series([99.0, 101.0, 100.0, 101.0, 104.0], dtype="float32"),
        }
    )

    source = EpisodeSource(read_parquet_fn=lambda _: frame.copy())
    spec = EpisodeSpec(
        scope="partition",
        partition="train",
        source_rel="st_geometry_fail.parquet",
        fold_id=None,
        output_path=Path("/tmp/st_geometry_fail.parquet"),
        row_count=5,
    )

    try:
        source.load_episode(
            spec=spec,
            expected_columns=["timestamp", "ST_trend", "ST_active_line", "ST_distance_to_active_line", "evt_st_buy", "close"],
            observation_columns=["ST_trend", "ST_active_line", "ST_distance_to_active_line", "evt_st_buy"],
            strict_post_valid_numeric_columns=["ST_trend", "ST_active_line", "ST_distance_to_active_line", "evt_st_buy"],
            expected_dtypes={
                "timestamp": "datetime64[ns, UTC]",
                "ST_trend": "float32",
                "ST_active_line": "float32",
                "ST_distance_to_active_line": "float32",
                "evt_st_buy": "uint8",
                "close": "float32",
            },
            timestamp_column="timestamp",
            execution_price_column="close",
            mark_to_market_column="close",
            include_timestamp_in_observation=False,
            observation_output_dtype="float32",
            allowed_safe_casts={"uint8->float32"},
            valid_observation_start_row=1,
            valid_observation_start_timestamp="2024-01-01T00:01:00+00:00",
            warmup_head_nan_profile={"ST_active_line": 1, "ST_distance_to_active_line": 1},
        )
    except ValueError as exc:
        assert str(exc) == "POST_VALID_OBSERVATION_NAN:ST_active_line:3"
    else:  # pragma: no cover
        raise AssertionError("Expected geometry post-valid observation NaN failure.")


def test_episode_source_fails_when_too_short_after_warmup() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC"),
            "EMA_200": pd.Series([np.nan, 2.0], dtype="float32"),
            "evt_flag": pd.Series([0, 1], dtype="uint8"),
            "close": pd.Series([100.0, 101.0], dtype="float32"),
        }
    )

    source = EpisodeSource(read_parquet_fn=lambda _: frame.copy())
    spec = EpisodeSpec(
        scope="partition",
        partition="train",
        source_rel="too_short.parquet",
        fold_id=None,
        output_path=Path("/tmp/too_short.parquet"),
        row_count=2,
    )

    try:
        source.load_episode(
            spec=spec,
            expected_columns=["timestamp", "EMA_200", "evt_flag", "close"],
            observation_columns=["EMA_200", "evt_flag"],
            strict_post_valid_numeric_columns=["EMA_200", "evt_flag"],
            expected_dtypes={
                "timestamp": "datetime64[ns, UTC]",
                "EMA_200": "float32",
                "evt_flag": "uint8",
                "close": "float32",
            },
            timestamp_column="timestamp",
            execution_price_column="close",
            mark_to_market_column="close",
            include_timestamp_in_observation=False,
            observation_output_dtype="float32",
            allowed_safe_casts={"uint8->float32"},
            valid_observation_start_row=1,
            valid_observation_start_timestamp="2024-01-01T00:01:00+00:00",
            warmup_head_nan_profile={"EMA_200": 1},
        )
    except ValueError as exc:
        assert str(exc) == "EPISODE_TOO_SHORT_AFTER_WARMUP:1:2"
    else:  # pragma: no cover
        raise AssertionError("Expected too-short-after-warmup failure.")
