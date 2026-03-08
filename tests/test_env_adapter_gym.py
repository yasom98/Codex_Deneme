"""Unit tests for Gymnasium env adapter contract (Milestone 4.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rl.env_adapter_gym import TradingEnvGym
from rl.env_contract import (
    ENV_CONTRACT_POST_VALID_OBSERVATION_NAN,
    ENV_CONTRACT_RUNTIME_PRICE_CONFIG_MISMATCH,
    parse_env_config,
)


def _seed_state_run(
    tmp_path: Path,
    run_id: str,
    *,
    float64_feature: bool = False,
    warmup_rows: int = 0,
    post_valid_nan_row: int | None = None,
    row_count: int = 4,
    supertrend_geometry: bool = False,
) -> tuple[Path, Path, dict[str, pd.DataFrame]]:
    state_root = tmp_path / "runs" / run_id / "data_states"
    reports_root = state_root / "reports"
    parquet_root = state_root / "parquet" / "partitions" / "train"
    reports_root.mkdir(parents=True, exist_ok=True)
    parquet_root.mkdir(parents=True, exist_ok=True)

    parquet_path = parquet_root / "sample.parquet"
    parquet_path.write_text("state-placeholder", encoding="utf-8")

    if supertrend_geometry:
        close_values = [100.0 + float(index) for index in range(row_count)]
        active_line_values = [99.0 + float(index) for index in range(row_count)]
        distance_values = [close_values[index] - active_line_values[index] for index in range(row_count)]
        for index in range(min(warmup_rows, row_count)):
            active_line_values[index] = float("nan")
            distance_values[index] = float("nan")
        if post_valid_nan_row is not None:
            active_line_values[post_valid_nan_row] = float("nan")
            distance_values[post_valid_nan_row] = float("nan")

        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=row_count, freq="1min", tz="UTC"),
                "ST_trend": pd.Series(([1.0, -1.0] * ((row_count // 2) + 1))[:row_count], dtype="float32"),
                "ST_active_line": pd.Series(active_line_values, dtype="float32"),
                "ST_distance_to_active_line": pd.Series(distance_values, dtype="float32"),
                "evt_st_buy": pd.Series(([0, 1] * ((row_count // 2) + 1))[:row_count], dtype="uint8"),
                "close": pd.Series(close_values, dtype="float32"),
            }
        )
        selected_state_columns = ["timestamp", "ST_trend", "ST_up", "ST_dn", "evt_st_buy"]
        selected_state_dtypes = {
            "timestamp": "datetime64[ns, UTC]",
            "ST_trend": "float32",
            "ST_up": "float32",
            "ST_dn": "float32",
            "evt_st_buy": "uint8",
        }
        state_feature_columns = ["ST_trend", "ST_active_line", "ST_distance_to_active_line", "evt_st_buy"]
        event_columns = ["evt_st_buy"]
        regime_columns = ["ST_trend"]
        geometry_columns = ["ST_active_line", "ST_distance_to_active_line"]
        strict_post_valid_numeric_columns = list(state_feature_columns)
        conditional_raw_columns = ["ST_up", "ST_dn"]
        conditional_column_replacements = {
            "ST_up": ["ST_active_line", "ST_distance_to_active_line"],
            "ST_dn": ["ST_active_line", "ST_distance_to_active_line"],
        }
        geometry_feature_formulas = {
            "ST_active_line_formula": "deterministic_single_finite_band_with_trend_consistency",
            "ST_distance_to_active_line_formula": "close_minus_active_line",
        }
        required_observation_columns = (
            ["ST_active_line", "ST_distance_to_active_line"] if warmup_rows > 0 else []
        )
        head_nan_profile = (
            {"ST_active_line": warmup_rows, "ST_distance_to_active_line": warmup_rows} if warmup_rows > 0 else {}
        )
    else:
        feature_name = "EMA_200" if warmup_rows > 0 or post_valid_nan_row is not None else "feat_x"
        feat_dtype = "float64" if float64_feature else "float32"
        feature_values: list[float] = [float(index + 1) for index in range(row_count)]
        for index in range(min(warmup_rows, row_count)):
            feature_values[index] = float("nan")
        if post_valid_nan_row is not None:
            feature_values[post_valid_nan_row] = float("nan")

        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=row_count, freq="1min", tz="UTC"),
                feature_name: pd.Series(feature_values, dtype=feat_dtype),
                "evt_flag": pd.Series(([0, 1] * ((row_count // 2) + 1))[:row_count], dtype="uint8"),
                "close": pd.Series([100.0 + float(index) for index in range(row_count)], dtype="float32"),
            }
        )
        selected_state_columns = ["timestamp", feature_name, "evt_flag"]
        selected_state_dtypes = {
            "timestamp": "datetime64[ns, UTC]",
            feature_name: feat_dtype,
            "evt_flag": "uint8",
        }
        state_feature_columns = [feature_name, "evt_flag"]
        event_columns = ["evt_flag"]
        regime_columns = []
        geometry_columns = []
        strict_post_valid_numeric_columns = list(state_feature_columns)
        conditional_raw_columns = []
        conditional_column_replacements = {}
        geometry_feature_formulas = {}
        required_observation_columns = [feature_name] if warmup_rows > 0 else []
        head_nan_profile = {feature_name: warmup_rows} if warmup_rows > 0 else {}

    valid_from_timestamp = None
    if warmup_rows < row_count:
        valid_from_timestamp = pd.Timestamp(frame["timestamp"].iloc[warmup_rows]).isoformat()

    manifest = {
        "manifest_version": "states.manifest.v1",
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "builder_version": "state_builder.v1",
        "state_build_id": "state-build-id-1",
        "source_lineage": {
            "dataset_manifest_path": str(state_root / "dummy_dataset_manifest.json"),
            "dataset_build_report_path": str(state_root / "dummy_dataset_report.json"),
        },
        "source_hashes": {
            "dataset_manifest_hash": "hash-a",
            "dataset_build_report_hash": "hash-b",
            "source_file_inventory_hash": "hash-c",
        },
        "split_mode": "ratio_chrono",
        "output_semantics": {
            "mode": "standard_partitions",
            "fold_outputs_generated": False,
            "top_level_partitions_generated": True,
            "aggregate_walk_forward": False,
        },
        "column_selection_contract": {
            "timestamp_column": "timestamp",
            "selected_state_columns": selected_state_columns,
            "selected_state_dtypes": selected_state_dtypes,
        },
        "observation_contract": {
            "selected_input_columns": ["timestamp", *state_feature_columns],
            "state_feature_columns": state_feature_columns,
            "event_columns": event_columns,
            "regime_columns": regime_columns,
            "geometry_columns": geometry_columns,
            "strict_post_valid_numeric_columns": strict_post_valid_numeric_columns,
            "conditional_raw_columns": conditional_raw_columns,
            "conditional_column_policy": "exclude_from_core_and_replace_with_geometry",
            "conditional_column_replacements": conditional_column_replacements,
            "geometry_feature_version": "geometry.features.v1",
            "geometry_feature_formulas": geometry_feature_formulas,
            "future_feature_hooks": {
                "trend_age_context": {
                    "implemented": False,
                    "planned_columns": ["bars_since_AT_flip", "bars_since_ST_flip"],
                }
            },
            "dtype_policy": {
                "selected_dtypes": {
                    "timestamp": "datetime64[ns, UTC]",
                    **{column: str(frame[column].dtype) for column in state_feature_columns},
                }
            },
            "row_order_policy": {"name": "timestamp_ascending", "stable_tie_breaker": "source_row_position"},
            "timestamp_policy": {"timestamp_column": "timestamp", "required_timezone": "UTC"},
        },
        "runtime_price_contract": {
            "timestamp_column": "timestamp",
            "execution_price_column": "close",
            "mark_to_market_column": "close",
            "required_runtime_columns": ["close"],
            "runtime_price_dtypes": {"close": "float32"},
            "artifact_columns": ["timestamp", *state_feature_columns, "close"],
        },
        "partition_metadata": [
            {
                "scope": "partition",
                "source_rel": "sample.parquet",
                "partition": "train",
                "fold_id": None,
                "output_path": str(parquet_path.resolve()),
                "row_count": row_count,
                "timestamp_min_utc": "2024-01-01T00:00:00+00:00",
                "timestamp_max_utc": pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat(),
                "duplicate_timestamp_count": 0,
                "timestamp_unique_ok": True,
                "file_sha256": "placeholder",
                "warmup_contract": {
                    "enabled": bool(warmup_rows > 0),
                    "required_observation_columns": required_observation_columns,
                    "policy": "drop_head_until_all_required_obs_numeric",
                    "valid_from_row": warmup_rows,
                    "valid_from_timestamp": valid_from_timestamp,
                    "post_valid_nan_policy": "fail_closed",
                    "head_nan_profile": head_nan_profile,
                },
            }
        ],
        "walk_forward_fold_metadata": [],
        "output_completeness_ok": True,
    }
    (reports_root / "state_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = {
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "builder_version": "state_builder.v1",
        "state_build_overall": True,
        "output_completeness_ok": True,
        "state_build_id": "state-build-id-1",
        "output_semantics": manifest["output_semantics"],
        "source_hashes": manifest["source_hashes"],
    }
    (reports_root / "state_build_report.json").write_text(json.dumps(report), encoding="utf-8")
    return state_root, parquet_path, {str(parquet_path.resolve()): frame}


def _base_config_payload(run_id: str, state_root: Path) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "state_root": str(state_root),
        "episode_ref": {
            "scope": "partition",
            "partition": "train",
            "source_rel": "sample.parquet",
            "fold_id": None,
        },
        "execution_price_column": "close",
        "mark_to_market_column": "close",
        "include_timestamp_in_observation": False,
        "observation_output_dtype": "float32",
        "observation_dtype_policy": "strict",
        "allowed_safe_casts": ["uint8->float32"],
        "initial_cash": 1000.0,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "max_steps": None,
        "seed": 42,
        "execution_timing_contract": {
            "observation_timestamp_policy": "row_t",
            "execution_price_policy": "close_t",
            "reward_accrual_interval_policy": "post_action_t_to_t_plus_1",
            "mark_to_market_policy": "next_row_close",
        },
        "action_semantics_contract": {
            "action_space_type": "Discrete",
            "action_space_n": 4,
            "invalid_action_policy": "noop_with_info_flag",
            "reversal_policy": "disallow_same_step",
            "position_model": "single_position_unit",
        },
        "reward_contract": {
            "reward_version": "reward.v1",
            "reward_formula_summary": "pnl_delta - fees - slippage_cost",
            "included_components": ["pnl_delta", "fees", "slippage_cost"],
            "reward_scale": 1.0,
            "reward_clip_min": None,
            "reward_clip_max": None,
        },
        "termination_contract": {
            "data_end_terminated": True,
            "max_steps_truncated": True,
        },
    }


def test_gym_reset_step_signatures_and_spaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    run_id = "env_adapter_shapes"
    state_root, _, frame_map = _seed_state_run(tmp_path, run_id)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    env = TradingEnvGym(config=parse_env_config(_base_config_payload(run_id, state_root)))
    obs, info = env.reset(seed=42)

    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert env.action_space.n == 4
    assert env.observation_space.shape == (2,)
    assert env.observation_space.contains(obs)
    assert info["seed"] == 42

    next_obs, reward, terminated, truncated, step_info = env.step(1)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_obs.shape == (2,)
    assert step_info["action_semantic"] == "OPEN_LONG"


def test_info_payload_keys_stable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    run_id = "env_adapter_info_keys"
    state_root, _, frame_map = _seed_state_run(tmp_path, run_id)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    env = TradingEnvGym(config=parse_env_config(_base_config_payload(run_id, state_root)))
    env.reset(seed=42)
    _, _, _, _, info = env.step(0)

    required = {
        "timestamp",
        "step_index",
        "position_before",
        "position_after",
        "action_raw",
        "action_semantic",
        "price_exec",
        "reward_total",
        "reward_components",
        "cost_components",
        "portfolio_value",
    }
    assert required.issubset(set(info.keys()))


def test_dtype_coercion_policy_enforced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    run_id = "env_adapter_dtype_policy"
    state_root, _, frame_map = _seed_state_run(tmp_path, run_id, float64_feature=True)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    disallow_payload = _base_config_payload(run_id, state_root)
    disallow_payload["allowed_safe_casts"] = ["uint8->float32"]
    with pytest.raises(ValueError, match="ENV_CONTRACT_OBSERVATION_DTYPE_MISMATCH"):
        TradingEnvGym(config=parse_env_config(disallow_payload))

    allow_payload = _base_config_payload(run_id, state_root)
    allow_payload["allowed_safe_casts"] = ["uint8->float32", "float64->float32"]
    env = TradingEnvGym(config=parse_env_config(allow_payload))
    obs, _ = env.reset(seed=42)
    assert obs.dtype == np.float32


def test_runtime_price_contract_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    run_id = "env_adapter_runtime_price_mismatch"
    state_root, _, frame_map = _seed_state_run(tmp_path, run_id)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    payload = _base_config_payload(run_id, state_root)
    payload["execution_price_column"] = "last_trade"
    payload["mark_to_market_column"] = "close"

    with pytest.raises(ValueError, match=ENV_CONTRACT_RUNTIME_PRICE_CONFIG_MISMATCH):
        TradingEnvGym(config=parse_env_config(payload))


def test_warmup_reset_starts_from_valid_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    run_id = "env_adapter_warmup_reset"
    state_root, _, frame_map = _seed_state_run(tmp_path, run_id, warmup_rows=2, row_count=5)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    env = TradingEnvGym(config=parse_env_config(_base_config_payload(run_id, state_root)))
    obs, info = env.reset(seed=42)

    assert np.allclose(obs, np.asarray([3.0, 0.0], dtype=np.float32))
    assert info["episode_valid_start_row"] == 2
    assert info["effective_episode_start_row"] == 2
    assert info["warmup_applied"] is True


def test_supertrend_geometry_contract_avoids_raw_band_preflight_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("gymnasium")
    run_id = "env_adapter_supertrend_geometry"
    state_root, _, frame_map = _seed_state_run(tmp_path, run_id, warmup_rows=1, row_count=5, supertrend_geometry=True)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    env = TradingEnvGym(config=parse_env_config(_base_config_payload(run_id, state_root)))
    obs, info = env.reset(seed=42)

    assert env.observation_space.shape == (4,)
    assert np.allclose(obs, np.asarray([-1.0, 100.0, 1.0, 1.0], dtype=np.float32))
    assert info["warmup_applied"] is True


def test_supertrend_geometry_corruption_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("gymnasium")
    run_id = "env_adapter_supertrend_geometry_fail"
    state_root, _, frame_map = _seed_state_run(
        tmp_path,
        run_id,
        warmup_rows=1,
        post_valid_nan_row=3,
        row_count=5,
        supertrend_geometry=True,
    )

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    with pytest.raises(ValueError, match=ENV_CONTRACT_POST_VALID_OBSERVATION_NAN):
        TradingEnvGym(config=parse_env_config(_base_config_payload(run_id, state_root)))


def test_post_valid_observation_nan_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    run_id = "env_adapter_post_valid_nan"
    state_root, _, frame_map = _seed_state_run(tmp_path, run_id, warmup_rows=2, post_valid_nan_row=3, row_count=5)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)

    with pytest.raises(ValueError, match=ENV_CONTRACT_POST_VALID_OBSERVATION_NAN):
        TradingEnvGym(config=parse_env_config(_base_config_payload(run_id, state_root)))
