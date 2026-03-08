"""Shared fixtures for Milestone 4.6 readiness tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


def seed_state_run(
    tmp_path: Path,
    run_id: str,
    *,
    entries: list[dict[str, Any]],
    selected_episode_ref: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, pd.DataFrame]]:
    """Seed a minimal validated state run with deterministic episode fixtures."""

    state_root = tmp_path / "runs" / run_id / "data_states"
    reports_root = state_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    frame_map: dict[str, pd.DataFrame] = {}
    partition_metadata: list[dict[str, Any]] = []

    for item in entries:
        partition = str(item["partition"])
        source_rel = str(item["source_rel"])
        row_count = int(item.get("row_count", 6))
        warmup_rows = int(item.get("warmup_rows", 0))
        post_valid_nan_row = item.get("post_valid_nan_row")
        scope = str(item.get("scope", "partition"))
        fold_id = item.get("fold_id")

        parquet_path = state_root / "parquet" / "partitions" / partition / source_rel
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("state-placeholder", encoding="utf-8")

        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=row_count, freq="1min", tz="UTC"),
                "feat_x": pd.Series(
                    [float(index + 1) if index >= warmup_rows else float("nan") for index in range(row_count)],
                    dtype="float32",
                ),
                "evt_flag": pd.Series(([0, 1] * ((row_count // 2) + 1))[:row_count], dtype="uint8"),
                "close": pd.Series([100.0 + float(index) for index in range(row_count)], dtype="float32"),
            }
        )
        if isinstance(post_valid_nan_row, int):
            frame.loc[int(post_valid_nan_row), "feat_x"] = float("nan")
        frame_map[str(parquet_path.resolve())] = frame

        valid_from_timestamp = None
        if warmup_rows < row_count:
            valid_from_timestamp = pd.Timestamp(frame["timestamp"].iloc[warmup_rows]).isoformat()

        partition_metadata.append(
            {
                "scope": scope,
                "source_rel": source_rel,
                "partition": partition,
                "fold_id": fold_id if scope == "fold" else None,
                "output_path": str(parquet_path.resolve()),
                "row_count": row_count,
                "timestamp_min_utc": pd.Timestamp(frame["timestamp"].iloc[0]).isoformat(),
                "timestamp_max_utc": pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat(),
                "duplicate_timestamp_count": 0,
                "timestamp_unique_ok": True,
                "file_sha256": "placeholder",
                "warmup_contract": {
                    "enabled": bool(warmup_rows > 0),
                    "required_observation_columns": ["feat_x"] if warmup_rows > 0 else [],
                    "policy": "drop_head_until_all_required_obs_numeric",
                    "valid_from_row": warmup_rows,
                    "valid_from_timestamp": valid_from_timestamp,
                    "post_valid_nan_policy": "fail_closed",
                    "head_nan_profile": {"feat_x": warmup_rows} if warmup_rows > 0 else {},
                },
            }
        )

    manifest = {
        "manifest_version": "states.manifest.v1",
        "generated_at_utc": "2026-03-08T00:00:00+00:00",
        "run_id": run_id,
        "builder_version": "state_builder.v1",
        "state_build_id": "state-build-id-1",
        "state_build_id_policy": "deterministic_hash_v1",
        "build_mode": "materialize_only",
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
            "selected_state_columns": ["timestamp", "feat_x", "evt_flag"],
            "selected_state_dtypes": {
                "timestamp": "datetime64[ns, UTC]",
                "feat_x": "float32",
                "evt_flag": "uint8",
            },
        },
        "observation_contract": {
            "selected_input_columns": ["timestamp", "feat_x", "evt_flag"],
            "state_feature_columns": ["feat_x", "evt_flag"],
            "event_columns": ["evt_flag"],
            "regime_columns": [],
            "geometry_columns": [],
            "strict_post_valid_numeric_columns": ["feat_x", "evt_flag"],
            "conditional_raw_columns": [],
            "conditional_column_policy": "exclude_from_core_and_replace_with_geometry",
            "conditional_column_replacements": {},
            "geometry_feature_version": "geometry.features.v1",
            "geometry_feature_formulas": {},
            "future_feature_hooks": {},
            "dtype_policy": {
                "selected_dtypes": {
                    "timestamp": "datetime64[ns, UTC]",
                    "feat_x": "float32",
                    "evt_flag": "uint8",
                }
            },
            "row_order_policy": {"name": "timestamp_ascending", "stable_tie_breaker": "source_row_position"},
            "timestamp_policy": {"timestamp_column": "timestamp", "required_timezone": "UTC"},
            "scaling_policy": {"enabled": False, "scaler_type": "none"},
        },
        "runtime_price_contract": {
            "timestamp_column": "timestamp",
            "execution_price_column": "close",
            "mark_to_market_column": "close",
            "required_runtime_columns": ["close"],
            "runtime_price_dtypes": {"close": "float32"},
            "artifact_columns": ["timestamp", "feat_x", "evt_flag", "close"],
        },
        "warmup_contract_summary": {
            "enabled": any(bool(item["warmup_contract"]["enabled"]) for item in partition_metadata),
            "policy": "drop_head_until_all_required_obs_numeric",
            "post_valid_nan_policy": "fail_closed",
            "artifacts_total": len(partition_metadata),
            "artifacts_with_warmup": sum(1 for item in partition_metadata if bool(item["warmup_contract"]["enabled"])),
            "max_valid_from_row": max(int(item["warmup_contract"]["valid_from_row"]) for item in partition_metadata),
        },
        "scaler_stats_ref": None,
        "partition_metadata": partition_metadata,
        "walk_forward_fold_metadata": [],
        "output_completeness_ok": True,
    }
    (reports_root / "state_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = {
        "generated_at_utc": "2026-03-08T00:00:00+00:00",
        "run_id": run_id,
        "builder_version": "state_builder.v1",
        "state_build_overall": True,
        "output_completeness_ok": True,
        "state_build_id": "state-build-id-1",
        "output_semantics": manifest["output_semantics"],
        "source_hashes": manifest["source_hashes"],
    }
    (reports_root / "state_build_report.json").write_text(json.dumps(report), encoding="utf-8")

    selected = selected_episode_ref or {
        "scope": "partition",
        "partition": str(entries[0]["partition"]),
        "source_rel": str(entries[0]["source_rel"]),
        "fold_id": None,
    }
    env_config = {
        "run_id": run_id,
        "state_root": str(state_root),
        "episode_ref": selected,
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
    config_path = tmp_path / f"{run_id}_env_config.json"
    config_path.write_text(json.dumps(env_config), encoding="utf-8")
    return state_root, config_path, frame_map


def patch_read_parquet(monkeypatch: pytest.MonkeyPatch, frame_map: dict[str, pd.DataFrame]) -> None:
    """Patch pandas parquet loading against the seeded frame map."""

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)
