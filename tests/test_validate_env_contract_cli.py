"""Integration-style tests for validate_env_contract CLI (Milestone 4.5)."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rl.env_contract import (
    ENV_CONTRACT_EPISODE_TOO_SHORT_AFTER_WARMUP,
    ENV_CONTRACT_POST_VALID_OBSERVATION_NAN,
    ENV_CONTRACT_RUNTIME_ERROR,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_env_contract.py"


def _seed_state_run(
    tmp_path: Path,
    run_id: str,
    *,
    with_manifest: bool = True,
    with_report: bool = True,
    warmup_rows: int = 0,
    post_valid_nan_row: int | None = None,
    row_count: int = 4,
    supertrend_geometry: bool = False,
) -> tuple[Path, Path, Path, dict[str, pd.DataFrame]]:
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
        feature_values: list[float] = [float(index + 1) for index in range(row_count)]
        for index in range(min(warmup_rows, row_count)):
            feature_values[index] = float("nan")
        if post_valid_nan_row is not None:
            feature_values[post_valid_nan_row] = float("nan")
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=row_count, freq="1min", tz="UTC"),
                feature_name: pd.Series(feature_values, dtype="float32"),
                "evt_flag": pd.Series(([0, 1] * ((row_count // 2) + 1))[:row_count], dtype="uint8"),
                "close": pd.Series([100.0 + float(index) for index in range(row_count)], dtype="float32"),
            }
        )
        selected_state_columns = ["timestamp", feature_name, "evt_flag"]
        selected_state_dtypes = {
            "timestamp": "datetime64[ns, UTC]",
            feature_name: "float32",
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

    if with_manifest:
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

    if with_report:
        report = {
            "generated_at_utc": "2026-02-21T00:00:00+00:00",
            "run_id": run_id,
            "builder_version": "state_builder.v1",
            "state_build_overall": True,
            "output_completeness_ok": True,
            "state_build_id": "state-build-id-1",
            "output_semantics": {
                "mode": "standard_partitions",
                "fold_outputs_generated": False,
                "top_level_partitions_generated": True,
                "aggregate_walk_forward": False,
            },
            "source_hashes": {
                "dataset_manifest_hash": "hash-a",
                "dataset_build_report_hash": "hash-b",
                "source_file_inventory_hash": "hash-c",
            },
        }
        (reports_root / "state_build_report.json").write_text(json.dumps(report), encoding="utf-8")

    config_payload: dict[str, Any] = {
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
    config_path = tmp_path / "env_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    return state_root, reports_root, config_path, {str(parquet_path.resolve()): frame}


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_help() -> None:
    main = _load_main()
    with pytest.raises(SystemExit) as exc:
        main.__globals__["sys"].argv = ["validate_env_contract.py", "--help"]
        main()
    assert int(exc.value.code) == 0


def test_cli_success_exit_zero_and_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "env_contract_cli_success"
    state_root, _, config_path, frame_map = _seed_state_run(tmp_path, run_id)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_env_contract.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
            "--smoke-step",
            "true",
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    report_path = tmp_path / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["env_contract_overall"] is True
    assert payload["run_id"] == run_id
    assert payload["invocation_args"]["run_id"] == run_id
    assert payload["warmup_applied"] is False
    assert payload["episode_valid_start_row"] == 0
    assert payload["effective_episode_start_row"] == 0
    assert payload["observation_space_metadata"]["observation_space_type"] == "Box"
    assert payload["action_space_metadata"]["action_space_type"] == "Discrete"


def test_cli_contract_fail_exit_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "env_contract_cli_fail"
    state_root, _, config_path, frame_map = _seed_state_run(tmp_path, run_id, with_manifest=False)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_env_contract.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
        ],
    )

    exit_code = int(main())
    assert exit_code == 2

    report_path = tmp_path / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["env_contract_overall"] is False


def test_cli_runtime_error_exit_three(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "env_contract_cli_runtime"
    state_root, _, config_path, _ = _seed_state_run(tmp_path, run_id)
    main = _load_main()

    def fake_validate(**_: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setitem(main.__globals__, "validate_env_contract", fake_validate)
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_env_contract.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
        ],
    )

    exit_code = int(main())
    assert exit_code == 3

    report_path = tmp_path / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["errors"][0]["code"] == ENV_CONTRACT_RUNTIME_ERROR


def test_cli_summary_update_non_blocking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "env_contract_summary_non_blocking"
    state_root, _, config_path, frame_map = _seed_state_run(tmp_path, run_id)
    summary_path = tmp_path / "runs" / run_id / "data_features" / "reports" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    def fake_atomic_write_json(payload: dict[str, Any], dest: Path) -> None:
        if dest == summary_path:
            raise RuntimeError("summary write fail")
        tmp = dest.with_suffix(f"{dest.suffix}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, dest)

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    main = _load_main()
    monkeypatch.setitem(main.__globals__, "atomic_write_json", fake_atomic_write_json)
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_env_contract.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
        ],
    )

    exit_code = int(main())
    assert exit_code == 0
    assert summary_path.exists()


def test_cli_warmup_success_records_effective_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "env_contract_cli_warmup"
    state_root, _, config_path, frame_map = _seed_state_run(tmp_path, run_id, warmup_rows=2, row_count=5)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_env_contract.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
            "--smoke-step",
            "true",
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    report_path = tmp_path / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["env_contract_overall"] is True
    assert payload["warmup_applied"] is True
    assert payload["episode_valid_start_row"] == 2
    assert payload["effective_episode_start_row"] == 2
    assert payload["warmup_contract"]["valid_from_timestamp"] == "2024-01-01T00:02:00+00:00"


def test_cli_supertrend_geometry_contract_passes_without_raw_band_nan_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = "env_contract_cli_supertrend_geometry"
    state_root, _, config_path, frame_map = _seed_state_run(
        tmp_path,
        run_id,
        warmup_rows=1,
        row_count=5,
        supertrend_geometry=True,
    )

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_env_contract.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
            "--smoke-step",
            "true",
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    report_path = tmp_path / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["env_contract_overall"] is True
    assert payload["warmup_applied"] is True
    assert payload["observation_space_metadata"]["observation_space_shape"] == [4]


def test_cli_supertrend_geometry_corruption_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = "env_contract_cli_supertrend_geometry_fail"
    state_root, _, config_path, frame_map = _seed_state_run(
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
    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_env_contract.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
        ],
    )

    exit_code = int(main())
    assert exit_code == 2

    report_path = tmp_path / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["env_contract_overall"] is False
    assert payload["errors"][0]["code"] == ENV_CONTRACT_POST_VALID_OBSERVATION_NAN


def test_cli_too_short_after_warmup_fails_deterministically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "env_contract_cli_too_short_warmup"
    state_root, _, config_path, frame_map = _seed_state_run(tmp_path, run_id, warmup_rows=1, row_count=2)

    def fake_read(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_env_contract.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
        ],
    )

    exit_code = int(main())
    assert exit_code == 2

    report_path = tmp_path / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["env_contract_overall"] is False
    assert payload["errors"][0]["code"] == ENV_CONTRACT_EPISODE_TOO_SHORT_AFTER_WARMUP
