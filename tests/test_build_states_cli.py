"""Integration-style tests for build_states CLI (Milestone 4.4)."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.state_builder import (
    STATE_BUILD_AGGREGATE_WALK_FORWARD_DEFERRED,
    STATE_BUILD_RUNTIME_ERROR,
    STATE_BUILD_RUNTIME_PRICE_CONTRACT_REQUIRED,
    STATE_BUILD_SCALER_TYPE_UNSUPPORTED,
    STATE_BUILD_SEQUENCE_MODE_DEFERRED,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_states.py"


def _base_frame(rows: int = 20, freq: str = "1min", start: str = "2024-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=rows, freq=freq, tz="UTC")
    frame = pd.DataFrame({"timestamp": ts})
    frame["close"] = pd.Series(np.linspace(100.0, 100.0 + float(rows - 1), rows), dtype="float32")
    frame["feat_a"] = pd.Series(np.linspace(1.0, float(rows), rows), dtype="float32")
    frame["feat_b"] = pd.Series(np.linspace(10.0, float(rows + 9), rows), dtype="float32")
    event = np.zeros(rows, dtype=np.uint8)
    event[::4] = np.uint8(1)
    frame["evt_flag"] = pd.Series(event, dtype="uint8")
    return frame


def _source_path(dataset_root: Path, rel_name: str, partition: str) -> Path:
    return dataset_root / "parquet" / "partitions" / partition / rel_name


def _seed_run(
    tmp_path: Path,
    run_id: str,
    *,
    with_manifest: bool = True,
    with_report: bool = True,
    dataset_build_overall: bool = True,
    output_completeness_ok: bool = True,
    with_summary: bool = False,
    frame_overrides: dict[str, pd.DataFrame] | None = None,
    selected_columns: list[str] | None = None,
    selected_dtypes: dict[str, str] | None = None,
) -> tuple[Path, dict[str, pd.DataFrame], Path, Path, Path]:
    dataset_root = tmp_path / "runs" / run_id / "data_datasets"
    reports_root = dataset_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    train = _base_frame(rows=8, freq="1min", start="2024-01-01 00:00:00")
    val = _base_frame(rows=4, freq="1min", start="2024-01-01 00:08:00")
    test = _base_frame(rows=4, freq="1min", start="2024-01-01 00:12:00")

    frame_map: dict[str, pd.DataFrame] = {}
    entries: list[dict[str, Any]] = []

    effective_selected_columns = list(selected_columns) if selected_columns is not None else ["timestamp", "feat_a", "feat_b", "evt_flag"]
    effective_selected_dtypes = (
        dict(selected_dtypes)
        if selected_dtypes is not None
        else {
            "timestamp": "datetime64[ns, UTC]",
            "feat_a": "float32",
            "feat_b": "float32",
            "evt_flag": "uint8",
        }
    )

    for partition, frame in (("train", train), ("val", val), ("test", test)):
        if frame_overrides is not None and partition in frame_overrides:
            frame = frame_overrides[partition].copy()
        src = _source_path(dataset_root, "a.parquet", partition)
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("dataset-placeholder", encoding="utf-8")
        frame_map[str(src.resolve())] = frame.copy()

        ts = pd.to_datetime(frame["timestamp"], utc=True)
        entries.append(
            {
                "scope": "partition",
                "source_rel": "a.parquet",
                "partition": partition,
                "fold_id": None,
                "output_path": str(src.resolve()),
                "row_count": int(len(frame)),
                "timestamp_min_utc": ts.iloc[0].isoformat(),
                "timestamp_max_utc": ts.iloc[-1].isoformat(),
                "duplicate_timestamp_count": 0,
                "timestamp_unique_ok": True,
                "file_sha256": "placeholder",
            }
        )

    dataset_manifest_path = reports_root / "dataset_manifest.json"
    if with_manifest:
        dataset_manifest = {
            "manifest_version": "datasets.manifest.v1",
            "generated_at_utc": "2026-02-21T00:00:00+00:00",
            "run_id": run_id,
            "builder_version": "dataset_builder.v1",
            "dataset_build_id": "dataset-build-id",
            "split_mode": "ratio_chrono",
            "output_semantics": {
                "mode": "standard_partitions",
                "fold_outputs_generated": False,
                "top_level_partitions_generated": True,
                "aggregate_walk_forward": False,
            },
            "column_selection_contract": {
                "timestamp_column": "timestamp",
                "selected_columns": effective_selected_columns,
                "selected_dtypes": effective_selected_dtypes,
                "column_selection_hash": "colhash",
                "dtype_hash": "dtypehash",
            },
            "source_hashes": {
                "feature_manifest_hash": "feat-hash",
                "train_input_report_hash": "train-hash",
                "split_report_hash": "split-hash",
            },
            "partition_metadata": entries,
            "output_completeness_ok": True,
        }
        dataset_manifest_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")

    dataset_report_path = reports_root / "dataset_build_report.json"
    if with_report:
        dataset_report = {
            "generated_at_utc": "2026-02-21T00:00:00+00:00",
            "run_id": run_id,
            "builder_version": "dataset_builder.v1",
            "dataset_build_id": "dataset-build-id",
            "dataset_build_overall": bool(dataset_build_overall),
            "output_completeness_ok": bool(output_completeness_ok),
            "split_mode": "ratio_chrono",
            "output_semantics": {
                "mode": "standard_partitions",
                "fold_outputs_generated": False,
                "top_level_partitions_generated": True,
                "aggregate_walk_forward": False,
            },
        }
        dataset_report_path.write_text(json.dumps(dataset_report), encoding="utf-8")

    summary_path = tmp_path / "runs" / run_id / "data_features" / "reports" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if with_summary:
        summary_path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    return dataset_root, frame_map, dataset_manifest_path, dataset_report_path, summary_path


def _patch_parquet_io(monkeypatch: object, frame_map: dict[str, pd.DataFrame]) -> None:
    def fake_read_parquet(path: Path) -> pd.DataFrame:
        key = str(Path(path).resolve())
        if key not in frame_map:
            raise ValueError(f"Unexpected parquet path: {key}")
        return frame_map[key].copy()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del index
        payload = {
            "rows": int(len(self)),
            "columns": list(self.columns),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def _runtime_price_args() -> list[str]:
    return ["--execution-price-column", "close", "--mark-to-market-column", "close"]


def test_cli_help() -> None:
    main = _load_main()
    with pytest.raises(SystemExit) as exc:
        main.__globals__["sys"].argv = ["build_states.py", "--help"]
        main()
    assert int(exc.value.code) == 0


def test_cli_success_exit_zero_and_outputs(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_success"
    dataset_root, frame_map, _, _, _ = _seed_run(tmp_path, run_id)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "build_states.py",
            "--run-id",
            run_id,
            "--input-root",
            str(dataset_root),
            "--enable-scaling",
            "true",
            "--scaler-type",
            "standard",
            *_runtime_price_args(),
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    output_root = tmp_path / "runs" / run_id / "data_states"
    report_path = output_root / "reports" / "state_build_report.json"
    manifest_path = output_root / "reports" / "state_manifest.json"
    scaler_path = output_root / "reports" / "scaler_stats.json"

    assert report_path.exists()
    assert manifest_path.exists()
    assert scaler_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["state_build_overall"] is True
    assert report["invocation_args"]["run_id"] == run_id
    assert report["invocation_args"]["enable_scaling"] is True
    assert report["runtime_price_contract"]["required_runtime_columns"] == ["close"]
    assert report["warmup_contract_summary"]["enabled"] is False
    assert manifest["observation_contract"]["state_feature_columns"] == ["feat_a", "feat_b", "evt_flag"]
    assert manifest["runtime_price_contract"]["artifact_columns"] == ["timestamp", "feat_a", "feat_b", "evt_flag", "close"]
    assert manifest["partition_metadata"][0]["warmup_contract"]["enabled"] is False


def test_cli_emits_warmup_contract_without_trimming(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_warmup"
    train_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC"),
            "EMA_200": pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32"),
            "EMA_600": pd.Series([np.nan, np.nan, 10.0, 11.0, 12.0, 13.0], dtype="float32"),
            "EMA_1200": pd.Series([np.nan, np.nan, np.nan, 20.0, 21.0, 22.0], dtype="float32"),
            "evt_flag": pd.Series([0, 1, 0, 1, 0, 1], dtype="uint8"),
            "close": pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], dtype="float32"),
        }
    )
    val_frame = train_frame.copy()
    val_frame["timestamp"] = pd.date_range("2024-01-02", periods=6, freq="1min", tz="UTC")
    test_frame = train_frame.copy()
    test_frame["timestamp"] = pd.date_range("2024-01-03", periods=6, freq="1min", tz="UTC")
    dataset_root, frame_map, _, _, _ = _seed_run(
        tmp_path,
        run_id,
        frame_overrides={"train": train_frame, "val": val_frame, "test": test_frame},
        selected_columns=["timestamp", "EMA_200", "EMA_600", "EMA_1200", "evt_flag"],
        selected_dtypes={
            "timestamp": "datetime64[ns, UTC]",
            "EMA_200": "float32",
            "EMA_600": "float32",
            "EMA_1200": "float32",
            "evt_flag": "uint8",
        },
    )
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "build_states.py",
            "--run-id",
            run_id,
            "--input-root",
            str(dataset_root),
            *_runtime_price_args(),
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    output_root = tmp_path / "runs" / run_id / "data_states"
    manifest = json.loads((output_root / "reports" / "state_manifest.json").read_text(encoding="utf-8"))
    artifact = manifest["partition_metadata"][0]
    assert artifact["row_count"] == 6
    assert artifact["warmup_contract"]["required_observation_columns"] == ["EMA_200", "EMA_600", "EMA_1200"]
    assert artifact["warmup_contract"]["valid_from_row"] == 3
    assert artifact["warmup_contract"]["valid_from_timestamp"] == "2024-01-01T00:03:00+00:00"


def test_cli_emits_supertrend_geometry_contract(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_supertrend_geometry"
    train_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
            "AlphaTrend": pd.Series([95.0, 96.0, 97.0, 98.0, 99.0], dtype="float32"),
            "AlphaTrend_2": pd.Series([94.0, 95.0, 96.0, 97.0, 98.0], dtype="float32"),
            "ST_trend": pd.Series([1, 1, -1, -1, -1], dtype="int8"),
            "ST_up": pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan], dtype="float32"),
            "ST_dn": pd.Series([np.nan, np.nan, 101.0, 102.0, 103.0], dtype="float32"),
            "evt_at_buy": pd.Series([0, 0, 0, 0, 0], dtype="uint8"),
            "evt_at_sell": pd.Series([0, 0, 0, 0, 0], dtype="uint8"),
            "evt_st_buy": pd.Series([0, 1, 0, 0, 0], dtype="uint8"),
            "evt_st_sell": pd.Series([0, 0, 1, 0, 0], dtype="uint8"),
            "close": pd.Series([99.0, 101.0, 100.0, 101.0, 104.0], dtype="float32"),
        }
    )
    val_frame = train_frame.copy()
    val_frame["timestamp"] = pd.date_range("2024-01-02", periods=5, freq="1min", tz="UTC")
    test_frame = train_frame.copy()
    test_frame["timestamp"] = pd.date_range("2024-01-03", periods=5, freq="1min", tz="UTC")

    dataset_root, frame_map, _, _, _ = _seed_run(
        tmp_path,
        run_id,
        frame_overrides={"train": train_frame, "val": val_frame, "test": test_frame},
        selected_columns=[
            "timestamp",
            "AlphaTrend",
            "AlphaTrend_2",
            "ST_trend",
            "ST_up",
            "ST_dn",
            "evt_at_buy",
            "evt_at_sell",
            "evt_st_buy",
            "evt_st_sell",
        ],
        selected_dtypes={
            "timestamp": "datetime64[ns, UTC]",
            "AlphaTrend": "float32",
            "AlphaTrend_2": "float32",
            "ST_trend": "int8",
            "ST_up": "float32",
            "ST_dn": "float32",
            "evt_at_buy": "uint8",
            "evt_at_sell": "uint8",
            "evt_st_buy": "uint8",
            "evt_st_sell": "uint8",
        },
    )
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "build_states.py",
            "--run-id",
            run_id,
            "--input-root",
            str(dataset_root),
            *_runtime_price_args(),
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    output_root = tmp_path / "runs" / run_id / "data_states"
    manifest = json.loads((output_root / "reports" / "state_manifest.json").read_text(encoding="utf-8"))
    observation_contract = manifest["observation_contract"]
    assert observation_contract["state_feature_columns"] == [
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_active_line",
        "ST_distance_to_active_line",
        "evt_at_buy",
        "evt_at_sell",
        "evt_st_buy",
        "evt_st_sell",
    ]
    assert observation_contract["conditional_raw_columns"] == ["ST_up", "ST_dn"]
    assert observation_contract["geometry_feature_formulas"]["ST_distance_to_active_line_formula"] == "close_minus_active_line"

    parquet_payload = json.loads((output_root / "parquet" / "partitions" / "train" / "a.parquet").read_text(encoding="utf-8"))
    assert parquet_payload["columns"] == [
        "timestamp",
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_active_line",
        "ST_distance_to_active_line",
        "evt_at_buy",
        "evt_at_sell",
        "evt_st_buy",
        "evt_st_sell",
        "close",
    ]


def test_cli_contract_fail_exit_two(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_contract_fail"
    dataset_root, frame_map, _, _, _ = _seed_run(tmp_path, run_id, with_manifest=False)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), *_runtime_price_args()],
    )

    exit_code = int(main())
    assert exit_code == 2


def test_cli_runtime_error_exit_three(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_runtime_error"
    dataset_root, _, _, _, _ = _seed_run(tmp_path, run_id)

    main = _load_main()

    def fake_build_states(_: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setitem(main.__globals__, "build_states", fake_build_states)
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), *_runtime_price_args()],
    )

    exit_code = int(main())
    assert exit_code == 3

    report_path = tmp_path / "runs" / run_id / "data_states" / "reports" / "state_build_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["errors"][0]["code"] == STATE_BUILD_RUNTIME_ERROR


def test_cli_summary_update_non_blocking(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_summary_non_blocking"
    dataset_root, frame_map, _, _, summary_path = _seed_run(tmp_path, run_id, with_summary=True)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()

    def fake_atomic_write_json(payload: dict[str, Any], dest: Path) -> None:
        if dest == summary_path:
            raise RuntimeError("summary write fail")
        tmp = dest.with_suffix(f"{dest.suffix}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, dest)

    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(main.__globals__, "atomic_write_json", fake_atomic_write_json)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), *_runtime_price_args()],
    )

    exit_code = int(main())
    assert exit_code == 0
    assert summary_path.exists()


def test_cli_overwrite_behavior(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_overwrite"
    dataset_root, frame_map, _, _, _ = _seed_run(tmp_path, run_id)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), "--overwrite", "false", *_runtime_price_args()],
    )
    first = int(main())
    assert first == 0

    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), "--overwrite", "false", *_runtime_price_args()],
    )
    second = int(main())
    assert second == 2

    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), "--overwrite", "true", *_runtime_price_args()],
    )
    third = int(main())
    assert third == 0


def test_cli_sequence_mode_deferred(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_sequence_defer"
    dataset_root, frame_map, _, _, _ = _seed_run(tmp_path, run_id)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), "--sequence-mode", "true", *_runtime_price_args()],
    )
    exit_code = int(main())
    assert exit_code == 2

    report = json.loads((tmp_path / "runs" / run_id / "data_states" / "reports" / "state_build_report.json").read_text(encoding="utf-8"))
    assert any(item.get("code") == STATE_BUILD_SEQUENCE_MODE_DEFERRED for item in report.get("errors", []))


def test_cli_aggregate_walk_forward_deferred(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_agg_defer"
    dataset_root, frame_map, _, _, _ = _seed_run(tmp_path, run_id)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), "--aggregate-walk-forward", "true", *_runtime_price_args()],
    )
    exit_code = int(main())
    assert exit_code == 2

    report = json.loads((tmp_path / "runs" / run_id / "data_states" / "reports" / "state_build_report.json").read_text(encoding="utf-8"))
    assert any(item.get("code") == STATE_BUILD_AGGREGATE_WALK_FORWARD_DEFERRED for item in report.get("errors", []))


def test_cli_unsupported_scaler_type_exit_two(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_scaler_unsupported"
    dataset_root, frame_map, _, _, _ = _seed_run(tmp_path, run_id)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "build_states.py",
            "--run-id",
            run_id,
            "--input-root",
            str(dataset_root),
            "--enable-scaling",
            "true",
            "--scaler-type",
            "unknown",
            *_runtime_price_args(),
        ],
    )
    exit_code = int(main())
    assert exit_code == 2

    report = json.loads((tmp_path / "runs" / run_id / "data_states" / "reports" / "state_build_report.json").read_text(encoding="utf-8"))
    assert any(item.get("code") == STATE_BUILD_SCALER_TYPE_UNSUPPORTED for item in report.get("errors", []))


def test_cli_scaling_default_explicitly_recorded(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_scaling_default"
    dataset_root, frame_map, _, _, _ = _seed_run(tmp_path, run_id)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root), *_runtime_price_args()],
    )

    exit_code = int(main())
    assert exit_code == 0

    output_root = tmp_path / "runs" / run_id / "data_states"
    report = json.loads((output_root / "reports" / "state_build_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "reports" / "state_manifest.json").read_text(encoding="utf-8"))

    assert report["invocation_args"]["enable_scaling"] is False
    assert manifest["observation_contract"]["scaling_policy"]["enabled"] is False
    assert not (output_root / "reports" / "scaler_stats.json").exists()


def test_cli_missing_runtime_price_args_fails_closed(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "state_cli_runtime_price_missing"
    dataset_root, frame_map, _, _, _ = _seed_run(tmp_path, run_id)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main.__globals__["sys"], "argv", ["build_states.py", "--run-id", run_id, "--input-root", str(dataset_root)])

    exit_code = int(main())
    assert exit_code == 2

    report = json.loads((tmp_path / "runs" / run_id / "data_states" / "reports" / "state_build_report.json").read_text(encoding="utf-8"))
    assert any(item.get("code") == STATE_BUILD_RUNTIME_PRICE_CONTRACT_REQUIRED for item in report.get("errors", []))
