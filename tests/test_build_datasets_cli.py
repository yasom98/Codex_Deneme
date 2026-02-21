"""Integration-style tests for build_datasets CLI."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.dataset_builder import DATASET_BUILD_RUNTIME_ERROR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_datasets.py"


def _base_frame(rows: int = 20, freq: str = "1min") -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq=freq, tz="UTC")
    frame = pd.DataFrame({"timestamp": ts})
    frame["feat_cont"] = pd.Series(np.linspace(1.0, 2.0, rows), dtype="float32")
    events = np.zeros(rows, dtype=np.uint8)
    events[::5] = np.uint8(1)
    frame["evt_flag"] = pd.Series(events, dtype="uint8")
    return frame


def _range_payload(ts: pd.Series, start: int, end: int) -> dict[str, Any]:
    return {
        "start_utc": ts.iloc[start].isoformat(),
        "end_exclusive_utc": ts.iloc[end].isoformat() if end < len(ts) else None,
        "end_inclusive_utc": ts.iloc[end - 1].isoformat(),
        "row_count": int(end - start),
        "internal_interval": "[start, end)",
    }


def _seed_run(
    tmp_path: Path,
    run_id: str,
    *,
    with_manifest: bool = True,
    with_train_input_report: bool = True,
    train_input_overall: bool = True,
    with_split_report: bool = True,
    split_overall: bool = True,
    walk_forward: bool = False,
    with_summary: bool = False,
) -> tuple[Path, Path, Path, dict[str, pd.DataFrame]]:
    input_root = tmp_path / "runs" / run_id / "data_features" / "parquet"
    reports_root = tmp_path / "runs" / run_id / "data_features" / "reports"
    input_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    src = input_root / "sample.parquet"
    src.write_text("source-placeholder", encoding="utf-8")
    frame = _base_frame()
    frame_map = {str(src.resolve()): frame.copy()}

    if with_manifest:
        manifest_payload = {
            "manifest_version": "features.manifest.v1",
            "generated_at_utc": "2026-02-21T00:00:00+00:00",
            "run_id": run_id,
            "timestamp_column": "timestamp",
            "continuous_columns": ["feat_cont"],
            "event_columns": ["evt_flag"],
            "column_dtypes": {
                "timestamp": "datetime64[ns, UTC]",
                "feat_cont": "float32",
                "evt_flag": "uint8",
            },
            "feature_groups": {
                "raw_ohlcv": ["timestamp"],
                "price_derived": [],
                "trend": ["feat_cont"],
                "regime": [],
                "event": ["evt_flag"],
                "placeholders": [],
            },
        }
        (reports_root / "feature_manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    if with_train_input_report:
        train_payload = {
            "run_id": run_id,
            "train_input_validation_overall": bool(train_input_overall),
        }
        (reports_root / "train_input_validation_report.json").write_text(json.dumps(train_payload), encoding="utf-8")

    if with_split_report:
        ts = pd.to_datetime(frame["timestamp"], utc=True)
        file_reports = [
            {
                "input_file": str(src.resolve()),
                "status": "success",
                "train_range": _range_payload(ts, 0, 12),
                "val_range": _range_payload(ts, 12, 16),
                "test_range": _range_payload(ts, 16, 20),
            }
        ]
        fold_reports: list[dict[str, Any]] = []
        split_mode = "ratio_chrono"
        if walk_forward:
            split_mode = "walk_forward"
            fold_reports = [
                {
                    "fold_id": 0,
                    "input_file": str(src.resolve()),
                    "train_range": _range_payload(ts, 0, 10),
                    "val_range": _range_payload(ts, 10, 14),
                    "test_range": _range_payload(ts, 14, 18),
                },
                {
                    "fold_id": 1,
                    "input_file": str(src.resolve()),
                    "train_range": _range_payload(ts, 0, 12),
                    "val_range": _range_payload(ts, 12, 16),
                    "test_range": _range_payload(ts, 16, 20),
                },
            ]

        split_payload = {
            "generated_at_utc": "2026-02-21T00:00:00+00:00",
            "run_id": run_id,
            "split_mode": split_mode,
            "split_validation_overall": bool(split_overall),
            "manifest_path": str(reports_root / "feature_manifest.json"),
            "train_input_validation_report_path": str(reports_root / "train_input_validation_report.json"),
            "file_reports": file_reports,
            "fold_reports": fold_reports,
        }
        (reports_root / "split_validation_report.json").write_text(json.dumps(split_payload), encoding="utf-8")

    if with_summary:
        (reports_root / "summary.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    return input_root, reports_root, src, frame_map


def _patch_parquet_io(monkeypatch: object, frame_map: dict[str, pd.DataFrame]) -> None:
    def fake_read_parquet(path: Path) -> pd.DataFrame:
        return frame_map[str(Path(path).resolve())].copy()

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del index
        Path(path).write_text(json.dumps({"rows": int(len(self))}), encoding="utf-8")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_help() -> None:
    main = _load_main()
    with pytest.raises(SystemExit) as exc:
        main.__globals__["sys"].argv = ["build_datasets.py", "--help"]
        main()
    assert int(exc.value.code) == 0


def test_cli_success_exit_zero_and_outputs(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_cli_success"
    input_root, reports_root, _, frame_map = _seed_run(tmp_path, run_id)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setattr(main.__globals__["sys"], "argv", ["build_datasets.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 0

    report_path = tmp_path / "runs" / run_id / "data_datasets" / "reports" / "dataset_build_report.json"
    manifest_path = tmp_path / "runs" / run_id / "data_datasets" / "reports" / "dataset_manifest.json"
    assert report_path.exists()
    assert manifest_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["dataset_build_overall"] is True
    assert payload["invocation_args"]["run_id"] == run_id


def test_cli_contract_fail_exit_two(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_cli_contract_fail"
    input_root, _, _, frame_map = _seed_run(tmp_path, run_id, with_manifest=False)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setattr(main.__globals__["sys"], "argv", ["build_datasets.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 2


def test_cli_runtime_error_exit_three(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_cli_runtime_fail"
    input_root, _, _, _ = _seed_run(tmp_path, run_id)

    main = _load_main()

    def fake_build(_: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setitem(main.__globals__, "build_datasets", fake_build)
    monkeypatch.setattr(main.__globals__["sys"], "argv", ["build_datasets.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 3

    report_path = tmp_path / "runs" / run_id / "data_datasets" / "reports" / "dataset_build_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["errors"][0]["code"] == DATASET_BUILD_RUNTIME_ERROR


def test_cli_summary_update_non_blocking(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_cli_summary_non_blocking"
    input_root, reports_root, _, frame_map = _seed_run(tmp_path, run_id, with_summary=True)
    _patch_parquet_io(monkeypatch, frame_map)

    summary_path = reports_root / "summary.json"
    main = _load_main()

    def fake_atomic_write_json(payload: dict[str, Any], dest: Path) -> None:
        if dest == summary_path:
            raise RuntimeError("summary write fail")
        tmp = dest.with_suffix(f"{dest.suffix}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, dest)

    monkeypatch.setitem(main.__globals__, "atomic_write_json", fake_atomic_write_json)
    monkeypatch.setattr(main.__globals__["sys"], "argv", ["build_datasets.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 0
    assert summary_path.exists()


def test_cli_overwrite_and_walk_forward_aggregate_flags(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "dataset_cli_flags"
    input_root, _, _, frame_map = _seed_run(tmp_path, run_id, walk_forward=True)
    _patch_parquet_io(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "build_datasets.py",
            "--run-id",
            run_id,
            "--input-root",
            str(input_root),
            "--aggregate-walk-forward",
            "true",
            "--overwrite",
            "false",
        ],
    )
    first_exit = int(main())
    assert first_exit == 0

    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "build_datasets.py",
            "--run-id",
            run_id,
            "--input-root",
            str(input_root),
            "--aggregate-walk-forward",
            "true",
            "--overwrite",
            "false",
        ],
    )
    second_exit = int(main())
    assert second_exit == 2

    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "build_datasets.py",
            "--run-id",
            run_id,
            "--input-root",
            str(input_root),
            "--aggregate-walk-forward",
            "true",
            "--overwrite",
            "true",
        ],
    )
    third_exit = int(main())
    assert third_exit == 0

    report_path = tmp_path / "runs" / run_id / "data_datasets" / "reports" / "dataset_build_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["output_semantics"]["mode"] == "walk_forward_fold_plus_aggregate"
