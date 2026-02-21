"""Integration-style tests for validate_splits CLI."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.split_validation import SPLIT_RUNTIME_ERROR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_splits.py"


def _base_frame(rows: int = 64, freq: str = "15min") -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq=freq, tz="UTC")
    out = pd.DataFrame({"timestamp": ts})
    out["feat_cont"] = pd.Series(np.linspace(1.0, 2.0, rows), dtype="float32")
    event = np.zeros(rows, dtype=np.uint8)
    event[::6] = np.uint8(1)
    out["evt_flag"] = pd.Series(event, dtype="uint8")
    return out


def _base_manifest(run_id: str) -> dict[str, Any]:
    return {
        "manifest_version": "features.manifest.v1",
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "timestamp_column": "timestamp",
        "continuous_columns": ["feat_cont"],
        "event_columns": ["evt_flag"],
    }


def _seed_run(
    tmp_path: Path,
    run_id: str,
    *,
    with_manifest: bool = True,
    with_train_input_report: bool = True,
    train_input_overall: bool = True,
    with_summary: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    runs_root = tmp_path / "runs"
    input_root = runs_root / run_id / "data_features" / "parquet"
    reports_root = runs_root / run_id / "data_features" / "reports"
    input_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    parquet_path = input_root / "sample.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    manifest_path = reports_root / "feature_manifest.json"
    if with_manifest:
        manifest_path.write_text(json.dumps(_base_manifest(run_id)), encoding="utf-8")

    train_input_report_path = reports_root / "train_input_validation_report.json"
    if with_train_input_report:
        train_input_report_path.write_text(
            json.dumps({"run_id": run_id, "train_input_validation_overall": bool(train_input_overall)}),
            encoding="utf-8",
        )

    if with_summary:
        summary_path = reports_root / "summary.json"
        summary_path.write_text(json.dumps({"run_id": run_id, "total_files": 1}), encoding="utf-8")

    return runs_root, input_root, reports_root, parquet_path, train_input_report_path


def _patch_read_parquet(monkeypatch: object, frame: pd.DataFrame) -> None:
    def fake_read_parquet(path: Path) -> pd.DataFrame:
        del path
        return frame.copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def _fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "sha256": digest,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def test_cli_pass_exit_zero_and_writes_report(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "split_cli_pass_run"
    _, input_root, reports_root, _, _ = _seed_run(tmp_path, run_id)
    _patch_read_parquet(monkeypatch, _base_frame(rows=90, freq="1min"))

    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_splits.py",
            "--run-id",
            run_id,
            "--input-root",
            str(input_root),
            "--split-mode",
            "ratio_chrono",
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.1",
        ],
    )
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 0

    report_path = reports_root / "split_validation_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["split_validation_overall"] is True


def test_cli_contract_fail_exit_two(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "split_cli_contract_fail_run"
    _, input_root, reports_root, _, _ = _seed_run(tmp_path, run_id, with_manifest=False)
    _patch_read_parquet(monkeypatch, _base_frame(rows=90, freq="1min"))

    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_splits.py",
            "--run-id",
            run_id,
            "--input-root",
            str(input_root),
            "--split-mode",
            "ratio_chrono",
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.1",
        ],
    )
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 2

    report_path = reports_root / "split_validation_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["split_validation_overall"] is False


def test_summary_update_non_blocking(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "split_cli_summary_non_blocking_run"
    _, input_root, reports_root, _, _ = _seed_run(tmp_path, run_id, with_summary=True)
    summary_path = reports_root / "summary.json"
    _patch_read_parquet(monkeypatch, _base_frame(rows=90, freq="1min"))

    main = _load_main()

    def fake_atomic_write_json(payload: dict[str, Any], dest: Path) -> None:
        if dest == summary_path:
            raise RuntimeError("summary write failure")
        tmp = dest.with_suffix(f"{dest.suffix}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, dest)

    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_splits.py",
            "--run-id",
            run_id,
            "--input-root",
            str(input_root),
            "--split-mode",
            "ratio_chrono",
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.1",
        ],
    )
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(main.__globals__, "atomic_write_json", fake_atomic_write_json)

    exit_code = int(main())
    assert exit_code == 0
    assert summary_path.exists()


def test_cli_runtime_error_exit_three(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "split_cli_runtime_error_run"
    _, input_root, reports_root, _, _ = _seed_run(tmp_path, run_id)

    main = _load_main()

    def fake_validate(_: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_splits.py",
            "--run-id",
            run_id,
            "--input-root",
            str(input_root),
            "--split-mode",
            "ratio_chrono",
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.1",
        ],
    )
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(main.__globals__, "validate_splits", fake_validate)

    exit_code = int(main())
    assert exit_code == 3

    report_path = reports_root / "split_validation_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["errors"][0]["code"] == SPLIT_RUNTIME_ERROR


def test_cli_does_not_mutate_manifest_parquet_or_prior_report(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "split_cli_no_mutation_run"
    _, input_root, reports_root, parquet_path, train_input_report_path = _seed_run(
        tmp_path,
        run_id,
        with_manifest=True,
        with_train_input_report=True,
        train_input_overall=True,
    )

    manifest_path = reports_root / "feature_manifest.json"
    before_manifest = _fingerprint(manifest_path)
    before_parquet = _fingerprint(parquet_path)
    before_train_input = _fingerprint(train_input_report_path)

    _patch_read_parquet(monkeypatch, _base_frame(rows=90, freq="1min"))
    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_splits.py",
            "--run-id",
            run_id,
            "--input-root",
            str(input_root),
            "--split-mode",
            "ratio_chrono",
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.1",
        ],
    )
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 0

    after_manifest = _fingerprint(manifest_path)
    after_parquet = _fingerprint(parquet_path)
    after_train_input = _fingerprint(train_input_report_path)

    assert before_manifest == after_manifest
    assert before_parquet == after_parquet
    assert before_train_input == after_train_input
