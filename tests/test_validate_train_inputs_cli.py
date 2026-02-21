"""Integration-style tests for validate_train_inputs CLI."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.train_input_validation import TRAIN_INPUT_RUNTIME_ERROR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_train_inputs.py"


def _all_columns() -> list[str]:
    return [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "PP",
        "R1",
        "S1",
        "R2",
        "S2",
        "R3",
        "S3",
        "R4",
        "S4",
        "R5",
        "S5",
        "EMA_200",
        "EMA_600",
        "EMA_1200",
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_up",
        "ST_dn",
        "log_return",
        "hl_range",
        "body_ratio",
        "rolling_vol_20",
        "rolling_vol_50",
        "zscore_return",
        "normalized_range",
        "trend_strength",
        "squeeze_proxy",
        "floating_pnl_placeholder",
        "drawdown_placeholder",
        "evt_at_buy_raw",
        "evt_at_sell_raw",
        "evt_at_buy",
        "evt_at_sell",
        "evt_st_buy",
        "evt_st_sell",
        "trend_regime",
        "volatility_regime",
        "position_placeholder",
        "market_state",
    ]


def _continuous_columns() -> list[str]:
    return [
        "PP",
        "R1",
        "S1",
        "R2",
        "S2",
        "R3",
        "S3",
        "R4",
        "S4",
        "R5",
        "S5",
        "EMA_200",
        "EMA_600",
        "EMA_1200",
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_up",
        "ST_dn",
        "log_return",
        "hl_range",
        "body_ratio",
        "rolling_vol_20",
        "rolling_vol_50",
        "zscore_return",
        "normalized_range",
        "trend_strength",
        "squeeze_proxy",
        "floating_pnl_placeholder",
        "drawdown_placeholder",
    ]


def _event_columns() -> list[str]:
    return [
        "evt_at_buy_raw",
        "evt_at_sell_raw",
        "evt_at_buy",
        "evt_at_sell",
        "evt_st_buy",
        "evt_st_sell",
        "trend_regime",
        "volatility_regime",
        "position_placeholder",
        "market_state",
    ]


def _placeholder_columns() -> list[str]:
    return [
        "position_placeholder",
        "floating_pnl_placeholder",
        "drawdown_placeholder",
        "market_state",
    ]


def _feature_groups() -> dict[str, list[str]]:
    return {
        "raw_ohlcv": ["timestamp", "open", "high", "low", "close", "volume"],
        "price_derived": ["log_return", "hl_range", "body_ratio", "rolling_vol_20", "rolling_vol_50", "zscore_return", "normalized_range"],
        "trend": ["EMA_200", "EMA_600", "EMA_1200", "AlphaTrend", "AlphaTrend_2", "ST_trend", "ST_up", "ST_dn", "trend_strength"],
        "regime": ["trend_regime", "volatility_regime", "squeeze_proxy"],
        "event": ["evt_at_buy_raw", "evt_at_sell_raw", "evt_at_buy", "evt_at_sell", "evt_st_buy", "evt_st_sell"],
        "placeholders": ["position_placeholder", "floating_pnl_placeholder", "drawdown_placeholder", "market_state"],
    }


def _base_frame(rows: int = 64) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    out = pd.DataFrame({"timestamp": ts})
    float_columns = [col for col in _all_columns() if col not in {"timestamp", *_event_columns()}]
    for idx, col in enumerate(float_columns):
        out[col] = pd.Series(np.linspace(1.0 + idx, 2.0 + idx, rows), dtype="float32")

    for idx, col in enumerate(_event_columns()):
        pattern = np.zeros(rows, dtype=np.uint8)
        pattern[idx % 5 :: 7] = np.uint8(1)
        out[col] = pd.Series(pattern, dtype="uint8")

    return out.loc[:, _all_columns()]


def _base_manifest(run_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "manifest_version": "features.manifest.v1",
        "generated_at_utc": "2026-02-21T00:00:00+00:00",
        "run_id": run_id,
        "feature_groups": _feature_groups(),
        "column_dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "event_columns": _event_columns(),
        "continuous_columns": _continuous_columns(),
        "placeholder_columns": _placeholder_columns(),
        "indicator_spec_version": "indicators.v2026-02-15.1",
        "config_hash": "abc123",
        "formula_fingerprint_bundle": "bundle123",
        "timestamp_column": "timestamp",
        "canonical_column_order": list(frame.columns),
    }


def _seed_run(tmp_path: Path, run_id: str, *, with_manifest: bool = True, with_summary: bool = False) -> tuple[Path, Path, Path, Path]:
    runs_root = tmp_path / "runs"
    input_root = runs_root / run_id / "data_features" / "parquet"
    reports_root = runs_root / run_id / "data_features" / "reports"
    input_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    parquet_path = input_root / "sample.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    manifest_path = reports_root / "feature_manifest.json"
    if with_manifest:
        frame = _base_frame()
        manifest_path.write_text(json.dumps(_base_manifest(run_id, frame)), encoding="utf-8")

    summary_path = reports_root / "summary.json"
    if with_summary:
        summary_path.write_text(json.dumps({"run_id": run_id, "total_files": 1}), encoding="utf-8")

    return runs_root, input_root, reports_root, parquet_path


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
    run_id = "cli_pass_run"
    _, input_root, reports_root, _ = _seed_run(tmp_path, run_id, with_manifest=True)
    _patch_read_parquet(monkeypatch, _base_frame())
    main = _load_main()

    monkeypatch.setattr(main.__globals__["sys"], "argv", ["validate_train_inputs.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 0

    report_path = reports_root / "train_input_validation_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["train_input_validation_overall"] is True
    assert payload["total_files"] == 1


def test_cli_contract_fail_exit_two(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "cli_fail_run"
    _, input_root, reports_root, _ = _seed_run(tmp_path, run_id, with_manifest=False)
    main = _load_main()

    monkeypatch.setattr(main.__globals__["sys"], "argv", ["validate_train_inputs.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 2

    report_path = reports_root / "train_input_validation_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["train_input_validation_overall"] is False


def test_summary_update_is_non_blocking(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "cli_summary_non_blocking_run"
    _, input_root, reports_root, _ = _seed_run(tmp_path, run_id, with_manifest=True, with_summary=True)
    _patch_read_parquet(monkeypatch, _base_frame())
    main = _load_main()
    summary_path = reports_root / "summary.json"

    def fake_atomic_write_json(payload: dict[str, Any], dest: Path) -> None:
        if dest == summary_path:
            raise RuntimeError("summary write failure")
        tmp = dest.with_suffix(f"{dest.suffix}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, dest)

    monkeypatch.setattr(main.__globals__["sys"], "argv", ["validate_train_inputs.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(main.__globals__, "atomic_write_json", fake_atomic_write_json)

    exit_code = int(main())
    assert exit_code == 0

    report_path = reports_root / "train_input_validation_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["train_input_validation_overall"] is True
    assert summary_path.exists()


def test_cli_runtime_error_exit_three(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "cli_runtime_error_run"
    _, input_root, reports_root, _ = _seed_run(tmp_path, run_id, with_manifest=True)
    main = _load_main()

    def fake_validate(_: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(main.__globals__["sys"], "argv", ["validate_train_inputs.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(main.__globals__, "validate_train_inputs", fake_validate)

    exit_code = int(main())
    assert exit_code == 3

    report_path = reports_root / "train_input_validation_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["errors"][0]["code"] == TRAIN_INPUT_RUNTIME_ERROR


def test_cli_does_not_mutate_manifest_or_parquet(monkeypatch: object, tmp_path: Path) -> None:
    run_id = "cli_no_mutation_run"
    _, input_root, reports_root, parquet_path = _seed_run(tmp_path, run_id, with_manifest=True, with_summary=True)
    manifest_path = reports_root / "feature_manifest.json"
    before_manifest = _fingerprint(manifest_path)
    before_parquet = _fingerprint(parquet_path)

    _patch_read_parquet(monkeypatch, _base_frame())
    main = _load_main()
    monkeypatch.setattr(main.__globals__["sys"], "argv", ["validate_train_inputs.py", "--run-id", run_id, "--input-root", str(input_root)])
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)

    exit_code = int(main())
    assert exit_code == 0

    after_manifest = _fingerprint(manifest_path)
    after_parquet = _fingerprint(parquet_path)
    assert before_manifest == after_manifest
    assert before_parquet == after_parquet
