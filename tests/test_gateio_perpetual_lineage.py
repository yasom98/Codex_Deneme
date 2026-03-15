"""Tests for the separate Gate.io perpetual lineage flow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pandas as pd

from data.gateio_perpetual_lineage import (
    GateioPerpetualLineageOptions,
    _run_processing_chain,
    extract_legacy_reference_points,
    run_separate_gateio_perpetual_lineage,
)


def _write_data_config(config_path: Path, input_root: Path, runs_root: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {input_root}",
                f"runs_root: {runs_root}",
                'csv_glob: "**/*.csv"',
                "timestamp_aliases:",
                "  - timestamp",
                "  - ts",
                "  - datetime",
                "  - time",
                "required_columns:",
                "  - open",
                "  - high",
                "  - low",
                "  - close",
                "  - volume",
                "float_columns:",
                "  - open",
                "  - high",
                "  - low",
                "  - close",
                "  - volume",
                "duplicate_policy: last",
                "seed: 42",
            ]
        ),
        encoding="utf-8",
    )


def _write_features_config(config_path: Path, runs_root: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {runs_root / 'placeholder' / 'data_standardized' / 'parquet'}",
                f"runs_root: {runs_root}",
                'parquet_glob: "*.parquet"',
                "seed: 42",
                "",
                "supertrend:",
                "  periods: 10",
                "  multiplier: 3.0",
                "  source: hl2",
                "  change_atr_method: true",
                "",
                "alphatrend:",
                "  coeff: 3.0",
                "  ap: 11",
                "  use_no_volume: false",
                "",
                "pivot:",
                "  pivot_tf: 1D",
                "  warmup_policy: allow_first_session_nan",
                "  first_session_fill: ffill_from_second_session",
                "",
                "parity:",
                "  enabled: true",
                "  sample_rows: 128",
                "  float_atol: 1.0e-6",
                "  float_rtol: 1.0e-6",
                "",
                "health:",
                "  warn_ratio: 1.0",
                "  critical_warn_ratio: 1.0",
                "  critical_columns:",
                "    - EMA_200",
            ]
        ),
        encoding="utf-8",
    )


def _write_legacy_csv(path: Path, timeframe: str) -> None:
    if timeframe == "1m":
        path.write_text(
            "\n".join(
                [
                    "ts,open,high,low,close,volume",
                    "2024-01-01 00:00:00+00:00,10,11,9,10.5,100",
                    "2024-01-01 00:01:00+00:00,11,12,10,11.5,101",
                ]
            ),
            encoding="utf-8",
        )
        return
    if timeframe == "5m":
        path.write_text(
            "\n".join(
                [
                    "timestamp,open,high,low,close,volume",
                    "2024-01-01 00:00:00+00:00,20,21,19,20.5,200",
                    "2024-01-01 00:05:00+00:00,21,22,20,21.5,201",
                ]
            ),
            encoding="utf-8",
        )
        return
    if timeframe == "15m":
        path.write_text(
            "\n".join(
                [
                    "ts;open;high;low;close;volume",
                    "2024-01-01 00:00:00+00:00;30;31;29;30.5;300",
                    "2024-01-01 00:15:00+00:00;31;32;30;31.5;301",
                ]
            ),
            encoding="utf-8",
        )
        return
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _feature_manifest(run_id: str) -> dict[str, object]:
    return {
        "manifest_version": "features.manifest.v1",
        "run_id": run_id,
        "feature_groups": {"raw_ohlcv": ["timestamp", "open", "high", "low", "close", "volume"]},
        "column_dtypes": {
            "timestamp": "datetime64[ns, UTC]",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float32",
        },
        "event_columns": [],
        "continuous_columns": ["open", "high", "low", "close", "volume"],
        "placeholder_columns": [],
        "warmup_policy": {"pivot_warmup_policy": "allow_first_session_nan"},
        "indicator_spec_version": "indicators.vtest",
        "config_hash": "cfg-hash",
        "formula_fingerprint_bundle": "bundle-hash",
    }


def _seed_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project_root = tmp_path
    runs_root = project_root / "runs"
    configs_root = project_root / "configs"
    configs_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    data_config_path = configs_root / "data.yaml"
    features_config_path = configs_root / "features.yaml"
    _write_data_config(data_config_path, project_root, runs_root)
    _write_features_config(features_config_path, runs_root)

    for timeframe in ("1m", "5m", "15m"):
        _write_legacy_csv(project_root / f"BTC_USDT_{timeframe}_price_data.csv", timeframe)

    accepted_feature_root = runs_root / "accepted_ref" / "data_features" / "reports"
    accepted_feature_root.mkdir(parents=True, exist_ok=True)
    accepted_feature_root.joinpath("feature_manifest.json").write_text(json.dumps(_feature_manifest("accepted_ref")), encoding="utf-8")

    return project_root, data_config_path, features_config_path, runs_root


def _options(project_root: Path, data_config_path: Path, features_config_path: Path) -> GateioPerpetualLineageOptions:
    return GateioPerpetualLineageOptions(
        project_root=project_root,
        accepted_run_id="accepted_ref",
        refresh_session_id="refresh_sep",
        legacy_input_root=project_root,
        data_config_path=data_config_path,
        features_config_path=features_config_path,
        request_limit=500,
        max_retries=1,
        retry_backoff_seconds=0.0,
        python_executable=project_root / ".venv" / "bin" / "python",
        recent_window_candles=4,
        log_level="INFO",
    )


def test_extract_legacy_reference_points_reads_last_timestamps(tmp_path: Path) -> None:
    project_root, data_config_path, features_config_path, _ = _seed_project(tmp_path)

    points = extract_legacy_reference_points(_options(project_root, data_config_path, features_config_path))
    by_timeframe = {point.timeframe: point for point in points}

    assert by_timeframe["1m"].legacy_last_timestamp.isoformat() == "2024-01-01T00:01:00+00:00"
    assert by_timeframe["1m"].download_start_timestamp.isoformat() == "2024-01-01T00:02:00+00:00"
    assert by_timeframe["5m"].legacy_last_timestamp.isoformat() == "2024-01-01T00:05:00+00:00"
    assert by_timeframe["5m"].download_start_timestamp.isoformat() == "2024-01-01T00:10:00+00:00"
    assert by_timeframe["15m"].legacy_last_timestamp.isoformat() == "2024-01-01T00:15:00+00:00"
    assert by_timeframe["15m"].download_start_timestamp.isoformat() == "2024-01-01T00:30:00+00:00"


def test_run_separate_lineage_writes_new_raw_files_without_touching_legacy(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config_path, features_config_path, _ = _seed_project(tmp_path)
    options = _options(project_root, data_config_path, features_config_path)
    legacy_before = {
        timeframe: (project_root / f"BTC_USDT_{timeframe}_price_data.csv").read_text(encoding="utf-8")
        for timeframe in ("1m", "5m", "15m")
    }

    class _FakeClient:
        def load_markets(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_fetch(*, timeframe: str, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp, **_: object) -> pd.DataFrame:
        freq = {"1m": "1min", "5m": "5min", "15m": "15min"}[timeframe]
        timestamps = pd.date_range(start=start_timestamp, end=end_timestamp, freq=freq, tz="UTC")
        rows = []
        for index, timestamp in enumerate(timestamps, start=1):
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": float(index),
                    "high": float(index) + 0.1,
                    "low": float(index) - 0.1,
                    "close": float(index) + 0.05,
                    "volume": float(index) * 10.0,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr("data.gateio_perpetual_lineage._utc_now", lambda: datetime(2024, 1, 1, 0, 46, tzinfo=timezone.utc))
    monkeypatch.setattr("data.gateio_perpetual_lineage._build_exchange_client", lambda exchange_id, market_type: _FakeClient())
    monkeypatch.setattr("data.gateio_perpetual_lineage._fetch_gateio_perpetual_timeframe", lambda client, **kwargs: fake_fetch(**kwargs))
    monkeypatch.setattr(
        "data.gateio_perpetual_lineage._run_processing_chain",
        lambda options, raw_input_root: {
            "status": "success",
            "report": {
                "status": "success",
                "refresh_session_id": options.refresh_session_id,
                "exchange_used": "gateio",
                "market_type": "perpetual",
                "source_lineage": "separate_parallel_lineage",
                "provider_track": "recent_window_smoke",
                "provider_choice_reason": "smoke",
                "legacy_lineage_touched": False,
                "raw_input_root": str(raw_input_root.resolve()),
                "feature_contract_compatibility": {"status": "success", "checks": {}},
                "stages": [],
                "errors": [],
            },
        },
    )

    result = run_separate_gateio_perpetual_lineage(options)

    assert result["status"] == "success"
    for timeframe in ("1m", "5m", "15m"):
        legacy_path = project_root / f"BTC_USDT_{timeframe}_price_data.csv"
        assert legacy_path.read_text(encoding="utf-8") == legacy_before[timeframe]

        output_path = (
            project_root
            / "runs"
            / "refresh_sep"
            / "data_tail_refresh"
            / "separate_parallel_lineage"
            / "raw"
            / "recent_window_smoke"
            / "gateio_perpetual"
            / f"BTC_USDT_{timeframe}_price_data.csv"
        )
        assert output_path.exists()

    report_path = project_root / "runs" / "refresh_sep" / "data_tail_refresh" / "reports" / "separate_parallel_lineage_download_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["refresh_session_id"] == "refresh_sep"
    assert payload["exchange_used"] == "gateio"
    assert payload["market_type"] == "perpetual"
    assert payload["source_lineage"] == "separate_parallel_lineage"
    assert payload["provider_track"] == "recent_window_smoke"
    assert payload["legacy_lineage_touched"] is False
    assert payload["provider_choice_reason"]
    assert len(payload["legacy_reference_files"]) == 3
    assert {item["timeframe"] for item in payload["timeframes"]} == {"1m", "5m", "15m"}
    for item in payload["timeframes"]:
        assert item["rows_downloaded"] > 0
        assert item["output_file"] is not None
        assert item["legacy_last_timestamp_utc"] is not None
        assert item["download_start_utc"] is not None
        assert item["download_end_utc"] is not None
        assert item["legacy_lineage_touched"] is False


def test_run_separate_lineage_fails_closed_on_partial_download(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config_path, features_config_path, _ = _seed_project(tmp_path)
    options = _options(project_root, data_config_path, features_config_path)

    class _FakeClient:
        def load_markets(self) -> None:
            return None

        def close(self) -> None:
            return None

    def partial_fetch(*, timeframe: str, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp, **_: object) -> pd.DataFrame:
        del timeframe
        timestamps = [start_timestamp, end_timestamp]
        return pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.05,
                    "volume": 10.0,
                }
                for timestamp in timestamps
            ]
        )

    monkeypatch.setattr("data.gateio_perpetual_lineage._utc_now", lambda: datetime(2024, 1, 1, 0, 46, tzinfo=timezone.utc))
    monkeypatch.setattr("data.gateio_perpetual_lineage._build_exchange_client", lambda exchange_id, market_type: _FakeClient())
    monkeypatch.setattr("data.gateio_perpetual_lineage._fetch_gateio_perpetual_timeframe", lambda client, **kwargs: partial_fetch(**kwargs))

    result = run_separate_gateio_perpetual_lineage(options)

    assert result["status"] == "failed"
    output_root = (
        project_root
        / "runs"
        / "refresh_sep"
        / "data_tail_refresh"
        / "separate_parallel_lineage"
        / "raw"
        / "recent_window_smoke"
        / "gateio_perpetual"
    )
    assert not output_root.exists()

    report_path = project_root / "runs" / "refresh_sep" / "data_tail_refresh" / "reports" / "separate_parallel_lineage_download_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["errors"][0]["code"] == "PERPETUAL_DOWNLOAD_FAILED"


def test_processing_chain_uses_explicit_input_roots(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config_path, features_config_path, runs_root = _seed_project(tmp_path)
    options = _options(project_root, data_config_path, features_config_path)
    raw_input_root = (
        project_root / "runs" / "refresh_sep" / "data_tail_refresh" / "separate_parallel_lineage" / "raw" / "recent_window_smoke"
    )
    raw_input_root.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []

    def fake_run_subprocess(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        del cwd
        commands.append(list(command))
        run_id = command[command.index("--run-id") + 1]
        if command[1] == "scripts/make_standardized_data.py":
            summary_path = runs_root / run_id / "data_standardized" / "reports" / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        elif command[1] == "scripts/make_features.py":
            reports_root = runs_root / run_id / "data_features" / "reports"
            reports_root.mkdir(parents=True, exist_ok=True)
            reports_root.joinpath("summary.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            reports_root.joinpath("feature_manifest.json").write_text(json.dumps(_feature_manifest("accepted_ref")), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("data.gateio_perpetual_lineage._run_subprocess", fake_run_subprocess)

    result = _run_processing_chain(options, raw_input_root=raw_input_root)

    assert result["status"] == "success"
    assert commands[0][1] == "scripts/make_standardized_data.py"
    assert commands[0][commands[0].index("--input-root") + 1] == str(raw_input_root.resolve())
    expected_feature_input_root = runs_root / "refresh_sep" / "data_standardized" / "parquet"
    assert commands[1][1] == "scripts/make_features.py"
    assert commands[1][commands[1].index("--input-root") + 1] == str(expected_feature_input_root.resolve())
    assert result["report"]["feature_contract_compatibility"]["status"] == "success"
