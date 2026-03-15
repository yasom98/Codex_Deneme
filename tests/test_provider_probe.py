"""Tests for provider capability and provenance gating."""

from __future__ import annotations

import json
from pathlib import Path

from data.provider_probe import ProviderProbeOptions, probe_provider_capabilities


def _write_data_config(config_path: Path, input_root: Path, runs_root: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {input_root}",
                f"runs_root: {runs_root}",
                'csv_glob: "*.csv"',
                "timestamp_aliases:",
                "  - timestamp",
                "  - ts",
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


def _write_raw_csv(path: Path, timeframe: str) -> None:
    if timeframe == "15m":
        payload = "\n".join(
            [
                "ts;open;high;low;close;volume",
                "2024-01-01 00:00:00+00:00;30;31;29;30.5;300",
                "2024-01-01 00:15:00+00:00;31;32;30;31.5;301",
            ]
        )
    elif timeframe == "5m":
        payload = "\n".join(
            [
                "timestamp,open,high,low,close,volume",
                "2024-01-01 00:00:00+00:00,20,21,19,20.5,200",
                "2024-01-01 00:05:00+00:00,21,22,20,21.5,201",
            ]
        )
    else:
        payload = "\n".join(
            [
                "ts,open,high,low,close,volume",
                "2024-01-01 00:00:00+00:00,10,11,9,10.5,100",
                "2024-01-01 00:01:00+00:00,11,12,10,11.5,101",
            ]
        )
    path.write_text(payload, encoding="utf-8")


def _seed_project(tmp_path: Path) -> Path:
    project_root = tmp_path
    configs_root = project_root / "configs"
    runs_root = project_root / "runs"
    configs_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    _write_data_config(configs_root / "data.yaml", project_root, runs_root)

    accepted_run = "accepted_ref"
    standardized_root = runs_root / accepted_run / "data_standardized"
    (standardized_root / "reports" / "per_file").mkdir(parents=True, exist_ok=True)
    (standardized_root / "parquet").mkdir(parents=True, exist_ok=True)
    (standardized_root / "reports" / "summary.json").write_text(
        json.dumps({"total_files": 3, "succeeded_files": 3, "failed_files": 0}),
        encoding="utf-8",
    )

    for timeframe in ("1m", "5m", "15m"):
        csv_path = project_root / f"BTC_USDT_{timeframe}_price_data.csv"
        _write_raw_csv(csv_path, timeframe)
        parquet_path = standardized_root / "parquet" / f"BTC_USDT_{timeframe}_price_data.parquet"
        parquet_path.write_text("placeholder", encoding="utf-8")
        (standardized_root / "reports" / "per_file" / f"BTC_USDT_{timeframe}_price_data.json").write_text(
            json.dumps({"input_file": str(csv_path.resolve()), "output_file": str(parquet_path.resolve())}),
            encoding="utf-8",
        )
    return project_root


def test_provider_probe_blocks_when_market_type_is_unresolved(tmp_path: Path) -> None:
    project_root = _seed_project(tmp_path)

    report = probe_provider_capabilities(
        ProviderProbeOptions(
            project_root=project_root,
            accepted_run_id="accepted_ref",
            probe_session_id="probe_unresolved",
            exchange=None,
            market_type=None,
            symbol=None,
            retry_backoff_seconds=0.0,
            max_retries=1,
        )
    )

    assert report["source_of_truth_verdict"]["status"] == "proven_canonical_raw"
    assert report["canonical_exchange_verdict"]["status"] == "unresolved"
    assert report["market_type_verdict"]["status"] == "unresolved"
    assert report["recommended_live_smoke_provider_order"] == []
    providers = {item["provider"]: item for item in report["provider_results"]}
    assert set(providers) == {"gateio", "okx", "bybit"}
    assert all(item["probe_status"] == "blocked" for item in providers.values())
