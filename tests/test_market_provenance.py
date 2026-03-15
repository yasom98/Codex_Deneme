"""Tests for canonical market-data provenance recovery."""

from __future__ import annotations

import json
from pathlib import Path

from data.market_provenance import (
    MarketProvenanceOptions,
    compare_forensic_candle_slices,
    provenance_report_path,
    recover_market_data_provenance,
)


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


def test_provenance_recovery_is_fail_closed_without_explicit_fields(tmp_path: Path) -> None:
    project_root = _seed_project(tmp_path)

    report = recover_market_data_provenance(
        MarketProvenanceOptions(
            project_root=project_root,
            accepted_run_id="accepted_ref",
            probe_session_id="probe_unresolved",
        )
    )

    assert report["source_of_truth_verdict"]["status"] == "proven"
    assert report["canonical_exchange_verdict"]["status"] == "unresolved"
    assert report["market_type_verdict"]["status"] == "unresolved"
    assert report["symbol_normalization_verdict"]["status"] == "unresolved"
    assert report["overall_verdict"] == "unresolved"
    assert report["live_refresh_gate"]["status"] == "blocked"
    assert report["repo_evidence"]["root_symbol_id"] == "BTC_USDT"
    assert report["accepted_run_evidence"]["explicit_provenance_matches"] == []
    report_path = provenance_report_path(project_root, "probe_unresolved")
    assert report_path.exists()


def test_provenance_recovery_returns_proven_when_explicit_manifest_exists(tmp_path: Path) -> None:
    project_root = _seed_project(tmp_path)
    manifest_path = project_root / "runs" / "accepted_ref" / "data_standardized" / "reports" / "market_data_provenance.json"
    manifest_path.write_text(
        json.dumps(
            {
                "canonical_exchange": "gateio",
                "market_type": "spot",
                "exchange_symbol": "BTC_USDT",
            }
        ),
        encoding="utf-8",
    )

    report = recover_market_data_provenance(
        MarketProvenanceOptions(
            project_root=project_root,
            accepted_run_id="accepted_ref",
            probe_session_id="probe_proven",
        )
    )

    assert report["canonical_exchange_verdict"]["status"] == "proven"
    assert report["canonical_exchange_verdict"]["value"] == "gateio"
    assert report["market_type_verdict"]["status"] == "proven"
    assert report["market_type_verdict"]["value"] == "spot"
    assert report["symbol_normalization_verdict"]["status"] == "proven"
    assert report["symbol_normalization_verdict"]["value"] == "BTC_USDT"
    assert report["overall_verdict"] == "proven"
    assert report["live_refresh_gate"]["status"] == "approved"


def test_compare_forensic_candle_slices_reports_volume_mismatch() -> None:
    result = compare_forensic_candle_slices(
        canonical_rows=[
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10.0,
            }
        ],
        candidate_rows=[
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 11.0,
            }
        ],
        abs_tolerance=0.0,
    )

    assert result["status"] == "mismatch"
    assert result["mismatches"][0]["fields"]["volume"] == {"canonical": 10.0, "candidate": 11.0}
