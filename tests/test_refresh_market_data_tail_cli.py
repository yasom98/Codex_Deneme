"""CLI tests for the market-data tail refresh orchestrator."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "refresh_market_data_tail.py"


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_returns_zero_on_full_success(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "_resolve_refresh_provenance",
        lambda args: {"exchange": "okx", "market_type": "spot", "symbol": "BTC/USDT"},
    )
    monkeypatch.setitem(
        main.__globals__,
        "run_refresh",
        lambda options: {"status": "success", "phase_a": {"status": "success"}, "phase_b": {"status": "success"}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_market_data_tail.py",
            "--accepted-run-id",
            "accepted_ref",
            "--refresh-session-id",
            "refresh_001",
            "--provenance-report-path",
            "provenance.json",
        ],
    )
    assert int(main()) == 0


def test_cli_returns_two_when_phase_a_fails(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "_resolve_refresh_provenance",
        lambda args: {"exchange": "okx", "market_type": "spot", "symbol": "BTC/USDT"},
    )
    monkeypatch.setitem(
        main.__globals__,
        "run_refresh",
        lambda options: {"status": "failed", "phase_a": {"status": "failed"}, "phase_b": {"status": "not_started"}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_market_data_tail.py",
            "--accepted-run-id",
            "accepted_ref",
            "--refresh-session-id",
            "refresh_001",
            "--provenance-report-path",
            "provenance.json",
        ],
    )
    assert int(main()) == 2


def test_cli_returns_three_when_phase_b_fails(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "_resolve_refresh_provenance",
        lambda args: {"exchange": "okx", "market_type": "spot", "symbol": "BTC/USDT"},
    )
    monkeypatch.setitem(
        main.__globals__,
        "run_refresh",
        lambda options: {"status": "failed", "phase_a": {"status": "success"}, "phase_b": {"status": "failed"}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_market_data_tail.py",
            "--accepted-run-id",
            "accepted_ref",
            "--refresh-session-id",
            "refresh_001",
            "--provenance-report-path",
            "provenance.json",
        ],
    )
    assert int(main()) == 3


def test_cli_returns_four_when_provenance_gate_blocks(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "_resolve_refresh_provenance",
        lambda args: (_ for _ in ()).throw(ValueError("blocked")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_market_data_tail.py",
            "--accepted-run-id",
            "accepted_ref",
            "--refresh-session-id",
            "refresh_001",
            "--provenance-report-path",
            "provenance.json",
        ],
    )
    assert int(main()) == 4


def test_cli_returns_zero_for_separate_gateio_perpetual_lineage_success(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "run_separate_gateio_perpetual_lineage",
        lambda options: {"status": "success", "download_report_path": "download.json", "processing_report_path": "processing.json"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_market_data_tail.py",
            "--mode",
            "separate_gateio_perpetual_lineage",
            "--accepted-run-id",
            "accepted_ref",
            "--refresh-session-id",
            "refresh_001",
            "--legacy-input-root",
            ".",
        ],
    )
    assert int(main()) == 0


def test_cli_returns_five_for_separate_gateio_perpetual_lineage_failure(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "run_separate_gateio_perpetual_lineage",
        lambda options: {"status": "failed", "download_report_path": "download.json", "processing_report_path": None},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_market_data_tail.py",
            "--mode",
            "separate_gateio_perpetual_lineage",
            "--accepted-run-id",
            "accepted_ref",
            "--refresh-session-id",
            "refresh_001",
            "--legacy-input-root",
            ".",
        ],
    )
    assert int(main()) == 5


def test_cli_returns_zero_for_separate_binance_perpetual_backfill_success(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "run_separate_binance_perpetual_backfill",
        lambda options: {
            "status": "success",
            "download_report_path": "download.json",
            "processing_report_path": "processing.json",
            "checkpoint_report_path": "checkpoint.json",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_market_data_tail.py",
            "--mode",
            "separate_binance_perpetual_backfill",
            "--accepted-run-id",
            "accepted_ref",
            "--refresh-session-id",
            "refresh_001",
            "--legacy-input-root",
            ".",
            "--historical-max-candles-per-timeframe",
            "100",
        ],
    )
    assert int(main()) == 0


def test_cli_returns_five_for_separate_binance_perpetual_backfill_failure(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "run_separate_binance_perpetual_backfill",
        lambda options: {
            "status": "failed",
            "download_report_path": "download.json",
            "processing_report_path": None,
            "checkpoint_report_path": "checkpoint.json",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_market_data_tail.py",
            "--mode",
            "separate_binance_perpetual_backfill",
            "--accepted-run-id",
            "accepted_ref",
            "--refresh-session-id",
            "refresh_001",
            "--legacy-input-root",
            ".",
            "--historical-max-candles-per-timeframe",
            "100",
        ],
    )
    assert int(main()) == 5
