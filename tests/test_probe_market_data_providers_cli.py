"""CLI tests for provider capability probing."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "probe_market_data_providers.py"


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_returns_zero(monkeypatch: object) -> None:
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "recover_market_data_provenance",
        lambda options: {
            "overall_verdict": "unresolved",
            "canonical_exchange_verdict": {"value": None},
            "market_type_verdict": {"value": None},
            "symbol_normalization_verdict": {"value": None},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_market_data_providers.py",
            "--accepted-run-id",
            "accepted_ref",
            "--probe-session-id",
            "probe_001",
        ],
    )
    assert int(main()) == 0
