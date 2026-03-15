"""CLI entrypoint for canonical market-data provenance recovery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.logging import get_logger, setup_logging
from data.market_provenance import MarketProvenanceOptions, provenance_report_path, recover_market_data_provenance

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Recover canonical exchange, market type and symbol provenance for root market-data lineage.")
    parser.add_argument("--accepted-run-id", type=str, required=True, help="Accepted immutable run id.")
    parser.add_argument("--probe-session-id", type=str, required=True, help="Run id under runs/<probe_session_id>.")
    parser.add_argument("--exchange", type=str, default=None, help="Optional operator hint recorded in the provenance report; it does not upgrade the verdict.")
    parser.add_argument("--market-type", type=str, default=None, help="Optional operator hint recorded in the provenance report; it does not upgrade the verdict.")
    parser.add_argument("--symbol", type=str, default=None, help="Optional operator hint recorded in the provenance report; it does not upgrade the verdict.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    """Run the provider capability probe and return a deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    options = MarketProvenanceOptions(
        project_root=PROJECT_ROOT,
        accepted_run_id=args.accepted_run_id.strip(),
        probe_session_id=args.probe_session_id.strip(),
        exchange_hint=args.exchange.strip().lower() if isinstance(args.exchange, str) and args.exchange.strip() else None,
        market_type_hint=args.market_type.strip() if isinstance(args.market_type, str) and args.market_type.strip() else None,
        symbol_hint=args.symbol.strip() if isinstance(args.symbol, str) and args.symbol.strip() else None,
    )

    report = recover_market_data_provenance(options)
    LOGGER.info(
        "Provenance recovery summary | probe_session_id=%s overall_verdict=%s exchange=%s market_type=%s symbol=%s report_path=%s",
        options.probe_session_id,
        report.get("overall_verdict"),
        report.get("canonical_exchange_verdict", {}).get("value"),
        report.get("market_type_verdict", {}).get("value"),
        report.get("symbol_normalization_verdict", {}).get("value"),
        provenance_report_path(PROJECT_ROOT, options.probe_session_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
