"""CLI entrypoint for safe canonical market-data tail refresh and rebuild."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.logging import get_logger, setup_logging
from data.binance_perpetual_backfill import BinancePerpetualBackfillOptions, run_separate_binance_perpetual_backfill
from data.gateio_perpetual_lineage import GateioPerpetualLineageOptions, run_separate_gateio_perpetual_lineage
from data.tail_refresh import RefreshOptions, run_refresh

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Refresh the missing tail of canonical market data and rebuild downstream layers.")
    parser.add_argument(
        "--mode",
        type=str,
        default="canonical_refresh",
        choices=("canonical_refresh", "separate_gateio_perpetual_lineage", "separate_binance_perpetual_backfill"),
        help=(
            "Execution mode. The default canonical_refresh path is unchanged; separate_gateio_perpetual_lineage "
            "runs a recent-window Gate.io smoke, while separate_binance_perpetual_backfill runs the historical "
            "perpetual backfill track."
        ),
    )
    parser.add_argument("--accepted-run-id", type=str, required=True, help="Accepted upstream run id used as the immutable reference lineage.")
    parser.add_argument("--refresh-session-id", type=str, required=True, help="New refresh session id under runs/<refresh_session_id>.")
    parser.add_argument(
        "--legacy-input-root",
        type=Path,
        default=None,
        help="Legacy immutable raw CSV root used only to derive the new perpetual lineage start timestamps.",
    )
    parser.add_argument(
        "--provenance-report-path",
        type=Path,
        default=None,
        help="Required provenance recovery report. Live refresh is allowed only when its overall verdict is exactly proven.",
    )
    parser.add_argument("--exchange", type=str, default=None, help="Explicit CCXT exchange id when repo provenance is missing.")
    parser.add_argument("--market-type", type=str, default=None, choices=("spot", "swap", "future", "futures", "perpetual"))
    parser.add_argument("--symbol", type=str, default=None, help="Explicit exchange symbol, for example BTC/USDT.")
    parser.add_argument("--fallback-exchanges", type=str, default="", help="Optional comma-separated fallback exchange ids.")
    parser.add_argument("--provider-capability-report-path", type=Path, default=None, help="Required provider capability gate report before live refresh.")
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml", help="Standardization data config path.")
    parser.add_argument("--features-config", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml", help="Feature config path.")
    parser.add_argument("--request-limit", type=int, default=500, help="Maximum OHLCV rows requested per CCXT call before exchange-specific capping.")
    parser.add_argument(
        "--recent-window-candles",
        type=int,
        default=1500,
        help="Recent closed-candle window size for the Gate.io smoke track.",
    )
    parser.add_argument(
        "--historical-max-candles-per-timeframe",
        type=int,
        default=0,
        help="Optional bounded-candle cap for the Binance historical backfill track. Zero means full backfill.",
    )
    parser.add_argument(
        "--target-end-utc",
        type=str,
        default=None,
        help="Optional explicit UTC cutoff for the Binance historical backfill track, for example 2026-03-14T12:00:00+00:00.",
    )
    parser.add_argument("--max-retries", type=int, default=4, help="Maximum fetch retries per request.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.5, help="Backoff multiplier for temporary fetch failures.")
    parser.add_argument("--overlap-abs-tolerance", type=float, default=0.0, help="Absolute OHLCV tolerance for overlap equivalence validation.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def _split_csv_arg(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Failed to read provenance report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid provenance report JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _normalized_cli_market_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "futures":
        return "future"
    if normalized == "perpetual":
        return "swap"
    return normalized or None


def _resolve_refresh_provenance(args: argparse.Namespace) -> dict[str, str]:
    if args.provenance_report_path is None:
        raise ValueError("provenance report path is required before live refresh.")

    payload = _load_json(args.provenance_report_path.resolve())
    if payload.get("overall_verdict") != "proven":
        raise ValueError("live refresh remains blocked because provenance overall_verdict is not proven.")

    exchange_verdict = payload.get("canonical_exchange_verdict", {})
    market_type_verdict = payload.get("market_type_verdict", {})
    symbol_verdict = payload.get("symbol_normalization_verdict", {})
    exchange = exchange_verdict.get("value")
    market_type = market_type_verdict.get("value")
    symbol = symbol_verdict.get("value")

    if not all(isinstance(item, str) and item.strip() for item in (exchange, market_type, symbol)):
        raise ValueError("provenance report is missing proven exchange, market_type or exchange symbol values.")

    cli_exchange = args.exchange.strip().lower() if isinstance(args.exchange, str) and args.exchange.strip() else None
    cli_market_type = _normalized_cli_market_type(args.market_type)
    cli_symbol = args.symbol.strip() if isinstance(args.symbol, str) and args.symbol.strip() else None

    resolved_exchange = str(exchange).strip().lower()
    resolved_market_type = str(market_type).strip().lower()
    resolved_symbol = str(symbol).strip()

    if cli_exchange is not None and cli_exchange != resolved_exchange:
        raise ValueError("CLI exchange override conflicts with the proven provenance report.")
    if cli_market_type is not None and cli_market_type != resolved_market_type:
        raise ValueError("CLI market-type override conflicts with the proven provenance report.")
    if cli_symbol is not None and cli_symbol != resolved_symbol:
        raise ValueError("CLI symbol override conflicts with the proven provenance report.")

    return {"exchange": resolved_exchange, "market_type": resolved_market_type, "symbol": resolved_symbol}


def main() -> int:
    """Run the tail-refresh flow and return a deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)
    if args.mode == "separate_gateio_perpetual_lineage":
        if args.legacy_input_root is None:
            LOGGER.error("Separate perpetual lineage blocked | reason=legacy-input-root is required in separate_gateio_perpetual_lineage mode")
            return 5
        result = run_separate_gateio_perpetual_lineage(
            GateioPerpetualLineageOptions(
                project_root=PROJECT_ROOT,
                accepted_run_id=args.accepted_run_id.strip(),
                refresh_session_id=args.refresh_session_id.strip(),
                legacy_input_root=args.legacy_input_root.resolve(),
                data_config_path=args.data_config.resolve(),
                features_config_path=args.features_config.resolve(),
                request_limit=int(args.request_limit),
                max_retries=int(args.max_retries),
                retry_backoff_seconds=float(args.retry_backoff_seconds),
                python_executable=Path(sys.executable).resolve(),
                recent_window_candles=int(args.recent_window_candles),
                log_level=str(args.log_level),
            )
        )
        LOGGER.info(
            "Separate perpetual lineage summary | refresh_session_id=%s overall=%s download_report=%s processing_report=%s",
            args.refresh_session_id,
            result.get("status"),
            result.get("download_report_path"),
            result.get("processing_report_path"),
        )
        return 0 if result.get("status") == "success" else 5

    if args.mode == "separate_binance_perpetual_backfill":
        if args.legacy_input_root is None:
            LOGGER.error("Separate perpetual lineage blocked | reason=legacy-input-root is required in separate_binance_perpetual_backfill mode")
            return 5
        result = run_separate_binance_perpetual_backfill(
            BinancePerpetualBackfillOptions(
                project_root=PROJECT_ROOT,
                accepted_run_id=args.accepted_run_id.strip(),
                refresh_session_id=args.refresh_session_id.strip(),
                legacy_input_root=args.legacy_input_root.resolve(),
                data_config_path=args.data_config.resolve(),
                features_config_path=args.features_config.resolve(),
                request_limit=int(args.request_limit),
                max_retries=int(args.max_retries),
                retry_backoff_seconds=float(args.retry_backoff_seconds),
                python_executable=Path(sys.executable).resolve(),
                max_candles_per_timeframe=int(args.historical_max_candles_per_timeframe),
                target_end_utc=args.target_end_utc.strip() if isinstance(args.target_end_utc, str) and args.target_end_utc.strip() else None,
                log_level=str(args.log_level),
            )
        )
        LOGGER.info(
            "Separate perpetual historical summary | refresh_session_id=%s overall=%s download_report=%s processing_report=%s checkpoint=%s",
            args.refresh_session_id,
            result.get("status"),
            result.get("download_report_path"),
            result.get("processing_report_path"),
            result.get("checkpoint_report_path"),
        )
        return 0 if result.get("status") == "success" else 5

    try:
        resolved_provenance = _resolve_refresh_provenance(args)
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("Tail refresh blocked by provenance gate | reason=%s", exc)
        return 4

    options = RefreshOptions(
        project_root=PROJECT_ROOT,
        accepted_run_id=args.accepted_run_id.strip(),
        refresh_session_id=args.refresh_session_id.strip(),
        data_config_path=args.data_config.resolve(),
        features_config_path=args.features_config.resolve(),
        provider_probe_report_path=args.provider_capability_report_path.resolve() if args.provider_capability_report_path is not None else None,
        exchange=resolved_provenance["exchange"],
        market_type=resolved_provenance["market_type"],
        symbol=resolved_provenance["symbol"],
        fallback_exchanges=_split_csv_arg(args.fallback_exchanges),
        request_limit=int(args.request_limit),
        max_retries=int(args.max_retries),
        retry_backoff_seconds=float(args.retry_backoff_seconds),
        overlap_abs_tolerance=float(args.overlap_abs_tolerance),
        python_executable=Path(sys.executable).resolve(),
        log_level=str(args.log_level),
    )

    result = run_refresh(options)
    phase_a = result.get("phase_a", {})
    phase_b = result.get("phase_b", {})

    LOGGER.info(
        "Tail refresh summary | refresh_session_id=%s overall=%s phase_a=%s phase_b=%s",
        options.refresh_session_id,
        result.get("status"),
        phase_a.get("status"),
        phase_b.get("status"),
    )

    if phase_a.get("status") != "success":
        return 2
    if phase_b.get("status") != "success":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
