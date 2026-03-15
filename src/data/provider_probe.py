"""Provider capability and provenance gate for market-data refresh."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence

from core.io_atomic import atomic_write_json
from core.logging import get_logger
from data.tail_refresh import OHLCV_FETCH_LIMITS, REQUIRED_TIMEFRAMES, _build_exchange_client, _exchange_limit, _normalize_market_type, verify_canonical_raw_lineage

LOGGER = get_logger(__name__)

PROVIDER_PROBE_VERSION = "provider_capability_probe.v1"
PROVIDER_CANDIDATES: tuple[str, ...] = ("gateio", "okx", "bybit")


@dataclass(frozen=True)
class ProviderProbeOptions:
    """Runtime options for the provider capability probe."""

    project_root: Path
    accepted_run_id: str
    probe_session_id: str
    exchange: str | None
    market_type: str | None
    symbol: str | None
    retry_backoff_seconds: float
    max_retries: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _probe_report_path(project_root: Path, probe_session_id: str) -> Path:
    return project_root / "runs" / probe_session_id / "data_tail_refresh" / "reports" / "provider_capability_report.json"


def _discover_repo_provenance(project_root: Path, accepted_run_id: str) -> dict[str, Any]:
    """Return repo-level exchange/market-type provenance verdicts."""

    del project_root, accepted_run_id
    return {
        "canonical_exchange_verdict": {
            "status": "unresolved",
            "value": None,
            "reason": "No exchange provenance field is encoded in current repo configs, manifests or accepted-run reports.",
        },
        "market_type_verdict": {
            "status": "unresolved",
            "value": None,
            "reason": "No market-type provenance field is encoded in current repo configs, manifests or accepted-run reports.",
        },
        "symbol_verdict": {
            "status": "partial",
            "value": "BTC_USDT",
            "reason": "Canonical symbol id is proven by canonical raw filenames, but exchange symbol normalization is not proven.",
            "exchange_symbol": None,
        },
    }


def _provider_order(canonical_exchange: str | None) -> tuple[str, ...]:
    providers: list[str] = []
    if isinstance(canonical_exchange, str) and canonical_exchange.strip():
        providers.append(canonical_exchange.strip().lower())
    for candidate in PROVIDER_CANDIDATES:
        if candidate not in providers:
            providers.append(candidate)
    return tuple(providers)


def _retry_fetch_ohlcv(client: Any, *, symbol: str, timeframe: str, max_retries: int, retry_backoff_seconds: float) -> Sequence[Sequence[Any]]:
    last_error: Exception | None = None
    limit = _exchange_limit(str(getattr(client, "id", "")), OHLCV_FETCH_LIMITS.get(str(getattr(client, "id", "")), 100))
    for attempt in range(1, max_retries + 1):
        try:
            return client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=min(limit, 2))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_backoff_seconds * float(attempt))
    raise RuntimeError(f"fetch_ohlcv probe failed for {getattr(client, 'id', 'unknown')} {symbol} {timeframe}") from last_error


def probe_provider_capabilities(options: ProviderProbeOptions) -> dict[str, Any]:
    """Run the provenance gate and, when possible, lightweight provider capability probes."""

    verification = verify_canonical_raw_lineage(options.project_root, options.accepted_run_id)
    repo_provenance = _discover_repo_provenance(options.project_root, options.accepted_run_id)
    canonical_exchange_value = repo_provenance["canonical_exchange_verdict"]["value"]
    market_type_value = repo_provenance["market_type_verdict"]["value"]
    symbol_value = repo_provenance["symbol_verdict"]["exchange_symbol"]

    if options.exchange is not None and options.exchange.strip():
        canonical_exchange_value = options.exchange.strip().lower()
        repo_provenance["canonical_exchange_verdict"] = {
            "status": "explicit_override",
            "value": canonical_exchange_value,
            "reason": "Operator-supplied explicit provider override.",
        }
    normalized_market_type = _normalize_market_type(options.market_type)
    if normalized_market_type is not None:
        market_type_value = normalized_market_type
        repo_provenance["market_type_verdict"] = {
            "status": "explicit_override",
            "value": normalized_market_type,
            "reason": "Operator-supplied explicit market-type override.",
        }
    if options.symbol is not None and options.symbol.strip():
        symbol_value = options.symbol.strip()
        repo_provenance["symbol_verdict"] = {
            "status": "explicit_override",
            "value": "BTC_USDT",
            "reason": "Operator-supplied explicit exchange symbol normalization.",
            "exchange_symbol": symbol_value,
        }

    providers = _provider_order(canonical_exchange_value if repo_provenance["canonical_exchange_verdict"]["status"] != "unresolved" else None)
    entries: list[dict[str, Any]] = []

    source_of_truth_verdict = {
        "status": verification.get("source_of_truth_status"),
        "details": {
            "config_path": verification.get("config_path"),
            "input_root": verification.get("input_root"),
            "discovered_csv_files": verification.get("discovered_csv_files"),
        },
    }

    if verification.get("status") != "success":
        report = {
            "report_version": PROVIDER_PROBE_VERSION,
            "generated_at_utc": _utc_now(),
            "probe_session_id": options.probe_session_id,
            "accepted_run_id": options.accepted_run_id,
            "source_of_truth_verdict": source_of_truth_verdict,
            "canonical_exchange_verdict": repo_provenance["canonical_exchange_verdict"],
            "market_type_verdict": repo_provenance["market_type_verdict"],
            "symbol_verdict": repo_provenance["symbol_verdict"],
            "provider_results": [],
            "recommended_live_smoke_provider_order": [],
            "errors": list(verification.get("errors", [])),
        }
        atomic_write_json(report, _probe_report_path(options.project_root, options.probe_session_id))
        return report

    market_type_resolved = isinstance(market_type_value, str) and market_type_value != ""
    symbol_resolved = isinstance(symbol_value, str) and symbol_value != ""
    if not market_type_resolved or not symbol_resolved:
        reason = "Market type is unresolved." if not market_type_resolved else "Exchange symbol normalization is unresolved."
        for provider in providers:
            entries.append(
                {
                    "provider": provider,
                    "market_type_tested": market_type_value,
                    "symbol_normalization_used": symbol_value,
                    "accessibility_success": False,
                    "retrieval_worked_in_practice": False,
                    "probe_status": "blocked",
                    "error_class": "ProvenanceGateError",
                    "error_message": reason,
                }
            )
        report = {
            "report_version": PROVIDER_PROBE_VERSION,
            "generated_at_utc": _utc_now(),
            "probe_session_id": options.probe_session_id,
            "accepted_run_id": options.accepted_run_id,
            "source_of_truth_verdict": source_of_truth_verdict,
            "canonical_exchange_verdict": repo_provenance["canonical_exchange_verdict"],
            "market_type_verdict": repo_provenance["market_type_verdict"],
            "symbol_verdict": repo_provenance["symbol_verdict"],
            "provider_results": entries,
            "recommended_live_smoke_provider_order": [],
            "errors": [],
        }
        atomic_write_json(report, _probe_report_path(options.project_root, options.probe_session_id))
        return report

    approved_order: list[str] = []
    for provider in providers:
        result = {
            "provider": provider,
            "market_type_tested": market_type_value,
            "symbol_normalization_used": symbol_value,
            "accessibility_success": False,
            "retrieval_worked_in_practice": False,
            "probe_status": "failed",
            "supported_timeframes": {},
            "error_class": None,
            "error_message": None,
        }
        client = None
        try:
            client = _build_exchange_client(provider, str(market_type_value))
            client.load_markets()
            for timeframe in REQUIRED_TIMEFRAMES:
                result["supported_timeframes"][timeframe] = bool(getattr(client, "timeframes", {}) and timeframe in client.timeframes)
                _retry_fetch_ohlcv(
                    client,
                    symbol=str(symbol_value),
                    timeframe=timeframe,
                    max_retries=options.max_retries,
                    retry_backoff_seconds=options.retry_backoff_seconds,
                )
            result["accessibility_success"] = True
            result["retrieval_worked_in_practice"] = True
            result["probe_status"] = "success"
            approved_order.append(provider)
        except Exception as exc:  # noqa: BLE001
            result["error_class"] = exc.__class__.__name__
            result["error_message"] = str(exc)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    LOGGER.info("Provider probe close failed | provider=%s", provider)
        entries.append(result)

    report = {
        "report_version": PROVIDER_PROBE_VERSION,
        "generated_at_utc": _utc_now(),
        "probe_session_id": options.probe_session_id,
        "accepted_run_id": options.accepted_run_id,
        "source_of_truth_verdict": source_of_truth_verdict,
        "canonical_exchange_verdict": repo_provenance["canonical_exchange_verdict"],
        "market_type_verdict": repo_provenance["market_type_verdict"],
        "symbol_verdict": repo_provenance["symbol_verdict"],
        "provider_results": entries,
        "recommended_live_smoke_provider_order": approved_order,
        "errors": [],
    }
    atomic_write_json(report, _probe_report_path(options.project_root, options.probe_session_id))
    return report
