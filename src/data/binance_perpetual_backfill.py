"""Separate Binance perpetual historical backfill helpers."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import json
from math import ceil
import os
from pathlib import Path
import socket
import ssl
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from core.io_atomic import atomic_write_json, atomic_write_text
from core.logging import get_logger
from data.gateio_perpetual_lineage import (
    DOWNLOAD_REPORT_NAME,
    PROCESSING_REPORT_NAME,
    PROVIDER_STRATEGY_REPORT_NAME,
    GateioPerpetualLineageOptions,
    LegacyReferencePoint,
    SEPARATE_SOURCE_LINEAGE,
    _base_download_report,
    _ensure_track_session_is_clean,
    _legacy_reference_files,
    _raw_input_root,
    _reference_point_payload,
    _render_raw_frame,
    _report_root,
    _run_processing_chain,
    _write_json_copies,
    _utc_now,
    build_provider_strategy_report,
    extract_legacy_reference_points,
)
from data.tail_refresh import (
    TIMEFRAME_SECONDS,
    _exchange_limit,
    _format_ts,
    _last_closed_candle_start,
    _retry_fetch_ohlcv,
    _to_canonical_ohlcv,
    _validate_timeframe_alignment,
)

LOGGER = get_logger(__name__)

try:
    import ccxt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in runtime smoke paths only
    ccxt = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent
    def tqdm(iterable: object | None = None, **_: object) -> object:  # type: ignore[no-redef]
        return iterable if iterable is not None else _NullTqdm()


class _NullTqdm:
    """Minimal tqdm-compatible fallback used when tqdm is unavailable."""

    def __enter__(self) -> _NullTqdm:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        del exc_type, exc, tb
        return False

    def update(self, _: int = 0) -> None:
        return None

    def set_postfix(self, *_: object, **__: object) -> None:
        return None

BINANCE_BACKFILL_MODE = "separate_binance_perpetual_backfill"
BINANCE_BACKFILL_VERSION = "binance_perpetual_parallel_lineage.v1"
BINANCE_TRACK = "historical_backfill"
BINANCE_TRACK_REPORT_NAME = "historical_backfill_report.json"
BINANCE_CHECKPOINT_REPORT_NAME = "historical_backfill_checkpoint.json"

BINANCE_EXCHANGE_USED = "binance"
BINANCE_BOOTSTRAP_CCXT_EXCHANGE_ID = "binanceusdm"
BINANCE_MARKET_TYPE = "perpetual"
BINANCE_CCXT_MARKET_TYPE = "swap"
BINANCE_EXCHANGE_SYMBOL = "BTCUSDT"
BINANCE_CCXT_SYMBOL = "BTC/USDT:USDT"
BINANCE_CONTRACT_TYPE = "PERPETUAL"
BINANCE_MAX_FETCH_LIMIT = 1000
BINANCE_PROVIDER_CHOICE_REASON = (
    "Binance USD-M perpetual klines support documented forward pagination and are suitable for deep historical backfill."
)
BINANCE_METADATA_SURFACE_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_KLINE_SURFACE_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_TESTNET_METADATA_SURFACE_URL = "https://demo-fapi.binance.com/fapi/v1/exchangeInfo"
NETWORK_PROBE_TIMEOUT_SECONDS = 8.0


@dataclass(frozen=True)
class BinancePerpetualBackfillOptions:
    """Runtime options for the separate Binance perpetual historical backfill."""

    project_root: Path
    accepted_run_id: str
    refresh_session_id: str
    legacy_input_root: Path
    data_config_path: Path
    features_config_path: Path
    request_limit: int
    max_retries: int
    retry_backoff_seconds: float
    python_executable: Path
    max_candles_per_timeframe: int = 0
    target_end_utc: str | None = None
    log_level: str = "INFO"


def _build_binance_usdm_client() -> Any:
    """Instantiate a Binance USD-M futures client with a futures-only market surface."""

    if ccxt is None:
        raise RuntimeError("ccxt is not installed; install it into the active virtual environment before Binance backfill.")

    exchange_ctor = getattr(ccxt, BINANCE_BOOTSTRAP_CCXT_EXCHANGE_ID, None)
    if exchange_ctor is None:
        raise ValueError(f"Unsupported CCXT exchange id: {BINANCE_BOOTSTRAP_CCXT_EXCHANGE_ID}")

    client = exchange_ctor(
        {
            "enableRateLimit": True,
            "options": {
                "defaultType": BINANCE_CCXT_MARKET_TYPE,
                "defaultSubType": "linear",
                "fetchMarkets": {"types": ["linear"]},
            },
        }
    )
    if not bool(getattr(client, "has", {}).get("fetchOHLCV", False)):
        raise ValueError(f"Exchange does not expose fetchOHLCV: {BINANCE_BOOTSTRAP_CCXT_EXCHANGE_ID}")
    return client


def _bootstrap_proof_template() -> dict[str, Any]:
    """Build the static portion of the Binance USD-M bootstrap proof payload."""

    return {
        "status": "not_started",
        "provider_surface": "binance_usdm_futures",
        "ccxt_exchange_id_expected": BINANCE_BOOTSTRAP_CCXT_EXCHANGE_ID,
        "market_type_expected": BINANCE_MARKET_TYPE,
        "ccxt_market_type_expected": BINANCE_CCXT_MARKET_TYPE,
        "metadata_surface_url_expected": BINANCE_METADATA_SURFACE_URL,
        "kline_surface_url_expected": BINANCE_KLINE_SURFACE_URL,
        "symbol_expected": {
            "exchange_symbol": BINANCE_EXCHANGE_SYMBOL,
            "ccxt_symbol": BINANCE_CCXT_SYMBOL,
            "contract_type": BINANCE_CONTRACT_TYPE,
        },
        "resolved": None,
        "error": None,
    }


def _network_probe_template(url: str) -> dict[str, Any]:
    """Build a machine-readable network probe template."""

    return {
        "url": url,
        "result": "not_run",
        "http_status": None,
        "error_class": None,
        "error_message": None,
    }


def _network_probe_suite_template() -> dict[str, Any]:
    """Build the default Binance network-probe suite payload."""

    return {
        "prod": _network_probe_template(BINANCE_METADATA_SURFACE_URL),
        "testnet": _network_probe_template(BINANCE_TESTNET_METADATA_SURFACE_URL),
    }


def _probe_network_endpoint(url: str, *, timeout_seconds: float = NETWORK_PROBE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Probe a public Binance futures metadata endpoint and classify failures."""

    payload = _network_probe_template(url)
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "Codex/market-data-bootstrap"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", response.getcode()))
            body = response.read()
    except HTTPError as exc:
        status = int(exc.code)
        body = exc.read()
        payload["result"] = "failed"
        payload["http_status"] = status
        body_text = body.decode("utf-8", errors="replace")[:2000]
        if status in {403, 418, 451} or any(token in body_text.lower() for token in ("forbidden", "access denied", "captcha", "waf")):
            payload["error_class"] = "waf_or_forbidden"
        elif 400 <= status <= 499:
            payload["error_class"] = "http_4xx"
        elif 500 <= status <= 599:
            payload["error_class"] = "http_5xx"
        else:
            payload["error_class"] = "unknown_network_failure"
        payload["error_message"] = f"HTTP {status}"
        return payload
    except ssl.SSLError as exc:
        payload["result"] = "failed"
        payload["error_class"] = "tls_failure"
        payload["error_message"] = str(exc)
        return payload
    except TimeoutError as exc:
        payload["result"] = "failed"
        payload["error_class"] = "timeout"
        payload["error_message"] = str(exc)
        return payload
    except URLError as exc:
        payload["result"] = "failed"
        error_class = _classify_url_error_reason(exc.reason)
        payload["error_class"] = error_class
        payload["error_message"] = str(exc.reason)
        return payload
    except Exception as exc:  # noqa: BLE001
        payload["result"] = "failed"
        payload["error_class"] = "unknown_network_failure"
        payload["error_message"] = str(exc)
        return payload

    payload["http_status"] = status
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        payload["result"] = "failed"
        payload["error_class"] = "malformed_response"
        payload["error_message"] = str(exc)
        return payload

    if not isinstance(decoded, dict) or not isinstance(decoded.get("symbols"), list):
        payload["result"] = "failed"
        payload["error_class"] = "malformed_response"
        payload["error_message"] = "Expected JSON object containing a symbols list."
        return payload

    payload["result"] = "success"
    return payload


def _classify_url_error_reason(reason: object) -> str:
    """Classify low-level URL/network failures into operator-useful buckets."""

    if isinstance(reason, socket.gaierror):
        return "dns_failure"
    if isinstance(reason, ssl.SSLError):
        return "tls_failure"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(reason, (ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return "connect_failure"
    if isinstance(reason, OSError):
        if reason.errno in {
            errno.ECONNREFUSED,
            errno.ECONNRESET,
            errno.EHOSTUNREACH,
            errno.ENETUNREACH,
            errno.ENETDOWN,
            errno.EHOSTDOWN,
        }:
            return "connect_failure"
        if reason.errno in {errno.ETIMEDOUT}:
            return "timeout"
        message = str(reason).lower()
        if "temporary failure in name resolution" in message or "name or service not known" in message:
            return "dns_failure"
        if "certificate verify failed" in message or "ssl" in message:
            return "tls_failure"
        if "timed out" in message:
            return "timeout"
    return "unknown_network_failure"


def _run_binance_network_probe_suite() -> dict[str, Any]:
    """Run narrow public connectivity probes for Binance USD-M futures metadata."""

    return {
        "prod": _probe_network_endpoint(BINANCE_METADATA_SURFACE_URL),
        "testnet": _probe_network_endpoint(BINANCE_TESTNET_METADATA_SURFACE_URL),
    }


def _classify_bootstrap_failure(
    bootstrap_proof: Mapping[str, Any],
    network_probe: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Classify bootstrap failure and recommend a narrow operator next step."""

    bootstrap_error = str(bootstrap_proof.get("error") or "")
    prod = network_probe.get("prod", {})
    testnet = network_probe.get("testnet", {})
    prod_class = prod.get("error_class")
    testnet_result = testnet.get("result")

    if bootstrap_proof.get("status") == "proven":
        return None, None
    if isinstance(prod_class, str) and prod_class:
        if prod_class == "waf_or_forbidden" and testnet_result == "success":
            return prod_class, "Prod Binance USD-M futures metadata is blocked while demo/testnet is reachable; inspect geo/IP or WAF restrictions."
        return prod_class, _recommended_action_for_failure_class(prod_class)
    if bootstrap_error:
        lowered = bootstrap_error.lower()
        if "not pinned to binance usd-m futures" in lowered or "unexpected binance usd-m" in lowered:
            return "bootstrap_contract_failure", "Inspect Binance USD-M bootstrap configuration and keep the client pinned to binanceusdm futures only."
        if "resolved binance market" in lowered or "failed to resolve binance usd-m market" in lowered:
            return "bootstrap_resolution_failure", "Inspect Binance USD-M market loading and deterministic symbol resolution for BTC/USDT:USDT."
    return "unknown_network_failure", _recommended_action_for_failure_class("unknown_network_failure")


def _recommended_action_for_failure_class(failure_class: str) -> str:
    """Return a concise operator next action for a classified bootstrap failure."""

    actions = {
        "dns_failure": "Check DNS resolution for fapi.binance.com and demo-fapi.binance.com from the runtime environment.",
        "connect_failure": "Check outbound firewall, proxy, or routing rules to Binance USD-M futures public endpoints.",
        "tls_failure": "Check TLS interception, CA trust, and certificate handling for Binance futures endpoints.",
        "timeout": "Check outbound latency or proxy timeout policy, then retry the public futures metadata probe.",
        "http_4xx": "Inspect public endpoint policy/region restrictions and review the returned 4xx response.",
        "http_5xx": "Provider-side futures metadata error; retry later and monitor Binance status.",
        "waf_or_forbidden": "Check geo/IP/WAF restrictions for Binance USD-M futures public access.",
        "malformed_response": "Inspect the public futures metadata response for proxy rewriting or unexpected content.",
        "unknown_network_failure": "Capture lower-level DNS/TCP/TLS diagnostics from the runtime environment and retry the futures metadata probe.",
        "bootstrap_contract_failure": "Inspect Binance USD-M bootstrap configuration and verify the client remains futures-only.",
        "bootstrap_resolution_failure": "Inspect deterministic BTC/USDT:USDT perpetual market resolution after futures metadata loads.",
    }
    return actions.get(failure_class, actions["unknown_network_failure"])


def _expected_rows_for_range(start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp, timeframe: str) -> int:
    """Return the inclusive expected candle count for a timeframe range."""

    return int(((end_timestamp - start_timestamp).total_seconds() / TIMEFRAME_SECONDS[timeframe]) + 1)


def _expected_pages_for_range(start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp, timeframe: str, limit: int) -> int:
    """Return the expected page count for a timeframe range at a fixed request limit."""

    return int(ceil(_expected_rows_for_range(start_timestamp, end_timestamp, timeframe) / int(limit)))


def _resolve_target_end_timestamp(
    *,
    timeframe: str,
    last_closed_ts: pd.Timestamp,
    target_end_utc: str | None,
) -> pd.Timestamp:
    """Resolve the target end timestamp for a timeframe, honoring an explicit UTC cutoff when provided."""

    if target_end_utc is None:
        return last_closed_ts

    resolved = pd.Timestamp(target_end_utc)
    if resolved.tzinfo is None:
        raise ValueError("target_end_utc must be timezone-aware and explicitly UTC.")
    resolved = resolved.tz_convert("UTC")

    timeframe_seconds = TIMEFRAME_SECONDS[timeframe]
    if int(resolved.timestamp()) % timeframe_seconds != 0 or resolved.nanosecond != 0:
        raise ValueError(f"target_end_utc is not aligned to the {timeframe} boundary: {_format_ts(resolved)}")
    if resolved > last_closed_ts:
        raise ValueError(
            f"target_end_utc exceeds the last fully closed Binance candle for {timeframe}: "
            f"target_end={_format_ts(resolved)} last_closed={_format_ts(last_closed_ts)}"
        )
    return resolved


def _prove_binance_usdm_bootstrap(client: Any) -> dict[str, Any]:
    """Load Binance USD-M markets and prove deterministic perpetual contract resolution."""

    proof = _bootstrap_proof_template()
    bootstrap_surface = str(getattr(client, "id", ""))
    metadata_surface_url = str(getattr(client, "urls", {}).get("api", {}).get("fapiPublic", ""))
    options = getattr(client, "options", {})
    fetch_markets = options.get("fetchMarkets", {})
    fetch_market_types = list(fetch_markets.get("types", [])) if isinstance(fetch_markets, dict) else []
    proof["resolved"] = {
        "ccxt_exchange_id": bootstrap_surface,
        "metadata_surface_url": metadata_surface_url,
        "client_default_type": options.get("defaultType"),
        "client_default_subtype": options.get("defaultSubType"),
        "client_fetch_market_types": fetch_market_types,
        "unified_symbol": None,
        "exchange_symbol": None,
        "market_type": None,
        "contract": None,
        "linear": None,
        "swap": None,
        "future": None,
        "spot": None,
        "base": None,
        "quote": None,
        "settle": None,
    }

    if bootstrap_surface != BINANCE_BOOTSTRAP_CCXT_EXCHANGE_ID:
        raise ValueError(f"Bootstrap client is not pinned to Binance USD-M futures: {bootstrap_surface}")
    if metadata_surface_url != "https://fapi.binance.com/fapi/v1":
        raise ValueError(f"Unexpected Binance USD-M metadata surface: {metadata_surface_url}")
    if options.get("defaultType") != BINANCE_CCXT_MARKET_TYPE:
        raise ValueError(f"Unexpected Binance USD-M defaultType: {options.get('defaultType')}")
    if options.get("defaultSubType") != "linear":
        raise ValueError(f"Unexpected Binance USD-M defaultSubType: {options.get('defaultSubType')}")
    if fetch_market_types != ["linear"]:
        raise ValueError(f"Unexpected Binance USD-M fetchMarkets types: {fetch_market_types}")

    markets = client.load_markets()

    market = markets.get(BINANCE_CCXT_SYMBOL)
    if not isinstance(market, dict):
        raise ValueError(f"Failed to resolve Binance USD-M market: {BINANCE_CCXT_SYMBOL}")

    market_id = str(market.get("id") or "")
    market_symbol = str(market.get("symbol") or "")
    market_type = str(market.get("type") or "")
    market_settle = str(market.get("settle") or "")
    market_quote = str(market.get("quote") or "")
    market_base = str(market.get("base") or "")
    proof["resolved"] = {
        "ccxt_exchange_id": bootstrap_surface,
        "metadata_surface_url": metadata_surface_url,
        "client_default_type": options.get("defaultType"),
        "client_default_subtype": options.get("defaultSubType"),
        "client_fetch_market_types": fetch_market_types,
        "unified_symbol": market_symbol,
        "exchange_symbol": market_id,
        "market_type": market_type,
        "contract": bool(market.get("contract", False)),
        "linear": bool(market.get("linear", False)),
        "swap": bool(market.get("swap", False)),
        "future": bool(market.get("future", False)),
        "spot": bool(market.get("spot", False)),
        "base": market_base,
        "quote": market_quote,
        "settle": market_settle,
    }
    if market_symbol != BINANCE_CCXT_SYMBOL or market_id != BINANCE_EXCHANGE_SYMBOL:
        raise ValueError(
            f"Resolved Binance perpetual market mismatch: symbol={market_symbol} exchange_symbol={market_id}"
        )
    if market_type != BINANCE_CCXT_MARKET_TYPE:
        raise ValueError(f"Resolved Binance market type is not perpetual/swap: {market_type}")
    if not bool(market.get("contract", False)):
        raise ValueError("Resolved Binance market is not a contract market.")
    if not bool(market.get("linear", False)):
        raise ValueError("Resolved Binance market is not a USD-M linear contract.")
    if not bool(market.get("swap", False)):
        raise ValueError("Resolved Binance market is not a perpetual/swap contract.")
    if bool(market.get("spot", False)):
        raise ValueError("Resolved Binance market incorrectly points to spot.")
    if market_quote.upper() != "USDT" or market_settle.upper() != "USDT" or market_base.upper() != "BTC":
        raise ValueError(
            f"Resolved Binance market normalization mismatch: base={market_base} quote={market_quote} settle={market_settle}"
        )

    proof["status"] = "proven"
    return proof


def run_separate_binance_perpetual_backfill(options: BinancePerpetualBackfillOptions) -> dict[str, Any]:
    """Run the Binance perpetual historical backfill track for the separate lineage."""

    if int(options.max_candles_per_timeframe) < 0:
        raise ValueError("max_candles_per_timeframe must be zero or a positive integer.")

    _ensure_track_session_is_clean(options.project_root, options.refresh_session_id, BINANCE_TRACK)
    legacy_reference_files = _legacy_reference_files(options.legacy_input_root)
    report_root = _report_root(options)
    download_report_path = report_root / DOWNLOAD_REPORT_NAME
    historical_report_path = report_root / BINANCE_TRACK_REPORT_NAME
    processing_report_path = report_root / PROCESSING_REPORT_NAME
    provider_strategy_report_path = report_root / PROVIDER_STRATEGY_REPORT_NAME
    checkpoint_path = report_root / BINANCE_CHECKPOINT_REPORT_NAME

    limit = int(_exchange_limit(BINANCE_EXCHANGE_USED, min(options.request_limit, BINANCE_MAX_FETCH_LIMIT)))
    strategy_payload = build_provider_strategy_report(
        report_version=BINANCE_BACKFILL_VERSION,
        mode=BINANCE_BACKFILL_MODE,
        refresh_session_id=options.refresh_session_id,
        exchange_used=BINANCE_EXCHANGE_USED,
        market_type=BINANCE_MARKET_TYPE,
        provider_track=BINANCE_TRACK,
        provider_choice_reason=BINANCE_PROVIDER_CHOICE_REASON,
        symbol_normalization={
            "exchange_symbol": BINANCE_EXCHANGE_SYMBOL,
            "ccxt_symbol": BINANCE_CCXT_SYMBOL,
            "contract_type": BINANCE_CONTRACT_TYPE,
        },
        legacy_reference_files=legacy_reference_files,
        rate_limit_strategy={
            "ccxt_enable_rate_limit": True,
            "page_size_limit": limit,
            "max_retries": int(options.max_retries),
            "retry_backoff_seconds": float(options.retry_backoff_seconds),
        },
        window_strategy={"kind": "explicit_target_end", "target_end_utc": options.target_end_utc}
        if options.target_end_utc is not None
        else None,
        pagination_strategy={
            "kind": "forward_since_limit_pagination",
            "page_size_limit": limit,
            "checkpoint_resume": True,
            "bounded_candles_per_timeframe": int(options.max_candles_per_timeframe),
        },
    )
    atomic_write_json(strategy_payload, provider_strategy_report_path)

    try:
        reference_points = extract_legacy_reference_points(_legacy_options(options))
    except Exception as exc:  # noqa: BLE001
        payload = _failed_download_payload(
            options,
            legacy_reference_files=legacy_reference_files,
            reference_points=(),
            checkpoint_path=checkpoint_path,
            code="LEGACY_REFERENCE_EXTRACTION_FAILED",
            message=str(exc),
        )
        _write_json_copies(payload, (download_report_path, historical_report_path))
        return {
            "status": "failed",
            "mode": BINANCE_BACKFILL_MODE,
            "provider_track": BINANCE_TRACK,
            "download_report_path": download_report_path,
            "historical_backfill_report_path": historical_report_path,
            "provider_strategy_report_path": provider_strategy_report_path,
            "processing_report_path": None,
            "checkpoint_report_path": checkpoint_path,
        }

    try:
        download_result = _download_historical_backfill(options, reference_points, checkpoint_path=checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        payload = _failed_download_payload(
            options,
            legacy_reference_files=legacy_reference_files,
            reference_points=reference_points,
            checkpoint_path=checkpoint_path,
            code="PERPETUAL_DOWNLOAD_FAILED",
            message=str(exc),
        )
        _write_json_copies(payload, (download_report_path, historical_report_path))
        return {
            "status": "failed",
            "mode": BINANCE_BACKFILL_MODE,
            "provider_track": BINANCE_TRACK,
            "download_report_path": download_report_path,
            "historical_backfill_report_path": historical_report_path,
            "provider_strategy_report_path": provider_strategy_report_path,
            "processing_report_path": None,
            "checkpoint_report_path": checkpoint_path,
        }

    _write_json_copies(download_result["report"], (download_report_path, historical_report_path))
    processing_result = _run_processing_chain(
        _legacy_options(options),
        raw_input_root=download_result["raw_input_root"],
        report_version=BINANCE_BACKFILL_VERSION,
        mode=BINANCE_BACKFILL_MODE,
        exchange_used=BINANCE_EXCHANGE_USED,
        market_type=BINANCE_MARKET_TYPE,
        provider_track=BINANCE_TRACK,
        provider_choice_reason=BINANCE_PROVIDER_CHOICE_REASON,
    )
    atomic_write_json(processing_result["report"], processing_report_path)
    return {
        "status": processing_result["status"],
        "mode": BINANCE_BACKFILL_MODE,
        "provider_track": BINANCE_TRACK,
        "download_report_path": download_report_path,
        "historical_backfill_report_path": historical_report_path,
        "provider_strategy_report_path": provider_strategy_report_path,
        "processing_report_path": processing_report_path,
        "checkpoint_report_path": checkpoint_path,
        "raw_input_root": download_result["raw_input_root"],
    }


def _download_historical_backfill(
    options: BinancePerpetualBackfillOptions,
    reference_points: Sequence[LegacyReferencePoint],
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    now_utc = _utc_now()
    raw_input_root = _raw_input_root(options, provider_track=BINANCE_TRACK)
    output_root = raw_input_root / "binance_perpetual"
    checkpoint_root = raw_input_root / ".checkpoint" / "binance_perpetual"

    target_end_by_timeframe: dict[str, pd.Timestamp] = {}
    for point in reference_points:
        last_closed_ts = _last_closed_candle_start(now_utc, point.timeframe)
        target_end = _resolve_target_end_timestamp(
            timeframe=point.timeframe,
            last_closed_ts=last_closed_ts,
            target_end_utc=options.target_end_utc,
        )
        if point.download_start_timestamp > target_end:
            raise ValueError(
                f"No fully closed Binance perpetual candles available after legacy reference for {point.timeframe}: "
                f"download_start={_format_ts(point.download_start_timestamp)} last_closed={_format_ts(target_end)}"
            )
        if int(options.max_candles_per_timeframe) > 0:
            bounded_end = point.download_start_timestamp + pd.Timedelta(
                seconds=TIMEFRAME_SECONDS[point.timeframe] * (int(options.max_candles_per_timeframe) - 1)
            )
            target_end_by_timeframe[point.timeframe] = min(target_end, bounded_end)
        else:
            target_end_by_timeframe[point.timeframe] = target_end

    checkpoint = _load_or_initialize_checkpoint(options, reference_points, checkpoint_path, target_end_by_timeframe)
    limit = int(_exchange_limit(BINANCE_EXCHANGE_USED, min(options.request_limit, BINANCE_MAX_FETCH_LIMIT)))
    timeframe_reports: list[dict[str, Any]] = []
    current_timeframe: str | None = None

    client = _build_binance_usdm_client()
    try:
        bootstrap_proof = _prove_binance_usdm_bootstrap(client)
        checkpoint["bootstrap_proof"] = bootstrap_proof
        checkpoint["bootstrap_status"] = str(bootstrap_proof["status"])
        checkpoint["generated_at_utc"] = _utc_now().isoformat()
        atomic_write_json(checkpoint, checkpoint_path)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        for point in reference_points:
            current_timeframe = point.timeframe
            target_end = target_end_by_timeframe[point.timeframe]
            final_output_file = output_root / point.legacy_file.name
            timeframe_state = checkpoint["timeframes"][point.timeframe]
            if timeframe_state.get("status") == "completed" and final_output_file.exists():
                timeframe_reports.append(
                    _reference_point_payload(
                        options,
                        point,
                        output_file=final_output_file,
                        rows_downloaded=int(timeframe_state.get("rows_downloaded", 0)),
                        download_end_utc=str(timeframe_state.get("download_end_utc")),
                        exchange_used=BINANCE_EXCHANGE_USED,
                        market_type=BINANCE_MARKET_TYPE,
                        provider_track=BINANCE_TRACK,
                    )
                )
                continue

            _backfill_single_timeframe(
                client,
                options=options,
                point=point,
                target_end=target_end,
                limit=limit,
                final_output_file=final_output_file,
                checkpoint_root=checkpoint_root,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
            )
            refreshed_state = checkpoint["timeframes"][point.timeframe]
            timeframe_reports.append(
                _reference_point_payload(
                    options,
                    point,
                    output_file=final_output_file,
                    rows_downloaded=int(refreshed_state.get("rows_downloaded", 0)),
                    download_end_utc=str(refreshed_state.get("download_end_utc")),
                    exchange_used=BINANCE_EXCHANGE_USED,
                    market_type=BINANCE_MARKET_TYPE,
                    provider_track=BINANCE_TRACK,
                )
            )
    except Exception as exc:  # noqa: BLE001
        checkpoint["status"] = "failed"
        checkpoint["generated_at_utc"] = _utc_now().isoformat()
        if checkpoint.get("bootstrap_status") != "proven":
            checkpoint["bootstrap_status"] = "failed"
            bootstrap_proof = dict(checkpoint.get("bootstrap_proof") or _bootstrap_proof_template())
            bootstrap_proof["status"] = "failed"
            bootstrap_proof["error"] = str(exc)
            checkpoint["bootstrap_proof"] = bootstrap_proof
            network_probe = _run_binance_network_probe_suite()
            checkpoint["network_probe"] = network_probe
            bootstrap_failure_class, recommended_next_action = _classify_bootstrap_failure(bootstrap_proof, network_probe)
            checkpoint["bootstrap_failure_class"] = bootstrap_failure_class
            checkpoint["recommended_next_action"] = recommended_next_action
        if current_timeframe is not None:
            timeframe_state = checkpoint["timeframes"][current_timeframe]
            timeframe_state["status"] = "failed"
            timeframe_state["error"] = str(exc)
        atomic_write_json(checkpoint, checkpoint_path)
        raise
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            LOGGER.info("Binance client close failed")

    checkpoint["status"] = "completed"
    checkpoint["generated_at_utc"] = _utc_now().isoformat()
    atomic_write_json(checkpoint, checkpoint_path)

    report = _base_download_report(
        options,
        legacy_reference_files=_legacy_reference_files(options.legacy_input_root),
        exchange_used=BINANCE_EXCHANGE_USED,
        market_type=BINANCE_MARKET_TYPE,
        provider_track=BINANCE_TRACK,
        provider_choice_reason=BINANCE_PROVIDER_CHOICE_REASON,
        report_version=BINANCE_BACKFILL_VERSION,
        mode=BINANCE_BACKFILL_MODE,
        symbol_normalization={
            "exchange_symbol": BINANCE_EXCHANGE_SYMBOL,
            "ccxt_symbol": BINANCE_CCXT_SYMBOL,
            "contract_type": BINANCE_CONTRACT_TYPE,
        },
        completion_scope="bounded_smoke" if int(options.max_candles_per_timeframe) > 0 else "full_backfill",
        pagination_strategy={
            "kind": "forward_since_limit_pagination",
            "page_size_limit": limit,
            "checkpoint_resume": True,
            "bounded_candles_per_timeframe": int(options.max_candles_per_timeframe),
        },
        window_strategy={"kind": "explicit_target_end", "target_end_utc": options.target_end_utc}
        if options.target_end_utc is not None
        else None,
    )
    report.update(
        {
            "status": "success",
            "raw_input_root": str(raw_input_root.resolve()),
            "checkpoint_report_path": str(checkpoint_path.resolve()),
            "bootstrap_proof": dict(checkpoint["bootstrap_proof"]),
            "bootstrap_failure_class": checkpoint.get("bootstrap_failure_class"),
            "recommended_next_action": checkpoint.get("recommended_next_action"),
            "network_probe": dict(checkpoint.get("network_probe") or _network_probe_suite_template()),
            "timeframes": timeframe_reports,
            "errors": [],
        }
    )
    return {"status": "success", "raw_input_root": raw_input_root, "report": report}


def _backfill_single_timeframe(
    client: Any,
    *,
    options: BinancePerpetualBackfillOptions,
    point: LegacyReferencePoint,
    target_end: pd.Timestamp,
    limit: int,
    final_output_file: Path,
    checkpoint_root: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> None:
    timeframe_state = checkpoint["timeframes"][point.timeframe]
    timeframe_state["status"] = "in_progress"
    timeframe_state["error"] = None
    atomic_write_json(checkpoint, checkpoint_path)

    timeframe_ms = TIMEFRAME_SECONDS[point.timeframe] * 1000
    next_since_utc = str(timeframe_state.get("next_since_utc") or _format_ts(point.download_start_timestamp))
    next_since = pd.Timestamp(next_since_utc)
    target_end_ms = int(target_end.timestamp() * 1000)
    expected_rows = int(timeframe_state.get("expected_rows", _expected_rows_for_range(point.download_start_timestamp, target_end, point.timeframe)))
    expected_pages = int(
        timeframe_state.get("expected_pages", _expected_pages_for_range(point.download_start_timestamp, target_end, point.timeframe, limit))
    )
    initial_rows = min(int(timeframe_state.get("rows_downloaded", 0)), expected_rows)

    LOGGER.info(
        "Binance backfill timeframe start | timeframe=%s start=%s end=%s expected_rows=%d expected_pages=%d",
        point.timeframe,
        _format_ts(point.download_start_timestamp),
        _format_ts(target_end),
        expected_rows,
        expected_pages,
    )

    with tqdm(
        total=expected_rows,
        initial=initial_rows,
        desc=f"Binance {point.timeframe}",
        unit="row",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix({"pages": f"{int(timeframe_state.get('pages_completed', 0))}/{expected_pages}"})
        while int(next_since.timestamp() * 1000) <= target_end_ms:
            page_index = int(timeframe_state.get("pages_completed", 0)) + 1
            batch = _fetch_binance_backfill_page(
                client,
                timeframe=point.timeframe,
                since_timestamp=next_since,
                end_timestamp=target_end,
                request_limit=limit,
                max_retries=options.max_retries,
                retry_backoff_seconds=options.retry_backoff_seconds,
            )
            if batch.empty:
                raise ValueError(f"Binance backfill returned no rows for {point.timeframe} at page {page_index}.")

            alignment_ok, gap_count, alignment_error = _validate_timeframe_alignment(batch, point.timeframe)
            if not alignment_ok:
                raise ValueError(f"Invalid Binance timeframe alignment for {point.timeframe}: {alignment_error} gap_count={gap_count}")

            first_ts = pd.Timestamp(batch["timestamp"].iloc[0])
            if first_ts != next_since:
                raise ValueError(
                    f"Binance backfill page start mismatch for {point.timeframe}: expected={_format_ts(next_since)} observed={_format_ts(first_ts)}"
                )

            chunk_path = checkpoint_root / point.timeframe / f"page_{page_index:06d}.csv"
            atomic_write_text(_render_raw_frame(batch), chunk_path)
            last_ts = pd.Timestamp(batch["timestamp"].iloc[-1])
            next_since = last_ts + pd.Timedelta(seconds=TIMEFRAME_SECONDS[point.timeframe])

            page_payload = {
                "page_index": page_index,
                "chunk_file": str(chunk_path.resolve()),
                "rows_downloaded": int(len(batch)),
                "page_start_utc": _format_ts(first_ts),
                "page_end_utc": _format_ts(last_ts),
            }
            timeframe_state.setdefault("pages", [])
            if len(timeframe_state["pages"]) < page_index:
                timeframe_state["pages"].append(page_payload)
            else:
                timeframe_state["pages"][page_index - 1] = page_payload
            timeframe_state["pages_completed"] = page_index
            timeframe_state["rows_downloaded"] = int(timeframe_state.get("rows_downloaded", 0)) + int(len(batch))
            timeframe_state["next_since_utc"] = _format_ts(next_since)
            timeframe_state["download_end_utc"] = _format_ts(last_ts)
            timeframe_state["status"] = "in_progress"
            atomic_write_json(checkpoint, checkpoint_path)

            progress.update(int(len(batch)))
            progress.set_postfix({"pages": f"{page_index}/{expected_pages}"})

            if last_ts >= target_end:
                break

    if int(timeframe_state.get("rows_downloaded", 0)) != expected_rows:
        raise ValueError(
            f"Partial Binance backfill detected for {point.timeframe}: expected_rows={expected_rows} observed_rows={timeframe_state.get('rows_downloaded', 0)}"
        )
    if str(timeframe_state.get("download_end_utc")) != _format_ts(target_end):
        raise ValueError(
            f"Binance backfill did not reach target end for {point.timeframe}: "
            f"expected={_format_ts(target_end)} observed={timeframe_state.get('download_end_utc')}"
        )

    chunk_files = [Path(item["chunk_file"]) for item in timeframe_state.get("pages", [])]
    _atomic_concatenate_csv_chunks(chunk_files, final_output_file)

    timeframe_state["status"] = "completed"
    timeframe_state["output_file"] = str(final_output_file.resolve())
    timeframe_state["error"] = None
    atomic_write_json(checkpoint, checkpoint_path)
    LOGGER.info(
        "Binance backfill timeframe completed | timeframe=%s rows_downloaded=%d pages_completed=%d output=%s",
        point.timeframe,
        int(timeframe_state.get("rows_downloaded", 0)),
        int(timeframe_state.get("pages_completed", 0)),
        final_output_file,
    )


def _fetch_binance_backfill_page(
    client: Any,
    *,
    timeframe: str,
    since_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    request_limit: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    rows = _retry_fetch_ohlcv(
        client,
        symbol=BINANCE_CCXT_SYMBOL,
        timeframe=timeframe,
        since_ms=int(since_timestamp.timestamp() * 1000),
        limit=request_limit,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    batch = _to_canonical_ohlcv(rows)
    batch = batch.loc[(batch["timestamp"] >= since_timestamp) & (batch["timestamp"] <= end_timestamp)].copy()
    if batch.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    batch = batch.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return batch.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()


def _legacy_options(options: BinancePerpetualBackfillOptions) -> GateioPerpetualLineageOptions:
    return GateioPerpetualLineageOptions(
        project_root=options.project_root,
        accepted_run_id=options.accepted_run_id,
        refresh_session_id=options.refresh_session_id,
        legacy_input_root=options.legacy_input_root,
        data_config_path=options.data_config_path,
        features_config_path=options.features_config_path,
        request_limit=options.request_limit,
        max_retries=options.max_retries,
        retry_backoff_seconds=options.retry_backoff_seconds,
        python_executable=options.python_executable,
        log_level=options.log_level,
    )


def _failed_download_payload(
    options: BinancePerpetualBackfillOptions,
    *,
    legacy_reference_files: Sequence[str],
    reference_points: Sequence[LegacyReferencePoint],
    checkpoint_path: Path,
    code: str,
    message: str,
) -> dict[str, Any]:
    payload = _base_download_report(
        options,
        legacy_reference_files=legacy_reference_files,
        exchange_used=BINANCE_EXCHANGE_USED,
        market_type=BINANCE_MARKET_TYPE,
        provider_track=BINANCE_TRACK,
        provider_choice_reason=BINANCE_PROVIDER_CHOICE_REASON,
        report_version=BINANCE_BACKFILL_VERSION,
        mode=BINANCE_BACKFILL_MODE,
        symbol_normalization={
            "exchange_symbol": BINANCE_EXCHANGE_SYMBOL,
            "ccxt_symbol": BINANCE_CCXT_SYMBOL,
            "contract_type": BINANCE_CONTRACT_TYPE,
        },
        completion_scope="bounded_smoke" if int(options.max_candles_per_timeframe) > 0 else "full_backfill",
        pagination_strategy={
            "kind": "forward_since_limit_pagination",
            "page_size_limit": int(_exchange_limit(BINANCE_EXCHANGE_USED, min(options.request_limit, BINANCE_MAX_FETCH_LIMIT))),
            "checkpoint_resume": True,
            "bounded_candles_per_timeframe": int(options.max_candles_per_timeframe),
        },
        window_strategy={"kind": "explicit_target_end", "target_end_utc": options.target_end_utc}
        if options.target_end_utc is not None
        else None,
    )
    payload.update(
        {
            "status": "failed",
            "checkpoint_report_path": str(checkpoint_path.resolve()),
            "timeframes": [
                _reference_point_payload(
                    options,
                    point,
                    output_file=None,
                    rows_downloaded=0,
                    download_end_utc=None,
                    exchange_used=BINANCE_EXCHANGE_USED,
                    market_type=BINANCE_MARKET_TYPE,
                    provider_track=BINANCE_TRACK,
                )
                for point in reference_points
            ],
            "errors": [{"code": code, "message": message}],
        }
    )
    if checkpoint_path.exists():
        try:
            payload["checkpoint_state"] = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(payload["checkpoint_state"], dict):
                payload["bootstrap_proof"] = dict(payload["checkpoint_state"].get("bootstrap_proof") or _bootstrap_proof_template())
                payload["bootstrap_failure_class"] = payload["checkpoint_state"].get("bootstrap_failure_class")
                payload["recommended_next_action"] = payload["checkpoint_state"].get("recommended_next_action")
                payload["network_probe"] = dict(payload["checkpoint_state"].get("network_probe") or _network_probe_suite_template())
        except (OSError, json.JSONDecodeError):
            payload["checkpoint_state"] = {"status": "unreadable"}
            payload["bootstrap_proof"] = _bootstrap_proof_template()
            payload["bootstrap_failure_class"] = None
            payload["recommended_next_action"] = None
            payload["network_probe"] = _network_probe_suite_template()
    else:
        payload["bootstrap_proof"] = _bootstrap_proof_template()
        payload["bootstrap_failure_class"] = None
        payload["recommended_next_action"] = None
        payload["network_probe"] = _network_probe_suite_template()
    return payload


def _load_or_initialize_checkpoint(
    options: BinancePerpetualBackfillOptions,
    reference_points: Sequence[LegacyReferencePoint],
    checkpoint_path: Path,
    target_end_by_timeframe: Mapping[str, pd.Timestamp],
) -> dict[str, Any]:
    if checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        _validate_checkpoint_payload(payload, options, reference_points, target_end_by_timeframe)
        return payload

    payload: dict[str, Any] = {
        "checkpoint_version": BINANCE_BACKFILL_VERSION,
        "generated_at_utc": _utc_now().isoformat(),
        "status": "pending",
        "mode": BINANCE_BACKFILL_MODE,
        "refresh_session_id": options.refresh_session_id,
        "exchange_used": BINANCE_EXCHANGE_USED,
        "market_type": BINANCE_MARKET_TYPE,
        "source_lineage": SEPARATE_SOURCE_LINEAGE,
        "provider_track": BINANCE_TRACK,
        "provider_choice_reason": BINANCE_PROVIDER_CHOICE_REASON,
        "legacy_lineage_touched": False,
        "symbol_normalization": {
            "exchange_symbol": BINANCE_EXCHANGE_SYMBOL,
            "ccxt_symbol": BINANCE_CCXT_SYMBOL,
            "contract_type": BINANCE_CONTRACT_TYPE,
        },
        "bootstrap_status": "not_started",
        "bootstrap_proof": _bootstrap_proof_template(),
        "bootstrap_failure_class": None,
        "recommended_next_action": None,
        "network_probe": _network_probe_suite_template(),
        "max_candles_per_timeframe": int(options.max_candles_per_timeframe),
        "target_end_utc": str(options.target_end_utc) if options.target_end_utc is not None else None,
        "timeframes": {},
    }
    for point in reference_points:
        expected_rows = _expected_rows_for_range(point.download_start_timestamp, target_end_by_timeframe[point.timeframe], point.timeframe)
        expected_pages = _expected_pages_for_range(
            point.download_start_timestamp,
            target_end_by_timeframe[point.timeframe],
            point.timeframe,
            int(_exchange_limit(BINANCE_EXCHANGE_USED, min(options.request_limit, BINANCE_MAX_FETCH_LIMIT))),
        )
        payload["timeframes"][point.timeframe] = {
            "status": "pending",
            "legacy_reference_file": str(point.legacy_file.resolve()),
            "legacy_last_timestamp_utc": _format_ts(point.legacy_last_timestamp),
            "download_start_utc": _format_ts(point.download_start_timestamp),
            "target_end_utc": _format_ts(target_end_by_timeframe[point.timeframe]),
            "expected_rows": expected_rows,
            "expected_pages": expected_pages,
            "next_since_utc": _format_ts(point.download_start_timestamp),
            "rows_downloaded": 0,
            "pages_completed": 0,
            "output_file": None,
            "download_end_utc": None,
            "pages": [],
            "error": None,
        }
    atomic_write_json(payload, checkpoint_path)
    return payload


def _validate_checkpoint_payload(
    payload: Mapping[str, Any],
    options: BinancePerpetualBackfillOptions,
    reference_points: Sequence[LegacyReferencePoint],
    target_end_by_timeframe: Mapping[str, pd.Timestamp],
) -> None:
    if payload.get("refresh_session_id") != options.refresh_session_id:
        raise ValueError("Checkpoint refresh_session_id does not match the requested historical backfill run.")
    if payload.get("provider_track") != BINANCE_TRACK:
        raise ValueError("Checkpoint provider_track does not match the Binance historical backfill track.")
    if payload.get("exchange_used") != BINANCE_EXCHANGE_USED:
        raise ValueError("Checkpoint exchange_used does not match Binance historical backfill.")
    if bool(payload.get("legacy_lineage_touched", True)):
        raise ValueError("Checkpoint reports an invalid legacy_lineage_touched state.")
    if int(payload.get("max_candles_per_timeframe", 0)) != int(options.max_candles_per_timeframe):
        raise ValueError("Checkpoint max_candles_per_timeframe does not match the requested run.")
    if payload.get("target_end_utc") != (str(options.target_end_utc) if options.target_end_utc is not None else None):
        raise ValueError("Checkpoint target_end_utc does not match the requested run.")
    if payload.get("bootstrap_status") not in {"not_started", "proven", "failed"}:
        raise ValueError("Checkpoint bootstrap_status is invalid.")
    bootstrap_proof = payload.get("bootstrap_proof")
    if not isinstance(bootstrap_proof, dict):
        raise ValueError("Checkpoint bootstrap_proof payload is invalid.")
    if bootstrap_proof.get("provider_surface") != "binance_usdm_futures":
        raise ValueError("Checkpoint bootstrap_proof provider_surface is invalid.")
    expected_symbol = bootstrap_proof.get("symbol_expected")
    if not isinstance(expected_symbol, dict) or expected_symbol.get("ccxt_symbol") != BINANCE_CCXT_SYMBOL:
        raise ValueError("Checkpoint bootstrap_proof symbol expectation is invalid.")
    if payload.get("bootstrap_failure_class") is not None and not isinstance(payload.get("bootstrap_failure_class"), str):
        raise ValueError("Checkpoint bootstrap_failure_class is invalid.")
    if payload.get("recommended_next_action") is not None and not isinstance(payload.get("recommended_next_action"), str):
        raise ValueError("Checkpoint recommended_next_action is invalid.")
    network_probe = payload.get("network_probe")
    if not isinstance(network_probe, dict):
        raise ValueError("Checkpoint network_probe payload is invalid.")
    for name, expected_url in (("prod", BINANCE_METADATA_SURFACE_URL), ("testnet", BINANCE_TESTNET_METADATA_SURFACE_URL)):
        probe = network_probe.get(name)
        if not isinstance(probe, dict):
            raise ValueError(f"Checkpoint network_probe.{name} payload is invalid.")
        if probe.get("url") != expected_url:
            raise ValueError(f"Checkpoint network_probe.{name}.url is invalid.")

    timeframe_states = payload.get("timeframes")
    if not isinstance(timeframe_states, dict):
        raise ValueError("Checkpoint timeframes payload is invalid.")

    for point in reference_points:
        state = timeframe_states.get(point.timeframe)
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint missing timeframe state: {point.timeframe}")
        if state.get("legacy_last_timestamp_utc") != _format_ts(point.legacy_last_timestamp):
            raise ValueError(f"Checkpoint legacy timestamp mismatch for {point.timeframe}")
        if state.get("download_start_utc") != _format_ts(point.download_start_timestamp):
            raise ValueError(f"Checkpoint download start mismatch for {point.timeframe}")
        if state.get("target_end_utc") != _format_ts(target_end_by_timeframe[point.timeframe]):
            raise ValueError(f"Checkpoint target end mismatch for {point.timeframe}")
        expected_rows = _expected_rows_for_range(point.download_start_timestamp, target_end_by_timeframe[point.timeframe], point.timeframe)
        expected_pages = _expected_pages_for_range(
            point.download_start_timestamp,
            target_end_by_timeframe[point.timeframe],
            point.timeframe,
            int(_exchange_limit(BINANCE_EXCHANGE_USED, min(options.request_limit, BINANCE_MAX_FETCH_LIMIT))),
        )
        if int(state.get("expected_rows", 0)) != expected_rows:
            raise ValueError(f"Checkpoint expected_rows mismatch for {point.timeframe}")
        if int(state.get("expected_pages", 0)) != expected_pages:
            raise ValueError(f"Checkpoint expected_pages mismatch for {point.timeframe}")
        for page in state.get("pages", []):
            chunk_file = Path(page["chunk_file"])
            if not chunk_file.exists():
                raise ValueError(f"Checkpoint chunk file is missing: {chunk_file}")


def _atomic_concatenate_csv_chunks(chunk_paths: Sequence[Path], destination: Path) -> None:
    tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            for index, chunk_path in enumerate(chunk_paths):
                with chunk_path.open("r", encoding="utf-8", newline="") as chunk_handle:
                    for line_index, line in enumerate(chunk_handle):
                        if index > 0 and line_index == 0:
                            continue
                        handle.write(line)
        os.replace(tmp_path, destination)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to atomically assemble Binance historical raw file: {destination}") from exc
