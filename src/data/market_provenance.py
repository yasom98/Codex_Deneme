"""Canonical market-data provenance recovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.io_atomic import atomic_write_json
from core.logging import get_logger
from data.tail_refresh import verify_canonical_raw_lineage

LOGGER = get_logger(__name__)

PROVENANCE_REPORT_VERSION = "market_data_provenance_recovery.v1"
PROVENANCE_REPORT_NAME = "provenance_recovery_report.json"
VERDICT_PROVEN = "proven"
VERDICT_LIKELY = "high_confidence_likely"
VERDICT_UNRESOLVED = "unresolved"

_YAML_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "exchange": re.compile(r"^\s*(canonical_exchange|canonical_exchange_id|exchange|exchange_id)\s*:\s*(?P<value>[^#\n]+?)\s*$"),
    "market_type": re.compile(r"^\s*(canonical_market_type|market_type)\s*:\s*(?P<value>[^#\n]+?)\s*$"),
    "exchange_symbol": re.compile(
        r"^\s*(canonical_exchange_symbol|exchange_symbol|symbol_normalization)\s*:\s*(?P<value>[^#\n]+?)\s*$"
    ),
}

_EXCHANGE_KEYS: tuple[str, ...] = ("canonical_exchange", "canonical_exchange_id", "exchange", "exchange_id")
_MARKET_TYPE_KEYS: tuple[str, ...] = ("canonical_market_type", "market_type")
_EXCHANGE_SYMBOL_KEYS: tuple[str, ...] = ("canonical_exchange_symbol", "exchange_symbol", "symbol_normalization")

_PROVIDER_CATALOG: tuple[dict[str, str], ...] = (
    {
        "provider": "gateio",
        "provider_market_family": "spot",
        "market_type_candidate": "spot",
        "exchange_symbol_candidate": "BTC_USDT",
        "request_surface": "GET /spot/candlesticks currency_pair=BTC_USDT",
        "doc_url": "https://www.gate.com/docs/developers/apiv4/en",
    },
    {
        "provider": "gateio",
        "provider_market_family": "derivatives",
        "market_type_candidate": "unresolved",
        "exchange_symbol_candidate": "BTC_USDT",
        "request_surface": "GET /futures/{settle}/candlesticks contract=BTC_USDT",
        "doc_url": "https://www.gate.com/docs/developers/apiv4/en",
    },
    {
        "provider": "okx",
        "provider_market_family": "spot",
        "market_type_candidate": "spot",
        "exchange_symbol_candidate": "BTC-USDT",
        "request_surface": "GET /api/v5/market/history-candles instId=BTC-USDT",
        "doc_url": "https://www.okx.com/docs-v5/en",
    },
    {
        "provider": "okx",
        "provider_market_family": "swap",
        "market_type_candidate": "swap",
        "exchange_symbol_candidate": "BTC-USDT-SWAP",
        "request_surface": "GET /api/v5/public/instruments instType=SWAP instId=BTC-USDT-SWAP",
        "doc_url": "https://www.okx.com/docs-v5/en",
    },
    {
        "provider": "okx",
        "provider_market_family": "future",
        "market_type_candidate": "future",
        "exchange_symbol_candidate": "BTC-USDT-<expiry>",
        "request_surface": "GET /api/v5/public/instruments instType=FUTURES instId=BTC-USDT-<expiry>",
        "doc_url": "https://www.okx.com/docs-v5/en",
    },
    {
        "provider": "bybit",
        "provider_market_family": "spot",
        "market_type_candidate": "spot",
        "exchange_symbol_candidate": "BTCUSDT",
        "request_surface": "GET /v5/market/kline category=spot symbol=BTCUSDT",
        "doc_url": "https://bybit-exchange.github.io/docs/v5/market/kline",
    },
    {
        "provider": "bybit",
        "provider_market_family": "linear",
        "market_type_candidate": "swap",
        "exchange_symbol_candidate": "BTCUSDT",
        "request_surface": "GET /v5/market/kline category=linear symbol=BTCUSDT",
        "doc_url": "https://bybit-exchange.github.io/docs/v5/market/kline",
    },
)


@dataclass(frozen=True)
class MarketProvenanceOptions:
    """Runtime options for provenance recovery."""

    project_root: Path
    accepted_run_id: str
    probe_session_id: str
    exchange_hint: str | None = None
    market_type_hint: str | None = None
    symbol_hint: str | None = None


def compare_forensic_candle_slices(
    canonical_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    abs_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare two tiny OHLCV slices for narrow forensic matching."""

    if abs_tolerance < 0.0:
        raise ValueError("abs_tolerance must be non-negative.")

    mismatches: list[dict[str, Any]] = []
    fields = ("open", "high", "low", "close", "volume")
    max_len = max(len(canonical_rows), len(candidate_rows))
    for index in range(max_len):
        canonical_row = canonical_rows[index] if index < len(canonical_rows) else None
        candidate_row = candidate_rows[index] if index < len(candidate_rows) else None
        if canonical_row is None or candidate_row is None:
            mismatches.append(
                {
                    "row_index": index,
                    "reason": "ROW_COUNT_MISMATCH",
                    "canonical_present": canonical_row is not None,
                    "candidate_present": candidate_row is not None,
                }
            )
            continue

        timestamp_lhs = str(canonical_row.get("timestamp"))
        timestamp_rhs = str(candidate_row.get("timestamp"))
        row_mismatch: dict[str, Any] = {"row_index": index, "timestamp": timestamp_lhs, "fields": {}}
        if timestamp_lhs != timestamp_rhs:
            row_mismatch["fields"]["timestamp"] = {"canonical": timestamp_lhs, "candidate": timestamp_rhs}

        for field in fields:
            lhs = canonical_row.get(field)
            rhs = candidate_row.get(field)
            try:
                lhs_value = float(lhs)
                rhs_value = float(rhs)
            except (TypeError, ValueError):
                if lhs != rhs:
                    row_mismatch["fields"][field] = {"canonical": lhs, "candidate": rhs}
                continue
            if abs(lhs_value - rhs_value) > abs_tolerance:
                row_mismatch["fields"][field] = {"canonical": lhs_value, "candidate": rhs_value}

        if row_mismatch["fields"]:
            mismatches.append(row_mismatch)

    return {
        "status": "match" if not mismatches else "mismatch",
        "rows_compared": min(len(canonical_rows), len(candidate_rows)),
        "abs_tolerance": float(abs_tolerance),
        "mismatches": mismatches,
    }


def provenance_report_path(project_root: Path, probe_session_id: str) -> Path:
    """Return the canonical provenance report path for a probe session."""

    return project_root / "runs" / probe_session_id / "data_tail_refresh" / "reports" / PROVENANCE_REPORT_NAME


def recover_market_data_provenance(options: MarketProvenanceOptions) -> dict[str, Any]:
    """Recover canonical raw market-data provenance and write a machine-readable report."""

    verification = verify_canonical_raw_lineage(options.project_root, options.accepted_run_id)
    repo_evidence = _build_repo_evidence(options.project_root, verification)
    accepted_run_evidence = _build_accepted_run_evidence(options.project_root, options.accepted_run_id)
    explicit_values = _collect_explicit_values(repo_evidence, accepted_run_evidence)

    canonical_exchange_verdict = _resolve_explicit_field_verdict(
        explicit_values["exchange_values"],
        explicit_values["exchange_evidence"],
        field_name="canonical_exchange",
    )
    market_type_verdict = _resolve_explicit_field_verdict(
        explicit_values["market_type_values"],
        explicit_values["market_type_evidence"],
        field_name="market_type",
    )
    symbol_normalization_verdict = _resolve_symbol_verdict(
        explicit_values["exchange_symbol_values"],
        explicit_values["exchange_symbol_evidence"],
        root_symbol_id=repo_evidence.get("root_symbol_id"),
    )

    provider_comparison_results = _build_provider_comparison_results(
        exchange_hint=options.exchange_hint,
        market_type_hint=options.market_type_hint,
        symbol_hint=options.symbol_hint,
    )
    overall_verdict = _overall_verdict(
        verification_status=str(verification.get("status")),
        canonical_exchange_verdict=canonical_exchange_verdict,
        market_type_verdict=market_type_verdict,
        symbol_normalization_verdict=symbol_normalization_verdict,
    )

    report = {
        "report_version": PROVENANCE_REPORT_VERSION,
        "generated_at_utc": _utc_now(),
        "probe_session_id": options.probe_session_id,
        "accepted_run_id": options.accepted_run_id,
        "source_of_truth_verdict": {
            "status": VERDICT_PROVEN if verification.get("status") == "success" else VERDICT_UNRESOLVED,
            "source_of_truth_status": verification.get("source_of_truth_status"),
            "reason": _source_of_truth_reason(verification),
        },
        "repo_evidence": repo_evidence,
        "accepted_run_evidence": accepted_run_evidence,
        "provider_comparison_results": provider_comparison_results,
        "canonical_exchange_verdict": canonical_exchange_verdict,
        "market_type_verdict": market_type_verdict,
        "symbol_normalization_verdict": symbol_normalization_verdict,
        "overall_verdict": overall_verdict,
        "live_refresh_gate": {
            "status": "approved" if overall_verdict == VERDICT_PROVEN else "blocked",
            "reason": (
                "Live refresh may only proceed when provenance verdict is exactly proven."
                if overall_verdict == VERDICT_PROVEN
                else "Live refresh remains blocked because provenance verdict is not proven."
            ),
        },
        "approved_live_refresh_command": None,
        "operator_hints": {
            "exchange": _clean_hint(options.exchange_hint),
            "market_type": _clean_hint(options.market_type_hint),
            "symbol": _clean_hint(options.symbol_hint),
        },
        "errors": _report_errors(verification),
    }
    atomic_write_json(report, provenance_report_path(options.project_root, options.probe_session_id))
    return report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_hint(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _source_of_truth_reason(verification: Mapping[str, Any]) -> str:
    if verification.get("status") == "success":
        return "Canonical root CSV lineage matches the accepted-run standardization artefacts."
    return "Canonical root CSV lineage could not be proven from the accepted-run artefacts."


def _report_errors(verification: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors = verification.get("errors", [])
    if not isinstance(errors, list):
        return []
    return [item for item in errors if isinstance(item, dict)]


def _build_repo_evidence(project_root: Path, verification: Mapping[str, Any]) -> dict[str, Any]:
    sources = verification.get("sources", [])
    root_symbol_ids = {
        str(item.get("symbol_id"))
        for item in sources
        if isinstance(item, Mapping) and isinstance(item.get("symbol_id"), str) and item.get("symbol_id")
    }
    config_matches = _scan_repo_config_evidence(project_root)
    return {
        "config_path": verification.get("config_path"),
        "input_root": verification.get("input_root"),
        "discovered_csv_files": verification.get("discovered_csv_files", []),
        "canonical_raw_sources": sources if isinstance(sources, list) else [],
        "root_symbol_id": sorted(root_symbol_ids)[0] if len(root_symbol_ids) == 1 else None,
        "explicit_provenance_matches": config_matches,
    }


def _build_accepted_run_evidence(project_root: Path, accepted_run_id: str) -> dict[str, Any]:
    accepted_run_root = project_root / "runs" / accepted_run_id
    scan_roots = (
        accepted_run_root / "data_standardized" / "reports",
        accepted_run_root / "data_features" / "reports",
        accepted_run_root / "data_datasets" / "reports",
        accepted_run_root / "data_states" / "reports",
    )
    json_paths: list[Path] = []
    for root in scan_roots:
        if root.exists():
            json_paths.extend(path for path in root.rglob("*.json") if path.is_file())
    json_paths = sorted(set(json_paths))
    explicit_matches: list[dict[str, Any]] = []
    for path in json_paths:
        explicit_matches.extend(_scan_json_file_for_provenance(path, source_type="accepted_run_json"))
    return {
        "accepted_run_root": str(accepted_run_root.resolve()),
        "scan_roots": [str(path.resolve()) for path in scan_roots if path.exists()],
        "json_files_scanned": [str(path.resolve()) for path in json_paths],
        "explicit_provenance_matches": explicit_matches,
    }


def _scan_repo_config_evidence(project_root: Path) -> list[dict[str, Any]]:
    config_root = project_root / "configs"
    if not config_root.exists():
        return []

    matches: list[dict[str, Any]] = []
    for path in sorted(config_root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            matches.extend(_scan_json_file_for_provenance(path, source_type="repo_config_json"))
            continue
        if suffix not in (".yaml", ".yml"):
            continue
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            LOGGER.info("Repo config provenance scan skipped unreadable file | path=%s error=%s", path, exc)
            continue
        file_matches: dict[str, str] = {}
        for field_name, pattern in _YAML_KEY_PATTERNS.items():
            for line in raw_text.splitlines():
                match = pattern.match(line)
                if match is not None:
                    file_matches[field_name] = _strip_quotes(match.group("value").strip())
                    break
        if file_matches:
            matches.append({"path": str(path.resolve()), "source_type": "repo_config_yaml", **file_matches})
    return matches


def _scan_json_file_for_provenance(path: Path, *, source_type: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if payload is None:
        return []

    matches: list[dict[str, Any]] = []
    for pointer, obj in _iter_dict_nodes(payload):
        entry: dict[str, Any] = {}
        for key in _EXCHANGE_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                entry["exchange"] = value.strip()
                break
        for key in _MARKET_TYPE_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                entry["market_type"] = value.strip()
                break
        for key in _EXCHANGE_SYMBOL_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                entry["exchange_symbol"] = value.strip()
                break
        if entry:
            entry["path"] = str(path.resolve())
            entry["json_pointer"] = pointer
            entry["source_type"] = source_type
            matches.append(entry)
    return matches


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_dict_nodes(value: Any, *, pointer: str = "$") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        yield pointer, value
        for key, nested_value in value.items():
            yield from _iter_dict_nodes(nested_value, pointer=f"{pointer}.{key}")
    elif isinstance(value, list):
        for idx, nested_value in enumerate(value):
            yield from _iter_dict_nodes(nested_value, pointer=f"{pointer}[{idx}]")


def _collect_explicit_values(repo_evidence: Mapping[str, Any], accepted_run_evidence: Mapping[str, Any]) -> dict[str, Any]:
    exchange_values: set[str] = set()
    market_type_values: set[str] = set()
    exchange_symbol_values: set[str] = set()
    exchange_evidence: list[dict[str, Any]] = []
    market_type_evidence: list[dict[str, Any]] = []
    exchange_symbol_evidence: list[dict[str, Any]] = []

    for entry in _all_explicit_matches(repo_evidence, accepted_run_evidence):
        if not isinstance(entry, Mapping):
            continue
        exchange_value = _normalize_exchange_value(entry.get("exchange"))
        if exchange_value is not None:
            exchange_values.add(exchange_value)
            exchange_evidence.append(dict(entry))

        market_type_value = _normalize_market_type_value(entry.get("market_type"))
        if market_type_value is not None:
            market_type_values.add(market_type_value)
            market_type_evidence.append(dict(entry))

        symbol_value = _normalize_exchange_symbol_value(entry.get("exchange_symbol"))
        if symbol_value is not None:
            exchange_symbol_values.add(symbol_value)
            exchange_symbol_evidence.append(dict(entry))

    return {
        "exchange_values": exchange_values,
        "market_type_values": market_type_values,
        "exchange_symbol_values": exchange_symbol_values,
        "exchange_evidence": exchange_evidence,
        "market_type_evidence": market_type_evidence,
        "exchange_symbol_evidence": exchange_symbol_evidence,
    }


def _all_explicit_matches(repo_evidence: Mapping[str, Any], accepted_run_evidence: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for container in (repo_evidence, accepted_run_evidence):
        matches = container.get("explicit_provenance_matches", [])
        if isinstance(matches, list):
            for entry in matches:
                if isinstance(entry, Mapping):
                    yield entry


def _normalize_exchange_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _normalize_market_type_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    aliases = {"futures": "future", "perpetual": "swap"}
    normalized = aliases.get(normalized, normalized)
    if normalized in {"spot", "swap", "future"}:
        return normalized
    return None


def _normalize_exchange_symbol_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_explicit_field_verdict(
    values: set[str],
    evidence: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
) -> dict[str, Any]:
    if len(values) == 1:
        value = next(iter(values))
        return {
            "status": VERDICT_PROVEN,
            "value": value,
            "reason": f"Explicit {field_name} provenance was found in repo or accepted-run artefacts.",
            "evidence": [dict(item) for item in evidence],
        }
    if len(values) > 1:
        return {
            "status": VERDICT_UNRESOLVED,
            "value": None,
            "reason": f"Conflicting explicit {field_name} provenance values were found.",
            "evidence": [dict(item) for item in evidence],
        }
    return {
        "status": VERDICT_UNRESOLVED,
        "value": None,
        "reason": f"No explicit {field_name} provenance field was found in repo or accepted-run artefacts.",
        "evidence": [],
    }


def _resolve_symbol_verdict(
    values: set[str],
    evidence: Sequence[Mapping[str, Any]],
    *,
    root_symbol_id: Any,
) -> dict[str, Any]:
    verdict = _resolve_explicit_field_verdict(values, evidence, field_name="exchange_symbol")
    verdict["root_symbol_id"] = root_symbol_id if isinstance(root_symbol_id, str) and root_symbol_id else None
    if verdict["status"] == VERDICT_UNRESOLVED and verdict["root_symbol_id"] is not None:
        verdict["reason"] = (
            "No explicit exchange symbol provenance field was found; root canonical symbol id is known but exchange normalization remains unresolved."
        )
    return verdict


def _build_provider_comparison_results(
    *,
    exchange_hint: str | None,
    market_type_hint: str | None,
    symbol_hint: str | None,
) -> list[dict[str, Any]]:
    normalized_exchange_hint = _normalize_exchange_value(exchange_hint)
    normalized_market_type_hint = _normalize_market_type_value(market_type_hint)
    normalized_symbol_hint = _normalize_exchange_symbol_value(symbol_hint)

    results: list[dict[str, Any]] = []
    for entry in _PROVIDER_CATALOG:
        provider = entry["provider"]
        market_type_candidate = entry["market_type_candidate"]
        exchange_symbol_candidate = entry["exchange_symbol_candidate"]
        results.append(
            {
                **entry,
                "comparison_status": "not_run",
                "forensic_support_level": VERDICT_UNRESOLVED,
                "reason": "No explicit tiny forensic candle slice was supplied for provider matching.",
                "operator_hint_match": bool(
                    (normalized_exchange_hint is None or normalized_exchange_hint == provider)
                    and (normalized_market_type_hint is None or normalized_market_type_hint == market_type_candidate)
                    and (normalized_symbol_hint is None or normalized_symbol_hint == exchange_symbol_candidate)
                ),
            }
        )
    return results


def _overall_verdict(
    *,
    verification_status: str,
    canonical_exchange_verdict: Mapping[str, Any],
    market_type_verdict: Mapping[str, Any],
    symbol_normalization_verdict: Mapping[str, Any],
) -> str:
    if verification_status != "success":
        return VERDICT_UNRESOLVED
    required_statuses = (
        canonical_exchange_verdict.get("status"),
        market_type_verdict.get("status"),
        symbol_normalization_verdict.get("status"),
    )
    if all(status == VERDICT_PROVEN for status in required_statuses):
        return VERDICT_PROVEN
    if all(status in {VERDICT_PROVEN, VERDICT_LIKELY} for status in required_statuses) and any(
        status == VERDICT_LIKELY for status in required_statuses
    ):
        return VERDICT_LIKELY
    return VERDICT_UNRESOLVED


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value
