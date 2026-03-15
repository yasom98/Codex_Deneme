"""Tail-refresh contract for canonical market-data lineage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import pandas as pd

from core.config import PipelineConfig, load_config, validate_config
from core.health import summarize_reports
from core.io_atomic import atomic_write_json, atomic_write_text
from core.logging import get_logger
from core.paths import discover_csv_files
from data.ingest import _detect_delimiter, read_csv_ohlcv
from data.schema import CANONICAL_COLUMNS, NUMERIC_COLUMNS, detect_timestamp_alias, map_to_canonical, normalize_column_names
from data.standardize import _coerce_numeric_series, standardize_all

LOGGER = get_logger(__name__)

TAIL_REFRESH_VERSION = "tail_refresh.v1"
TIMEFRAME_SECONDS: dict[str, int] = {"1m": 60, "5m": 300, "15m": 900}
SUPPORTED_MARKET_TYPES: tuple[str, ...] = ("spot", "swap", "future")
OVERLAP_COMPARE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
OHLCV_FETCH_LIMITS: dict[str, int] = {"okx": 100, "bybit": 200, "binance": 1000}
REQUIRED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m")
TIMEFRAME_FILE_RE = re.compile(r"(?P<symbol>[A-Z0-9_]+)_(?P<timeframe>1m|5m|15m)_price_data\.csv$")

try:
    import ccxt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in runtime smoke paths only
    ccxt = None


@dataclass(frozen=True)
class RefreshOptions:
    """User-specified runtime options for the tail-refresh flow."""

    project_root: Path
    accepted_run_id: str
    refresh_session_id: str
    data_config_path: Path
    features_config_path: Path
    provider_probe_report_path: Path | None
    exchange: str | None
    market_type: str | None
    symbol: str | None
    fallback_exchanges: tuple[str, ...]
    request_limit: int
    max_retries: int
    retry_backoff_seconds: float
    overlap_abs_tolerance: float
    python_executable: Path
    log_level: str = "INFO"


@dataclass(frozen=True)
class RawFileFormat:
    """Storage-format metadata required to preserve raw CSV conventions."""

    delimiter: str
    timestamp_column: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalRawSource:
    """Canonical raw source file proven by repo configs and accepted-run reports."""

    timeframe: str
    symbol_id: str
    source_csv: Path
    source_report_path: Path
    source_report_payload: dict[str, Any]
    output_parquet_path: Path
    raw_format: RawFileFormat
    row_count: int
    timestamp_min_utc: str
    timestamp_max_utc: str


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def _normalize_market_type(value: str | None) -> str | None:
    """Normalize supported market-type aliases."""

    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "":
        return None
    if normalized == "futures":
        return "future"
    if normalized == "perpetual":
        return "swap"
    if normalized not in SUPPORTED_MARKET_TYPES:
        raise ValueError(f"Unsupported market_type: {value}")
    return normalized


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Failed to read JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON payload: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _format_ts(ts: pd.Timestamp) -> str:
    """Return a stable UTC ISO timestamp string."""

    return ts.tz_convert("UTC").isoformat()


def _format_raw_timestamp(ts: pd.Timestamp) -> str:
    """Render a raw timestamp value without silently changing timezone semantics."""

    return ts.tz_convert("UTC").isoformat(sep=" ")


def _parse_timeframe_from_name(name: str) -> tuple[str, str] | None:
    """Parse symbol id and timeframe from a canonical raw file name."""

    match = TIMEFRAME_FILE_RE.fullmatch(name)
    if match is None:
        return None
    return str(match.group("symbol")), str(match.group("timeframe"))


def _parse_csv_to_canonical(source_csv: Path, timestamp_aliases: Sequence[str]) -> tuple[pd.DataFrame, RawFileFormat]:
    """Load a raw CSV, infer its format and return canonical OHLCV rows."""

    raw_df = read_csv_ohlcv(source_csv)
    if raw_df.empty:
        raise ValueError(f"Raw CSV is empty: {source_csv}")

    detection = detect_timestamp_alias(list(raw_df.columns), timestamp_aliases)
    canonical_df = map_to_canonical(raw_df, timestamp_col=detection.selected_column).copy()
    canonical_df["timestamp"] = pd.to_datetime(canonical_df["timestamp"], utc=True, errors="coerce")
    if canonical_df["timestamp"].isna().any():
        raise ValueError(f"Invalid timestamps detected in {source_csv}")

    for col in NUMERIC_COLUMNS:
        canonical_df[col] = _coerce_numeric_series(canonical_df[col])
    if canonical_df.loc[:, list(NUMERIC_COLUMNS)].isna().any().any():
        raise ValueError(f"Invalid numeric OHLCV values detected in {source_csv}")

    canonical_df = canonical_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if canonical_df["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps detected in {source_csv}")
    if not bool(canonical_df["timestamp"].is_monotonic_increasing):
        raise ValueError(f"Non-monotonic timestamps detected in {source_csv}")

    raw_format = RawFileFormat(
        delimiter=_detect_delimiter(source_csv),
        timestamp_column=detection.selected_column,
        columns=tuple(str(col) for col in raw_df.columns),
    )
    return canonical_df, raw_format


def _validate_timeframe_alignment(frame: pd.DataFrame, timeframe: str, *, allow_single_row: bool = True) -> tuple[bool, int, str | None]:
    """Validate boundary alignment and consecutive timestamp spacing."""

    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if frame.empty:
        return True, 0, None

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        return False, 0, "TIMESTAMP_PARSE_FAILED"

    seconds = TIMEFRAME_SECONDS[timeframe]
    # Normalize to nanosecond storage first so boundary checks stay correct across
    # pandas datetime units such as datetime64[ms, UTC] on Windows host runs.
    epoch_seconds = timestamps.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000_000
    if bool((epoch_seconds % seconds != 0).any()):
        return False, 0, "TIMEFRAME_BOUNDARY_MISALIGNED"

    deltas = timestamps.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return (allow_single_row, 0 if allow_single_row else 1, None if allow_single_row else "SINGLE_ROW_NOT_ALLOWED")

    gap_count = int((deltas != float(seconds)).sum())
    if gap_count > 0:
        return False, gap_count, "TIMEFRAME_GAPS_DETECTED"
    return True, 0, None


def _last_closed_candle_start(now_utc: datetime, timeframe: str) -> pd.Timestamp:
    """Return the latest fully closed candle start timestamp for a timeframe."""

    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    seconds = TIMEFRAME_SECONDS[timeframe]
    epoch_now = int(now_utc.timestamp())
    current_bucket_start = epoch_now - (epoch_now % seconds)
    return pd.Timestamp(current_bucket_start - seconds, unit="s", tz="UTC")


def _empty_canonical_frame() -> pd.DataFrame:
    """Return an empty canonical OHLCV dataframe."""

    frame = pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


def _to_canonical_ohlcv(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Convert CCXT OHLCV rows into the canonical dataframe shape."""

    if not rows:
        return _empty_canonical_frame()

    frame = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.loc[:, list(CANONICAL_COLUMNS)].copy()
    frame = frame.dropna(subset=["timestamp", *NUMERIC_COLUMNS]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return frame


def _material_mismatch_rows(existing: pd.DataFrame, fetched: pd.DataFrame, *, tolerance: float) -> list[dict[str, Any]]:
    """Return overlap rows where OHLCV values differ beyond the allowed tolerance."""

    merged = existing.merge(fetched, on="timestamp", suffixes=("_existing", "_fetched"))
    mismatches: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        diff_payload: dict[str, float] = {}
        for field in OVERLAP_COMPARE_FIELDS:
            lhs = float(row[f"{field}_existing"])
            rhs = float(row[f"{field}_fetched"])
            diff = abs(lhs - rhs)
            if diff > tolerance:
                diff_payload[field] = diff
        if diff_payload:
            mismatches.append({"timestamp": _format_ts(pd.Timestamp(row["timestamp"])), "diffs": diff_payload})
    return mismatches


def _render_raw_rows(frame: pd.DataFrame, raw_format: RawFileFormat, *, include_header: bool) -> str:
    """Render canonical rows using the source raw CSV format."""

    out = pd.DataFrame()
    for column in raw_format.columns:
        if column == raw_format.timestamp_column:
            out[column] = pd.to_datetime(frame["timestamp"], utc=True).map(_format_raw_timestamp)
        else:
            normalized = normalize_column_names([column])[0]
            if normalized not in frame.columns:
                raise ValueError(f"Source column is not supported for raw rendering: {column}")
            out[column] = frame[normalized]
    return out.to_csv(index=False, header=include_header, sep=raw_format.delimiter, lineterminator="\n")


def _atomic_copy_with_append(source: Path, dest: Path, append_payload: str) -> None:
    """Atomically copy a source file and append additional rows."""

    tmp = dest.with_suffix(f"{dest.suffix}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(source, tmp)
        if append_payload:
            with tmp.open("rb") as reader:
                reader.seek(0, os.SEEK_END)
                needs_newline = reader.tell() > 0
                if needs_newline:
                    reader.seek(-1, os.SEEK_END)
                    needs_newline = reader.read(1) not in (b"\n", b"\r")
            with tmp.open("a", encoding="utf-8", newline="") as writer:
                if needs_newline:
                    writer.write("\n")
                writer.write(append_payload)
        os.replace(tmp, dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to atomically create refreshed raw snapshot: {dest}") from exc


def verify_canonical_raw_lineage(project_root: Path, accepted_run_id: str) -> dict[str, Any]:
    """Verify that repo-root CSV files are the canonical raw lineage for the accepted run."""

    config_path = project_root / "configs" / "data.yaml"
    cfg = load_config(config_path)
    discovered_csv_files = tuple(path.resolve() for path in discover_csv_files(cfg.input_root, cfg.csv_glob))
    summary_path = project_root / "runs" / accepted_run_id / "data_standardized" / "reports" / "summary.json"
    per_file_reports_root = project_root / "runs" / accepted_run_id / "data_standardized" / "reports" / "per_file"

    payload: dict[str, Any] = {
        "status": "failed",
        "source_of_truth_status": "unresolved",
        "config_path": str(config_path.resolve()),
        "input_root": str(cfg.input_root.resolve()),
        "csv_glob": cfg.csv_glob,
        "accepted_run_id": accepted_run_id,
        "discovered_csv_files": [str(path) for path in discovered_csv_files],
        "required_timeframes": list(REQUIRED_TIMEFRAMES),
        "evidence": [],
        "errors": [],
    }

    if not summary_path.exists():
        payload["errors"].append(
            {"code": "ACCEPTED_STANDARDIZED_SUMMARY_MISSING", "message": f"Missing accepted-run standardization summary: {summary_path}"}
        )
        return payload

    try:
        summary = _load_json(summary_path)
    except (RuntimeError, ValueError) as exc:
        payload["errors"].append({"code": "ACCEPTED_STANDARDIZED_SUMMARY_INVALID", "message": str(exc)})
        return payload

    payload["evidence"].append({"check": "summary.total_files", "observed": summary.get("total_files"), "expected": len(discovered_csv_files)})
    if int(summary.get("total_files", -1)) != len(discovered_csv_files):
        payload["errors"].append(
            {
                "code": "DISCOVERY_SUMMARY_MISMATCH",
                "message": "Accepted standardized summary does not match currently discovered CSV inventory.",
                "context": {"summary_total_files": summary.get("total_files"), "discovered_csv_files": len(discovered_csv_files)},
            }
        )
        return payload

    sources: list[CanonicalRawSource] = []
    for timeframe in REQUIRED_TIMEFRAMES:
        report_path = per_file_reports_root / f"BTC_USDT_{timeframe}_price_data.json"
        if not report_path.exists():
            payload["errors"].append(
                {"code": "ACCEPTED_PER_FILE_REPORT_MISSING", "message": f"Missing accepted per-file report: {report_path}", "context": {"timeframe": timeframe}}
            )
            return payload

        report_payload = _load_json(report_path)
        input_file = Path(str(report_payload.get("input_file", ""))).resolve()
        output_file = Path(str(report_payload.get("output_file", ""))).resolve()
        parsed = _parse_timeframe_from_name(input_file.name)
        if parsed is None:
            payload["errors"].append(
                {"code": "UNSUPPORTED_SOURCE_FILENAME", "message": f"Cannot parse timeframe from source file: {input_file}"}
            )
            return payload
        symbol_id, report_timeframe = parsed
        if report_timeframe != timeframe:
            payload["errors"].append(
                {
                    "code": "PER_FILE_TIMEFRAME_MISMATCH",
                    "message": "Accepted per-file report timeframe does not match required timeframe.",
                    "context": {"required_timeframe": timeframe, "report_timeframe": report_timeframe, "input_file": str(input_file)},
                }
            )
            return payload
        if input_file not in discovered_csv_files:
            payload["errors"].append(
                {
                    "code": "PER_FILE_INPUT_NOT_DISCOVERED",
                    "message": "Accepted per-file report input is not part of the configured raw discovery inventory.",
                    "context": {"input_file": str(input_file)},
                }
            )
            return payload
        if not output_file.exists():
            payload["errors"].append(
                {"code": "ACCEPTED_STANDARDIZED_OUTPUT_MISSING", "message": f"Missing accepted standardized parquet: {output_file}"}
            )
            return payload

        canonical_df, raw_format = _parse_csv_to_canonical(input_file, cfg.timestamp_aliases)
        alignment_ok, gap_count, alignment_error = _validate_timeframe_alignment(canonical_df, timeframe)
        if not alignment_ok:
            payload["errors"].append(
                {
                    "code": alignment_error,
                    "message": "Canonical raw CSV failed timeframe alignment verification.",
                    "context": {"timeframe": timeframe, "input_file": str(input_file), "gap_count": gap_count},
                }
            )
            return payload

        sources.append(
            CanonicalRawSource(
                timeframe=timeframe,
                symbol_id=symbol_id,
                source_csv=input_file,
                source_report_path=report_path.resolve(),
                source_report_payload=report_payload,
                output_parquet_path=output_file,
                raw_format=raw_format,
                row_count=int(len(canonical_df)),
                timestamp_min_utc=_format_ts(pd.Timestamp(canonical_df["timestamp"].iloc[0])),
                timestamp_max_utc=_format_ts(pd.Timestamp(canonical_df["timestamp"].iloc[-1])),
            )
        )

        payload["evidence"].append(
            {
                "check": "accepted_per_file_input",
                "timeframe": timeframe,
                "input_file": str(input_file),
                "row_count": int(len(canonical_df)),
                "timestamp_min_utc": _format_ts(pd.Timestamp(canonical_df["timestamp"].iloc[0])),
                "timestamp_max_utc": _format_ts(pd.Timestamp(canonical_df["timestamp"].iloc[-1])),
                "delimiter": raw_format.delimiter,
                "timestamp_column": raw_format.timestamp_column,
            }
        )

    payload["status"] = "success"
    payload["source_of_truth_status"] = "proven_canonical_raw"
    payload["sources"] = [
        {
            "timeframe": source.timeframe,
            "symbol_id": source.symbol_id,
            "source_csv": str(source.source_csv),
            "source_report_path": str(source.source_report_path),
            "output_parquet_path": str(source.output_parquet_path),
            "row_count": source.row_count,
            "timestamp_min_utc": source.timestamp_min_utc,
            "timestamp_max_utc": source.timestamp_max_utc,
            "delimiter": source.raw_format.delimiter,
            "timestamp_column": source.raw_format.timestamp_column,
            "columns": list(source.raw_format.columns),
        }
        for source in sources
    ]
    payload["_cfg"] = cfg
    payload["_sources"] = sources
    return payload


def resolve_market_provenance(project_root: Path, *, exchange: str | None, market_type: str | None, symbol: str | None) -> dict[str, Any]:
    """Resolve exchange, market-type and symbol provenance with explicit fail-closed statuses."""

    del project_root
    normalized_market_type = _normalize_market_type(market_type)
    payload = {
        "status": "failed",
        "exchange_provenance_status": "missing",
        "market_type_provenance_status": "missing",
        "symbol_provenance_status": "missing",
        "exchange_used": None,
        "market_type": None,
        "symbol": None,
        "errors": [],
    }

    if exchange is None or exchange.strip() == "":
        payload["errors"].append({"code": "EXCHANGE_PROVENANCE_MISSING", "message": "Exchange provenance is not encoded in the repo and no explicit override was provided."})
    else:
        payload["exchange_used"] = exchange.strip().lower()
        payload["exchange_provenance_status"] = "explicit_override"

    if normalized_market_type is None:
        payload["errors"].append(
            {"code": "MARKET_TYPE_PROVENANCE_MISSING", "message": "Market-type provenance is not encoded in the repo and no explicit override was provided."}
        )
    else:
        payload["market_type"] = normalized_market_type
        payload["market_type_provenance_status"] = "explicit_override"

    if symbol is None or symbol.strip() == "":
        payload["errors"].append({"code": "SYMBOL_PROVENANCE_MISSING", "message": "No explicit exchange symbol override was provided."})
    else:
        payload["symbol"] = symbol.strip()
        payload["symbol_provenance_status"] = "explicit_override"

    payload["status"] = "success" if not payload["errors"] else "failed"
    return payload


def _ccxt_default_type(market_type: str) -> str:
    """Map normalized market type to the CCXT option value."""

    return "future" if market_type == "future" else market_type


def _build_exchange_client(exchange_id: str, market_type: str) -> Any:
    """Instantiate a CCXT exchange client with conservative defaults."""

    if ccxt is None:
        raise RuntimeError("ccxt is not installed; install it into the active virtual environment before tail refresh.")

    exchange_ctor = getattr(ccxt, exchange_id, None)
    if exchange_ctor is None:
        raise ValueError(f"Unsupported CCXT exchange id: {exchange_id}")

    options = {"defaultType": _ccxt_default_type(market_type)}
    client = exchange_ctor({"enableRateLimit": True, "options": options})
    if not bool(getattr(client, "has", {}).get("fetchOHLCV", False)):
        raise ValueError(f"Exchange does not expose fetchOHLCV: {exchange_id}")
    return client


def _exchange_limit(exchange_id: str, requested_limit: int) -> int:
    """Apply exchange-specific upper bounds to fetchOHLCV pagination."""

    hard_cap = OHLCV_FETCH_LIMITS.get(exchange_id, requested_limit)
    return max(1, min(requested_limit, hard_cap))


def _retry_fetch_ohlcv(
    client: Any,
    *,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> Sequence[Sequence[Any]]:
    """Fetch OHLCV rows with bounded retries and backoff."""

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since_ms, limit=limit)
        except Exception as exc:  # noqa: BLE001 - CCXT raises heterogeneous transport/runtime exceptions
            last_error = exc
            LOGGER.info(
                "fetch_ohlcv retry | exchange=%s symbol=%s timeframe=%s since_ms=%d attempt=%d/%d error=%s",
                getattr(client, "id", "unknown"),
                symbol,
                timeframe,
                since_ms,
                attempt,
                max_retries,
                exc,
            )
            if attempt == max_retries:
                break
            time.sleep(retry_backoff_seconds * float(attempt))
    raise RuntimeError(
        f"fetch_ohlcv failed after {max_retries} attempts | exchange={getattr(client, 'id', 'unknown')} timeframe={timeframe} since_ms={since_ms}"
    ) from last_error


def _fetch_tail_for_timeframe(
    client: Any,
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    last_existing_ts: pd.Timestamp,
    last_closed_ts: pd.Timestamp,
    request_limit: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Fetch OHLCV tail rows from the overlap boundary up to the latest closed candle."""

    if last_existing_ts >= last_closed_ts:
        return _empty_canonical_frame()

    timeframe_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    next_since_ms = max(0, int(last_existing_ts.timestamp() * 1000) - timeframe_ms)
    end_ms = int(last_closed_ts.timestamp() * 1000)
    limit = _exchange_limit(exchange_id, request_limit)
    frames: list[pd.DataFrame] = []

    while next_since_ms <= end_ms:
        rows = _retry_fetch_ohlcv(
            client,
            symbol=symbol,
            timeframe=timeframe,
            since_ms=next_since_ms,
            limit=limit,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        batch = _to_canonical_ohlcv(rows)
        if batch.empty:
            break

        batch = batch.loc[batch["timestamp"] <= last_closed_ts].copy()
        if batch.empty:
            break

        frames.append(batch)
        last_batch_ts = pd.Timestamp(batch["timestamp"].iloc[-1])
        next_since_ms = int(last_batch_ts.timestamp() * 1000) + timeframe_ms
        if len(batch) < limit and last_batch_ts >= last_closed_ts:
            break

        pause_seconds = max(float(getattr(client, "rateLimit", 0)) / 1000.0, retry_backoff_seconds)
        time.sleep(pause_seconds)

    if not frames:
        return _empty_canonical_frame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return out.loc[:, list(CANONICAL_COLUMNS)].copy()


def _download_timeframes(
    sources: Sequence[CanonicalRawSource],
    *,
    timestamp_aliases: Sequence[str],
    exchange_id: str,
    market_type: str,
    symbol: str,
    request_limit: int,
    max_retries: int,
    retry_backoff_seconds: float,
    now_utc: datetime,
) -> dict[str, Any]:
    """Download overlap-inclusive tail rows for all required timeframes from one exchange."""

    client = _build_exchange_client(exchange_id, market_type)
    attempts: list[dict[str, Any]] = []
    try:
        client.load_markets()
        timeframe_frames: dict[str, pd.DataFrame] = {}
        for source in sources:
            existing_frame, _ = _parse_csv_to_canonical(source.source_csv, timestamp_aliases)
            last_existing_ts = pd.Timestamp(existing_frame["timestamp"].iloc[-1])
            last_closed_ts = _last_closed_candle_start(now_utc, source.timeframe)
            timeframe_frames[source.timeframe] = _fetch_tail_for_timeframe(
                client,
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=source.timeframe,
                last_existing_ts=last_existing_ts,
                last_closed_ts=last_closed_ts,
                request_limit=request_limit,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        attempts.append({"exchange": exchange_id, "status": "success"})
        return {"status": "success", "exchange_used": exchange_id, "attempts": attempts, "frames": timeframe_frames}
    except Exception as exc:  # noqa: BLE001
        attempts.append({"exchange": exchange_id, "status": "failed", "error": str(exc)})
        return {"status": "failed", "exchange_used": exchange_id, "attempts": attempts, "error": str(exc)}
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            LOGGER.info("Exchange client close failed | exchange=%s", exchange_id)


def _stage_file_path(project_root: Path, refresh_session_id: str, source_name: str) -> Path:
    return project_root / "runs" / refresh_session_id / "data_tail_refresh" / "staging" / source_name


def _raw_snapshot_path(project_root: Path, refresh_session_id: str, source_name: str) -> Path:
    return project_root / "runs" / refresh_session_id / "data_tail_refresh" / "canonical_raw_snapshot" / source_name


def run_phase_a(options: RefreshOptions) -> dict[str, Any]:
    """Execute source verification, staging download, validation and canonical raw snapshot merge."""

    report_root = options.project_root / "runs" / options.refresh_session_id / "data_tail_refresh" / "reports"
    main_report_path = report_root / "data_tail_refresh_report.json"
    staging_report_path = report_root / "staging_validation_report.json"
    merge_report_path = report_root / "canonical_merge_report.json"

    verification = verify_canonical_raw_lineage(options.project_root, options.accepted_run_id)
    base_report: dict[str, Any] = {
        "report_version": TAIL_REFRESH_VERSION,
        "generated_at_utc": _utc_now().isoformat(),
        "refresh_session_id": options.refresh_session_id,
        "accepted_run_id": options.accepted_run_id,
        "phase_a_status": "failed",
        "phase_b_status": "not_started",
        "rebuild_status": "not_started",
        "source_of_truth_status": verification.get("source_of_truth_status"),
        "exchange_used": None,
        "market_type": None,
        "symbol": None,
        "fallback_used": False,
        "refresh_lineage_root": str((options.project_root / "runs" / options.refresh_session_id).resolve()),
        "tail_start_utc": None,
        "tail_end_utc": None,
        "merged_rows": 0,
        "errors": [],
    }

    if verification.get("status") != "success":
        base_report["errors"] = list(verification.get("errors", []))
        atomic_write_json(base_report, main_report_path)
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "timeframes": [],
                "errors": list(verification.get("errors", [])),
            },
            staging_report_path,
        )
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "files": [],
                "errors": list(verification.get("errors", [])),
            },
            merge_report_path,
        )
        return {
            "status": "failed",
            "phase_a_status": "failed",
            "phase_b_status": "not_started",
            "main_report_path": main_report_path,
            "staging_report_path": staging_report_path,
            "merge_report_path": merge_report_path,
        }

    provenance = resolve_market_provenance(
        options.project_root,
        exchange=options.exchange,
        market_type=options.market_type,
        symbol=options.symbol,
    )
    if provenance["status"] != "success":
        base_report["errors"] = list(provenance.get("errors", []))
        base_report["source_of_truth_status"] = verification["source_of_truth_status"]
        atomic_write_json(base_report, main_report_path)
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "timeframes": [],
                "errors": list(provenance.get("errors", [])),
            },
            staging_report_path,
        )
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "files": [],
                "errors": list(provenance.get("errors", [])),
            },
            merge_report_path,
        )
        return {
            "status": "failed",
            "phase_a_status": "failed",
            "phase_b_status": "not_started",
            "main_report_path": main_report_path,
            "staging_report_path": staging_report_path,
            "merge_report_path": merge_report_path,
        }

    if options.provider_probe_report_path is None:
        gate_error = {"code": "PROVIDER_CAPABILITY_GATE_REQUIRED", "message": "provider capability report path is required before live tail refresh."}
        base_report["errors"] = [gate_error]
        atomic_write_json(base_report, main_report_path)
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "timeframes": [],
                "errors": [gate_error],
            },
            staging_report_path,
        )
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "files": [],
                "errors": [gate_error],
            },
            merge_report_path,
        )
        return {
            "status": "failed",
            "phase_a_status": "failed",
            "phase_b_status": "not_started",
            "main_report_path": main_report_path,
            "staging_report_path": staging_report_path,
            "merge_report_path": merge_report_path,
        }

    probe_payload = _load_json(options.provider_probe_report_path)
    provider_results = probe_payload.get("provider_results", [])
    selected_provider_result = None
    if isinstance(provider_results, list):
        for item in provider_results:
            if isinstance(item, dict) and item.get("provider") == provenance["exchange_used"]:
                selected_provider_result = item
                break
    market_type_verdict = probe_payload.get("market_type_verdict", {})
    selected_probe_ok = (
        isinstance(selected_provider_result, dict)
        and selected_provider_result.get("probe_status") == "success"
        and bool(selected_provider_result.get("retrieval_worked_in_practice"))
        and isinstance(market_type_verdict, dict)
        and market_type_verdict.get("value") == provenance["market_type"]
    )
    if not selected_probe_ok:
        gate_error = {
            "code": "PROVIDER_CAPABILITY_GATE_FAILED",
            "message": "Selected provider is not approved by the provider capability gate for the requested market type.",
            "context": {
                "provider_probe_report_path": str(options.provider_probe_report_path),
                "requested_exchange": provenance["exchange_used"],
                "requested_market_type": provenance["market_type"],
                "selected_provider_result": selected_provider_result,
                "market_type_verdict": market_type_verdict,
            },
        }
        base_report["errors"] = [gate_error]
        atomic_write_json(base_report, main_report_path)
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "timeframes": [],
                "errors": [gate_error],
            },
            staging_report_path,
        )
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "files": [],
                "errors": [gate_error],
            },
            merge_report_path,
        )
        return {
            "status": "failed",
            "phase_a_status": "failed",
            "phase_b_status": "not_started",
            "main_report_path": main_report_path,
            "staging_report_path": staging_report_path,
            "merge_report_path": merge_report_path,
        }

    if options.refresh_session_id.strip() == options.accepted_run_id.strip():
        message = "refresh_session_id must differ from accepted_run_id to preserve historical run immutability."
        base_report["errors"] = [{"code": "REFRESH_SESSION_ID_COLLISION", "message": message}]
        atomic_write_json(base_report, main_report_path)
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "timeframes": [],
                "errors": [{"code": "REFRESH_SESSION_ID_COLLISION", "message": message}],
            },
            staging_report_path,
        )
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "files": [],
                "errors": [{"code": "REFRESH_SESSION_ID_COLLISION", "message": message}],
            },
            merge_report_path,
        )
        return {
            "status": "failed",
            "phase_a_status": "failed",
            "phase_b_status": "not_started",
            "main_report_path": main_report_path,
            "staging_report_path": staging_report_path,
            "merge_report_path": merge_report_path,
        }

    cfg = verification["_cfg"]
    sources: list[CanonicalRawSource] = list(verification["_sources"])
    attempted_exchanges = tuple(
        exchange_id for exchange_id in (provenance["exchange_used"], *options.fallback_exchanges) if isinstance(exchange_id, str) and exchange_id.strip()
    )
    now_utc = _utc_now()
    download_result: dict[str, Any] | None = None
    download_errors: list[dict[str, Any]] = []
    for idx, exchange_id in enumerate(attempted_exchanges):
        result = _download_timeframes(
            sources,
            timestamp_aliases=cfg.timestamp_aliases,
            exchange_id=exchange_id,
            market_type=str(provenance["market_type"]),
            symbol=str(provenance["symbol"]),
            request_limit=options.request_limit,
            max_retries=options.max_retries,
            retry_backoff_seconds=options.retry_backoff_seconds,
            now_utc=now_utc,
        )
        download_errors.extend(result.get("attempts", []))
        if result.get("status") == "success":
            download_result = result
            base_report["fallback_used"] = idx > 0
            break

    if download_result is None:
        base_report["errors"] = [
            {"code": "OHLCV_DOWNLOAD_FAILED", "message": "All exchange attempts failed.", "context": {"attempts": download_errors}}
        ]
        base_report["exchange_used"] = provenance["exchange_used"]
        base_report["market_type"] = provenance["market_type"]
        base_report["symbol"] = provenance["symbol"]
        atomic_write_json(base_report, main_report_path)
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "exchange_attempts": download_errors,
                "timeframes": [],
                "errors": base_report["errors"],
            },
            staging_report_path,
        )
        atomic_write_json(
            {
                "report_version": TAIL_REFRESH_VERSION,
                "generated_at_utc": _utc_now().isoformat(),
                "refresh_session_id": options.refresh_session_id,
                "status": "failed",
                "source_of_truth_status": verification.get("source_of_truth_status"),
                "files": [],
                "errors": base_report["errors"],
            },
            merge_report_path,
        )
        return {
            "status": "failed",
            "phase_a_status": "failed",
            "phase_b_status": "not_started",
            "main_report_path": main_report_path,
            "staging_report_path": staging_report_path,
            "merge_report_path": merge_report_path,
        }

    staging_results: list[dict[str, Any]] = []
    merge_results: list[dict[str, Any]] = []
    pending_writes: list[dict[str, Any]] = []
    aggregate_tail_start: pd.Timestamp | None = None
    aggregate_tail_end: pd.Timestamp | None = None
    total_merged_rows = 0
    phase_errors: list[dict[str, Any]] = []

    for source in sources:
        existing_frame, _ = _parse_csv_to_canonical(source.source_csv, cfg.timestamp_aliases)
        fetched_frame = download_result["frames"].get(source.timeframe, _empty_canonical_frame()).copy()
        last_existing_ts = pd.Timestamp(existing_frame["timestamp"].iloc[-1])
        last_closed_ts = _last_closed_candle_start(now_utc, source.timeframe)

        staging_file_path = _stage_file_path(options.project_root, options.refresh_session_id, source.source_csv.name)
        snapshot_file_path = _raw_snapshot_path(options.project_root, options.refresh_session_id, source.source_csv.name)

        if fetched_frame.empty and last_closed_ts > last_existing_ts:
            phase_errors.append(
                {
                    "code": "NO_TAIL_ROWS_FETCHED",
                    "message": "Expected new closed candles but download returned no rows.",
                    "context": {"timeframe": source.timeframe, "last_existing_ts": _format_ts(last_existing_ts), "last_closed_ts": _format_ts(last_closed_ts)},
                }
            )
            continue

        alignment_ok, gap_count, alignment_error = _validate_timeframe_alignment(fetched_frame, source.timeframe)
        overlap_frame = fetched_frame.loc[fetched_frame["timestamp"] <= last_existing_ts].copy()
        new_rows = fetched_frame.loc[fetched_frame["timestamp"] > last_existing_ts].copy()
        if not fetched_frame.empty and overlap_frame.empty:
            phase_errors.append(
                {
                    "code": "MISSING_OVERLAP_CONFIRMATION",
                    "message": "Fetched tail did not include any overlap rows for material equivalence validation.",
                    "context": {"timeframe": source.timeframe, "last_existing_ts": _format_ts(last_existing_ts)},
                }
            )
            continue

        mismatch_rows = _material_mismatch_rows(existing_frame.loc[existing_frame["timestamp"].isin(overlap_frame["timestamp"])], overlap_frame, tolerance=options.overlap_abs_tolerance)
        if mismatch_rows:
            phase_errors.append(
                {
                    "code": "OVERLAP_MISMATCH",
                    "message": "Overlap candles disagreed beyond the configured material tolerance.",
                    "context": {"timeframe": source.timeframe, "mismatch_rows": mismatch_rows[:5], "overlap_abs_tolerance": options.overlap_abs_tolerance},
                }
            )
            continue

        expected_next_ts = last_existing_ts + pd.Timedelta(seconds=TIMEFRAME_SECONDS[source.timeframe])
        if not new_rows.empty and pd.Timestamp(new_rows["timestamp"].iloc[0]) != expected_next_ts:
            phase_errors.append(
                {
                    "code": "TAIL_GAP_AT_BOUNDARY",
                    "message": "Fetched new rows do not begin exactly at the next expected timeframe boundary.",
                    "context": {
                        "timeframe": source.timeframe,
                        "expected_next_ts": _format_ts(expected_next_ts),
                        "observed_first_new_ts": _format_ts(pd.Timestamp(new_rows["timestamp"].iloc[0])),
                    },
                }
            )
            continue
        if last_closed_ts > last_existing_ts and (new_rows.empty or pd.Timestamp(new_rows["timestamp"].iloc[-1]) != last_closed_ts):
            phase_errors.append(
                {
                    "code": "TAIL_END_INCOMPLETE",
                    "message": "Fetched tail does not reach the latest fully closed candle.",
                    "context": {
                        "timeframe": source.timeframe,
                        "last_existing_ts": _format_ts(last_existing_ts),
                        "last_closed_ts": _format_ts(last_closed_ts),
                        "observed_tail_end": _format_ts(pd.Timestamp(new_rows["timestamp"].iloc[-1])) if not new_rows.empty else None,
                    },
                }
            )
            continue

        if not alignment_ok:
            phase_errors.append(
                {
                    "code": alignment_error,
                    "message": "Fetched tail failed timeframe boundary validation.",
                    "context": {"timeframe": source.timeframe, "gap_count": gap_count},
                }
            )
            continue

        stage_payload = _render_raw_rows(fetched_frame, source.raw_format, include_header=True) if not fetched_frame.empty else ""
        append_payload = _render_raw_rows(new_rows, source.raw_format, include_header=False) if not new_rows.empty else ""
        pending_writes.append(
            {
                "staging_file_path": staging_file_path,
                "stage_payload": stage_payload,
                "source_csv": source.source_csv,
                "snapshot_file_path": snapshot_file_path,
                "append_payload": append_payload,
            }
        )

        staging_results.append(
            {
                "timeframe": source.timeframe,
                "source_csv": str(source.source_csv),
                "staging_file_path": str(staging_file_path),
                "last_existing_ts": _format_ts(last_existing_ts),
                "last_closed_ts": _format_ts(last_closed_ts),
                "tail_start_utc": _format_ts(pd.Timestamp(new_rows["timestamp"].iloc[0])) if not new_rows.empty else None,
                "tail_end_utc": _format_ts(pd.Timestamp(new_rows["timestamp"].iloc[-1])) if not new_rows.empty else None,
                "staging_row_count": int(len(fetched_frame)),
                "overlap_row_count": int(len(overlap_frame)),
                "new_row_count": int(len(new_rows)),
                "boundary_alignment_ok": bool(alignment_ok),
                "gap_count": gap_count,
                "partial_candles_excluded": True,
                "overlap_equivalence_ok": len(mismatch_rows) == 0,
                "overlap_abs_tolerance": options.overlap_abs_tolerance,
            }
        )
        merge_results.append(
            {
                "timeframe": source.timeframe,
                "source_csv": str(source.source_csv),
                "snapshot_file_path": str(snapshot_file_path),
                "before_rows": int(len(existing_frame)),
                "after_rows": int(len(existing_frame) + len(new_rows)),
                "merged_rows": int(len(new_rows)),
                "tail_start_utc": _format_ts(pd.Timestamp(new_rows["timestamp"].iloc[0])) if not new_rows.empty else None,
                "tail_end_utc": _format_ts(pd.Timestamp(new_rows["timestamp"].iloc[-1])) if not new_rows.empty else None,
                "format_preserved": {
                    "delimiter": source.raw_format.delimiter,
                    "timestamp_column": source.raw_format.timestamp_column,
                    "columns": list(source.raw_format.columns),
                },
            }
        )

        if not new_rows.empty:
            total_merged_rows += int(len(new_rows))
            start_ts = pd.Timestamp(new_rows["timestamp"].iloc[0])
            end_ts = pd.Timestamp(new_rows["timestamp"].iloc[-1])
            aggregate_tail_start = start_ts if aggregate_tail_start is None else min(aggregate_tail_start, start_ts)
            aggregate_tail_end = end_ts if aggregate_tail_end is None else max(aggregate_tail_end, end_ts)

    phase_a_status = "success" if not phase_errors else "failed"
    if phase_a_status == "success":
        for item in pending_writes:
            atomic_write_text(str(item["stage_payload"]), Path(item["staging_file_path"]))
            _atomic_copy_with_append(Path(item["source_csv"]), Path(item["snapshot_file_path"]), str(item["append_payload"]))

    base_report.update(
        {
            "phase_a_status": phase_a_status,
            "exchange_used": download_result["exchange_used"],
            "market_type": provenance["market_type"],
            "symbol": provenance["symbol"],
            "tail_start_utc": _format_ts(aggregate_tail_start) if aggregate_tail_start is not None else None,
            "tail_end_utc": _format_ts(aggregate_tail_end) if aggregate_tail_end is not None else None,
            "merged_rows": total_merged_rows,
            "errors": phase_errors,
        }
    )

    atomic_write_json(base_report, main_report_path)
    atomic_write_json(
        {
            "report_version": TAIL_REFRESH_VERSION,
            "generated_at_utc": _utc_now().isoformat(),
            "refresh_session_id": options.refresh_session_id,
            "status": phase_a_status,
            "source_of_truth_status": verification.get("source_of_truth_status"),
            "exchange_used": download_result["exchange_used"],
            "market_type": provenance["market_type"],
            "symbol": provenance["symbol"],
            "exchange_attempts": download_errors,
            "timeframes": staging_results,
            "errors": phase_errors,
        },
        staging_report_path,
    )
    atomic_write_json(
        {
            "report_version": TAIL_REFRESH_VERSION,
            "generated_at_utc": _utc_now().isoformat(),
            "refresh_session_id": options.refresh_session_id,
            "status": phase_a_status,
            "source_of_truth_status": verification.get("source_of_truth_status"),
            "exchange_used": download_result["exchange_used"],
            "market_type": provenance["market_type"],
            "symbol": provenance["symbol"],
            "files": merge_results,
            "merged_rows": total_merged_rows,
            "errors": phase_errors,
        },
        merge_report_path,
    )
    return {
        "status": phase_a_status,
        "phase_a_status": phase_a_status,
        "phase_b_status": "not_started",
        "main_report_path": main_report_path,
        "staging_report_path": staging_report_path,
        "merge_report_path": merge_report_path,
        "source_of_truth_status": verification.get("source_of_truth_status"),
        "raw_snapshot_root": options.project_root / "runs" / options.refresh_session_id / "data_tail_refresh" / "canonical_raw_snapshot",
        "exchange_used": download_result["exchange_used"],
        "market_type": provenance["market_type"],
        "symbol": provenance["symbol"],
        "tail_start_utc": _format_ts(aggregate_tail_start) if aggregate_tail_start is not None else None,
        "tail_end_utc": _format_ts(aggregate_tail_end) if aggregate_tail_end is not None else None,
        "merged_rows": total_merged_rows,
    }


def _run_subprocess(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess stage and capture its logs."""

    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)


def _accepted_stage_arg(report_path: Path, key: str) -> Any:
    payload = _load_json(report_path)
    invocation_args = payload.get("invocation_args")
    if not isinstance(invocation_args, Mapping):
        raise ValueError(f"Missing invocation_args in accepted report: {report_path}")
    return invocation_args.get(key)


def _build_split_stage_command(options: RefreshOptions, *, refresh_session_id: str, feature_root: Path) -> list[str]:
    accepted_split_report = options.project_root / "runs" / options.accepted_run_id / "data_features" / "reports" / "split_validation_report.json"
    accepted_payload = _load_json(accepted_split_report)
    invocation_args = accepted_payload.get("invocation_args")
    if not isinstance(invocation_args, Mapping):
        raise ValueError("accepted split_validation_report invocation_args missing")

    command = [
        str(options.python_executable),
        "scripts/validate_splits.py",
        "--run-id",
        refresh_session_id,
        "--input-root",
        str(feature_root),
        "--split-mode",
        str(accepted_payload.get("split_mode")),
        "--require-train-input-validation",
        str(invocation_args.get("require_train_input_validation", True)).lower(),
        "--min-train-rows",
        str(invocation_args.get("min_train_rows", 1)),
        "--min-val-rows",
        str(invocation_args.get("min_val_rows", 1)),
        "--min-test-rows",
        str(invocation_args.get("min_test_rows", 1)),
        "--warmup-rows",
        str(invocation_args.get("warmup_rows", 0)),
        "--log-level",
        options.log_level,
    ]
    split_overrides = invocation_args.get("split_overrides", {})
    if isinstance(split_overrides, Mapping):
        if accepted_payload.get("split_mode") == "ratio_chrono":
            for key in ("train_ratio", "val_ratio", "test_ratio"):
                value = split_overrides.get(key)
                if isinstance(value, str) and value.strip():
                    command.extend([f"--{key.replace('_', '-')}", value])
    return command


def _build_dataset_stage_command(
    options: RefreshOptions,
    *,
    refresh_session_id: str,
    feature_root: Path,
    feature_reports_root: Path,
    dataset_root: Path,
) -> list[str]:
    accepted_dataset_report = options.project_root / "runs" / options.accepted_run_id / "data_datasets" / "reports" / "dataset_build_report.json"
    invocation_args = _load_json(accepted_dataset_report).get("invocation_args", {})
    if not isinstance(invocation_args, Mapping):
        raise ValueError("accepted dataset_build_report invocation_args missing")
    return [
        str(options.python_executable),
        "scripts/build_datasets.py",
        "--run-id",
        refresh_session_id,
        "--input-root",
        str(feature_root),
        "--output-root",
        str(dataset_root),
        "--feature-manifest-path",
        str(feature_reports_root / "feature_manifest.json"),
        "--train-input-report-path",
        str(feature_reports_root / "train_input_validation_report.json"),
        "--split-report-path",
        str(feature_reports_root / "split_validation_report.json"),
        "--overwrite",
        str(invocation_args.get("overwrite", False)).lower(),
        "--require-train-input-validation",
        str(invocation_args.get("require_train_input_validation", True)).lower(),
        "--require-split-validation",
        str(invocation_args.get("require_split_validation", True)).lower(),
        "--aggregate-walk-forward",
        str(invocation_args.get("aggregate_walk_forward", False)).lower(),
        "--timestamp-column",
        "timestamp",
        "--execution-price-column",
        str(invocation_args.get("execution_price_column", "close")),
        "--mark-to-market-column",
        str(invocation_args.get("mark_to_market_column", "close")),
        "--log-level",
        options.log_level,
    ]


def _build_state_stage_command(
    options: RefreshOptions,
    *,
    refresh_session_id: str,
    dataset_root: Path,
    state_root: Path,
) -> list[str]:
    accepted_state_report = options.project_root / "runs" / options.accepted_run_id / "data_states" / "reports" / "state_build_report.json"
    invocation_args = _load_json(accepted_state_report).get("invocation_args", {})
    if not isinstance(invocation_args, Mapping):
        raise ValueError("accepted state_build_report invocation_args missing")
    return [
        str(options.python_executable),
        "scripts/build_states.py",
        "--run-id",
        refresh_session_id,
        "--input-root",
        str(dataset_root),
        "--output-root",
        str(state_root),
        "--dataset-manifest-path",
        str(dataset_root / "reports" / "dataset_manifest.json"),
        "--dataset-build-report-path",
        str(dataset_root / "reports" / "dataset_build_report.json"),
        "--overwrite",
        str(invocation_args.get("overwrite", False)).lower(),
        "--enable-scaling",
        str(invocation_args.get("enable_scaling", False)).lower(),
        "--scaler-type",
        str(invocation_args.get("scaler_type", "none")),
        "--timestamp-column",
        "timestamp",
        "--build-mode",
        str(invocation_args.get("build_mode", "materialize_only")),
        "--strict-column-selection",
        str(invocation_args.get("strict_column_selection", True)).lower(),
        "--sequence-mode",
        str(invocation_args.get("sequence_mode", False)).lower(),
        "--aggregate-walk-forward",
        str(invocation_args.get("aggregate_walk_forward", False)).lower(),
        "--execution-price-column",
        str(invocation_args.get("execution_price_column", "close")),
        "--mark-to-market-column",
        str(invocation_args.get("mark_to_market_column", "close")),
        "--log-level",
        options.log_level,
    ]


def _feature_contract_compatibility(accepted_manifest_path: Path, refreshed_manifest_path: Path) -> dict[str, Any]:
    """Check feature schema and semantic contract compatibility across runs."""

    accepted = _load_json(accepted_manifest_path)
    refreshed = _load_json(refreshed_manifest_path)
    checks = {
        "feature_groups_match": accepted.get("feature_groups") == refreshed.get("feature_groups"),
        "column_dtypes_match": accepted.get("column_dtypes") == refreshed.get("column_dtypes"),
        "event_columns_match": accepted.get("event_columns") == refreshed.get("event_columns"),
        "continuous_columns_match": accepted.get("continuous_columns") == refreshed.get("continuous_columns"),
        "placeholder_columns_match": accepted.get("placeholder_columns") == refreshed.get("placeholder_columns"),
        "warmup_policy_match": accepted.get("warmup_policy") == refreshed.get("warmup_policy"),
        "indicator_spec_version_match": accepted.get("indicator_spec_version") == refreshed.get("indicator_spec_version"),
        "config_hash_match": accepted.get("config_hash") == refreshed.get("config_hash"),
        "formula_fingerprint_bundle_match": accepted.get("formula_fingerprint_bundle") == refreshed.get("formula_fingerprint_bundle"),
    }
    return {
        "status": "success" if all(checks.values()) else "failed",
        "checks": checks,
        "accepted_manifest_path": str(accepted_manifest_path),
        "refreshed_manifest_path": str(refreshed_manifest_path),
    }


def run_phase_b(options: RefreshOptions, phase_a_result: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild downstream layers under a new refresh lineage using explicit input paths."""

    if phase_a_result.get("status") != "success":
        raise ValueError("Phase B requires a successful Phase A result.")

    report_root = options.project_root / "runs" / options.refresh_session_id / "data_tail_refresh" / "reports"
    rebuild_summary_path = report_root / "rebuild_summary.json"
    phase_a_main_report_path = Path(str(phase_a_result["main_report_path"]))

    cfg = load_config(options.data_config_path)
    refreshed_cfg: PipelineConfig = replace(cfg, input_root=Path(str(phase_a_result["raw_snapshot_root"])).resolve())
    validate_config(refreshed_cfg)

    standardize_run_root = cfg.runs_root / options.refresh_session_id / "data_standardized"
    standardize_reports_root = standardize_run_root / "reports"
    standardize_per_file_root = standardize_reports_root / "per_file"
    standardize_reports = standardize_all(
        cfg=refreshed_cfg,
        parquet_root=standardize_run_root / "parquet",
        per_file_reports_root=standardize_per_file_root,
        dry_run=False,
    )
    standardize_summary = summarize_reports(standardize_reports)
    standardize_summary["run_id"] = options.refresh_session_id
    standardize_summary["run_root"] = str(standardize_run_root.resolve())
    standardize_summary["input_root_resolved"] = str(refreshed_cfg.input_root.resolve())
    standardize_summary["output_root_resolved"] = str((standardize_run_root / "parquet").resolve())
    atomic_write_json(standardize_summary, standardize_reports_root / "summary.json")
    standardize_status = "success" if standardize_summary["failed_files"] == 0 else "failed"

    feature_root = cfg.runs_root / options.refresh_session_id / "data_features"
    feature_parquet_root = feature_root / "parquet"
    feature_reports_root = feature_root / "reports"
    dataset_root = cfg.runs_root / options.refresh_session_id / "data_datasets"
    state_root = cfg.runs_root / options.refresh_session_id / "data_states"

    stages: list[dict[str, Any]] = [
        {
            "stage": "standardized_build",
            "status": standardize_status,
            "command": [
                str(options.python_executable),
                "scripts/make_standardized_data.py",
                "--config",
                str(options.data_config_path.resolve()),
                "--run-id",
                options.refresh_session_id,
                "--input-root",
                str(Path(str(phase_a_result["raw_snapshot_root"])).resolve()),
            ],
            "input_root": str(Path(str(phase_a_result["raw_snapshot_root"])).resolve()),
            "output_root": str((standardize_run_root / "parquet").resolve()),
            "report_path": str((standardize_reports_root / "summary.json").resolve()),
            "exit_code": 0 if standardize_status == "success" else 2,
        }
    ]

    if standardize_status != "success":
        summary_payload = {
            "report_version": TAIL_REFRESH_VERSION,
            "generated_at_utc": _utc_now().isoformat(),
            "refresh_session_id": options.refresh_session_id,
            "accepted_run_id": options.accepted_run_id,
            "status": "failed",
            "feature_contract_compatibility": {"status": "not_started", "checks": {}},
            "stages": stages,
            "errors": [{"code": "STANDARDIZED_BUILD_FAILED", "message": "Standardized rebuild failed for refreshed raw snapshot."}],
        }
        atomic_write_json(summary_payload, rebuild_summary_path)
        phase_a_main_report = _load_json(phase_a_main_report_path)
        phase_a_main_report["phase_b_status"] = "failed"
        phase_a_main_report["rebuild_status"] = "failed"
        atomic_write_json(phase_a_main_report, phase_a_main_report_path)
        return {"status": "failed", "rebuild_summary_path": rebuild_summary_path}

    feature_command = [
        str(options.python_executable),
        "scripts/make_features.py",
        "--config",
        str(options.features_config_path.resolve()),
        "--run-id",
        options.refresh_session_id,
        "--input-root",
        str((standardize_run_root / "parquet").resolve()),
        "--strict-parity",
        "true",
        "--log-level",
        options.log_level,
    ]
    feature_result = _run_subprocess(feature_command, cwd=options.project_root)
    stages.append(
        {
            "stage": "feature_build",
            "status": "success" if feature_result.returncode == 0 else "failed",
            "command": feature_command,
            "input_root": str((standardize_run_root / "parquet").resolve()),
            "output_root": str(feature_parquet_root.resolve()),
            "report_path": str((feature_reports_root / "summary.json").resolve()),
            "exit_code": int(feature_result.returncode),
            "stdout_tail": feature_result.stdout[-4000:],
            "stderr_tail": feature_result.stderr[-4000:],
        }
    )

    feature_manifest_compatibility = {"status": "not_started", "checks": {}}
    if feature_result.returncode == 0:
        feature_manifest_compatibility = _feature_contract_compatibility(
            options.project_root / "runs" / options.accepted_run_id / "data_features" / "reports" / "feature_manifest.json",
            feature_reports_root / "feature_manifest.json",
        )

    if feature_result.returncode != 0 or feature_manifest_compatibility["status"] != "success":
        summary_payload = {
            "report_version": TAIL_REFRESH_VERSION,
            "generated_at_utc": _utc_now().isoformat(),
            "refresh_session_id": options.refresh_session_id,
            "accepted_run_id": options.accepted_run_id,
            "status": "failed",
            "feature_contract_compatibility": feature_manifest_compatibility,
            "stages": stages,
            "errors": [
                {
                    "code": "FEATURE_STAGE_FAILED" if feature_result.returncode != 0 else "FEATURE_CONTRACT_MISMATCH",
                    "message": "Feature rebuild failed or produced a contract-incompatible manifest.",
                }
            ],
        }
        atomic_write_json(summary_payload, rebuild_summary_path)
        phase_a_main_report = _load_json(phase_a_main_report_path)
        phase_a_main_report["phase_b_status"] = "failed"
        phase_a_main_report["rebuild_status"] = "failed"
        atomic_write_json(phase_a_main_report, phase_a_main_report_path)
        return {"status": "failed", "rebuild_summary_path": rebuild_summary_path}

    other_stage_commands = [
        (
            "train_input_validation",
            [
                str(options.python_executable),
                "scripts/validate_train_inputs.py",
                "--run-id",
                options.refresh_session_id,
                "--input-root",
                str(feature_parquet_root.resolve()),
                "--strict-extra-columns",
                "true",
                "--strict-column-order",
                "false",
                "--log-level",
                options.log_level,
            ],
            str(feature_parquet_root.resolve()),
            str((feature_reports_root / "train_input_validation_report.json").resolve()),
        ),
        ("split_validation", _build_split_stage_command(options, refresh_session_id=options.refresh_session_id, feature_root=feature_parquet_root), str(feature_parquet_root.resolve()), str((feature_reports_root / "split_validation_report.json").resolve())),
        ("dataset_build", _build_dataset_stage_command(options, refresh_session_id=options.refresh_session_id, feature_root=feature_parquet_root, feature_reports_root=feature_reports_root, dataset_root=dataset_root), str(feature_parquet_root.resolve()), str((dataset_root / "reports" / "dataset_build_report.json").resolve())),
        ("state_build", _build_state_stage_command(options, refresh_session_id=options.refresh_session_id, dataset_root=dataset_root, state_root=state_root), str(dataset_root.resolve()), str((state_root / "reports" / "state_build_report.json").resolve())),
    ]

    overall_success = True
    for stage_name, command, input_root, report_path in other_stage_commands:
        result = _run_subprocess(command, cwd=options.project_root)
        success = result.returncode == 0
        stages.append(
            {
                "stage": stage_name,
                "status": "success" if success else "failed",
                "command": command,
                "input_root": input_root,
                "report_path": report_path,
                "exit_code": int(result.returncode),
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
        if not success:
            overall_success = False
            break

    summary_payload = {
        "report_version": TAIL_REFRESH_VERSION,
        "generated_at_utc": _utc_now().isoformat(),
        "refresh_session_id": options.refresh_session_id,
        "accepted_run_id": options.accepted_run_id,
        "status": "success" if overall_success else "failed",
        "feature_contract_compatibility": feature_manifest_compatibility,
        "stages": stages,
        "errors": [] if overall_success else [{"code": "REBUILD_STAGE_FAILED", "message": "One or more rebuild stages failed."}],
    }
    atomic_write_json(summary_payload, rebuild_summary_path)

    phase_a_main_report = _load_json(phase_a_main_report_path)
    phase_a_main_report["phase_b_status"] = "success" if overall_success else "failed"
    phase_a_main_report["rebuild_status"] = "success" if overall_success else "failed"
    atomic_write_json(phase_a_main_report, phase_a_main_report_path)
    return {"status": "success" if overall_success else "failed", "rebuild_summary_path": rebuild_summary_path}


def run_refresh(options: RefreshOptions) -> dict[str, Any]:
    """Run Phase A and, when successful, Phase B for the refreshed lineage."""

    phase_a_result = run_phase_a(options)
    if phase_a_result.get("status") != "success":
        return {
            "status": "failed",
            "phase_a": phase_a_result,
            "phase_b": {"status": "not_started"},
        }
    phase_b_result = run_phase_b(options, phase_a_result)
    return {"status": "success" if phase_b_result.get("status") == "success" else "failed", "phase_a": phase_a_result, "phase_b": phase_b_result}
