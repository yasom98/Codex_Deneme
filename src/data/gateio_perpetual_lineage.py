"""Separate perpetual lineage helpers with a Gate.io recent-smoke track."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import pandas as pd

from core.config import load_config
from core.io_atomic import atomic_write_json, atomic_write_text
from core.logging import get_logger
from data.ingest import _detect_delimiter
from data.schema import detect_timestamp_alias
from data.tail_refresh import (
    REQUIRED_TIMEFRAMES,
    TIMEFRAME_SECONDS,
    _build_exchange_client,
    _exchange_limit,
    _feature_contract_compatibility,
    _format_ts,
    _last_closed_candle_start,
    _retry_fetch_ohlcv,
    _to_canonical_ohlcv,
    _validate_timeframe_alignment,
)

LOGGER = get_logger(__name__)

SEPARATE_SOURCE_LINEAGE = "separate_parallel_lineage"

SEPARATE_LINEAGE_MODE = "separate_gateio_perpetual_lineage"
SEPARATE_LINEAGE_VERSION = "gateio_perpetual_parallel_lineage.v2"
RECENT_SMOKE_TRACK = "recent_window_smoke"
RECENT_SMOKE_REPORT_NAME = "recent_smoke_report.json"
PROCESSING_REPORT_NAME = "separate_parallel_lineage_processing_report.json"
DOWNLOAD_REPORT_NAME = "separate_parallel_lineage_download_report.json"
PROVIDER_STRATEGY_REPORT_NAME = "provider_strategy_report.json"

GATEIO_EXCHANGE_USED = "gateio"
GATEIO_MARKET_TYPE = "perpetual"
GATEIO_CCXT_MARKET_TYPE = "swap"
GATEIO_EXCHANGE_SYMBOL = "BTC_USDT"
GATEIO_CCXT_SYMBOL = "BTC/USDT:USDT"
GATEIO_SETTLE = "usdt"
GATEIO_MAX_FETCH_LIMIT = 1999
DEFAULT_RECENT_WINDOW_CANDLES = 1500
GATEIO_PROVIDER_CHOICE_REASON = (
    "Gate.io perpetual candles are restricted to a recent safe window; the provider must not be used for deep historical backfill."
)


@dataclass(frozen=True)
class GateioPerpetualLineageOptions:
    """Runtime options for the separate Gate.io perpetual recent-smoke lineage."""

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
    recent_window_candles: int = DEFAULT_RECENT_WINDOW_CANDLES
    log_level: str = "INFO"


@dataclass(frozen=True)
class LegacyReferencePoint:
    """Legacy timestamp anchor derived from the immutable root lineage."""

    timeframe: str
    legacy_file: Path
    timestamp_column: str
    legacy_last_timestamp: pd.Timestamp
    download_start_timestamp: pd.Timestamp


def run_separate_gateio_perpetual_lineage(options: GateioPerpetualLineageOptions) -> dict[str, Any]:
    """Run the Gate.io recent-window smoke track for the separate perpetual lineage."""

    if int(options.recent_window_candles) <= 0:
        raise ValueError("recent_window_candles must be a positive integer.")

    _ensure_track_session_is_clean(options.project_root, options.refresh_session_id, RECENT_SMOKE_TRACK)
    legacy_reference_files = _legacy_reference_files(options.legacy_input_root)
    report_root = _report_root(options)
    download_report_path = report_root / DOWNLOAD_REPORT_NAME
    recent_smoke_report_path = report_root / RECENT_SMOKE_REPORT_NAME
    processing_report_path = report_root / PROCESSING_REPORT_NAME
    provider_strategy_report_path = report_root / PROVIDER_STRATEGY_REPORT_NAME

    strategy_payload = build_provider_strategy_report(
        report_version=SEPARATE_LINEAGE_VERSION,
        mode=SEPARATE_LINEAGE_MODE,
        refresh_session_id=options.refresh_session_id,
        exchange_used=GATEIO_EXCHANGE_USED,
        market_type=GATEIO_MARKET_TYPE,
        provider_track=RECENT_SMOKE_TRACK,
        provider_choice_reason=GATEIO_PROVIDER_CHOICE_REASON,
        symbol_normalization={
            "exchange_symbol": GATEIO_EXCHANGE_SYMBOL,
            "ccxt_symbol": GATEIO_CCXT_SYMBOL,
            "settle": GATEIO_SETTLE,
        },
        legacy_reference_files=legacy_reference_files,
        rate_limit_strategy={
            "ccxt_enable_rate_limit": True,
            "page_size_limit": int(_exchange_limit(GATEIO_EXCHANGE_USED, min(options.request_limit, GATEIO_MAX_FETCH_LIMIT))),
            "max_retries": int(options.max_retries),
            "retry_backoff_seconds": float(options.retry_backoff_seconds),
        },
        window_strategy={
            "kind": "recent_closed_window",
            "recent_window_candles": int(options.recent_window_candles),
            "requires_fully_closed_candles": True,
        },
        pagination_strategy={
            "kind": "forward_since_limit_pagination",
            "page_size_limit": int(_exchange_limit(GATEIO_EXCHANGE_USED, min(options.request_limit, GATEIO_MAX_FETCH_LIMIT))),
        },
    )
    atomic_write_json(strategy_payload, provider_strategy_report_path)

    try:
        reference_points = extract_legacy_reference_points(options)
    except Exception as exc:  # noqa: BLE001
        payload = _base_download_report(
            options,
            legacy_reference_files=legacy_reference_files,
            exchange_used=GATEIO_EXCHANGE_USED,
            market_type=GATEIO_MARKET_TYPE,
            provider_track=RECENT_SMOKE_TRACK,
            provider_choice_reason=GATEIO_PROVIDER_CHOICE_REASON,
            report_version=SEPARATE_LINEAGE_VERSION,
            mode=SEPARATE_LINEAGE_MODE,
            symbol_normalization={
                "exchange_symbol": GATEIO_EXCHANGE_SYMBOL,
                "ccxt_symbol": GATEIO_CCXT_SYMBOL,
                "settle": GATEIO_SETTLE,
            },
            completion_scope="recent_smoke",
        )
        payload.update(
            {
                "status": "failed",
                "timeframes": [],
                "errors": [{"code": "LEGACY_REFERENCE_EXTRACTION_FAILED", "message": str(exc)}],
            }
        )
        _write_json_copies(payload, (download_report_path, recent_smoke_report_path))
        return {
            "status": "failed",
            "mode": SEPARATE_LINEAGE_MODE,
            "provider_track": RECENT_SMOKE_TRACK,
            "download_report_path": download_report_path,
            "recent_smoke_report_path": recent_smoke_report_path,
            "provider_strategy_report_path": provider_strategy_report_path,
            "processing_report_path": None,
        }

    try:
        download_result = _download_recent_smoke_lineage(options, reference_points)
    except Exception as exc:  # noqa: BLE001
        payload = _base_download_report(
            options,
            legacy_reference_files=legacy_reference_files,
            exchange_used=GATEIO_EXCHANGE_USED,
            market_type=GATEIO_MARKET_TYPE,
            provider_track=RECENT_SMOKE_TRACK,
            provider_choice_reason=GATEIO_PROVIDER_CHOICE_REASON,
            report_version=SEPARATE_LINEAGE_VERSION,
            mode=SEPARATE_LINEAGE_MODE,
            symbol_normalization={
                "exchange_symbol": GATEIO_EXCHANGE_SYMBOL,
                "ccxt_symbol": GATEIO_CCXT_SYMBOL,
                "settle": GATEIO_SETTLE,
            },
            completion_scope="recent_smoke",
        )
        payload.update(
            {
                "status": "failed",
                "timeframes": [
                    _reference_point_payload(
                        options,
                        point,
                        output_file=None,
                        rows_downloaded=0,
                        download_end_utc=None,
                        exchange_used=GATEIO_EXCHANGE_USED,
                        market_type=GATEIO_MARKET_TYPE,
                        provider_track=RECENT_SMOKE_TRACK,
                    )
                    for point in reference_points
                ],
                "errors": [{"code": "PERPETUAL_DOWNLOAD_FAILED", "message": str(exc)}],
            }
        )
        _write_json_copies(payload, (download_report_path, recent_smoke_report_path))
        return {
            "status": "failed",
            "mode": SEPARATE_LINEAGE_MODE,
            "provider_track": RECENT_SMOKE_TRACK,
            "download_report_path": download_report_path,
            "recent_smoke_report_path": recent_smoke_report_path,
            "provider_strategy_report_path": provider_strategy_report_path,
            "processing_report_path": None,
        }

    _write_json_copies(download_result["report"], (download_report_path, recent_smoke_report_path))
    processing_result = _run_processing_chain(options, raw_input_root=download_result["raw_input_root"])
    atomic_write_json(processing_result["report"], processing_report_path)
    return {
        "status": processing_result["status"],
        "mode": SEPARATE_LINEAGE_MODE,
        "provider_track": RECENT_SMOKE_TRACK,
        "download_report_path": download_report_path,
        "recent_smoke_report_path": recent_smoke_report_path,
        "provider_strategy_report_path": provider_strategy_report_path,
        "processing_report_path": processing_report_path,
        "raw_input_root": download_result["raw_input_root"],
    }


def extract_legacy_reference_points(options: GateioPerpetualLineageOptions) -> list[LegacyReferencePoint]:
    """Derive timeframe-specific anchor timestamps from immutable legacy CSVs."""

    cfg = load_config(options.data_config_path)
    reference_points: list[LegacyReferencePoint] = []
    for timeframe in REQUIRED_TIMEFRAMES:
        legacy_file = _legacy_file_path(options.legacy_input_root, timeframe)
        timestamp_column, legacy_last_timestamp = _extract_legacy_last_timestamp(legacy_file, cfg.timestamp_aliases)
        download_start_timestamp = legacy_last_timestamp + pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
        reference_points.append(
            LegacyReferencePoint(
                timeframe=timeframe,
                legacy_file=legacy_file,
                timestamp_column=timestamp_column,
                legacy_last_timestamp=legacy_last_timestamp,
                download_start_timestamp=download_start_timestamp,
            )
        )
    return reference_points


def compare_perpetual_candle_slice(
    legacy_rows: Sequence[Mapping[str, Any]],
    perpetual_rows: Sequence[Mapping[str, Any]],
    *,
    abs_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare two small OHLCV slices for focused perpetual-lineage forensics."""

    if abs_tolerance < 0.0:
        raise ValueError("abs_tolerance must be non-negative.")

    mismatches: list[dict[str, Any]] = []
    fields = ("open", "high", "low", "close", "volume")
    max_len = max(len(legacy_rows), len(perpetual_rows))
    for index in range(max_len):
        legacy_row = legacy_rows[index] if index < len(legacy_rows) else None
        perpetual_row = perpetual_rows[index] if index < len(perpetual_rows) else None
        if legacy_row is None or perpetual_row is None:
            mismatches.append(
                {
                    "row_index": index,
                    "reason": "ROW_COUNT_MISMATCH",
                    "legacy_present": legacy_row is not None,
                    "perpetual_present": perpetual_row is not None,
                }
            )
            continue

        row_mismatch: dict[str, Any] = {"row_index": index, "timestamp": str(legacy_row.get("timestamp")), "fields": {}}
        if str(legacy_row.get("timestamp")) != str(perpetual_row.get("timestamp")):
            row_mismatch["fields"]["timestamp"] = {"legacy": legacy_row.get("timestamp"), "perpetual": perpetual_row.get("timestamp")}
        for field in fields:
            lhs = legacy_row.get(field)
            rhs = perpetual_row.get(field)
            try:
                lhs_value = float(lhs)
                rhs_value = float(rhs)
            except (TypeError, ValueError):
                if lhs != rhs:
                    row_mismatch["fields"][field] = {"legacy": lhs, "perpetual": rhs}
                continue
            if abs(lhs_value - rhs_value) > abs_tolerance:
                row_mismatch["fields"][field] = {"legacy": lhs_value, "perpetual": rhs_value}
        if row_mismatch["fields"]:
            mismatches.append(row_mismatch)

    return {
        "status": "match" if not mismatches else "mismatch",
        "rows_compared": min(len(legacy_rows), len(perpetual_rows)),
        "abs_tolerance": float(abs_tolerance),
        "mismatches": mismatches,
    }


def build_provider_strategy_report(
    *,
    report_version: str,
    mode: str,
    refresh_session_id: str,
    exchange_used: str,
    market_type: str,
    provider_track: str,
    provider_choice_reason: str,
    symbol_normalization: Mapping[str, Any],
    legacy_reference_files: Sequence[str],
    rate_limit_strategy: Mapping[str, Any],
    window_strategy: Mapping[str, Any] | None,
    pagination_strategy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the machine-readable provider strategy report payload."""

    return {
        "report_version": report_version,
        "generated_at_utc": _utc_now().isoformat(),
        "mode": mode,
        "refresh_session_id": refresh_session_id,
        "exchange_used": exchange_used,
        "market_type": market_type,
        "source_lineage": SEPARATE_SOURCE_LINEAGE,
        "provider_track": provider_track,
        "provider_choice_reason": provider_choice_reason,
        "legacy_lineage_touched": False,
        "symbol_normalization": dict(symbol_normalization),
        "legacy_reference_files": list(legacy_reference_files),
        "rate_limit_strategy": dict(rate_limit_strategy),
        "window_strategy": dict(window_strategy) if window_strategy is not None else None,
        "pagination_strategy": dict(pagination_strategy) if pagination_strategy is not None else None,
    }


def _download_recent_smoke_lineage(
    options: GateioPerpetualLineageOptions,
    reference_points: Sequence[LegacyReferencePoint],
) -> dict[str, Any]:
    now_utc = _utc_now()
    raw_input_root = _raw_input_root(options, provider_track=RECENT_SMOKE_TRACK)
    output_root = raw_input_root / "gateio_perpetual"
    payloads: list[tuple[Path, str]] = []
    timeframe_reports: list[dict[str, Any]] = []

    client = _build_exchange_client(GATEIO_EXCHANGE_USED, GATEIO_CCXT_MARKET_TYPE)
    try:
        client.load_markets()
        for point in reference_points:
            last_closed_ts = _last_closed_candle_start(now_utc, point.timeframe)
            recent_start_ts = last_closed_ts - pd.Timedelta(
                seconds=TIMEFRAME_SECONDS[point.timeframe] * (int(options.recent_window_candles) - 1)
            )
            frame = _fetch_gateio_perpetual_timeframe(
                client,
                timeframe=point.timeframe,
                start_timestamp=recent_start_ts,
                end_timestamp=last_closed_ts,
                request_limit=options.request_limit,
                max_retries=options.max_retries,
                retry_backoff_seconds=options.retry_backoff_seconds,
            )

            if frame.empty:
                raise ValueError(f"Recent smoke returned no rows for {point.timeframe}.")

            alignment_ok, gap_count, alignment_error = _validate_timeframe_alignment(frame, point.timeframe)
            if not alignment_ok:
                raise ValueError(f"Invalid timeframe alignment for {point.timeframe}: {alignment_error} gap_count={gap_count}")

            first_ts = pd.Timestamp(frame["timestamp"].iloc[0])
            last_ts = pd.Timestamp(frame["timestamp"].iloc[-1])
            if first_ts != recent_start_ts:
                raise ValueError(
                    f"Recent smoke for {point.timeframe} did not start at the safe recent window boundary: "
                    f"expected={_format_ts(recent_start_ts)} observed={_format_ts(first_ts)}"
                )
            if last_ts != last_closed_ts:
                raise ValueError(
                    f"Recent smoke for {point.timeframe} did not reach the last closed candle: "
                    f"expected={_format_ts(last_closed_ts)} observed={_format_ts(last_ts)}"
                )

            expected_rows = int(options.recent_window_candles)
            if int(len(frame)) != expected_rows:
                raise ValueError(
                    f"Recent smoke row count mismatch for {point.timeframe}: expected_rows={expected_rows} observed_rows={len(frame)}"
                )

            output_file = output_root / point.legacy_file.name
            payloads.append((output_file, _render_raw_frame(frame)))
            timeframe_reports.append(
                _reference_point_payload(
                    options,
                    point,
                    output_file=output_file,
                    rows_downloaded=int(len(frame)),
                    download_end_utc=_format_ts(last_ts),
                    download_start_utc=_format_ts(recent_start_ts),
                    exchange_used=GATEIO_EXCHANGE_USED,
                    market_type=GATEIO_MARKET_TYPE,
                    provider_track=RECENT_SMOKE_TRACK,
                )
            )
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            LOGGER.info("Gate.io client close failed")

    for output_file, payload in payloads:
        atomic_write_text(payload, output_file)

    report = _base_download_report(
        options,
        legacy_reference_files=_legacy_reference_files(options.legacy_input_root),
        exchange_used=GATEIO_EXCHANGE_USED,
        market_type=GATEIO_MARKET_TYPE,
        provider_track=RECENT_SMOKE_TRACK,
        provider_choice_reason=GATEIO_PROVIDER_CHOICE_REASON,
        report_version=SEPARATE_LINEAGE_VERSION,
        mode=SEPARATE_LINEAGE_MODE,
        symbol_normalization={
            "exchange_symbol": GATEIO_EXCHANGE_SYMBOL,
            "ccxt_symbol": GATEIO_CCXT_SYMBOL,
            "settle": GATEIO_SETTLE,
        },
        completion_scope="recent_smoke",
        window_strategy={
            "kind": "recent_closed_window",
            "recent_window_candles": int(options.recent_window_candles),
            "requires_fully_closed_candles": True,
        },
        pagination_strategy={
            "kind": "forward_since_limit_pagination",
            "page_size_limit": int(_exchange_limit(GATEIO_EXCHANGE_USED, min(options.request_limit, GATEIO_MAX_FETCH_LIMIT))),
        },
    )
    report.update(
        {
            "status": "success",
            "raw_input_root": str(raw_input_root.resolve()),
            "timeframes": timeframe_reports,
            "errors": [],
        }
    )
    return {"status": "success", "raw_input_root": raw_input_root, "report": report}


def _fetch_gateio_perpetual_timeframe(
    client: Any,
    *,
    timeframe: str,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    request_limit: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    timeframe_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    next_since_ms = int(start_timestamp.timestamp() * 1000)
    end_ms = int(end_timestamp.timestamp() * 1000)
    limit = _exchange_limit(GATEIO_EXCHANGE_USED, min(request_limit, GATEIO_MAX_FETCH_LIMIT))
    frames: list[pd.DataFrame] = []

    while next_since_ms <= end_ms:
        rows = _retry_fetch_ohlcv(
            client,
            symbol=GATEIO_CCXT_SYMBOL,
            timeframe=timeframe,
            since_ms=next_since_ms,
            limit=limit,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        batch = _to_canonical_ohlcv(rows)
        batch = batch.loc[(batch["timestamp"] >= start_timestamp) & (batch["timestamp"] <= end_timestamp)].copy()
        if batch.empty:
            break
        frames.append(batch)
        last_batch_ts = pd.Timestamp(batch["timestamp"].iloc[-1])
        next_since_ms = int(last_batch_ts.timestamp() * 1000) + timeframe_ms
        if len(batch) < limit and last_batch_ts >= end_timestamp:
            break

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()


def _run_processing_chain(
    options: GateioPerpetualLineageOptions,
    *,
    raw_input_root: Path,
    report_version: str = SEPARATE_LINEAGE_VERSION,
    mode: str = SEPARATE_LINEAGE_MODE,
    exchange_used: str = GATEIO_EXCHANGE_USED,
    market_type: str = GATEIO_MARKET_TYPE,
    provider_track: str = RECENT_SMOKE_TRACK,
    provider_choice_reason: str = GATEIO_PROVIDER_CHOICE_REASON,
) -> dict[str, Any]:
    data_cfg = load_config(options.data_config_path)
    run_root = data_cfg.runs_root / options.refresh_session_id
    standardize_summary_path = run_root / "data_standardized" / "reports" / "summary.json"
    feature_summary_path = run_root / "data_features" / "reports" / "summary.json"
    feature_manifest_path = run_root / "data_features" / "reports" / "feature_manifest.json"
    accepted_feature_manifest_path = data_cfg.runs_root / options.accepted_run_id / "data_features" / "reports" / "feature_manifest.json"

    if standardize_summary_path.exists() or feature_summary_path.exists() or feature_manifest_path.exists():
        return {
            "status": "failed",
            "report": _processing_report(
                options,
                raw_input_root=raw_input_root,
                stages=[],
                feature_contract_compatibility={"status": "not_started", "checks": {}},
                errors=[
                    {
                        "code": "PROCESSING_OUTPUT_ALREADY_EXISTS",
                        "message": "Processing outputs already exist for this refresh_session_id; use a fresh session id per provider track.",
                    }
                ],
                report_version=report_version,
                mode=mode,
                exchange_used=exchange_used,
                market_type=market_type,
                provider_track=provider_track,
                provider_choice_reason=provider_choice_reason,
            ),
        }

    standardize_command = [
        str(options.python_executable),
        "scripts/make_standardized_data.py",
        "--config",
        str(options.data_config_path.resolve()),
        "--run-id",
        options.refresh_session_id,
        "--input-root",
        str(raw_input_root.resolve()),
        "--log-level",
        options.log_level,
    ]
    feature_input_root = run_root / "data_standardized" / "parquet"
    feature_command = [
        str(options.python_executable),
        "scripts/make_features.py",
        "--config",
        str(options.features_config_path.resolve()),
        "--run-id",
        options.refresh_session_id,
        "--input-root",
        str(feature_input_root.resolve()),
        "--strict-parity",
        "true",
        "--log-level",
        options.log_level,
    ]

    stages: list[dict[str, Any]] = []

    standardize_result = _run_subprocess(standardize_command, cwd=options.project_root)
    stages.append(_stage_payload("standardize", standardize_command, raw_input_root, standardize_summary_path, standardize_result))
    if standardize_result.returncode != 0 or not standardize_summary_path.exists():
        return {
            "status": "failed",
            "report": _processing_report(
                options,
                raw_input_root=raw_input_root,
                stages=stages,
                feature_contract_compatibility={"status": "not_started", "checks": {}},
                errors=[{"code": "STANDARDIZATION_FAILED", "message": "Standardization failed for the new perpetual lineage."}],
                report_version=report_version,
                mode=mode,
                exchange_used=exchange_used,
                market_type=market_type,
                provider_track=provider_track,
                provider_choice_reason=provider_choice_reason,
            ),
        }

    feature_result = _run_subprocess(feature_command, cwd=options.project_root)
    stages.append(_stage_payload("feature_build", feature_command, feature_input_root, feature_summary_path, feature_result))
    if feature_result.returncode != 0 or not feature_summary_path.exists() or not feature_manifest_path.exists():
        return {
            "status": "failed",
            "report": _processing_report(
                options,
                raw_input_root=raw_input_root,
                stages=stages,
                feature_contract_compatibility={"status": "not_started", "checks": {}},
                errors=[{"code": "FEATURE_BUILD_FAILED", "message": "Feature build failed for the new perpetual lineage."}],
                report_version=report_version,
                mode=mode,
                exchange_used=exchange_used,
                market_type=market_type,
                provider_track=provider_track,
                provider_choice_reason=provider_choice_reason,
            ),
        }

    if not accepted_feature_manifest_path.exists():
        return {
            "status": "failed",
            "report": _processing_report(
                options,
                raw_input_root=raw_input_root,
                stages=stages,
                feature_contract_compatibility={"status": "failed", "checks": {}},
                errors=[{"code": "ACCEPTED_FEATURE_MANIFEST_MISSING", "message": f"Missing accepted feature manifest: {accepted_feature_manifest_path}"}],
                report_version=report_version,
                mode=mode,
                exchange_used=exchange_used,
                market_type=market_type,
                provider_track=provider_track,
                provider_choice_reason=provider_choice_reason,
            ),
        }

    feature_contract_compatibility = _feature_contract_compatibility(accepted_feature_manifest_path, feature_manifest_path)
    status = "success" if feature_contract_compatibility.get("status") == "success" else "failed"
    errors = [] if status == "success" else [{"code": "FEATURE_CONTRACT_MISMATCH", "message": "Feature manifest is not contract-compatible with the accepted lineage."}]
    return {
        "status": status,
        "report": _processing_report(
            options,
            raw_input_root=raw_input_root,
            stages=stages,
            feature_contract_compatibility=feature_contract_compatibility,
            errors=errors,
            report_version=report_version,
            mode=mode,
            exchange_used=exchange_used,
            market_type=market_type,
            provider_track=provider_track,
            provider_choice_reason=provider_choice_reason,
        ),
    }


def _run_subprocess(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)


def _stage_payload(
    stage_name: str,
    command: Sequence[str],
    input_root: Path,
    report_path: Path,
    result: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    return {
        "stage": stage_name,
        "status": "success" if result.returncode == 0 else "failed",
        "command": list(command),
        "input_root": str(input_root.resolve()),
        "report_path": str(report_path.resolve()),
        "exit_code": int(result.returncode),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _processing_report(
    options: GateioPerpetualLineageOptions,
    *,
    raw_input_root: Path,
    stages: Sequence[Mapping[str, Any]],
    feature_contract_compatibility: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
    report_version: str,
    mode: str,
    exchange_used: str,
    market_type: str,
    provider_track: str,
    provider_choice_reason: str,
) -> dict[str, Any]:
    return {
        "report_version": report_version,
        "generated_at_utc": _utc_now().isoformat(),
        "mode": mode,
        "status": "success" if not errors else "failed",
        "refresh_session_id": options.refresh_session_id,
        "exchange_used": exchange_used,
        "market_type": market_type,
        "source_lineage": SEPARATE_SOURCE_LINEAGE,
        "provider_track": provider_track,
        "provider_choice_reason": provider_choice_reason,
        "legacy_lineage_touched": False,
        "raw_input_root": str(raw_input_root.resolve()),
        "feature_contract_compatibility": dict(feature_contract_compatibility),
        "stages": [dict(item) for item in stages],
        "errors": [dict(item) for item in errors],
    }


def _base_download_report(
    options: GateioPerpetualLineageOptions | Any,
    *,
    legacy_reference_files: Sequence[str],
    exchange_used: str,
    market_type: str,
    provider_track: str,
    provider_choice_reason: str,
    report_version: str,
    mode: str,
    symbol_normalization: Mapping[str, Any],
    completion_scope: str,
    window_strategy: Mapping[str, Any] | None = None,
    pagination_strategy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "report_version": report_version,
        "generated_at_utc": _utc_now().isoformat(),
        "mode": mode,
        "refresh_session_id": options.refresh_session_id,
        "exchange_used": exchange_used,
        "market_type": market_type,
        "source_lineage": SEPARATE_SOURCE_LINEAGE,
        "provider_track": provider_track,
        "provider_choice_reason": provider_choice_reason,
        "legacy_lineage_touched": False,
        "completion_scope": completion_scope,
        "symbol_normalization": dict(symbol_normalization),
        "legacy_reference_files": list(legacy_reference_files),
        "window_strategy": dict(window_strategy) if window_strategy is not None else None,
        "pagination_strategy": dict(pagination_strategy) if pagination_strategy is not None else None,
    }


def _reference_point_payload(
    options: GateioPerpetualLineageOptions | Any,
    point: LegacyReferencePoint,
    *,
    output_file: Path | None,
    rows_downloaded: int,
    download_end_utc: str | None,
    download_start_utc: str | None = None,
    exchange_used: str,
    market_type: str,
    provider_track: str,
) -> dict[str, Any]:
    legacy_reference_files = _legacy_reference_files(options.legacy_input_root)
    return {
        "refresh_session_id": options.refresh_session_id,
        "exchange_used": exchange_used,
        "market_type": market_type,
        "source_lineage": SEPARATE_SOURCE_LINEAGE,
        "provider_track": provider_track,
        "legacy_lineage_touched": False,
        "legacy_reference_files": legacy_reference_files,
        "timeframe": point.timeframe,
        "legacy_reference_file": str(point.legacy_file.resolve()),
        "legacy_last_timestamp_utc": _format_ts(point.legacy_last_timestamp),
        "download_start_utc": download_start_utc or _format_ts(point.download_start_timestamp),
        "download_end_utc": download_end_utc,
        "rows_downloaded": int(rows_downloaded),
        "output_file": str(output_file.resolve()) if output_file is not None else None,
    }


def _extract_legacy_last_timestamp(path: Path, timestamp_aliases: Sequence[str]) -> tuple[str, pd.Timestamp]:
    if not path.exists():
        raise FileNotFoundError(f"Missing legacy reference file: {path}")
    if not path.is_file():
        raise ValueError(f"Legacy reference path is not a file: {path}")

    delimiter = _detect_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline().strip()
    if not header_line:
        raise ValueError(f"Legacy reference file is empty: {path}")

    header = next(csv.reader([header_line], delimiter=delimiter))
    detection = detect_timestamp_alias(header, timestamp_aliases)
    last_line = _read_last_non_empty_line(path)
    values = next(csv.reader([last_line], delimiter=delimiter))
    if len(values) != len(header):
        raise ValueError(f"Legacy reference last row does not match header width: {path}")

    timestamp_index = header.index(detection.selected_column)
    parsed = pd.to_datetime(values[timestamp_index], utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unreadable legacy timestamp in {path}: {values[timestamp_index]}")
    return detection.selected_column, pd.Timestamp(parsed)


def _read_last_non_empty_line(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        if position == 0:
            raise ValueError(f"Legacy reference file is empty: {path}")

        buffer = b""
        while position > 0:
            read_size = min(4096, position)
            position -= read_size
            handle.seek(position)
            buffer = handle.read(read_size) + buffer
            lines = buffer.splitlines()
            if position > 0 and lines:
                candidate_lines = lines[1:]
                buffer = lines[0]
            else:
                candidate_lines = lines
                buffer = b""
            non_empty = [line for line in candidate_lines if line.strip()]
            if non_empty:
                return non_empty[-1].decode("utf-8")

        if buffer.strip():
            return buffer.decode("utf-8")
    raise ValueError(f"Legacy reference file has no data rows: {path}")


def _render_raw_frame(frame: pd.DataFrame) -> str:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    return out.to_csv(index=False, lineterminator="\n")


def _legacy_file_path(legacy_input_root: Path, timeframe: str) -> Path:
    return legacy_input_root / f"BTC_USDT_{timeframe}_price_data.csv"


def _legacy_reference_files(legacy_input_root: Path) -> list[str]:
    return [str(_legacy_file_path(legacy_input_root, timeframe).resolve()) for timeframe in REQUIRED_TIMEFRAMES]


def _report_root(options: GateioPerpetualLineageOptions | Any) -> Path:
    return options.project_root / "runs" / options.refresh_session_id / "data_tail_refresh" / "reports"


def _raw_input_root(options: GateioPerpetualLineageOptions | Any, *, provider_track: str) -> Path:
    return options.project_root / "runs" / options.refresh_session_id / "data_tail_refresh" / SEPARATE_SOURCE_LINEAGE / "raw" / provider_track


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_track_session_is_clean(project_root: Path, refresh_session_id: str, provider_track: str) -> None:
    raw_root = project_root / "runs" / refresh_session_id / "data_tail_refresh" / SEPARATE_SOURCE_LINEAGE / "raw"
    if not raw_root.exists():
        return
    other_tracks = sorted(path.name for path in raw_root.iterdir() if path.is_dir() and path.name != provider_track)
    if other_tracks:
        raise RuntimeError(
            f"refresh_session_id already contains a different separate lineage track: {', '.join(other_tracks)}"
        )


def _write_json_copies(payload: Mapping[str, Any], destinations: Sequence[Path]) -> None:
    for destination in destinations:
        atomic_write_json(dict(payload), destination)
