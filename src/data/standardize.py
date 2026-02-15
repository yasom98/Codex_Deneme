"""End-to-end standardization engine with strict QC gates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import PipelineConfig
from core.health import FileHealthReport, add_error, finalize_report
from core.io_atomic import atomic_write_json, atomic_write_parquet
from core.logging import get_logger
from core.paths import build_output_parquet_path, build_report_path, discover_csv_files
from data.ingest import read_csv_ohlcv
from data.schema import (
    NUMERIC_COLUMNS,
    detect_timestamp_alias,
    final_gate_checks,
    map_to_canonical,
    normalize_column_names,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):  # type: ignore[no-redef]
        return iterable


LOGGER = get_logger(__name__)


def _parse_and_drop_invalid_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Parse timestamp as UTC and drop invalid timestamp rows."""
    out = df.copy()
    parsed = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    invalid_mask = parsed.isna()
    dropped = int(invalid_mask.sum())
    out = out.loc[~invalid_mask].copy()
    out["timestamp"] = parsed.loc[out.index]
    return out, dropped


def _drop_duplicate_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate timestamps keeping the last occurrence."""
    out = df.sort_values("timestamp", kind="mergesort")
    duplicated_mask = out.duplicated(subset=["timestamp"], keep="last")
    dropped = int(duplicated_mask.sum())
    out = out.loc[~duplicated_mask].copy()
    return out, dropped


def _convert_numeric_and_drop_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Convert OHLCV numeric columns and drop rows with conversion failures."""
    out = df.copy()
    converted: dict[str, pd.Series] = {}
    invalid_mask = pd.Series(False, index=out.index)

    for col in NUMERIC_COLUMNS:
        LOGGER.info("Converting numeric column | column=%s", col)
        series = pd.to_numeric(out[col], errors="coerce")
        converted[col] = series
        invalid_mask |= series.isna()

    dropped = int(invalid_mask.sum())
    out = out.loc[~invalid_mask].copy()

    for col in NUMERIC_COLUMNS:
        out[col] = converted[col].loc[out.index].astype("float32")

    return out, dropped


def standardize_file(
    src_csv: Path,
    cfg: PipelineConfig,
    parquet_root: Path,
    per_file_reports_root: Path,
    dry_run: bool = False,
) -> FileHealthReport:
    """Standardize one CSV file and return its health report."""
    report = FileHealthReport(input_file=str(src_csv))
    output_path = build_output_parquet_path(src_csv=src_csv, input_root=cfg.input_root, output_root=parquet_root)
    report_path = build_report_path(src_csv=src_csv, input_root=cfg.input_root, reports_root=per_file_reports_root)

    try:
        raw_df = read_csv_ohlcv(src_csv)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        add_error(report, stage="ingest", code="CSV_READ_FAILED", message=str(exc))
        finalize_report(report)
        if not dry_run:
            atomic_write_json(report.to_dict(), report_path)
        return report

    report.rows_in = int(len(raw_df))
    if raw_df.empty:
        add_error(report, stage="ingest", code="EMPTY_INPUT", message="Input CSV is empty.")
        finalize_report(report)
        if not dry_run:
            atomic_write_json(report.to_dict(), report_path)
        return report

    normalized_columns = normalize_column_names(list(raw_df.columns))
    if len(set(normalized_columns)) != len(normalized_columns):
        add_error(
            report,
            stage="normalize",
            code="COLUMN_COLLISION",
            message="Column collision after normalization.",
            normalized_columns=normalized_columns,
        )
        finalize_report(report)
        if not dry_run:
            atomic_write_json(report.to_dict(), report_path)
        return report
    raw_df.columns = normalized_columns

    try:
        detection = detect_timestamp_alias(raw_df.columns, cfg.timestamp_aliases)
    except ValueError as exc:
        add_error(report, stage="timestamp_alias", code="TIMESTAMP_ALIAS_NOT_FOUND", message=str(exc))
        finalize_report(report)
        if not dry_run:
            atomic_write_json(report.to_dict(), report_path)
        return report

    report.timestamp_alias = detection.selected_alias
    report.alias_confidence = detection.confidence
    LOGGER.info(
        "Timestamp alias selected | file=%s alias=%s column=%s confidence=%.2f",
        src_csv,
        detection.selected_alias,
        detection.selected_column,
        detection.confidence,
    )

    try:
        canonical_df = map_to_canonical(raw_df, timestamp_col=detection.selected_column)
    except ValueError as exc:
        add_error(report, stage="schema_map", code="CANONICAL_MAPPING_FAILED", message=str(exc))
        finalize_report(report)
        if not dry_run:
            atomic_write_json(report.to_dict(), report_path)
        return report

    ts_cleaned, dropped_invalid_ts = _parse_and_drop_invalid_timestamps(canonical_df)
    report.dropped_invalid_ts = dropped_invalid_ts
    if dropped_invalid_ts > 0:
        LOGGER.info("Dropped invalid timestamps | file=%s dropped=%d", src_csv, dropped_invalid_ts)

    deduped, dropped_duplicates = _drop_duplicate_timestamps(ts_cleaned)
    report.dropped_duplicates = dropped_duplicates
    if dropped_duplicates > 0:
        LOGGER.info("Dropped duplicate timestamps | file=%s dropped=%d", src_csv, dropped_duplicates)

    numeric_cleaned, dropped_invalid_numeric = _convert_numeric_and_drop_invalid(deduped)
    report.dropped_invalid_numeric = dropped_invalid_numeric
    if dropped_invalid_numeric > 0:
        LOGGER.info("Dropped invalid numeric rows | file=%s dropped=%d", src_csv, dropped_invalid_numeric)

    report.rows_out = int(len(numeric_cleaned))
    gates = final_gate_checks(numeric_cleaned)
    report.schema_ok = gates["schema_ok"]
    report.monotonic_ok = gates["monotonic_ok"]
    report.unique_ts_ok = gates["unique_ts_ok"]
    report.dtype_ok = gates["dtype_ok"]
    report.no_nan_ok = gates["no_nan_ok"]

    if report.rows_out == 0:
        add_error(report, stage="gates", code="EMPTY_OUTPUT", message="Output dataframe has no rows after cleaning.")
    if not report.schema_ok:
        add_error(report, stage="gates", code="SCHEMA_INVALID", message="Required canonical columns are missing.")
    if not report.monotonic_ok:
        add_error(report, stage="gates", code="TIMESTAMP_NOT_MONOTONIC", message="Timestamp is not monotonic increasing.")
    if not report.unique_ts_ok:
        add_error(report, stage="gates", code="TIMESTAMP_NOT_UNIQUE", message="Timestamp is not unique.")
    if not report.dtype_ok:
        add_error(report, stage="gates", code="DTYPE_INVALID", message="OHLCV columns are not float32.")
    if not report.no_nan_ok:
        add_error(report, stage="gates", code="NAN_IN_REQUIRED", message="NaN exists in canonical required columns.")

    finalize_report(report)

    if report.status == "success" and not dry_run:
        try:
            atomic_write_parquet(numeric_cleaned, output_path)
            report.output_file = str(output_path)
        except RuntimeError as exc:
            add_error(report, stage="write", code="PARQUET_WRITE_FAILED", message=str(exc))
            finalize_report(report)

    if not dry_run:
        atomic_write_json(report.to_dict(), report_path)

    return report


def standardize_all(
    cfg: PipelineConfig,
    parquet_root: Path,
    per_file_reports_root: Path,
    dry_run: bool = False,
) -> list[FileHealthReport]:
    """Run standardization for all discovered CSV files."""
    csv_files = discover_csv_files(cfg.input_root, cfg.csv_glob)
    LOGGER.info("Discovered CSV files | count=%d", len(csv_files))

    reports: list[FileHealthReport] = []
    for src_csv in tqdm(csv_files, desc="Standardizing CSV"):
        report = standardize_file(
            src_csv=src_csv,
            cfg=cfg,
            parquet_root=parquet_root,
            per_file_reports_root=per_file_reports_root,
            dry_run=dry_run,
        )
        reports.append(report)
        LOGGER.info("Processed file | file=%s status=%s rows_in=%d rows_out=%d", src_csv, report.status, report.rows_in, report.rows_out)

    return reports
