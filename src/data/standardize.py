"""End-to-end file standardization pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import PipelineConfig
from core.health import FileHealthReport, HealthIssue, health_check, write_health_json_atomic
from core.io_atomic import atomic_write_parquet
from core.logging import get_logger
from core.paths import build_output_parquet_path, build_report_path, discover_csv_files
from data.ingest import parse_timestamp_utc, read_csv_ohlcv
from data.schema import detect_timestamp_alias, enforce_schema, map_to_canonical, normalize_column_names

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):  # type: ignore[no-redef]
        return iterable


LOGGER = get_logger(__name__)


def _add_issue(report: FileHealthReport, code: str, severity: str, message: str, **context: object) -> None:
    report.issues.append(HealthIssue(code=code, severity=severity, message=message, context=dict(context)))


def _nan_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def standardize_file(src_csv: Path, cfg: PipelineConfig, dry_run: bool = False) -> FileHealthReport:
    """Standardize one CSV file and produce a health report."""
    output_path = build_output_parquet_path(src_csv=src_csv, input_root=cfg.input_root, output_root=cfg.output_root)
    report_path = build_report_path(src_csv=src_csv, input_root=cfg.input_root, reports_root=cfg.reports_root)

    report = FileHealthReport(
        source_file=str(src_csv),
        output_file=None,
        row_count_in=0,
        row_count_out=0,
        issues=[],
        success=False,
    )
    standardized_df: pd.DataFrame | None = None

    try:
        raw_df = read_csv_ohlcv(src_csv)
        report.row_count_in = int(len(raw_df))
        if raw_df.empty:
            _add_issue(report, "EMPTY_INPUT", "critical", "Input CSV has no rows.")
        else:
            normalized_columns = normalize_column_names(list(raw_df.columns))
            if len(set(normalized_columns)) != len(normalized_columns):
                _add_issue(report, "COLUMN_COLLISION", "critical", "Normalized columns are not unique.")
            raw_df.columns = normalized_columns

            detection = detect_timestamp_alias(raw_df.columns, cfg.timestamp_aliases)
            LOGGER.info(
                "Timestamp alias selected | file=%s alias=%s column=%s confidence=%.2f matched=%s",
                src_csv,
                detection.selected_alias,
                detection.selected_column,
                detection.confidence,
                ",".join(detection.matched_aliases),
            )
            if detection.selected_column != "timestamp":
                _add_issue(
                    report,
                    "TIMESTAMP_ALIAS_MAPPED",
                    "warning",
                    "Timestamp column mapped from alias.",
                    selected_column=detection.selected_column,
                    selected_alias=detection.selected_alias,
                    confidence=detection.confidence,
                )

            canonical_df = map_to_canonical(raw_df, timestamp_col=detection.selected_column)
            parsed_df = parse_timestamp_utc(canonical_df, "timestamp")

            invalid_timestamps = int(parsed_df["timestamp"].isna().sum())
            if invalid_timestamps > 0:
                _add_issue(
                    report,
                    "TIMESTAMP_PARSE_FAILED",
                    "critical",
                    "Failed to parse some timestamps.",
                    invalid_count=invalid_timestamps,
                )

            if not parsed_df["timestamp"].is_monotonic_increasing:
                _add_issue(
                    report,
                    "TIMESTAMP_SORTED",
                    "warning",
                    "Timestamps were not monotonic; output will be sorted.",
                )

            duplicate_count = int(parsed_df.duplicated(subset=["timestamp"]).sum())
            if duplicate_count > 0:
                _add_issue(
                    report,
                    "DUPLICATE_TIMESTAMP_DROPPED",
                    "warning",
                    "Duplicate timestamps detected and will be dropped with keep='last'.",
                    duplicate_count=duplicate_count,
                )

            for col in cfg.float_columns:
                if col in parsed_df.columns:
                    LOGGER.info("Coercing numeric column for NaN analysis | file=%s column=%s", src_csv, col)
                    coerced = pd.to_numeric(parsed_df[col], errors="coerce")
                    ratio = _nan_ratio(coerced)
                    if ratio > 0.005:
                        _add_issue(
                            report,
                            "HIGH_NAN_RATIO",
                            "warning",
                            f"NaN ratio is above threshold for {col}.",
                            column=col,
                            nan_ratio=ratio,
                        )

            if health_check(report, fail_on_critical=cfg.fail_on_critical):
                try:
                    standardized_df = enforce_schema(parsed_df)
                except ValueError as exc:
                    _add_issue(
                        report,
                        "SCHEMA_ENFORCEMENT_FAILED",
                        "critical",
                        "Schema enforcement failed.",
                        error=str(exc),
                    )
                else:
                    report.row_count_out = int(len(standardized_df))
                    if standardized_df.empty:
                        _add_issue(report, "EMPTY_OUTPUT", "critical", "Output became empty after standardization.")
            else:
                _add_issue(
                    report,
                    "WRITE_BLOCKED_BY_HEALTH_GATE",
                    "critical",
                    "Health gate blocked parquet write.",
                )

    except Exception as exc:
        _add_issue(
            report,
            "STANDARDIZATION_ERROR",
            "critical",
            "Unhandled exception during standardization.",
            error=str(exc),
        )

    report.success = health_check(report, fail_on_critical=cfg.fail_on_critical)

    if report.success and standardized_df is not None and not dry_run:
        try:
            atomic_write_parquet(standardized_df, output_path)
            report.output_file = str(output_path)
        except Exception as exc:
            _add_issue(report, "WRITE_FAILED", "critical", "Failed to write parquet output.", error=str(exc))
            report.success = False

    if not dry_run:
        write_health_json_atomic(report.to_dict(), report_path)

    return report


def standardize_all(cfg: PipelineConfig, dry_run: bool = False) -> list[FileHealthReport]:
    """Standardize all discovered CSV files."""
    csv_files = discover_csv_files(cfg.input_root, cfg.csv_glob)
    LOGGER.info("Discovered %d CSV file(s).", len(csv_files))

    reports: list[FileHealthReport] = []
    for src_csv in tqdm(csv_files, desc="Standardizing CSV"):
        report = standardize_file(src_csv=src_csv, cfg=cfg, dry_run=dry_run)
        reports.append(report)
        LOGGER.info("Processed %s | success=%s", src_csv, report.success)

    return reports
