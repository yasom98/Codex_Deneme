"""End-to-end standardization engine with strict QC gates."""

from __future__ import annotations

from dataclasses import dataclass
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


_TIMESTAMP_STRATEGY_ORDER: tuple[str, ...] = (
    "epoch_milliseconds",
    "epoch_seconds",
    "epoch_microseconds",
    "excel_serial",
    "generic_datetime",
)
_MIN_PLAUSIBLE_YEAR = 2010
_MAX_PLAUSIBLE_YEAR = 2100
_LARGE_FILE_MIN_ROWS = 500


@dataclass(frozen=True)
class TimestampParseResult:
    """Selected timestamp parsing strategy and quality metrics."""

    strategy: str
    parsed: pd.Series
    parse_valid_ratio: float
    plausible_year_ratio: float
    unique_days: int
    timestamp_min: pd.Timestamp | None
    timestamp_max: pd.Timestamp | None

    @property
    def score(self) -> tuple[float, float, int]:
        """Score tuple used for strategy comparison."""
        return (self.parse_valid_ratio, self.plausible_year_ratio, self.unique_days)


def _parse_with_strategy(series: pd.Series, strategy: str) -> pd.Series:
    """Parse timestamp series with a named strategy."""
    if strategy == "epoch_milliseconds":
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    if strategy == "epoch_seconds":
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    if strategy == "epoch_microseconds":
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(numeric, unit="us", utc=True, errors="coerce")
    if strategy == "excel_serial":
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", utc=True, errors="coerce")
    if strategy == "generic_datetime":
        return pd.to_datetime(series, utc=True, errors="coerce")
    raise ValueError(f"Unknown timestamp strategy: {strategy}")


def _evaluate_parse_quality(
    parsed: pd.Series,
    *,
    min_year: int = _MIN_PLAUSIBLE_YEAR,
    max_year: int = _MAX_PLAUSIBLE_YEAR,
) -> tuple[float, float, int, pd.Timestamp | None, pd.Timestamp | None]:
    """Return quality metrics for parsed timestamps."""
    total = int(len(parsed))
    if total == 0:
        return 0.0, 0.0, 0, None, None

    valid = parsed.dropna()
    valid_count = int(len(valid))
    if valid_count == 0:
        return 0.0, 0.0, 0, None, None

    parse_valid_ratio = valid_count / total
    years = valid.dt.year
    plausible_year_ratio = float(years.between(min_year, max_year, inclusive="both").mean())
    unique_days = int(valid.dt.floor("D").nunique())
    return parse_valid_ratio, plausible_year_ratio, unique_days, valid.min(), valid.max()


def _select_timestamp_strategy(series: pd.Series) -> TimestampParseResult:
    """Select best timestamp parsing strategy by quality score."""
    best_result: TimestampParseResult | None = None

    for strategy in _TIMESTAMP_STRATEGY_ORDER:
        parsed = _parse_with_strategy(series, strategy=strategy)
        parse_valid_ratio, plausible_year_ratio, unique_days, timestamp_min, timestamp_max = _evaluate_parse_quality(parsed)
        candidate = TimestampParseResult(
            strategy=strategy,
            parsed=parsed,
            parse_valid_ratio=parse_valid_ratio,
            plausible_year_ratio=plausible_year_ratio,
            unique_days=unique_days,
            timestamp_min=timestamp_min,
            timestamp_max=timestamp_max,
        )
        if best_result is None or candidate.score > best_result.score:
            best_result = candidate

    if best_result is None:
        raise RuntimeError("Timestamp strategy selection produced no candidates.")
    return best_result


def _is_implausible_year_range(
    parse_result: TimestampParseResult,
    *,
    min_year: int = _MIN_PLAUSIBLE_YEAR,
    max_year: int = _MAX_PLAUSIBLE_YEAR,
) -> bool:
    """Return True when parsed timestamp range falls outside plausible years."""
    if parse_result.timestamp_min is None or parse_result.timestamp_max is None:
        return True

    min_ts = parse_result.timestamp_min
    max_ts = parse_result.timestamp_max
    return bool(min_ts.year < min_year or max_ts.year > max_year)


def _parse_and_drop_invalid_timestamps(df: pd.DataFrame, parse_result: TimestampParseResult) -> tuple[pd.DataFrame, int]:
    """Apply selected parser output and drop invalid timestamp rows."""
    out = df.copy()
    parsed = parse_result.parsed
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

    parse_result = _select_timestamp_strategy(canonical_df["timestamp"])
    report.selected_timestamp_strategy = parse_result.strategy
    report.parse_valid_ratio = parse_result.parse_valid_ratio
    report.timestamp_min = parse_result.timestamp_min.isoformat() if parse_result.timestamp_min is not None else None
    report.timestamp_max = parse_result.timestamp_max.isoformat() if parse_result.timestamp_max is not None else None
    report.unique_days = parse_result.unique_days

    LOGGER.info(
        "Timestamp strategy selected | file=%s strategy=%s valid_ratio=%.4f plausible_ratio=%.4f unique_days=%d min=%s max=%s",
        src_csv,
        parse_result.strategy,
        parse_result.parse_valid_ratio,
        parse_result.plausible_year_ratio,
        parse_result.unique_days,
        report.timestamp_min,
        report.timestamp_max,
    )

    ts_cleaned, dropped_invalid_ts = _parse_and_drop_invalid_timestamps(canonical_df, parse_result=parse_result)
    report.dropped_invalid_ts = dropped_invalid_ts
    if dropped_invalid_ts > 0:
        LOGGER.info("Dropped invalid timestamps | file=%s dropped=%d", src_csv, dropped_invalid_ts)

    is_large_file = report.rows_in >= _LARGE_FILE_MIN_ROWS
    has_implausible_year_range = _is_implausible_year_range(parse_result)
    if (is_large_file and parse_result.unique_days < 2) or has_implausible_year_range:
        add_error(
            report,
            stage="timestamp_parse",
            code="TIMESTAMP_PARSE_INVALID",
            message="Timestamp parsing failed integrity gates.",
            selected_timestamp_strategy=parse_result.strategy,
            parse_valid_ratio=parse_result.parse_valid_ratio,
            timestamp_min=report.timestamp_min,
            timestamp_max=report.timestamp_max,
            unique_days=parse_result.unique_days,
            is_large_file=is_large_file,
            plausible_year_range=f"{_MIN_PLAUSIBLE_YEAR}-{_MAX_PLAUSIBLE_YEAR}",
        )
        finalize_report(report)
        if not dry_run:
            atomic_write_json(report.to_dict(), report_path)
        return report

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
