"""CSV schema and timestamp integrity inspection utilities."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.io_atomic import atomic_write_json
from core.logging import get_logger
from core.paths import discover_csv_files
from data.schema import normalize_column_names

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


LOGGER = get_logger(__name__)

_TIMESTAMP_ALIASES: tuple[str, ...] = ("timestamp", "ts", "time", "date")
_TIMESTAMP_STRATEGY_ORDER: tuple[str, ...] = (
    "epoch_milliseconds",
    "epoch_seconds",
    "epoch_microseconds",
    "excel_serial",
    "generic_datetime",
)
_MIN_PLAUSIBLE_YEAR = 2010
_MAX_PLAUSIBLE_YEAR = 2100
_FULL_PASS_MAX_BYTES = 1_000_000_000
_CHUNK_SIZE = 200_000

_OHLCV_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "price_close", "close_price", "last", "price"),
    "volume": ("volume", "vol", "v", "base_volume", "quote_volume", "qty", "quantity", "amount"),
}


@dataclass(frozen=True)
class TimestampParseResult:
    """Selected timestamp parsing strategy and sample quality metrics."""

    strategy: str
    parse_valid_ratio: float
    plausible_year_ratio: float
    min_ts: pd.Timestamp | None
    max_ts: pd.Timestamp | None
    unique_days_sample: int
    parsed_sample: pd.Series

    @property
    def score(self) -> tuple[float, float, int]:
        """Score tuple for strategy comparison."""
        return (self.parse_valid_ratio, self.plausible_year_ratio, self.unique_days_sample)


@dataclass(frozen=True)
class CsvInspectionResult:
    """Inspection output for one CSV file."""

    file: str
    delimiter: str
    encoding: str
    columns_exact: list[str]
    duplicate_header_names: list[str]
    timestamp_alias: str | None
    timestamp_column: str | None
    ohlcv_mapping: dict[str, str | None]
    indicator_columns: list[str]
    rows_sampled: int
    parse_valid_ratio: float
    min_ts: str | None
    max_ts: str | None
    unique_days_sample: int
    suspicious_flags: list[str]
    failure_reason_codes: list[str]
    parse_scope: str

    @property
    def passed(self) -> bool:
        """Return True when no failure reason codes are present."""
        return len(self.failure_reason_codes) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize result as dict."""
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class CsvInspectionRunReport:
    """JSON report payload for full inspection run."""

    run_id: str
    generated_at_utc: str
    input_root: str
    total_files: int
    passed_files: int
    failed_files: int
    inspections: list[dict[str, Any]]


def _detect_encoding(path: Path) -> str:
    """Detect encoding with a conservative fallback chain."""
    raw = path.read_bytes()[:65536]
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _detect_delimiter(path: Path, encoding: str) -> str:
    """Detect delimiter from file sample."""
    with path.open("r", encoding=encoding, errors="ignore", newline="") as handle:
        sample = handle.read(8192)

    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return ","
    return dialect.delimiter


def _parse_with_strategy(series: pd.Series, strategy: str) -> pd.Series:
    """Parse timestamp series with selected strategy."""
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


def _timestamp_quality(parsed: pd.Series) -> tuple[float, float, pd.Timestamp | None, pd.Timestamp | None, int]:
    """Compute parse-valid/plausibility metrics."""
    total = int(len(parsed))
    if total == 0:
        return 0.0, 0.0, None, None, 0

    valid = parsed.dropna()
    valid_count = int(len(valid))
    if valid_count == 0:
        return 0.0, 0.0, None, None, 0

    parse_valid_ratio = valid_count / total
    plausible_year_ratio = float(valid.dt.year.between(_MIN_PLAUSIBLE_YEAR, _MAX_PLAUSIBLE_YEAR, inclusive="both").mean())
    min_ts = valid.min()
    max_ts = valid.max()
    unique_days = int(valid.dt.floor("D").nunique())
    return parse_valid_ratio, plausible_year_ratio, min_ts, max_ts, unique_days


def _select_timestamp_strategy(series: pd.Series) -> TimestampParseResult:
    """Select best parse strategy from sample."""
    best: TimestampParseResult | None = None

    for strategy in _TIMESTAMP_STRATEGY_ORDER:
        parsed = _parse_with_strategy(series, strategy)
        parse_valid_ratio, plausible_year_ratio, min_ts, max_ts, unique_days = _timestamp_quality(parsed)
        candidate = TimestampParseResult(
            strategy=strategy,
            parse_valid_ratio=parse_valid_ratio,
            plausible_year_ratio=plausible_year_ratio,
            min_ts=min_ts,
            max_ts=max_ts,
            unique_days_sample=unique_days,
            parsed_sample=parsed,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        raise RuntimeError("Timestamp strategy selection failed.")
    return best


def _choose_timestamp_column(columns_exact: list[str]) -> tuple[str | None, str | None, list[str]]:
    """Detect timestamp column and matched alias."""
    normalized = normalize_column_names(columns_exact)
    candidates: list[tuple[str, str]] = []
    for idx, norm in enumerate(normalized):
        if norm in _TIMESTAMP_ALIASES:
            candidates.append((columns_exact[idx], norm))

    if not candidates:
        return None, None, []

    for alias in _TIMESTAMP_ALIASES:
        for raw_col, matched_alias in candidates:
            if matched_alias == alias:
                candidate_columns = [col for col, _ in candidates]
                return raw_col, alias, candidate_columns

    return None, None, []


def _map_ohlcv(columns_exact: list[str]) -> dict[str, str | None]:
    """Map canonical OHLCV fields to input columns by normalized aliases."""
    normalized_pairs = list(zip(columns_exact, normalize_column_names(columns_exact)))
    mapping: dict[str, str | None] = {}

    for canonical, aliases in _OHLCV_ALIASES.items():
        selected: str | None = None
        for raw_col, norm_col in normalized_pairs:
            if norm_col in aliases:
                selected = raw_col
                break
        mapping[canonical] = selected

    return mapping


def _find_duplicate_headers(columns_exact: list[str]) -> list[str]:
    """Return duplicate header names, preserving first-seen order."""
    counts = Counter(columns_exact)
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in columns_exact:
        if counts[name] > 1 and name not in seen:
            duplicates.append(name)
            seen.add(name)
    return duplicates


def _safe_read_sample(path: Path, delimiter: str, encoding: str, sample_size: int) -> pd.DataFrame:
    """Read a bounded sample safely without loading full file."""
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    try:
        return pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            nrows=sample_size,
            engine="python",
            dtype="string",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read sample rows: {path}") from exc


def _read_header_columns(path: Path, delimiter: str, encoding: str) -> list[str]:
    """Read raw header names exactly as written in CSV."""
    with path.open("r", encoding=encoding, errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            return [str(item) for item in next(reader)]
        except StopIteration:
            return []


def _full_timestamp_pass(
    path: Path,
    delimiter: str,
    encoding: str,
    timestamp_column: str,
    strategy: str,
) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None, float]:
    """Run lightweight full-column pass for timestamp quality."""
    total_rows = 0
    valid_rows = 0
    count_year_1970 = 0
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None

    chunk_iter = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        usecols=[timestamp_column],
        dtype="string",
        chunksize=_CHUNK_SIZE,
        engine="python",
    )
    for chunk in chunk_iter:
        ts_series = chunk[timestamp_column]
        parsed = _parse_with_strategy(ts_series, strategy)
        valid = parsed.dropna()

        chunk_total = int(len(parsed))
        chunk_valid = int(len(valid))
        total_rows += chunk_total
        valid_rows += chunk_valid
        if chunk_valid > 0:
            count_year_1970 += int((valid.dt.year == 1970).sum())
            chunk_min = valid.min()
            chunk_max = valid.max()
            min_ts = chunk_min if min_ts is None or chunk_min < min_ts else min_ts
            max_ts = chunk_max if max_ts is None or chunk_max > max_ts else max_ts

    if total_rows == 0:
        return 0.0, None, None, 0.0

    valid_ratio = valid_rows / total_rows
    year_1970_ratio = (count_year_1970 / valid_rows) if valid_rows > 0 else 0.0
    return valid_ratio, min_ts, max_ts, year_1970_ratio


def _to_iso(ts: pd.Timestamp | None) -> str | None:
    """Convert pandas timestamp to ISO-8601 string."""
    return ts.isoformat() if ts is not None else None


def _validate_semantics(sample_df: pd.DataFrame, mapping: dict[str, str | None]) -> list[str]:
    """Validate OHLCV semantics on sampled data."""
    flags: list[str] = []

    required = ("open", "high", "low", "close", "volume")
    if any(mapping[key] is None for key in required):
        return flags

    converted: dict[str, pd.Series] = {}
    for key in required:
        raw_col = mapping[key]
        if raw_col is None:
            continue
        converted[key] = pd.to_numeric(sample_df[raw_col], errors="coerce")

    numeric_df = pd.DataFrame(converted).dropna()
    if numeric_df.empty:
        flags.append("NUMERIC_SAMPLE_EMPTY")
        return flags

    if bool((numeric_df["high"] < numeric_df["low"]).any()):
        flags.append("HIGH_BELOW_LOW")
    if bool(((numeric_df["open"] < numeric_df["low"]) | (numeric_df["open"] > numeric_df["high"])).any()):
        flags.append("OPEN_OUTSIDE_RANGE")
    if bool(((numeric_df["close"] < numeric_df["low"]) | (numeric_df["close"] > numeric_df["high"])).any()):
        flags.append("CLOSE_OUTSIDE_RANGE")
    if bool((numeric_df["volume"] < 0).any()):
        flags.append("NEGATIVE_VOLUME")
    if bool((numeric_df[["open", "high", "low", "close"]] <= 0).any().any()):
        flags.append("NONPOSITIVE_PRICE")

    return flags


def inspect_csv_file(path: Path, sample_size: int = 2000) -> CsvInspectionResult:
    """Inspect one CSV file for schema/timestamp integrity."""
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    failure_reason_codes: list[str] = []
    suspicious_flags: list[str] = []

    try:
        encoding = _detect_encoding(path)
        delimiter = _detect_delimiter(path, encoding)
        raw_header_columns = _read_header_columns(path, delimiter, encoding)
        sample_df = _safe_read_sample(path, delimiter, encoding, sample_size=sample_size)
    except Exception as exc:  # pragma: no cover - defensive boundary
        LOGGER.error("Failed to inspect file %s: %s", path, exc)
        return CsvInspectionResult(
            file=str(path),
            delimiter=",",
            encoding="unknown",
            columns_exact=[],
            duplicate_header_names=[],
            timestamp_alias=None,
            timestamp_column=None,
            ohlcv_mapping={"open": None, "high": None, "low": None, "close": None, "volume": None},
            indicator_columns=[],
            rows_sampled=0,
            parse_valid_ratio=0.0,
            min_ts=None,
            max_ts=None,
            unique_days_sample=0,
            suspicious_flags=["FILE_READ_FAILED"],
            failure_reason_codes=["FILE_READ_FAILED"],
            parse_scope="none",
        )

    sample_columns = [str(col) for col in sample_df.columns]
    columns_exact = raw_header_columns if len(raw_header_columns) == len(sample_columns) else sample_columns
    raw_by_working: dict[str, str] = {}
    for idx, col in enumerate(sample_columns):
        if idx < len(columns_exact):
            raw_by_working[col] = columns_exact[idx]
        else:
            raw_by_working[col] = col

    duplicates = _find_duplicate_headers(columns_exact)
    if duplicates:
        failure_reason_codes.append("DUPLICATE_HEADER_NAMES")

    timestamp_column, timestamp_alias, timestamp_candidates = _choose_timestamp_column(sample_columns)
    if timestamp_column is None:
        failure_reason_codes.append("TIMESTAMP_ALIAS_NOT_FOUND")

    if len(timestamp_candidates) > 1:
        suspicious_flags.append("MULTIPLE_TIMESTAMP_CANDIDATES")

    ohlcv_mapping = _map_ohlcv(sample_columns)
    missing_ohlcv = [key for key, val in ohlcv_mapping.items() if val is None]
    if missing_ohlcv:
        failure_reason_codes.append("MISSING_OHLCV_COLUMNS")
        suspicious_flags.append(f"MISSING_FIELDS:{','.join(missing_ohlcv)}")

    mapped_columns = {col for col in ohlcv_mapping.values() if col is not None}
    if timestamp_column is not None:
        mapped_columns.add(timestamp_column)
    indicator_columns = [raw_by_working[col] for col in sample_columns if col not in mapped_columns]

    parse_valid_ratio = 0.0
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None
    unique_days_sample = 0
    parse_scope = "sample"
    year_1970_ratio = 0.0

    if timestamp_column is not None:
        sample_parse = _select_timestamp_strategy(sample_df[timestamp_column])
        parse_valid_ratio = sample_parse.parse_valid_ratio
        min_ts = sample_parse.min_ts
        max_ts = sample_parse.max_ts
        unique_days_sample = sample_parse.unique_days_sample

        file_size = path.stat().st_size
        if file_size <= _FULL_PASS_MAX_BYTES:
            try:
                full_ratio, full_min, full_max, year_1970_ratio = _full_timestamp_pass(
                    path=path,
                    delimiter=delimiter,
                    encoding=encoding,
                    timestamp_column=timestamp_column,
                    strategy=sample_parse.strategy,
                )
                parse_valid_ratio = full_ratio
                min_ts = full_min
                max_ts = full_max
                parse_scope = "full"
            except Exception as exc:  # pragma: no cover - defensive boundary
                LOGGER.warning("Full timestamp pass failed for %s: %s", path, exc)
                suspicious_flags.append("FULL_PASS_FALLBACK_TO_SAMPLE")

        if parse_valid_ratio < 0.99:
            failure_reason_codes.append("TIMESTAMP_PARSE_RATIO_LOW")
        if min_ts is None or max_ts is None:
            failure_reason_codes.append("TIMESTAMP_PARSE_EMPTY")
        else:
            if min_ts.year < _MIN_PLAUSIBLE_YEAR or max_ts.year > _MAX_PLAUSIBLE_YEAR:
                failure_reason_codes.append("TIMESTAMP_IMPLAUSIBLE_RANGE")
            if year_1970_ratio >= 0.8:
                failure_reason_codes.append("TIMESTAMP_COLLAPSED_1970")

    semantic_flags = _validate_semantics(sample_df=sample_df, mapping=ohlcv_mapping)
    suspicious_flags.extend(semantic_flags)
    if semantic_flags:
        failure_reason_codes.append("OHLCV_SEMANTIC_ANOMALY")

    mapped_raw: dict[str, str | None] = {
        key: (raw_by_working[val] if val is not None else None) for key, val in ohlcv_mapping.items()
    }
    timestamp_column_raw = raw_by_working.get(timestamp_column, timestamp_column) if timestamp_column is not None else None

    return CsvInspectionResult(
        file=str(path),
        delimiter=delimiter,
        encoding=encoding,
        columns_exact=columns_exact,
        duplicate_header_names=duplicates,
        timestamp_alias=timestamp_alias,
        timestamp_column=timestamp_column_raw,
        ohlcv_mapping=mapped_raw,
        indicator_columns=indicator_columns,
        rows_sampled=int(len(sample_df)),
        parse_valid_ratio=float(parse_valid_ratio),
        min_ts=_to_iso(min_ts),
        max_ts=_to_iso(max_ts),
        unique_days_sample=unique_days_sample,
        suspicious_flags=sorted(set(suspicious_flags)),
        failure_reason_codes=sorted(set(failure_reason_codes)),
        parse_scope=parse_scope,
    )


def inspection_health_check(payload: CsvInspectionRunReport) -> bool:
    """Health gate before writing inspection report."""
    if not payload.run_id.strip():
        return False
    if payload.total_files != len(payload.inspections):
        return False
    if payload.failed_files + payload.passed_files != payload.total_files:
        return False
    return True


def inspect_all_csvs(input_root: Path, run_id: str, sample_size: int = 2000) -> CsvInspectionRunReport:
    """Inspect all CSV files under input root recursively."""
    csv_files = discover_csv_files(input_root)
    inspections: list[dict[str, Any]] = []

    for csv_path in tqdm(csv_files, desc="Inspecting CSV files", unit="file"):
        result = inspect_csv_file(csv_path, sample_size=sample_size)
        inspections.append(result.to_dict())

    passed_files = sum(1 for item in inspections if bool(item.get("passed")))
    failed_files = len(inspections) - passed_files

    return CsvInspectionRunReport(
        run_id=run_id,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        input_root=str(input_root.resolve()),
        total_files=len(inspections),
        passed_files=passed_files,
        failed_files=failed_files,
        inspections=inspections,
    )


def write_inspection_report(report: CsvInspectionRunReport, destination: Path) -> None:
    """Write inspection report atomically after health gate."""
    if not inspection_health_check(report):
        raise RuntimeError("Inspection health check failed; report will not be written.")
    atomic_write_json(asdict(report), destination)
