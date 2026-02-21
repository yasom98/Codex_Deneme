"""Health report models and QC gates for feature engineering outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data.features import CONTINUOUS_FEATURE_COLUMNS, EVENT_FLAG_COLUMNS, PIVOT_FEATURE_COLUMNS, validate_shift_one

_EMA_WARMUP_ROWS: dict[str, int] = {
    "EMA_200": 199,
    "EMA_600": 599,
    "EMA_1200": 1199,
}


@dataclass
class StructuredError:
    """Structured health error payload."""

    stage: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureHealthReport:
    """Per-file health report for feature outputs."""

    input_file: str
    rows_in: int = 0
    rows_out: int = 0
    schema_ok: bool = False
    monotonic_ok: bool = False
    unique_ts_ok: bool = False
    dtype_ok: bool = False
    leakfree_ok: bool = False
    nan_ratio_ok: bool = False
    parity_ok: bool = False
    parity_gate_ok: bool = False
    strict_parity_enabled: bool = True
    status: str = "failed"
    output_file: str | None = None
    input_run_id_resolved: str | None = None
    output_run_id: str | None = None
    input_root_resolved: str | None = None
    output_root_resolved: str | None = None
    indicator_parity_status: str = "not_checked"
    indicator_parity_details: dict[str, bool] = field(default_factory=dict)
    indicator_validation_status: str = "not_checked"
    indicator_validation_details: dict[str, bool] = field(default_factory=dict)
    indicator_validation_ok: bool = False
    strict_parity_gate_passed_but_indicator_validation_failed: bool = False
    formula_fingerprints: dict[str, str] = field(default_factory=dict)
    formula_fingerprint_bundle: str = ""
    session_count: int = 0
    first_session_row_count: int = 0
    pivot_first_session_allowed_nan: bool = False
    pivot_first_session_rows: int = 0
    pivot_nonnull_ratio_after_first_session: float = 1.0
    pivot_fill_strategy_applied: str = "none"
    nan_ratios: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[StructuredError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report into a JSON-ready dictionary."""

        return asdict(self)


def add_error(report: FeatureHealthReport, stage: str, code: str, message: str, **context: Any) -> None:
    """Append a structured error to a report."""

    report.errors.append(StructuredError(stage=stage, code=code, message=message, context=context))


def add_warning(report: FeatureHealthReport, message: str) -> None:
    """Append a warning message to a report."""

    report.warnings.append(message)


def _expected_columns() -> tuple[str, ...]:
    return (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *CONTINUOUS_FEATURE_COLUMNS,
        *EVENT_FLAG_COLUMNS,
    )


def _continuous_columns() -> tuple[str, ...]:
    return ("open", "high", "low", "close", "volume", *CONTINUOUS_FEATURE_COLUMNS)


def evaluate_feature_health(
    report: FeatureHealthReport,
    feature_df: pd.DataFrame,
    raw_events: pd.DataFrame,
    shifted_events: pd.DataFrame,
    warn_ratio: float,
    critical_warn_ratio: float,
    critical_columns: Sequence[str],
    pivot_warmup_policy: str,
    pivot_first_session_fill: str,
    indicator_parity_status: str,
    indicator_parity_details: dict[str, bool],
    indicator_validation_status: str,
    indicator_validation_details: dict[str, bool],
    formula_fingerprints: dict[str, str],
    formula_fingerprint_bundle: str,
    strict_parity: bool,
) -> FeatureHealthReport:
    """Run strict QC gates for feature outputs and update report in place."""

    expected = _expected_columns()
    missing = [col for col in expected if col not in feature_df.columns]
    report.schema_ok = len(missing) == 0
    if missing:
        add_error(report, stage="schema", code="MISSING_COLUMNS", message="Required feature columns are missing.", missing=missing)

    if report.schema_ok:
        timestamp = feature_df["timestamp"]
        report.monotonic_ok = bool(timestamp.is_monotonic_increasing)
        report.unique_ts_ok = bool(timestamp.is_unique)

        if not report.monotonic_ok:
            add_error(report, stage="gates", code="TIMESTAMP_NOT_MONOTONIC", message="timestamp is not monotonic increasing")
        if not report.unique_ts_ok:
            add_error(report, stage="gates", code="TIMESTAMP_NOT_UNIQUE", message="timestamp is not unique")

        dtype_ok_continuous = all(str(feature_df[col].dtype) == "float32" for col in _continuous_columns())
        dtype_ok_flags = all(str(feature_df[col].dtype) == "uint8" for col in EVENT_FLAG_COLUMNS)
        report.dtype_ok = bool(dtype_ok_continuous and dtype_ok_flags)
        if not report.dtype_ok:
            add_error(report, stage="gates", code="DTYPE_POLICY_FAILED", message="Dtype policy failed for continuous/event columns")

    report.leakfree_ok = validate_shift_one(raw_events, shifted_events)
    if not report.leakfree_ok:
        add_error(report, stage="gates", code="LEAKFREE_SHIFT_FAILED", message="Event flags are not strict shift(1)")

    report.indicator_parity_status = indicator_parity_status
    report.indicator_parity_details = dict(indicator_parity_details)
    report.indicator_validation_status = indicator_validation_status
    report.indicator_validation_details = dict(indicator_validation_details)
    report.formula_fingerprints = dict(formula_fingerprints)
    report.formula_fingerprint_bundle = str(formula_fingerprint_bundle)
    report.strict_parity_enabled = bool(strict_parity)
    report.parity_ok = indicator_parity_status in {"passed", "disabled"}
    report.parity_gate_ok = report.parity_ok if report.strict_parity_enabled else True
    if not report.parity_ok and report.strict_parity_enabled:
        failed = sorted(key for key, passed in report.indicator_parity_details.items() if not passed)
        add_error(
            report,
            stage="gates",
            code="INDICATOR_PARITY_FAILED",
            message="Indicator parity check failed.",
            failed_columns=failed,
        )
    elif not report.parity_ok:
        failed = sorted(key for key, passed in report.indicator_parity_details.items() if not passed)
        add_warning(
            report,
            "Indicator parity mismatch ignored due to strict_parity=false"
            f" failed_columns={failed}",
        )

    report.indicator_validation_ok = report.indicator_validation_status == "passed"
    report.strict_parity_gate_passed_but_indicator_validation_failed = bool(
        report.parity_gate_ok and (not report.indicator_validation_ok)
    )
    if not report.indicator_validation_ok:
        failed_checks = sorted(key for key, passed in report.indicator_validation_details.items() if not passed)
        add_error(
            report,
            stage="gates",
            code="INDICATOR_VALIDATION_FAILED",
            message="Mandatory indicator validation failed.",
            failed_checks=failed_checks,
            strict_parity_enabled=report.strict_parity_enabled,
            parity_gate_ok=report.parity_gate_ok,
        )
        if report.strict_parity_gate_passed_but_indicator_validation_failed:
            add_warning(
                report,
                "strict_parity gate passed but indicator_validation failed; write blocked by mandatory validation gate.",
            )

    report.nan_ratio_ok = True
    critical_set = {col.strip() for col in critical_columns if str(col).strip()}
    normalized_pivot_policy = pivot_warmup_policy.strip().lower()
    normalized_pivot_fill = pivot_first_session_fill.strip().lower()
    report.pivot_first_session_allowed_nan = normalized_pivot_policy == "allow_first_session_nan"
    report.pivot_fill_strategy_applied = "none"

    first_session_mask = pd.Series(False, index=feature_df.index, dtype=bool)
    if len(feature_df) > 0:
        sessions = feature_df["timestamp"].dt.floor("D")
        report.session_count = int(sessions.nunique())
        first_session = sessions.iloc[0]
        first_session_mask = sessions.eq(first_session)
        report.pivot_first_session_rows = int(first_session_mask.sum())
        report.first_session_row_count = report.pivot_first_session_rows

    pivot_columns = list(PIVOT_FEATURE_COLUMNS)
    pivot_present = all(col in feature_df.columns for col in pivot_columns)
    if pivot_present and len(feature_df) > 0:
        pivot_frame = feature_df.loc[:, pivot_columns]
        after_first_session = pivot_frame.loc[~first_session_mask]
        after_first_rows = int(after_first_session.shape[0])
        all_nan_after_first = bool(after_first_rows > 0 and after_first_session.isna().all(axis=0).all())
        all_nan_all_rows = bool(pivot_frame.isna().all(axis=0).all())

        if all_nan_after_first:
            report.nan_ratio_ok = False
            add_error(
                report,
                stage="gates",
                code="PIVOT_ALL_NAN_AFTER_FIRST_SESSION",
                message="All pivot columns are NaN from second session onward.",
                session_count=report.session_count,
                first_session_row_count=report.first_session_row_count,
            )

        if all_nan_all_rows:
            if report.session_count <= 1:
                add_warning(report, "Pivot columns are NaN only in single-session warmup window.")
            else:
                report.nan_ratio_ok = False
                add_error(
                    report,
                    stage="gates",
                    code="PIVOT_MAPPING_EMPTY",
                    message="All pivot mapping columns are fully NaN across the file.",
                )

        first_session_pivots = pivot_frame.loc[first_session_mask]
        if (
            normalized_pivot_fill == "ffill_from_second_session"
            and after_first_rows > 0
            and bool(first_session_pivots.notna().all(axis=0).all())
        ):
            report.pivot_fill_strategy_applied = "ffill_from_second_session"

        first_session_has_nan = bool(first_session_pivots.isna().any(axis=0).any())
        if report.pivot_first_session_allowed_nan and first_session_has_nan:
            add_warning(
                report,
                "Pivot first-session NaN warmup exception used."
                f" rows={report.pivot_first_session_rows}",
            )

        if (not report.pivot_first_session_allowed_nan) and first_session_has_nan:
            report.nan_ratio_ok = False
            add_error(
                report,
                stage="gates",
                code="PIVOT_FIRST_SESSION_NULL_NOT_ALLOWED",
                message="Pivot columns contain NaN in first session but warmup policy disallows it.",
            )

        total_after_first = int(after_first_session.shape[0] * after_first_session.shape[1])
        if total_after_first > 0:
            nonnull_after_first = int(after_first_session.notna().sum().sum())
            ratio_after_first = nonnull_after_first / float(total_after_first)
            report.pivot_nonnull_ratio_after_first_session = ratio_after_first
            if ratio_after_first < 1.0 and (not all_nan_after_first):
                report.nan_ratio_ok = False
                add_error(
                    report,
                    stage="gates",
                    code="PIVOT_AFTER_FIRST_SESSION_NULL",
                    message="Pivot columns must be fully non-null from second session onward.",
                    nonnull_ratio=ratio_after_first,
                )
        else:
            report.pivot_nonnull_ratio_after_first_session = 1.0
    else:
        report.pivot_nonnull_ratio_after_first_session = 1.0

    if "ST_up" in feature_df.columns and "ST_dn" in feature_df.columns and len(feature_df) > 0:
        st_nonnull_union = feature_df["ST_up"].notna() | feature_df["ST_dn"].notna()
        if bool(st_nonnull_union.any()):
            first_valid_pos = int(np.flatnonzero(st_nonnull_union.to_numpy(dtype=bool, copy=False))[0])
            post_warmup_union = st_nonnull_union.iloc[first_valid_pos:]
            invalid_ratio = float((~post_warmup_union).mean())
            if invalid_ratio > 0.0:
                report.nan_ratio_ok = False
                add_error(
                    report,
                    stage="gates",
                    code="SUPERTREND_BAND_COVERAGE_FAILED",
                    message="ST_up/ST_dn coverage invalid after warmup.",
                    invalid_ratio=invalid_ratio,
                )
        else:
            report.nan_ratio_ok = False
            add_error(
                report,
                stage="gates",
                code="SUPERTREND_BAND_EMPTY",
                message="Both ST_up and ST_dn are NaN for all rows.",
            )

    for col in _continuous_columns():
        if col not in feature_df.columns:
            continue

        ratio = float(feature_df[col].isna().mean())
        report.nan_ratios[col] = ratio
        gate_ratio = ratio

        if col in _EMA_WARMUP_ROWS:
            warmup_rows = _EMA_WARMUP_ROWS[col]
            if len(feature_df) > warmup_rows:
                gate_ratio = float(feature_df[col].iloc[warmup_rows:].isna().mean())
            else:
                gate_ratio = 0.0

        if col in {"ST_up", "ST_dn"}:
            gate_ratio = 0.0

        if (
            col in PIVOT_FEATURE_COLUMNS
            and report.pivot_first_session_allowed_nan
            and report.pivot_first_session_rows > 0
        ):
            if len(feature_df) > report.pivot_first_session_rows:
                gate_ratio = float(feature_df.loc[~first_session_mask, col].isna().mean())
            else:
                gate_ratio = 0.0

        if gate_ratio > warn_ratio:
            add_warning(report, f"NaN ratio high | column={col} ratio={gate_ratio:.6f} warn_ratio={warn_ratio:.6f}")

        col_threshold = critical_warn_ratio if col in critical_set else warn_ratio
        if gate_ratio > col_threshold:
            report.nan_ratio_ok = False
            add_error(
                report,
                stage="gates",
                code="NAN_RATIO_TOO_HIGH",
                message="NaN ratio exceeded threshold",
                column=col,
                ratio=gate_ratio,
                ratio_all_rows=ratio,
                threshold=col_threshold,
            )

    report.rows_out = int(len(feature_df))
    if report.rows_out <= 0:
        add_error(report, stage="gates", code="EMPTY_OUTPUT", message="Feature output has no rows")

    report.status = "success" if health_check(report) else "failed"
    return report


def health_check(report: FeatureHealthReport) -> bool:
    """Return True when strict QC gates pass."""

    return (
        report.schema_ok
        and report.monotonic_ok
        and report.unique_ts_ok
        and report.dtype_ok
        and report.leakfree_ok
        and report.nan_ratio_ok
        and report.parity_gate_ok
        and report.indicator_validation_ok
        and report.rows_out > 0
        and len(report.errors) == 0
    )


def summarize_feature_reports(
    reports: list[FeatureHealthReport],
    formula_fingerprints: dict[str, str],
    formula_fingerprint_bundle: str,
    strict_parity: bool,
) -> dict[str, Any]:
    """Create global feature summary report."""

    total_files = len(reports)
    succeeded_files = sum(1 for report in reports if report.status == "success")
    failed_files = total_files - succeeded_files
    parity_status_overall = all(report.parity_ok for report in reports)
    indicator_validation_overall = all(report.indicator_validation_ok for report in reports)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": total_files,
        "succeeded_files": succeeded_files,
        "failed_files": failed_files,
        "total_rows_in": sum(report.rows_in for report in reports),
        "total_rows_out": sum(report.rows_out for report in reports),
        "failed_inputs": [report.input_file for report in reports if report.status == "failed"],
        "parity_status_overall": parity_status_overall,
        "indicator_validation_overall": indicator_validation_overall,
        "strict_parity": bool(strict_parity),
        "formula_fingerprints": dict(formula_fingerprints),
        "formula_fingerprint_bundle": str(formula_fingerprint_bundle),
    }
