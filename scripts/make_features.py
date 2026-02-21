"""CLI entrypoint for Feature Engineering v1 on standardized OHLCV parquet files."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.io_atomic import atomic_write_json, atomic_write_parquet
from core.logging import get_logger, setup_logging
from core.paths import build_report_path
from data.feature_health import FeatureHealthReport, add_error, evaluate_feature_health, summarize_feature_reports
from data.features import (
    build_feature_artifacts,
    build_feature_manifest_payload,
    build_feature_output_path,
    compute_formula_fingerprint_bundle,
    compute_formula_fingerprints,
    discover_parquet_files,
    get_expected_column_dtypes,
    get_feature_groups,
    load_feature_config,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):  # type: ignore[no-redef]
        return iterable


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build feature-engineered parquet tables from standardized OHLCV parquet files.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml", help="YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Run feature generation without writing output parquet files.")
    parser.add_argument("--run-id", type=str, default="", help="Custom run id. Default: current UTC timestamp.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional standardized parquet input root. Default: runs/{run_id}/data_standardized/parquet",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    parser.add_argument(
        "--strict-parity",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Fail run on parity mismatch when true (default).",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for random libraries used in the pipeline."""

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        LOGGER.info("numpy not installed; skipping numpy seed.")
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        LOGGER.info("torch not installed; skipping torch seed.")
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _default_input_root(runs_root: Path, run_id: str) -> Path:
    """Build default standardized input root for a run id."""
    return runs_root / run_id / "data_standardized" / "parquet"


def _resolve_input_root(runs_root: Path, run_id: str, cli_input_root: Path | None) -> Path:
    """Resolve input root from CLI override or run-scoped default."""
    if cli_input_root is not None:
        return cli_input_root.resolve()
    return _default_input_root(runs_root, run_id).resolve()


def _resolve_input_run_id(input_root: Path, runs_root: Path) -> str | None:
    """Resolve input run id from standardized parquet root path."""
    resolved_input = input_root.resolve()
    resolved_runs_root = runs_root.resolve()
    try:
        rel = resolved_input.relative_to(resolved_runs_root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 3:
        return None
    if parts[1] != "data_standardized" or parts[2] != "parquet":
        return None
    return parts[0]


def _build_mismatch_summary(
    run_id: str,
    run_root: Path,
    input_root: Path,
    output_root: Path,
    input_run_id_resolved: str | None,
) -> dict[str, object]:
    """Build summary payload for input/output run wiring mismatch."""
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": 0,
        "succeeded_files": 0,
        "failed_files": 1,
        "total_rows_in": 0,
        "total_rows_out": 0,
        "failed_inputs": [str(input_root)],
        "parity_status_overall": False,
        "indicator_validation_overall": False,
        "pivot_reference_validation_overall": False,
        "alphatrend_reference_validation_overall": False,
        "supertrend_reference_validation_overall": False,
        "strict_parity": True,
        "formula_fingerprints": {},
        "formula_fingerprint_bundle": "",
        "feature_count": 0,
        "manifest_generated": False,
        "manifest_path": None,
        "manifest_error": None,
        "run_id": run_id,
        "run_root": str(run_root),
        "input_run_id_resolved": input_run_id_resolved,
        "output_run_id": run_id,
        "input_root_resolved": str(input_root.resolve()),
        "output_root_resolved": str(output_root.resolve()),
        "errors": [
            {
                "stage": "run_wiring",
                "code": "INPUT_OUTPUT_RUN_MISMATCH",
                "message": "Input root run_id does not match output run_id.",
                "context": {
                    "input_run_id_resolved": input_run_id_resolved,
                    "output_run_id": run_id,
                    "input_root_resolved": str(input_root.resolve()),
                    "output_root_resolved": str(output_root.resolve()),
                },
            }
        ],
    }
    return summary


def main() -> int:
    """Run feature build pipeline and return process exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    cfg = load_feature_config(args.config)
    set_global_seed(cfg.seed)
    strict_parity = args.strict_parity.strip().lower() == "true"
    formula_fingerprints = compute_formula_fingerprints(cfg)
    formula_fingerprint_bundle = compute_formula_fingerprint_bundle(formula_fingerprints)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = cfg.runs_root / run_id / "data_features"
    parquet_root = run_root / "parquet"
    reports_root = run_root / "reports"
    per_file_reports_root = reports_root / "per_file"
    resolved_input_root = _resolve_input_root(cfg.runs_root, run_id=run_id, cli_input_root=args.input_root)
    input_run_id_resolved = _resolve_input_run_id(resolved_input_root, cfg.runs_root)

    if input_run_id_resolved != run_id:
        LOGGER.error(
            "INPUT_OUTPUT_RUN_MISMATCH | output_run_id=%s input_run_id_resolved=%s input_root=%s output_root=%s discovered_files=%d",
            run_id,
            input_run_id_resolved,
            resolved_input_root,
            parquet_root,
            0,
        )
        if not args.dry_run:
            summary_path = reports_root / "summary.json"
            mismatch_summary = _build_mismatch_summary(
                run_id=run_id,
                run_root=run_root,
                input_root=resolved_input_root,
                output_root=parquet_root,
                input_run_id_resolved=input_run_id_resolved,
            )
            mismatch_summary["indicator_spec_version"] = cfg.indicator_spec_version
            mismatch_summary["config_hash"] = cfg.config_hash
            mismatch_summary["strict_parity"] = bool(strict_parity)
            mismatch_summary["formula_fingerprints"] = dict(formula_fingerprints)
            mismatch_summary["formula_fingerprint_bundle"] = str(formula_fingerprint_bundle)
            atomic_write_json(mismatch_summary, summary_path)
        return 1

    if not resolved_input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {resolved_input_root}")
    if not resolved_input_root.is_dir():
        raise NotADirectoryError(f"input_root is not a directory: {resolved_input_root}")

    src_files = discover_parquet_files(resolved_input_root, cfg.parquet_glob)
    LOGGER.info(
        "Feature wiring | run_id=%s input_root=%s output_root=%s discovered_files=%d",
        run_id,
        resolved_input_root,
        parquet_root,
        len(src_files),
    )

    reports: list[FeatureHealthReport] = []
    manifest_feature_groups: dict[str, tuple[str, ...]] | None = None
    manifest_column_dtypes: dict[str, str] | None = None
    manifest_rows: int = 0
    manifest_date_min_utc: str | None = None
    manifest_date_max_utc: str | None = None
    manifest_error: str | None = None

    for src_path in tqdm(src_files, desc="Building features"):
        report = FeatureHealthReport(input_file=str(src_path))
        report.input_run_id_resolved = input_run_id_resolved
        report.output_run_id = run_id
        report.input_root_resolved = str(resolved_input_root)
        report.output_root_resolved = str(parquet_root.resolve())

        output_path = build_feature_output_path(src_path, resolved_input_root, parquet_root)
        report_path = build_report_path(src_path, resolved_input_root, per_file_reports_root)
        feature_df: pd.DataFrame | None = None

        try:
            src_df = pd.read_parquet(src_path)
            report.rows_in = int(len(src_df))
            artifacts = build_feature_artifacts(src_df, cfg)
            feature_df = artifacts.frame
            evaluate_feature_health(
                report=report,
                feature_df=artifacts.frame,
                raw_events=artifacts.raw_events,
                shifted_events=artifacts.shifted_events,
                warn_ratio=cfg.health.warn_ratio,
                critical_warn_ratio=cfg.health.critical_warn_ratio,
                critical_columns=cfg.health.critical_columns,
                pivot_warmup_policy=cfg.pivot.warmup_policy,
                pivot_first_session_fill=cfg.pivot.first_session_fill,
                indicator_parity_status=artifacts.indicator_parity_status,
                indicator_parity_details=artifacts.indicator_parity_details,
                indicator_validation_status=artifacts.indicator_validation_status,
                indicator_validation_details=artifacts.indicator_validation_details,
                formula_fingerprints=artifacts.formula_fingerprints,
                formula_fingerprint_bundle=artifacts.formula_fingerprint_bundle,
                strict_parity=strict_parity,
                continuous_feature_columns=artifacts.continuous_feature_columns,
                flag_feature_columns=artifacts.flag_feature_columns,
                warmup_rows_by_column=artifacts.warmup_rows_by_column,
                raw_regime_flags=artifacts.raw_regime_flags,
                shifted_regime_flags=artifacts.shifted_regime_flags,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            add_error(report, stage="build", code="FEATURE_BUILD_FAILED", message=str(exc))
            report.status = "failed"

        if report.status == "success" and not args.dry_run and feature_df is not None:
            try:
                atomic_write_parquet(feature_df, output_path)
                report.output_file = str(output_path)
            except RuntimeError as exc:
                add_error(report, stage="write", code="FEATURE_WRITE_FAILED", message=str(exc))
                report.status = "failed"

        if report.status == "success" and feature_df is not None:
            current_dtypes = {col: str(dtype) for col, dtype in feature_df.dtypes.items()}
            if manifest_column_dtypes is None:
                manifest_column_dtypes = current_dtypes
            elif manifest_column_dtypes != current_dtypes and manifest_error is None:
                manifest_error = "INCONSISTENT_OUTPUT_SCHEMA: feature output columns/dtypes differ across files"

            if manifest_feature_groups is None:
                manifest_feature_groups = artifacts.feature_groups

            manifest_rows += int(len(feature_df))
            current_min = feature_df["timestamp"].iloc[0].isoformat()
            current_max = feature_df["timestamp"].iloc[-1].isoformat()
            manifest_date_min_utc = current_min if manifest_date_min_utc is None else min(manifest_date_min_utc, current_min)
            manifest_date_max_utc = current_max if manifest_date_max_utc is None else max(manifest_date_max_utc, current_max)

        reports.append(report)
        LOGGER.info(
            "Feature file processed | input=%s status=%s rows_in=%d rows_out=%d",
            src_path,
            report.status,
            report.rows_in,
            report.rows_out,
        )

        if not args.dry_run:
            atomic_write_json(report.to_dict(), report_path)

    summary = summarize_feature_reports(
        reports,
        formula_fingerprints=formula_fingerprints,
        formula_fingerprint_bundle=formula_fingerprint_bundle,
        strict_parity=strict_parity,
    )
    summary["run_id"] = run_id
    summary["run_root"] = str(run_root)
    summary["input_run_id_resolved"] = input_run_id_resolved
    summary["output_run_id"] = run_id
    summary["input_root_resolved"] = str(resolved_input_root)
    summary["output_root_resolved"] = str(parquet_root.resolve())
    summary["indicator_spec_version"] = cfg.indicator_spec_version
    summary["config_hash"] = cfg.config_hash

    manifest_path = reports_root / "feature_manifest.json"
    manifest_generated = False
    if not args.dry_run:
        if manifest_error is None:
            try:
                manifest_payload = build_feature_manifest_payload(
                    run_id=run_id,
                    cfg=cfg,
                    feature_groups=manifest_feature_groups or get_feature_groups(cfg),
                    column_dtypes=manifest_column_dtypes or get_expected_column_dtypes(cfg),
                    row_count=manifest_rows,
                    date_min_utc=manifest_date_min_utc,
                    date_max_utc=manifest_date_max_utc,
                    formula_fingerprints=formula_fingerprints,
                    formula_fingerprint_bundle=formula_fingerprint_bundle,
                )
                atomic_write_json(manifest_payload, manifest_path)
                manifest_generated = True
            except (RuntimeError, ValueError, OSError) as exc:
                manifest_error = str(exc)

    summary["manifest_generated"] = manifest_generated
    summary["manifest_path"] = str(manifest_path) if manifest_generated else None
    summary["manifest_error"] = manifest_error

    LOGGER.info(
        "Feature build summary | run_id=%s total=%d success=%d failed=%d",
        run_id,
        summary["total_files"],
        summary["succeeded_files"],
        summary["failed_files"],
    )

    if not args.dry_run:
        summary_path = reports_root / "summary.json"
        atomic_write_json(summary, summary_path)

    return 0 if summary["failed_files"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
