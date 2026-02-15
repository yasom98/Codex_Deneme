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
from data.features import build_feature_artifacts, build_feature_output_path, discover_parquet_files, load_feature_config

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
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
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


def main() -> int:
    """Run feature build pipeline and return process exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    cfg = load_feature_config(args.config)
    set_global_seed(cfg.seed)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = cfg.runs_root / run_id / "data_features"
    parquet_root = run_root / "parquet"
    reports_root = run_root / "reports"
    per_file_reports_root = reports_root / "per_file"

    src_files = discover_parquet_files(cfg.input_root, cfg.parquet_glob)
    LOGGER.info("Discovered standardized parquet files | count=%d input_root=%s", len(src_files), cfg.input_root)

    reports: list[FeatureHealthReport] = []

    for src_path in tqdm(src_files, desc="Building features"):
        report = FeatureHealthReport(input_file=str(src_path))
        output_path = build_feature_output_path(src_path, cfg.input_root, parquet_root)
        report_path = build_report_path(src_path, cfg.input_root, per_file_reports_root)
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

    summary = summarize_feature_reports(reports)
    summary["run_id"] = run_id
    summary["run_root"] = str(run_root)
    summary["indicator_spec_version"] = cfg.indicator_spec_version
    summary["config_hash"] = cfg.config_hash

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
