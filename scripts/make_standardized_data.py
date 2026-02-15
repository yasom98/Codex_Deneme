"""CLI entrypoint for OHLCV standardization pipeline."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.config import load_config
from core.health import summarize_reports
from core.io_atomic import atomic_write_json
from core.logging import get_logger, setup_logging
from data.standardize import standardize_all

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Standardize crypto OHLCV CSV files to parquet.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data.yaml", help="YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Run validation and health checks without writing outputs.")
    parser.add_argument("--run-id", type=str, default="", help="Custom run id. Default: current UTC timestamp.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for common random libraries."""
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
    """Run pipeline and return process exit code."""
    args = parse_args()
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = cfg.runs_root / run_id / "data_standardized"
    parquet_root = run_root / "parquet"
    reports_root = run_root / "reports"
    per_file_reports_root = reports_root / "per_file"

    LOGGER.info(
        "Run layout | run_id=%s parquet_root=%s reports_root=%s per_file_reports_root=%s",
        run_id,
        parquet_root,
        reports_root,
        per_file_reports_root,
    )

    reports = standardize_all(
        cfg=cfg,
        parquet_root=parquet_root,
        per_file_reports_root=per_file_reports_root,
        dry_run=args.dry_run,
    )
    summary = summarize_reports(reports)
    summary["run_id"] = run_id
    summary["run_root"] = str(run_root)

    LOGGER.info("Summary: total=%d success=%d failed=%d", summary["total_files"], summary["succeeded_files"], summary["failed_files"])

    if not args.dry_run:
        summary_path = reports_root / "summary.json"
        atomic_write_json(summary, summary_path)
        LOGGER.info("Summary report written to %s", summary_path)

    return 0 if summary["failed_files"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
