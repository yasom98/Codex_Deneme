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

from core.io_atomic import atomic_write_parquet
from core.logging import get_logger, setup_logging
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

    src_files = discover_parquet_files(cfg.input_root, cfg.parquet_glob)
    LOGGER.info("Discovered standardized parquet files | count=%d input_root=%s", len(src_files), cfg.input_root)

    succeeded = 0
    failed = 0

    for src_path in tqdm(src_files, desc="Building features"):
        output_path = build_feature_output_path(src_path, cfg.input_root, parquet_root)

        try:
            src_df = pd.read_parquet(src_path)
            artifacts = build_feature_artifacts(src_df, cfg)
        except (ValueError, RuntimeError, OSError) as exc:
            failed += 1
            LOGGER.error("Feature build failed | input=%s error=%s", src_path, exc)
            continue

        if not args.dry_run:
            try:
                atomic_write_parquet(artifacts.frame, output_path)
            except RuntimeError as exc:
                failed += 1
                LOGGER.error("Feature parquet write failed | output=%s error=%s", output_path, exc)
                continue

        succeeded += 1
        LOGGER.info("Feature file processed | input=%s output=%s rows=%d", src_path, output_path, len(artifacts.frame))

    LOGGER.info("Feature build summary | run_id=%s succeeded=%d failed=%d", run_id, succeeded, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
