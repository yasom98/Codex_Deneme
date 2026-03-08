"""CLI entrypoint for Milestone 4.8 evaluation/backtest gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.logging import get_logger, setup_logging
from rl.evaluation_backtest import execute_evaluation_backtest

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse strict 4.8 CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the strict 4.8 evaluation/backtest gate.")
    parser.add_argument("--run-id", type=str, required=True, help="Explicit run id.")
    parser.add_argument("--model-artifact", type=Path, required=True, help="Explicit SB3 PPO .zip model artifact path.")
    parser.add_argument("--env-config", type=Path, required=True, help="Explicit env config JSON path.")
    parser.add_argument("--eval-config", type=Path, required=True, help="Explicit eval config JSON path.")
    parser.add_argument("--state-manifest", type=Path, required=True, help="Explicit state manifest JSON path.")
    parser.add_argument("--env-contract-report", type=Path, required=True, help="Explicit env contract report JSON path.")
    parser.add_argument("--readiness-report", type=Path, required=True, help="Explicit readiness report JSON path.")
    parser.add_argument("--episode-catalog", type=Path, required=True, help="Explicit episode catalog JSON path.")
    parser.add_argument("--split-report", type=Path, required=True, help="Explicit split report JSON path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh output dir for 4.8 reports.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    """Run the evaluation gate and return the exact 4.8 exit code contract."""

    args = parse_args()
    setup_logging(args.log_level)

    try:
        result = execute_evaluation_backtest(
            run_id=args.run_id,
            model_artifact_path=args.model_artifact.resolve(),
            env_config_path=args.env_config.resolve(),
            eval_config_path=args.eval_config.resolve(),
            state_manifest_path=args.state_manifest.resolve(),
            env_contract_report_path=args.env_contract_report.resolve(),
            readiness_report_path=args.readiness_report.resolve(),
            episode_catalog_path=args.episode_catalog.resolve(),
            split_report_path=args.split_report.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        LOGGER.info(
            "Evaluation/backtest summary | run_id=%s exit_code=%d reports_written=%s output_dir=%s",
            args.run_id,
            result.exit_code,
            result.reports_written,
            args.output_dir.resolve(),
        )
        return int(result.exit_code)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Evaluation/backtest runtime error | run_id=%s error=%s", args.run_id, exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
