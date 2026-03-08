"""CLI entrypoint for Milestone 4.7 training launch gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.logging import get_logger, setup_logging
from rl.training_launcher import execute_training_launch

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse strict launcher CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the strict 4.7 training launch gate. "
            "4.7 configs are starter/validation configs only: use the smoke example for bounded launch validation "
            "and the baseline example as an initial PPO starting point. Hyperparameter optimization is deferred to 4.9."
        )
    )
    parser.add_argument("--run-id", type=str, required=True, help="Explicit run id.")
    parser.add_argument("--env-config", type=Path, required=True, help="Explicit env config JSON path.")
    parser.add_argument(
        "--training-config",
        type=Path,
        required=True,
        help=(
            "Explicit training config JSON path. "
            "See configs/training_config.launch_smoke.example.json and "
            "configs/training_config.baseline_train.example.json."
        ),
    )
    parser.add_argument("--state-manifest", type=Path, required=True, help="Explicit state manifest JSON path.")
    parser.add_argument("--env-contract-report", type=Path, required=True, help="Explicit env contract report JSON path.")
    parser.add_argument("--readiness-report", type=Path, required=True, help="Explicit readiness report JSON path.")
    parser.add_argument("--episode-catalog", type=Path, required=True, help="Explicit episode catalog JSON path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh output dir for 4.7 reports.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    """Run the launcher gate and return deterministic exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    try:
        result = execute_training_launch(
            run_id=args.run_id,
            env_config_path=args.env_config.resolve(),
            training_config_path=args.training_config.resolve(),
            state_manifest_path=args.state_manifest.resolve(),
            env_contract_report_path=args.env_contract_report.resolve(),
            readiness_report_path=args.readiness_report.resolve(),
            episode_catalog_path=args.episode_catalog.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        LOGGER.info(
            "Training launcher summary | run_id=%s exit_code=%d reports_written=%s output_dir=%s",
            args.run_id,
            result.exit_code,
            result.reports_written,
            args.output_dir.resolve(),
        )
        return int(result.exit_code)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Training launcher runtime error | run_id=%s error=%s", args.run_id, exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
