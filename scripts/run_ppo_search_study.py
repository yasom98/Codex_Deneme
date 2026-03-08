"""CLI entrypoint for Milestone 4.9 PPO search studies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.logging import get_logger, setup_logging
from rl.ppo_search_orchestrator import execute_ppo_search_study

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse strict 4.9 CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the strict Milestone 4.9 PPO search study orchestrator.")
    parser.add_argument("--study-config", type=Path, required=True, help="Explicit 4.9 study config JSON path.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    """Run the 4.9 study and return the exact CLI exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    try:
        result = execute_ppo_search_study(study_config_path=args.study_config.resolve())
        LOGGER.info(
            "PPO search study summary | study_config=%s exit_code=%d reports_written=%s",
            args.study_config.resolve(),
            result.exit_code,
            result.reports_written,
        )
        return int(result.exit_code)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("PPO search study runtime error | study_config=%s error=%s", args.study_config.resolve(), exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
