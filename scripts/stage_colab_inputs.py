"""CLI entrypoint for explicit Colab input staging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.logging import get_logger, setup_logging
from rl.colab_runtime import STAGING_MANIFEST_FILENAME, stage_explicit_inputs

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse strict explicit Colab staging arguments."""

    parser = argparse.ArgumentParser(description="Stage the minimal explicit input set from canonical storage into local Colab VM storage.")
    parser.add_argument("--staging-root", type=Path, required=True, help="Fresh local staging root, for example /content/codex_run_001.")
    parser.add_argument("--env-config", type=Path, required=True, help="Explicit env config JSON path.")
    parser.add_argument("--training-config", type=Path, required=True, help="Explicit canonical PPO artifact-production training config JSON path.")
    parser.add_argument("--state-manifest", type=Path, required=True, help="Explicit state manifest JSON path.")
    parser.add_argument("--env-contract-report", type=Path, required=True, help="Explicit env contract report JSON path.")
    parser.add_argument("--readiness-report", type=Path, required=True, help="Explicit readiness report JSON path.")
    parser.add_argument("--episode-catalog", type=Path, required=True, help="Explicit episode catalog JSON path.")
    parser.add_argument("--split-report", type=Path, required=True, help="Explicit split validation report JSON path.")
    parser.add_argument("--eval-config", type=Path, required=False, default=None, help="Optional explicit evaluation config JSON path.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    """Stage the minimal explicit input set and return a strict exit code."""

    args = parse_args()
    setup_logging(args.log_level)

    source_paths = {
        "env_config": args.env_config.resolve(),
        "training_config": args.training_config.resolve(),
        "state_manifest": args.state_manifest.resolve(),
        "env_contract_report": args.env_contract_report.resolve(),
        "readiness_report": args.readiness_report.resolve(),
        "episode_catalog": args.episode_catalog.resolve(),
        "split_report": args.split_report.resolve(),
    }
    if args.eval_config is not None:
        source_paths["eval_config"] = args.eval_config.resolve()

    try:
        manifest_payload = stage_explicit_inputs(
            staging_root=args.staging_root.resolve(),
            source_paths=source_paths,
        )
        LOGGER.info(
            "Colab input staging summary | status=%s manifest_path=%s staging_root=%s",
            manifest_payload["status"],
            args.staging_root.resolve() / STAGING_MANIFEST_FILENAME,
            args.staging_root.resolve(),
        )
        return 0
    except (ValueError, RuntimeError) as exc:
        LOGGER.error("Colab input staging validation failed | error=%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Colab input staging runtime error | error=%s", exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
