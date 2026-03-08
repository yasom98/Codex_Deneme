"""CLI tests for Milestone 4.7 training launcher."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from tests.training_launcher_fixtures import seed_training_launcher_run, write_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "launch_training.py"


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_prelaunch_only_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "launch_training_cli_success"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(tmp_path, run_id)
    output_dir = tmp_path / "cli_prelaunch_out"

    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "launch_training.py",
            "--run-id",
            run_id,
            "--env-config",
            str(seeded["env_config_path"]),
            "--training-config",
            str(training_config_path),
            "--state-manifest",
            str(seeded["state_manifest_path"]),
            "--env-contract-report",
            str(seeded["env_contract_report_path"]),
            "--readiness-report",
            str(seeded["readiness_report_path"]),
            "--episode-catalog",
            str(seeded["episode_catalog_path"]),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    validation = json.loads((output_dir / "training_launch_validation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "training_launch_manifest.json").read_text(encoding="utf-8"))
    smoke = json.loads((output_dir / "training_smoke_report.json").read_text(encoding="utf-8"))

    assert validation["overall_pass"] is True
    assert manifest["smoke_mode"] == "prelaunch_only"
    assert smoke["smoke_success"] is True
    assert smoke["smoke_requested"] is False


def test_cli_output_conflict_returns_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "launch_training_cli_output_conflict"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(tmp_path, run_id)
    output_dir = tmp_path / "cli_existing_output"
    output_dir.mkdir()

    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "launch_training.py",
            "--run-id",
            run_id,
            "--env-config",
            str(seeded["env_config_path"]),
            "--training-config",
            str(training_config_path),
            "--state-manifest",
            str(seeded["state_manifest_path"]),
            "--env-contract-report",
            str(seeded["env_contract_report_path"]),
            "--readiness-report",
            str(seeded["readiness_report_path"]),
            "--episode-catalog",
            str(seeded["episode_catalog_path"]),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = int(main())
    assert exit_code == 2
