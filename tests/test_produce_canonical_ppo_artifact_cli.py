"""CLI tests for canonical PPO artifact production."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from tests.ppo_artifact_production_fixtures import (
    FakeArtifactPpo,
    FakeArtifactTrainingEnv,
    seed_artifact_production_run,
    write_artifact_training_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "produce_canonical_ppo_artifact.py"


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_success_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "artifact_production_cli_success"
    seeded = seed_artifact_production_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_artifact_training_config(tmp_path, run_id)
    output_dir = tmp_path / "cli_artifact_out"

    monkeypatch.setattr("rl.ppo_artifact_production.TradingEnvGym", FakeArtifactTrainingEnv)
    monkeypatch.setattr("rl.ppo_artifact_production._import_ppo_class", lambda: FakeArtifactPpo)

    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "produce_canonical_ppo_artifact.py",
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
            "--split-report",
            str(seeded["split_report_path"]),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    report = json.loads((output_dir / "artifact_production_report.json").read_text(encoding="utf-8"))
    assert report["canonical_artifact_ready"] is True


def test_cli_runtime_failure_returns_three(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "artifact_production_cli_runtime_failure"
    seeded = seed_artifact_production_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_artifact_training_config(tmp_path, run_id)
    output_dir = tmp_path / "cli_artifact_runtime_failure_out"

    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "execute_ppo_artifact_production",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("runtime-boom")),
    )
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "produce_canonical_ppo_artifact.py",
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
            "--split-report",
            str(seeded["split_report_path"]),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = int(main())
    assert exit_code == 3
