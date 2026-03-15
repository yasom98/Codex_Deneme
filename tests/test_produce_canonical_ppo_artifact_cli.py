"""CLI tests for canonical PPO artifact production."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ppo_artifact_production_fixtures import (
    FakeArtifactPpo,
    FakeArtifactTrainingEnv,
    seed_artifact_production_run,
    write_artifact_training_config,
)
from tests.test_colab_runtime import _load_json, _patch_runtime_dependencies, _seed_closure_source, _write_json
from rl.colab_staging_closure import CLOSURE_REPORT_FILENAME

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


def test_cli_passes_optional_progress_and_memory_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "artifact_production_cli_progress_flags"
    seeded = seed_artifact_production_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_artifact_training_config(tmp_path, run_id)
    output_dir = tmp_path / "cli_artifact_progress_flags_out"
    captured: dict[str, object] = {}

    main = _load_main()

    def _fake_execute(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(exit_code=0, reports_written=True)

    monkeypatch.setitem(main.__globals__, "execute_ppo_artifact_production", _fake_execute)
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
            "--progress-mode",
            "text",
            "--memory-log-interval-steps",
            "128",
        ],
    )

    exit_code = int(main())
    assert exit_code == 0
    assert captured["progress_mode"] == "text"
    assert captured["memory_log_interval_steps"] == 128


def test_cli_returns_two_when_colab_stage_preflight_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "artifact_production_cli_stage_invalid"
    _patch_runtime_dependencies(monkeypatch)
    source_paths = _seed_closure_source(monkeypatch, tmp_path, run_id=run_id)
    from rl.colab_runtime import stage_explicit_inputs

    staging_root = tmp_path / "cli_stage_invalid"
    stage_explicit_inputs(staging_root=staging_root, source_paths=source_paths)
    closure_report_path = staging_root / CLOSURE_REPORT_FILENAME
    closure_report = _load_json(closure_report_path)
    closure_report["overall_closure_valid"] = False
    _write_json(closure_report_path, closure_report)

    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "produce_canonical_ppo_artifact.py",
            "--run-id",
            run_id,
            "--env-config",
            str(staging_root / "env_contract" / "tmp" / source_paths["env_config"].name),
            "--training-config",
            str(staging_root / "configs" / "training_config.json"),
            "--state-manifest",
            str(staging_root / "data_states" / "reports" / "state_manifest.json"),
            "--env-contract-report",
            str(staging_root / "env_contract" / "reports" / "env_contract_report.json"),
            "--readiness-report",
            str(staging_root / "env_readiness" / "reports" / "training_env_readiness_report.json"),
            "--episode-catalog",
            str(staging_root / "env_readiness" / "reports" / "episode_catalog.json"),
            "--split-report",
            str(staging_root / "data_features" / "reports" / "split_validation_report.json"),
            "--output-dir",
            str(tmp_path / "cli_stage_invalid_out"),
        ],
    )

    exit_code = int(main())
    assert exit_code == 2
