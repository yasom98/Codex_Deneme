"""CLI tests for Milestone 4.8 evaluation/backtest."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from tests.evaluation_backtest_fixtures import FakePredictModel, seed_evaluation_run, write_eval_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_policy.py"


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_success_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "evaluate_policy_cli_success"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)
    output_dir = tmp_path / "cli_eval_out"

    monkeypatch.setattr(
        "rl.evaluation_backtest._load_ppo_model",
        lambda model_artifact_path, device: FakePredictModel(actions=[1, 0, 3, 0]),
    )

    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "evaluate_policy.py",
            "--run-id",
            run_id,
            "--model-artifact",
            str(seeded["model_artifact_path"]),
            "--env-config",
            str(seeded["env_config_path"]),
            "--eval-config",
            str(eval_config_path),
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

    validation = json.loads((output_dir / "evaluation_validation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    backtest = json.loads((output_dir / "evaluation_backtest_report.json").read_text(encoding="utf-8"))
    assert validation["overall_pass"] is True
    assert manifest["selected_algorithm"] == "ppo"
    assert backtest["evaluation_success"] is True


def test_cli_output_conflict_returns_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "evaluate_policy_cli_output_conflict"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)
    output_dir = tmp_path / "cli_eval_existing_out"
    output_dir.mkdir()

    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "evaluate_policy.py",
            "--run-id",
            run_id,
            "--model-artifact",
            str(seeded["model_artifact_path"]),
            "--env-config",
            str(seeded["env_config_path"]),
            "--eval-config",
            str(eval_config_path),
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
    assert exit_code == 2


def test_cli_runtime_failure_returns_three(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "evaluate_policy_cli_runtime_failure"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)
    output_dir = tmp_path / "cli_eval_runtime_failure_out"

    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "execute_evaluation_backtest",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("runtime-boom")),
    )
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "evaluate_policy.py",
            "--run-id",
            run_id,
            "--model-artifact",
            str(seeded["model_artifact_path"]),
            "--env-config",
            str(seeded["env_config_path"]),
            "--eval-config",
            str(eval_config_path),
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
