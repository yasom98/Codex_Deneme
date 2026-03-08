"""Unit/integration tests for Milestone 4.7 training launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.training_launcher import (
    TRAIN_LAUNCH_ALGO_PARAMS_INVALID,
    TRAIN_LAUNCH_NO_ELIGIBLE_TRAINING_EPISODES,
    TRAIN_LAUNCH_OUTPUT_CONFLICT,
    execute_training_launch,
)
from tests.training_launcher_fixtures import FakePpo, FakeTrainingEnv, seed_training_launcher_run, write_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_launcher(
    *,
    run_id: str,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    output_dir: Path,
):
    return execute_training_launch(
        run_id=run_id,
        env_config_path=env_config_path,
        training_config_path=training_config_path,
        state_manifest_path=state_manifest_path,
        env_contract_report_path=env_contract_report_path,
        readiness_report_path=readiness_report_path,
        episode_catalog_path=episode_catalog_path,
        output_dir=output_dir,
    )


def test_prelaunch_only_writes_validation_manifest_and_no_learn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_prelaunch_only"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(tmp_path, run_id)

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "prelaunch_out",
    )

    assert result.exit_code == 0
    assert result.manifest_payload is not None
    assert result.smoke_payload is not None
    assert result.validation_payload["overall_pass"] is True
    assert result.smoke_payload["smoke_requested"] is False
    assert result.smoke_payload["smoke_success"] is True
    assert result.smoke_payload["smoke_rollout_summary"]["learn_invoked"] is False
    assert result.smoke_payload["launch_guard_results"]["smoke_learn_timesteps_unused"] is True
    assert result.report_paths.validation_report_path.exists()
    assert result.report_paths.manifest_path.exists()
    assert result.report_paths.smoke_report_path.exists()


def test_launch_smoke_uses_stubbed_env_and_ppo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_launch_smoke"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(
        tmp_path,
        run_id,
        overrides={"smoke_mode": "launch_smoke", "smoke_learn_timesteps": 6},
    )

    monkeypatch.setattr("rl.training_launcher.TradingEnvGym", FakeTrainingEnv)
    monkeypatch.setattr("rl.training_launcher._import_ppo_class", lambda: FakePpo)

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "launch_smoke_out",
    )

    assert result.exit_code == 0
    assert result.smoke_payload is not None
    assert result.smoke_payload["smoke_requested"] is True
    assert result.smoke_payload["smoke_success"] is True
    assert result.smoke_payload["smoke_rollout_summary"]["learn_invoked"] is True
    assert result.smoke_payload["smoke_rollout_summary"]["smoke_learn_timesteps_used"] == 6
    phase_names = [item["phase"] for item in result.smoke_payload["startup_phase_trace"]]
    assert phase_names == [
        "prelaunch_validation",
        "env_init",
        "algo_init",
        "learn_start",
        "learn_finish",
        "report_write",
    ]
    assert result.smoke_payload["startup_phase_trace"][1]["status"] == "completed"
    assert result.smoke_payload["startup_phase_trace"][2]["status"] == "completed"
    assert result.smoke_payload["startup_phase_trace"][4]["status"] == "completed"


def test_fixed_episode_launch_does_not_depend_on_readiness_selection_policy_or_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_fixed_episode"
    seeded = seed_training_launcher_run(
        monkeypatch,
        tmp_path,
        run_id,
        readiness_selection_policy="seeded_random_episode",
        selected_episode_ref={
            "scope": "partition",
            "partition": "train",
            "source_rel": "a_train.parquet",
            "fold_id": None,
        },
    )
    training_config_path = write_training_config(
        tmp_path,
        run_id,
        overrides={"episode_selection_mode": "fixed_episode"},
    )

    readiness_payload = json.loads(seeded["readiness_report_path"].read_text(encoding="utf-8"))
    readiness_payload["selection_policy"] = "unrelated_selection_policy"
    readiness_payload["seed"] = 999
    seeded["readiness_report_path"].write_text(json.dumps(readiness_payload), encoding="utf-8")

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "fixed_mode_out",
    )

    assert result.exit_code == 0
    assert result.validation_payload["overall_pass"] is True
    assert result.manifest_payload is not None
    assert result.manifest_payload["lineages"]["selected_episode_ref"]["source_rel"] == "a_train.parquet"


def test_seeded_random_requires_non_empty_training_domain_even_if_readiness_report_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_empty_training_domain"
    seeded = seed_training_launcher_run(
        monkeypatch,
        tmp_path,
        run_id,
        entries=[{"partition": "val", "source_rel": "only_val.parquet", "row_count": 8}],
        readiness_selection_policy="fixed_episode",
    )
    training_config_path = write_training_config(tmp_path, run_id)

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "empty_training_domain_out",
    )

    assert result.exit_code == 2
    assert TRAIN_LAUNCH_NO_ELIGIBLE_TRAINING_EPISODES in result.validation_payload["failure_codes"]


def test_invalid_algo_params_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_invalid_algo_params"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(
        tmp_path,
        run_id,
        overrides={"algo_params": {"learning_rate": 0.0003}},
    )

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "invalid_algo_params_out",
    )

    assert result.exit_code == 2
    assert TRAIN_LAUNCH_ALGO_PARAMS_INVALID in result.validation_payload["failure_codes"]


def test_output_dir_existing_file_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_output_conflict_file"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(tmp_path, run_id)
    output_path = tmp_path / "existing_output"
    output_path.write_text("conflict", encoding="utf-8")

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=output_path,
    )

    assert result.exit_code == 2
    assert result.reports_written is False
    assert TRAIN_LAUNCH_OUTPUT_CONFLICT in result.validation_payload["failure_codes"]


def test_output_dir_existing_hidden_only_directory_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_output_conflict_hidden_dir"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(tmp_path, run_id)
    output_dir = tmp_path / "hidden_only_dir"
    output_dir.mkdir()
    (output_dir / ".keep").write_text("x", encoding="utf-8")

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=output_dir,
    )

    assert result.exit_code == 2
    assert result.reports_written is False
    assert TRAIN_LAUNCH_OUTPUT_CONFLICT in result.validation_payload["failure_codes"]


def test_output_dir_existing_empty_directory_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_output_conflict_empty_dir"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(tmp_path, run_id)
    output_dir = tmp_path / "empty_existing_dir"
    output_dir.mkdir()

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=output_dir,
    )

    assert result.exit_code == 2
    assert TRAIN_LAUNCH_OUTPUT_CONFLICT in result.validation_payload["failure_codes"]


def test_same_seed_repeat_keeps_stable_launch_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_same_seed_repeat"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_training_config(
        tmp_path,
        run_id,
        overrides={"smoke_mode": "launch_smoke", "smoke_learn_timesteps": 5},
    )

    monkeypatch.setattr("rl.training_launcher.TradingEnvGym", FakeTrainingEnv)
    monkeypatch.setattr("rl.training_launcher._import_ppo_class", lambda: FakePpo)

    first = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "repeat_out_a",
    )
    second = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "repeat_out_b",
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.validation_payload["selected_algorithm"] == second.validation_payload["selected_algorithm"]
    assert first.validation_payload["selected_episode_mode"] == second.validation_payload["selected_episode_mode"]
    assert first.validation_payload["effective_seed"] == second.validation_payload["effective_seed"]
    assert first.validation_payload["config_hash"] == second.validation_payload["config_hash"]
    assert first.validation_payload["readiness_hash"] == second.validation_payload["readiness_hash"]
    assert first.validation_payload["env_contract_hash"] == second.validation_payload["env_contract_hash"]
    assert first.validation_payload["state_manifest_hash"] == second.validation_payload["state_manifest_hash"]
    assert first.validation_payload["episode_catalog_hash"] == second.validation_payload["episode_catalog_hash"]
    assert [item["phase"] for item in first.smoke_payload["startup_phase_trace"]] == [
        item["phase"] for item in second.smoke_payload["startup_phase_trace"]
    ]


def test_launch_smoke_example_config_is_valid_with_stubbed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_example_smoke_config"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    example_config = PROJECT_ROOT / "configs" / "training_config.launch_smoke.example.json"

    monkeypatch.setattr("rl.training_launcher.TradingEnvGym", FakeTrainingEnv)
    monkeypatch.setattr("rl.training_launcher._import_ppo_class", lambda: FakePpo)

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=example_config,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "example_smoke_config_out",
    )

    assert result.exit_code == 0
    assert result.validation_payload["overall_pass"] is True
    assert result.smoke_payload["smoke_success"] is True
    assert result.smoke_payload["smoke_rollout_summary"]["smoke_learn_timesteps_used"] == 8


def test_baseline_example_config_is_valid_with_stubbed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "launcher_example_baseline_config"
    seeded = seed_training_launcher_run(monkeypatch, tmp_path, run_id)
    example_config = PROJECT_ROOT / "configs" / "training_config.baseline_train.example.json"

    monkeypatch.setattr("rl.training_launcher.TradingEnvGym", FakeTrainingEnv)
    monkeypatch.setattr("rl.training_launcher._import_ppo_class", lambda: FakePpo)

    result = _run_launcher(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=example_config,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        output_dir=tmp_path / "example_baseline_config_out",
    )

    assert result.exit_code == 0
    assert result.validation_payload["overall_pass"] is True
    assert result.manifest_payload["selected_algorithm"] == "ppo"
    assert result.smoke_payload["smoke_success"] is True
    assert result.smoke_payload["smoke_rollout_summary"]["smoke_learn_timesteps_used"] == 2048
