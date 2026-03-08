"""Tests for Milestone 4.9 PPO search orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.ppo_search_orchestrator import (
    TRIAL_STATUS_COMPLETED_CANDIDATE,
    TRIAL_STATUS_COMPLETED_NONCOMPETITIVE,
    TRIAL_STATUS_INVALID,
    TRIAL_STATUS_PRUNED,
    execute_ppo_search_study,
)
from tests.ppo_search_fixtures import build_search_study


def test_execute_study_writes_trial_reports_and_ranks_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seeded = build_search_study(monkeypatch, tmp_path, "ppo_search_success")

    result = execute_ppo_search_study(study_config_path=seeded["study_config_path"])

    assert result.exit_code == 0
    assert result.reports_written is True

    summary_path = seeded["output_root"] / "study_summary.json"
    progress_path = seeded["output_root"] / "study_progress.json"
    manifest_path = seeded["output_root"] / "study_manifest.json"
    assert summary_path.exists()
    assert progress_path.exists()
    assert manifest_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    statuses = [trial["status"] for trial in summary["trials"]]
    assert TRIAL_STATUS_COMPLETED_CANDIDATE in statuses
    assert TRIAL_STATUS_COMPLETED_NONCOMPETITIVE in statuses

    for trial in summary["trials"]:
        trial_root = Path(trial["output_dir"])
        assert (trial_root / "trial_manifest.json").exists()
        assert (trial_root / "trial_training_report.json").exists()
        assert (trial_root / "trial_evaluation_report.json").exists()
        assert (trial_root / "trial_objective_report.json").exists()
        assert (trial_root / "trial_guardrail_report.json").exists()
        assert (trial_root / "trial_status.json").exists()


def test_invalid_trial_is_reported_for_incompatible_rollout_batch_combo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seeded = build_search_study(
        monkeypatch,
        tmp_path,
        "ppo_search_invalid_combo",
        search_space_overrides={"batch_size": [3], "learning_rate": [0.0003]},
        resource_budget_overrides={"max_trials": 1},
    )

    result = execute_ppo_search_study(study_config_path=seeded["study_config_path"])

    assert result.exit_code == 0
    summary = json.loads((seeded["output_root"] / "study_summary.json").read_text(encoding="utf-8"))
    assert summary["trial_counts"][TRIAL_STATUS_INVALID] == 1
    trial = summary["trials"][0]
    assert trial["status"] == TRIAL_STATUS_INVALID

    trial_status = json.loads((Path(trial["output_dir"]) / "trial_status.json").read_text(encoding="utf-8"))
    assert trial_status["status"] == TRIAL_STATUS_INVALID
    assert trial_status["ranking_eligible"] is False


def test_probe_stage_can_prune_trial_without_running_final_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seeded = build_search_study(
        monkeypatch,
        tmp_path,
        "ppo_search_pruned",
        eval_actions=[0, 0, 0, 0],
        resource_budget_overrides={
            "max_trials": 1,
            "probe_train_total_timesteps": 8,
            "full_train_total_timesteps": 16,
        },
        objective_overrides={"primary_metric": "total_return"},
        pruning_overrides={
            "enabled": True,
            "warmup_trials": 0,
            "min_completed_probe_trials": 0,
            "min_probe_objective_score": 0.01,
            "relative_to_best_completed_margin": 0.0,
        },
        guardrail_overrides={"min_num_trades_hard": 0, "min_num_trades_soft": 0},
    )

    result = execute_ppo_search_study(study_config_path=seeded["study_config_path"])

    assert result.exit_code == 0
    summary = json.loads((seeded["output_root"] / "study_summary.json").read_text(encoding="utf-8"))
    assert summary["trial_counts"][TRIAL_STATUS_PRUNED] == 1

    trial = summary["trials"][0]
    assert trial["status"] == TRIAL_STATUS_PRUNED
    trial_root = Path(trial["output_dir"])
    assert (trial_root / "probe_artifact_production").exists()
    assert not (trial_root / "final_artifact_production").exists()
