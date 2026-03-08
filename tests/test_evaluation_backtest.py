"""Unit/integration tests for Milestone 4.8 evaluation/backtest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.evaluation_backtest import (
    ALIAS_WARNING_CODE,
    EVAL_MODEL_ARTIFACT_INVALID,
    EVAL_MODEL_LOAD_FAILED,
    EVAL_OUTPUT_CONFLICT,
    execute_evaluation_backtest,
)
from tests.evaluation_backtest_fixtures import FakePredictModel, seed_evaluation_run, write_eval_config


def _run_evaluation(
    *,
    run_id: str,
    model_artifact_path: Path,
    env_config_path: Path,
    eval_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
    output_dir: Path,
):
    return execute_evaluation_backtest(
        run_id=run_id,
        model_artifact_path=model_artifact_path,
        env_config_path=env_config_path,
        eval_config_path=eval_config_path,
        state_manifest_path=state_manifest_path,
        env_contract_report_path=env_contract_report_path,
        readiness_report_path=readiness_report_path,
        episode_catalog_path=episode_catalog_path,
        split_report_path=split_report_path,
        output_dir=output_dir,
    )


def test_single_path_success_writes_reports_and_proxy_metric(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "eval_single_path_success"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id, overrides={"write_step_trace": True})

    monkeypatch.setattr(
        "rl.evaluation_backtest._load_ppo_model",
        lambda model_artifact_path, device: FakePredictModel(actions=[1, 0, 3, 0]),
    )
    monkeypatch.setattr(
        "rl.evaluation_backtest.atomic_write_parquet",
        lambda df, dest: dest.write_text(f"rows={len(df)}", encoding="utf-8"),
    )

    result = _run_evaluation(
        run_id=run_id,
        model_artifact_path=seeded["model_artifact_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "eval_single_out",
    )

    assert result.exit_code == 0
    assert result.report_paths.validation_report_path.exists()
    assert result.report_paths.manifest_path.exists()
    assert result.report_paths.backtest_report_path.exists()
    assert result.report_paths.step_trace_path.exists()

    backtest = json.loads(result.report_paths.backtest_report_path.read_text(encoding="utf-8"))
    assert backtest["evaluation_success"] is True
    assert backtest["strategy_metrics"]["num_trades"] == 1
    assert backtest["benchmark_metrics"] is not None
    assert backtest["relative_metrics"] is not None
    assert backtest["metric_status"]["strategy"]["avg_trade_return"]["detail"]["metric_policy"] == "narrow_v1_proxy"
    assert [item["phase"] for item in backtest["startup_phase_trace"]] == [
        "validation",
        "model_load",
        "env_init",
        "eval_start",
        "eval_finish",
        "report_write",
    ]


def test_explicit_partition_alias_is_reported_and_relative_metrics_use_aggregate_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "eval_partition_alias"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(
        tmp_path,
        run_id,
        overrides={
            "evaluation_mode": "episodic_eval_backtest",
            "target_mode": "explicit_partition",
            "target_partition": "validation",
            "target_episode_refs": None,
            "max_eval_episodes": 2,
        },
    )

    monkeypatch.setattr(
        "rl.evaluation_backtest._load_ppo_model",
        lambda model_artifact_path, device: FakePredictModel(actions=[1, 0, 3, 0]),
    )

    result = _run_evaluation(
        run_id=run_id,
        model_artifact_path=seeded["model_artifact_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "eval_alias_out",
    )

    assert result.exit_code == 0

    validation = json.loads(result.report_paths.validation_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.report_paths.manifest_path.read_text(encoding="utf-8"))
    backtest = json.loads(result.report_paths.backtest_report_path.read_text(encoding="utf-8"))

    alias_check = next(item for item in validation["validation_checks"] if item["check_name"] == "partition_alias_compatibility_rule")
    assert alias_check["reason_code"] == ALIAS_WARNING_CODE
    assert manifest["selected_partition"] == "validation"
    assert manifest["lineages"]["partition_alias_resolution"]
    assert backtest["relative_metrics"]["excess_total_return"] == pytest.approx(
        backtest["strategy_metrics"]["total_return"] - backtest["benchmark_metrics"]["total_return"]
    )


def test_model_load_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "eval_model_load_fail"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)

    def _raise_model_load(model_artifact_path: Path, device: str | None) -> None:
        del model_artifact_path, device
        raise RuntimeError("boom")

    monkeypatch.setattr("rl.evaluation_backtest._load_ppo_model", _raise_model_load)

    result = _run_evaluation(
        run_id=run_id,
        model_artifact_path=seeded["model_artifact_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "eval_model_fail_out",
    )

    assert result.exit_code == 2
    assert EVAL_MODEL_LOAD_FAILED in result.backtest_payload["failure_codes"]


def test_invalid_model_artifact_is_rejected_before_model_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "eval_invalid_model_artifact"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)

    result = _run_evaluation(
        run_id=run_id,
        model_artifact_path=seeded["state_manifest_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "eval_invalid_model_out",
    )

    assert result.exit_code == 2
    assert EVAL_MODEL_ARTIFACT_INVALID in result.validation_payload["failure_codes"]


def test_output_conflict_returns_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "eval_output_conflict"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)
    output_dir = tmp_path / "eval_existing_output"
    output_dir.mkdir()

    result = _run_evaluation(
        run_id=run_id,
        model_artifact_path=seeded["model_artifact_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=output_dir,
    )

    assert result.exit_code == 2
    assert EVAL_OUTPUT_CONFLICT in result.validation_payload["failure_codes"]
    assert result.reports_written is False


def test_same_seed_repeat_keeps_stable_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "eval_same_seed_repeat"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)

    monkeypatch.setattr(
        "rl.evaluation_backtest._load_ppo_model",
        lambda model_artifact_path, device: FakePredictModel(actions=[1, 0, 3, 0]),
    )

    first = _run_evaluation(
        run_id=run_id,
        model_artifact_path=seeded["model_artifact_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "repeat_eval_a",
    )
    second = _run_evaluation(
        run_id=run_id,
        model_artifact_path=seeded["model_artifact_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "repeat_eval_b",
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.validation_payload["selected_algorithm"] == second.validation_payload["selected_algorithm"]
    assert first.validation_payload["deterministic"] == second.validation_payload["deterministic"]
    assert first.validation_payload["effective_seed"] == second.validation_payload["effective_seed"]
    assert first.validation_payload["model_artifact_hash"] == second.validation_payload["model_artifact_hash"]
    assert first.validation_payload["eval_config_hash"] == second.validation_payload["eval_config_hash"]
    assert first.validation_payload["readiness_hash"] == second.validation_payload["readiness_hash"]
    assert first.validation_payload["env_contract_hash"] == second.validation_payload["env_contract_hash"]
    assert first.validation_payload["state_manifest_hash"] == second.validation_payload["state_manifest_hash"]
    assert first.validation_payload["episode_catalog_hash"] == second.validation_payload["episode_catalog_hash"]
    assert first.validation_payload["split_report_hash"] == second.validation_payload["split_report_hash"]
    assert first.manifest_payload["evaluation_mode"] == second.manifest_payload["evaluation_mode"]
    assert first.manifest_payload["target_mode"] == second.manifest_payload["target_mode"]
    assert first.manifest_payload["selected_partition"] == second.manifest_payload["selected_partition"]
    assert first.manifest_payload["selected_fold_id"] == second.manifest_payload["selected_fold_id"]
    assert first.manifest_payload["selected_episode_refs"] == second.manifest_payload["selected_episode_refs"]
    assert first.manifest_payload["benchmark_mode"] == second.manifest_payload["benchmark_mode"]
    assert [item["phase"] for item in first.backtest_payload["startup_phase_trace"]] == [
        item["phase"] for item in second.backtest_payload["startup_phase_trace"]
    ]


def test_no_trades_marks_proxy_metric_unsupported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "eval_no_trades"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)

    monkeypatch.setattr(
        "rl.evaluation_backtest._load_ppo_model",
        lambda model_artifact_path, device: FakePredictModel(actions=[0, 0, 0, 0]),
    )

    result = _run_evaluation(
        run_id=run_id,
        model_artifact_path=seeded["model_artifact_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "eval_no_trade_out",
    )

    assert result.exit_code == 0
    metric_status = result.backtest_payload["metric_status"]["strategy"]["avg_trade_return"]
    assert metric_status["supported"] is False
    assert metric_status["detail"]["metric_policy"] == "narrow_v1_proxy"
