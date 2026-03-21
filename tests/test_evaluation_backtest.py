"""Unit/integration tests for Milestone 4.8 evaluation/backtest."""

from __future__ import annotations

import csv
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
from rl.passivity_diagnostics import (
    DETERMINISTIC_ACTION_RANKING_TRACE_FILENAME,
    PASSIVITY_DIAGNOSTICS_REPORT_FILENAME,
    build_deterministic_action_ranking_row,
    build_deterministic_action_ranking_summary,
)
from tests.evaluation_backtest_fixtures import FakePredictModel, seed_evaluation_run, write_eval_config, write_risk_overlay_config


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
    risk_overlay_config_path: Path | None = None,
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
        risk_overlay_config_path=risk_overlay_config_path,
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

    validation = json.loads(result.report_paths.validation_report_path.read_text(encoding="utf-8"))
    backtest = json.loads(result.report_paths.backtest_report_path.read_text(encoding="utf-8"))
    assert backtest["evaluation_success"] is True
    assert backtest["strategy_metrics"]["num_trades"] == 1
    assert backtest["benchmark_metrics"] is not None
    assert backtest["relative_metrics"] is not None
    assert validation["runtime"]["progress"]["active_mode"] == "disabled"
    assert backtest["runtime"]["execution_bounds"]["max_eval_episodes"] == 1
    assert backtest["metric_status"]["strategy"]["avg_trade_return"]["detail"]["metric_policy"] == "narrow_v1_proxy"
    assert [item["phase"] for item in backtest["startup_phase_trace"]] == [
        "validation",
        "model_load",
        "env_init",
        "eval_start",
        "eval_finish",
        "report_write",
    ]


def test_masked_single_path_uses_action_masks_during_predict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "eval_masked_single_path_success"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id, overrides={"write_step_trace": True, "action_masking": True})
    model = FakePredictModel(actions=[1, 0, 3, 0])

    def _plain_loader_should_not_run(model_artifact_path: Path, device: str | None) -> FakePredictModel:
        del model_artifact_path, device
        raise AssertionError("legacy PPO loader should not be used when action_masking=true")

    monkeypatch.setattr("rl.evaluation_backtest._load_ppo_model", _plain_loader_should_not_run)
    monkeypatch.setattr(
        "rl.evaluation_backtest._load_maskable_ppo_model",
        lambda model_artifact_path, device: model,
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
        output_dir=tmp_path / "eval_masked_single_out",
    )

    backtest = json.loads(result.report_paths.backtest_report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert backtest["action_masking_enabled"] is True
    assert backtest["diagnostic_artifacts"]["deterministic_action_ranking_trace_path"] is None
    assert model.action_masks_seen
    assert model.action_masks_seen[0].tolist() == [True, True, True, False]
    assert model.action_masks_seen[1].tolist() == [True, False, False, True]


def test_passivity_diagnostics_compares_deterministic_and_stochastic_policy_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "eval_passivity_diagnostics"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(
        tmp_path,
        run_id,
        overrides={"action_masking": True, "passivity_diagnostics": True},
    )
    model = FakePredictModel(
        deterministic_actions=[0, 0, 0, 0],
        stochastic_actions=[1, 3, 0, 0],
        deterministic_action_probabilities=[
            [0.40, 0.35, 0.05, 0.20],
            [0.42, 0.33, 0.05, 0.20],
            [0.39, 0.36, 0.05, 0.20],
            [0.41, 0.34, 0.05, 0.20],
        ],
    )

    def _plain_loader_should_not_run(model_artifact_path: Path, device: str | None) -> FakePredictModel:
        del model_artifact_path, device
        raise AssertionError("legacy PPO loader should not be used when action_masking=true")

    monkeypatch.setattr("rl.evaluation_backtest._load_ppo_model", _plain_loader_should_not_run)
    monkeypatch.setattr(
        "rl.evaluation_backtest._load_maskable_ppo_model",
        lambda model_artifact_path, device: model,
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
        output_dir=tmp_path / "eval_passivity_diagnostics_out",
    )

    backtest = json.loads(result.report_paths.backtest_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.report_paths.manifest_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(result.report_paths.passivity_diagnostics_report_path.read_text(encoding="utf-8"))
    ranking_rows = list(csv.DictReader(result.report_paths.deterministic_action_ranking_trace_path.open("r", encoding="utf-8")))

    assert result.exit_code == 0
    assert result.report_paths.passivity_diagnostics_report_path.name == PASSIVITY_DIAGNOSTICS_REPORT_FILENAME
    assert result.report_paths.deterministic_action_ranking_trace_path.name == DETERMINISTIC_ACTION_RANKING_TRACE_FILENAME
    assert backtest["passivity_diagnostics_enabled"] is True
    assert manifest["passivity_diagnostics_enabled"] is True
    assert backtest["diagnostic_artifacts"]["passivity_diagnostics_report_path"] == str(
        result.report_paths.passivity_diagnostics_report_path
    )
    assert backtest["diagnostic_artifacts"]["deterministic_action_ranking_trace_path"] == str(
        result.report_paths.deterministic_action_ranking_trace_path
    )
    assert diagnostics["deterministic_eval"]["action_semantic_counts"]["HOLD"] > 0
    assert diagnostics["deterministic_eval"]["num_trades"] == 0
    assert diagnostics["stochastic_eval"]["num_trades"] == 1
    assert diagnostics["stochastic_eval"]["action_semantic_counts"]["OPEN_LONG"] == 1
    assert diagnostics["deterministic_vs_stochastic"]["stochastic_more_active_than_deterministic"] is True
    assert diagnostics["deterministic_eval"]["action_ranking_summary"]["fraction_of_steps_hold_is_top1"] == pytest.approx(1.0)
    assert diagnostics["deterministic_eval"]["action_ranking_summary"]["hold_dominance_margin_band"] == "narrow"
    assert diagnostics["deterministic_eval"]["action_ranking_summary"]["step_count"] == len(ranking_rows)
    assert diagnostics["deterministic_eval"]["action_ranking_summary"]["top2_runner_up_counts"]["OPEN_LONG"] == len(
        ranking_rows
    )
    assert len(ranking_rows) > 0
    assert ranking_rows[0]["top_1_action_semantic"] == "HOLD"
    assert ranking_rows[0]["top_2_action_semantic"] == "OPEN_LONG"
    assert ranking_rows[0]["hold_gap_band"] == "small"


def test_deterministic_action_ranking_summary_classifies_wide_hold_dominance() -> None:
    rows = [
        build_deterministic_action_ranking_row(
            episode_index=0,
            episode_ref={"scope": "partition", "partition": "val", "source_rel": "val_a.parquet", "fold_id": None},
            step_ordinal=index,
            step_index=index,
            timestamp=f"2026-03-15T00:00:0{index}Z",
            position_before=0,
            selected_action_semantic="HOLD",
            action_probabilities={
                "HOLD": 0.72,
                "OPEN_LONG": 0.12,
                "OPEN_SHORT": 0.08,
                "CLOSE_POSITION": 0.08,
            },
        )
        for index in range(4)
    ]

    summary = build_deterministic_action_ranking_summary(rows)

    assert summary["fraction_of_steps_hold_is_top1"] == pytest.approx(1.0)
    assert summary["fraction_of_steps_gap_below_threshold"] == pytest.approx(0.0)
    assert summary["hold_dominance_margin_band"] == "wide"


def test_deterministic_action_ranking_summary_is_inconclusive_when_hold_is_not_consistently_top1() -> None:
    rows = [
        build_deterministic_action_ranking_row(
            episode_index=0,
            episode_ref={"scope": "partition", "partition": "val", "source_rel": "val_a.parquet", "fold_id": None},
            step_ordinal=0,
            step_index=0,
            timestamp="2026-03-15T00:00:00Z",
            position_before=0,
            selected_action_semantic="HOLD",
            action_probabilities={
                "HOLD": 0.40,
                "OPEN_LONG": 0.35,
                "OPEN_SHORT": 0.05,
                "CLOSE_POSITION": 0.20,
            },
        ),
        build_deterministic_action_ranking_row(
            episode_index=0,
            episode_ref={"scope": "partition", "partition": "val", "source_rel": "val_a.parquet", "fold_id": None},
            step_ordinal=1,
            step_index=1,
            timestamp="2026-03-15T00:00:01Z",
            position_before=0,
            selected_action_semantic="OPEN_LONG",
            action_probabilities={
                "HOLD": 0.20,
                "OPEN_LONG": 0.50,
                "OPEN_SHORT": 0.10,
                "CLOSE_POSITION": 0.20,
            },
        ),
    ]

    summary = build_deterministic_action_ranking_summary(rows)

    assert summary["fraction_of_steps_hold_is_top1"] == pytest.approx(0.5)
    assert summary["hold_dominance_margin_band"] == "inconclusive"


def test_passivity_diagnostics_fail_closed_when_ranking_probabilities_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "eval_passivity_diagnostics_missing_probs"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(
        tmp_path,
        run_id,
        overrides={"action_masking": True, "passivity_diagnostics": True},
    )
    model = FakePredictModel(
        deterministic_actions=[0, 0, 0, 0],
        distribution_probabilities_available=False,
    )

    monkeypatch.setattr("rl.evaluation_backtest._load_ppo_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(
        "rl.evaluation_backtest._load_maskable_ppo_model",
        lambda model_artifact_path, device: model,
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
        output_dir=tmp_path / "eval_passivity_diagnostics_missing_probs_out",
    )

    backtest = json.loads(result.report_paths.backtest_report_path.read_text(encoding="utf-8"))

    assert result.exit_code == 2
    assert backtest["evaluation_success"] is False
    assert "EVAL_PASSIVITY_DIAGNOSTICS_FAILED" in backtest["failure_codes"]


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


def test_risk_overlay_writes_minimal_artifacts_and_additive_report_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "eval_risk_overlay_artifacts"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)
    risk_overlay_config_path = write_risk_overlay_config(
        tmp_path,
        instrument="val_a.parquet",
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
        output_dir=tmp_path / "eval_risk_overlay_out",
        risk_overlay_config_path=risk_overlay_config_path,
    )

    assert result.exit_code == 0
    assert result.report_paths.risk_decision_log_path is not None
    assert result.report_paths.risk_overlay_summary_path is not None
    assert result.report_paths.risk_state_transition_log_path is not None
    assert result.report_paths.risk_decision_log_path.exists()
    assert result.report_paths.risk_overlay_summary_path.exists()
    assert result.report_paths.risk_state_transition_log_path.exists()

    manifest = json.loads(result.report_paths.manifest_path.read_text(encoding="utf-8"))
    backtest = json.loads(result.report_paths.backtest_report_path.read_text(encoding="utf-8"))
    summary = json.loads(result.report_paths.risk_overlay_summary_path.read_text(encoding="utf-8"))

    assert manifest["risk_overlay"]["enabled"] is True
    assert backtest["risk_overlay"]["enabled"] is True
    assert summary["overlay_enabled"] is True
    assert summary["decision_counts"]["ALLOW"] >= 1


def test_text_progress_mode_records_text_tqdm_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "eval_text_progress_runtime"
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

    result = execute_evaluation_backtest(
        run_id=run_id,
        model_artifact_path=seeded["model_artifact_path"],
        env_config_path=seeded["env_config_path"],
        eval_config_path=eval_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "eval_text_progress_out",
        progress_mode="text",
    )

    assert result.exit_code == 0
    assert result.validation_payload["runtime"]["progress"]["active_mode"] == "text_tqdm"
    assert result.backtest_payload["runtime"]["progress"]["requested_mode"] == "text"


def test_risk_overlay_drawdown_state_machine_clamps_short_under_freeze_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "eval_risk_overlay_freeze"
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    eval_config_path = write_eval_config(tmp_path, run_id)
    risk_overlay_config_path = write_risk_overlay_config(
        tmp_path,
        instrument="val_a.parquet",
        overrides={
            "config_version": "risk_overlay.v1",
            "allowed_instruments": ["val_a.parquet"],
            "freshness_limits": {
                "max_market_data_age_seconds": 0,
                "max_portfolio_state_age_seconds": 0,
                "max_proposal_age_seconds": 0,
            },
            "exposure_limits": {
                "max_abs_target_exposure": 1.0,
                "max_gross_exposure": 1.0,
                "max_net_exposure": 1.0,
                "max_instrument_exposure": 1.0,
                "defensive_scale_down": 1.0,
            },
            "leverage_limits": {
                "max_leverage": 1.0,
                "defensive_scale_down": 1.0,
            },
            "drawdown_thresholds": {
                "defensive_enter_pct": 0.0005,
                "defensive_exit_pct": 0.0001,
                "freeze_enter_pct": 0.0015,
                "freeze_exit_pct": 0.0010,
                "kill_pct": 0.01,
            },
            "hysteresis_bands": {"min_steps_in_state": 1},
            "recovery_policy": {
                "freeze_cooldown_steps": 1,
                "systemic_failure_kill_threshold": 2,
                "kill_requires_recovery_token": True,
            },
        },
    )

    monkeypatch.setattr(
        "rl.evaluation_backtest._load_ppo_model",
        lambda model_artifact_path, device: FakePredictModel(actions=[2, 0, 0, 0]),
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
        output_dir=tmp_path / "eval_risk_overlay_freeze_out",
        risk_overlay_config_path=risk_overlay_config_path,
    )

    assert result.exit_code == 0

    decision_log_lines = result.report_paths.risk_decision_log_path.read_text(encoding="utf-8").strip().splitlines()  # type: ignore[union-attr]
    decision_rows = [json.loads(line) for line in decision_log_lines if line.strip()]
    summary = json.loads(result.report_paths.risk_overlay_summary_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    transition_lines = result.report_paths.risk_state_transition_log_path.read_text(encoding="utf-8").strip().splitlines()  # type: ignore[union-attr]
    transitions = [json.loads(line) for line in transition_lines if line.strip()]

    assert any(row["state_transition"] and row["state_transition"]["to_mode"] == "DEFENSIVE" for row in decision_rows)
    assert any(row["state_transition"] and row["state_transition"]["to_mode"] == "FREEZE_ENTRIES" for row in decision_rows)
    assert any(row["decision_type"] == "CLAMP" and row["approved_action"]["action_semantic"] == "CLOSE_POSITION" for row in decision_rows)
    assert summary["decision_counts"]["CLAMP"] >= 1
    assert any(item["to_mode"] == "FREEZE_ENTRIES" for item in transitions)
