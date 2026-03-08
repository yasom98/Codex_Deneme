"""Fixtures for Milestone 4.9 PPO search orchestrator tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.evaluation_backtest_fixtures import FakePredictModel, seed_evaluation_run, write_eval_config
from tests.ppo_artifact_production_fixtures import (
    FakeArtifactPpo,
    FakeArtifactTrainingEnv,
    write_artifact_training_config,
)


def patch_search_runtime(monkeypatch: pytest.MonkeyPatch, *, eval_actions: list[int]) -> None:
    """Patch runtime dependencies for deterministic 4.9 tests."""

    monkeypatch.setattr("rl.training_launcher.TradingEnvGym", FakeArtifactTrainingEnv)
    monkeypatch.setattr("rl.training_launcher._import_ppo_class", lambda: FakeArtifactPpo)
    monkeypatch.setattr("rl.ppo_artifact_production.TradingEnvGym", FakeArtifactTrainingEnv)
    monkeypatch.setattr("rl.ppo_artifact_production._import_ppo_class", lambda: FakeArtifactPpo)
    monkeypatch.setattr(
        "rl.evaluation_backtest._load_ppo_model",
        lambda model_artifact_path, device: FakePredictModel(actions=list(eval_actions)),
    )

    def fake_atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(df.to_json(orient="records"), encoding="utf-8")

    def fake_read_parquet(path: str | Path) -> pd.DataFrame:
        return pd.read_json(Path(path), orient="records")

    monkeypatch.setattr("rl.evaluation_backtest.atomic_write_parquet", fake_atomic_write_parquet)
    monkeypatch.setattr("rl.ppo_search_orchestrator._read_step_trace", fake_read_parquet)


def build_search_study(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_id: str,
    *,
    eval_actions: list[int] | None = None,
    search_space_overrides: dict[str, Any] | None = None,
    resource_budget_overrides: dict[str, Any] | None = None,
    objective_overrides: dict[str, Any] | None = None,
    guardrail_overrides: dict[str, Any] | None = None,
    pruning_overrides: dict[str, Any] | None = None,
    promotion_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a deterministic 4.9 study config wired to test fixtures."""

    patch_search_runtime(monkeypatch, eval_actions=eval_actions or [1, 0, 3, 0])
    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    training_template_path = write_artifact_training_config(
        tmp_path,
        run_id,
        overrides={"total_timesteps": 16, "device": "cpu"},
    )
    eval_template_path = write_eval_config(
        tmp_path,
        run_id,
        overrides={
            "deterministic": True,
            "device": "cpu",
            "benchmark_mode": "buy_and_hold",
            "write_step_trace": False,
            "max_eval_steps": 16,
            "max_eval_episodes": 1,
        },
    )

    payload: dict[str, Any] = {
        "study_id": f"{run_id}_study",
        "milestone": "Milestone 4.9 — Constrained PPO Hyperparameter Search Orchestrator Contract",
        "study_mode": "ppo_hparam_search",
        "search_method": "grid_product_v1",
        "sampler_seed": 42,
        "trial_seed": 42,
        "upstream_refs": {
            "run_id": run_id,
            "env_config_path": str(seeded["env_config_path"].resolve()),
            "state_manifest_path": str(seeded["state_manifest_path"].resolve()),
            "env_contract_report_path": str(seeded["env_contract_report_path"].resolve()),
            "readiness_report_path": str(seeded["readiness_report_path"].resolve()),
            "episode_catalog_path": str(seeded["episode_catalog_path"].resolve()),
            "split_report_path": str(seeded["split_report_path"].resolve()),
            "artifact_training_config_template_path": str(training_template_path.resolve()),
            "eval_config_template_path": str(eval_template_path.resolve()),
        },
        "search_space": {
            "learning_rate": [0.0002, 0.0003],
            "n_steps": [8],
            "batch_size": [4],
            "n_epochs": [2],
            "gamma": [0.99],
            "gae_lambda": [0.95],
            "clip_range": [0.2],
            "ent_coef": [0.0],
        },
        "resource_budget": {
            "max_trials": 2,
            "launcher_smoke_learn_timesteps": 4,
            "probe_train_total_timesteps": None,
            "full_train_total_timesteps": 16,
            "max_eval_episodes": 2,
            "max_eval_steps": 16,
        },
        "objective_spec": {
            "primary_metric": "excess_total_return",
            "turnover_penalty_weight": 0.05,
            "instability_penalty_weight": 0.05,
            "low_trade_count_penalty_weight": 0.05,
            "soft_trade_rate_target": 0.1,
        },
        "guardrail_spec": {
            "require_step_trace": True,
            "max_strategy_max_drawdown": 1.0,
            "min_num_trades_hard": 0,
            "min_num_trades_soft": 1,
            "max_trade_rate_hard": 1.0,
        },
        "pruning_spec": {
            "enabled": False,
            "warmup_trials": 0,
            "min_completed_probe_trials": 0,
            "min_probe_objective_score": None,
            "relative_to_best_completed_margin": 0.0,
        },
        "promotion_spec": {
            "candidate_top_k": 1,
            "promotion_min_distinct_seeds": 2,
            "require_positive_objective": True,
            "max_strategy_max_drawdown": 0.5,
            "min_num_trades": 1,
        },
        "output_root": str((tmp_path / f"{run_id}_study_output").resolve()),
    }

    if search_space_overrides:
        payload["search_space"].update(search_space_overrides)
    if resource_budget_overrides:
        payload["resource_budget"].update(resource_budget_overrides)
    if objective_overrides:
        payload["objective_spec"].update(objective_overrides)
    if guardrail_overrides:
        payload["guardrail_spec"].update(guardrail_overrides)
    if pruning_overrides:
        payload["pruning_spec"].update(pruning_overrides)
    if promotion_overrides:
        payload["promotion_spec"].update(promotion_overrides)

    study_config_path = tmp_path / f"{run_id}_study_config.json"
    study_config_path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "study_config_path": study_config_path,
        "output_root": Path(payload["output_root"]),
        "seeded": seeded,
        "training_template_path": training_template_path,
        "eval_template_path": eval_template_path,
    }
