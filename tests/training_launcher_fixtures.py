"""Shared fixtures for Milestone 4.7 training launcher tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rl.env_contract import parse_env_config, validate_env_contract
from rl.env_readiness import START_POLICY_VALID_FROM_ROW, validate_training_env_readiness
from rl.episode_selector import SELECTION_POLICY_FIXED, SELECTION_POLICY_SEEDED_RANDOM
from tests.rl_readiness_fixtures import patch_read_parquet, seed_state_run


def seed_training_launcher_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_id: str,
    *,
    entries: list[dict[str, Any]] | None = None,
    readiness_selection_policy: str = SELECTION_POLICY_SEEDED_RANDOM,
    selected_episode_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create canonical 4.6 artifacts that 4.7 can consume explicitly."""

    state_entries = entries or [
        {"partition": "train", "source_rel": "b_train.parquet", "row_count": 8},
        {"partition": "train", "source_rel": "a_train.parquet", "row_count": 8},
    ]
    state_root, env_config_path, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=state_entries,
        selected_episode_ref=selected_episode_ref,
    )
    patch_read_parquet(monkeypatch, frame_map)

    env_config_payload = json.loads(env_config_path.read_text(encoding="utf-8"))
    env_config = parse_env_config(env_config_payload)
    env_contract_result = validate_env_contract(
        config=env_config,
        smoke_step=False,
        invocation_args={"test_fixture": True},
    )
    env_contract_report_path = tmp_path / "runs" / run_id / "env_contract" / "reports" / "env_contract_report.json"
    env_contract_report_path.parent.mkdir(parents=True, exist_ok=True)
    env_contract_report_path.write_text(json.dumps(env_contract_result.report_payload), encoding="utf-8")

    readiness_result = validate_training_env_readiness(
        run_id=run_id,
        state_root=state_root,
        env_config_payload=env_config_payload,
        selection_policy=readiness_selection_policy,
        start_policy=START_POLICY_VALID_FROM_ROW,
        min_remaining_steps=2,
        seed=42,
    )
    episode_catalog_path = tmp_path / "runs" / run_id / "env_readiness" / "reports" / "episode_catalog.json"
    readiness_report_path = tmp_path / "runs" / run_id / "env_readiness" / "reports" / "training_env_readiness_report.json"
    episode_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    episode_catalog_path.write_text(json.dumps(readiness_result.catalog_payload), encoding="utf-8")
    readiness_report_path.write_text(json.dumps(readiness_result.readiness_payload), encoding="utf-8")

    return {
        "run_id": run_id,
        "state_root": state_root,
        "env_config_path": env_config_path,
        "training_config_path": None,
        "state_manifest_path": state_root / "reports" / "state_manifest.json",
        "env_contract_report_path": env_contract_report_path,
        "readiness_report_path": readiness_report_path,
        "episode_catalog_path": episode_catalog_path,
    }


def write_training_config(
    tmp_path: Path,
    run_id: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a strict 4.7 training config JSON file."""

    payload: dict[str, Any] = {
        "algorithm": "ppo",
        "seed": 42,
        "total_timesteps": 32,
        "device": "cpu",
        "episode_selection_mode": "seeded_random_episode",
        "startup_policy": "fresh_only",
        "smoke_mode": "prelaunch_only",
        "smoke_learn_timesteps": 8,
        "algo_params": {
            "learning_rate": 0.0003,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    }
    if overrides:
        payload.update(overrides)

    config_path = tmp_path / f"{run_id}_training_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


class FakeTrainingEnv:
    """Minimal env stub for launch_smoke tests."""

    def __init__(self, *, config: Any, validate_on_init: bool) -> None:
        del validate_on_init
        self.config = config
        self.closed = False

    def close(self) -> None:
        """Close the fake env."""

        self.closed = True


class FakePpo:
    """Minimal PPO stub for launch_smoke tests."""

    def __init__(self, policy: str, env: Any, seed: int, device: str | None, verbose: int, **kwargs: Any) -> None:
        self.policy = policy
        self.env = env
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.kwargs = kwargs
        self.num_timesteps = 0

    def learn(self, total_timesteps: int) -> "FakePpo":
        """Record bounded learn execution."""

        self.num_timesteps = int(total_timesteps)
        return self
