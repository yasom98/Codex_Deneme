"""Fixtures for canonical PPO artifact production tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

import pytest

from tests.evaluation_backtest_fixtures import seed_evaluation_run


def seed_artifact_production_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_id: str) -> dict[str, Any]:
    """Create explicit upstream artifacts consumable by artifact production."""

    return seed_evaluation_run(monkeypatch, tmp_path, run_id)


def write_artifact_training_config(
    tmp_path: Path,
    run_id: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a strict artifact production config JSON file."""

    payload: dict[str, Any] = {
        "algorithm": "ppo",
        "policy": "MlpPolicy",
        "seed": 42,
        "total_timesteps": 16,
        "device": "cpu",
        "episode_selection_mode": "seeded_random_episode",
        "startup_policy": "fresh_only",
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

    config_path = tmp_path / f"{run_id}_artifact_training_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


class FakeArtifactTrainingEnv:
    """Minimal env stub for canonical artifact production tests."""

    def __init__(self, *, config: Any, validate_on_init: bool) -> None:
        del validate_on_init
        self.config = config
        self.closed = False

    def close(self) -> None:
        """Close the fake env."""

        self.closed = True


class FakeArtifactPpo:
    """Minimal PPO stub with save/load behavior."""

    def __init__(self, policy: str, env: Any, seed: int, device: str | None, verbose: int, **kwargs: Any) -> None:
        self.policy = policy
        self.env = env
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.kwargs = kwargs
        self.num_timesteps = 0

    def learn(self, total_timesteps: int) -> "FakeArtifactPpo":
        """Record learn execution."""

        self.num_timesteps = int(total_timesteps)
        return self

    def save(self, path: str) -> None:
        """Persist a small zip artifact."""

        payload = {
            "policy": self.policy,
            "seed": self.seed,
            "device": self.device,
            "verbose": self.verbose,
            "kwargs": self.kwargs,
            "num_timesteps": self.num_timesteps,
        }
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("fake_model.json", json.dumps(payload))

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "FakeArtifactPpo":
        """Load the persisted fake artifact."""

        with zipfile.ZipFile(path, "r") as archive:
            payload = json.loads(archive.read("fake_model.json").decode("utf-8"))
        model = cls(
            policy=str(payload["policy"]),
            env=None,
            seed=int(payload["seed"]),
            device=device,
            verbose=int(payload["verbose"]),
            **dict(payload["kwargs"]),
        )
        model.num_timesteps = int(payload["num_timesteps"])
        return model
