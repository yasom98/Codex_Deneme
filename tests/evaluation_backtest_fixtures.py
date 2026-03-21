"""Shared fixtures for Milestone 4.8 evaluation/backtest tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pytest

from rl.env_contract import parse_env_config, validate_env_contract
from rl.env_readiness import START_POLICY_VALID_FROM_ROW, validate_training_env_readiness
from tests.rl_readiness_fixtures import patch_read_parquet, seed_state_run


def seed_evaluation_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_id: str,
    *,
    entries: list[dict[str, Any]] | None = None,
    split_mode: str = "ratio_chrono",
) -> dict[str, Any]:
    """Create explicit upstream artifacts consumable by the 4.8 gate."""

    state_entries = entries or [
        {"partition": "train", "source_rel": "train_a.parquet", "row_count": 8},
        {"partition": "val", "source_rel": "val_a.parquet", "row_count": 8},
        {"partition": "val", "source_rel": "val_b.parquet", "row_count": 8},
        {"partition": "test", "source_rel": "test_a.parquet", "row_count": 8},
    ]
    state_root, env_config_path, frame_map = seed_state_run(tmp_path, run_id, entries=state_entries)
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
        selection_policy="seeded_random_episode",
        start_policy=START_POLICY_VALID_FROM_ROW,
        min_remaining_steps=2,
        seed=42,
    )
    episode_catalog_path = tmp_path / "runs" / run_id / "env_readiness" / "reports" / "episode_catalog.json"
    readiness_report_path = tmp_path / "runs" / run_id / "env_readiness" / "reports" / "training_env_readiness_report.json"
    episode_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    episode_catalog_path.write_text(json.dumps(readiness_result.catalog_payload), encoding="utf-8")
    readiness_report_path.write_text(json.dumps(readiness_result.readiness_payload), encoding="utf-8")

    split_report_path = _write_split_report(
        tmp_path=tmp_path,
        run_id=run_id,
        entries=state_entries,
        split_mode=split_mode,
    )
    model_artifact_path = _write_dummy_model_zip(tmp_path, run_id)

    return {
        "run_id": run_id,
        "state_root": state_root,
        "env_config_path": env_config_path,
        "state_manifest_path": state_root / "reports" / "state_manifest.json",
        "env_contract_report_path": env_contract_report_path,
        "readiness_report_path": readiness_report_path,
        "episode_catalog_path": episode_catalog_path,
        "split_report_path": split_report_path,
        "model_artifact_path": model_artifact_path,
    }


def write_eval_config(tmp_path: Path, run_id: str, *, overrides: dict[str, Any] | None = None) -> Path:
    """Write a strict 4.8 evaluation config JSON file."""

    payload: dict[str, Any] = {
        "algorithm": "ppo",
        "seed": 42,
        "deterministic": True,
        "device": "cpu",
        "evaluation_mode": "single_path_backtest",
        "target_mode": "explicit_episode_refs",
        "target_partition": None,
        "target_fold_id": None,
        "target_episode_refs": [
            {"scope": "partition", "partition": "val", "source_rel": "val_a.parquet", "fold_id": None}
        ],
        "benchmark_mode": "buy_and_hold",
        "startup_policy": "fresh_only",
        "max_eval_episodes": 1,
        "max_eval_steps": 16,
        "write_step_trace": False,
        "backtest_metrics": [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "num_steps",
            "num_trades",
            "win_rate",
            "avg_trade_return",
            "final_equity",
        ],
    }
    if overrides:
        payload.update(overrides)

    config_path = tmp_path / f"{run_id}_eval_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def write_risk_overlay_config(
    tmp_path: Path,
    *,
    instrument: str,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a strict 4.10 risk overlay config JSON file."""

    payload: dict[str, Any] = {
        "config_version": "risk_overlay.v1",
        "allowed_instruments": [instrument],
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
            "defensive_scale_down": 0.5,
        },
        "leverage_limits": {
            "max_leverage": 1.0,
            "defensive_scale_down": 0.5,
        },
        "drawdown_thresholds": {
            "defensive_enter_pct": 0.02,
            "defensive_exit_pct": 0.01,
            "freeze_enter_pct": 0.04,
            "freeze_exit_pct": 0.025,
            "kill_pct": 0.08,
        },
        "hysteresis_bands": {
            "min_steps_in_state": 1,
        },
        "recovery_policy": {
            "freeze_cooldown_steps": 1,
            "systemic_failure_kill_threshold": 2,
            "kill_requires_recovery_token": True,
        },
    }
    if overrides:
        payload.update(overrides)

    config_path = tmp_path / f"{instrument.replace('/', '_')}_risk_overlay_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


class FakePredictModel:
    """Simple scripted model stub for deterministic evaluation tests."""

    def __init__(
        self,
        actions: list[int] | None = None,
        *,
        deterministic_actions: list[int] | None = None,
        stochastic_actions: list[int] | None = None,
        deterministic_action_probabilities: list[list[float]] | None = None,
        distribution_probabilities_available: bool = True,
    ) -> None:
        default_actions = actions or [1, 0, 3]
        self._deterministic_actions = deterministic_actions or default_actions
        self._stochastic_actions = stochastic_actions or default_actions
        self._deterministic_action_probabilities = deterministic_action_probabilities
        self._deterministic_index = 0
        self._stochastic_index = 0
        self.random_seed: int | None = None
        self.action_masks_seen: list[Any] = []
        self.distribution_action_masks_seen: list[Any] = []
        self._distribution_probabilities_available = bool(distribution_probabilities_available)
        self.policy = _FakePredictPolicy(owner=self)

    def set_random_seed(self, seed: int) -> None:
        """Record the provided seed."""

        self.random_seed = int(seed)

    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
        action_masks: Any | None = None,
    ) -> tuple[int, None]:
        """Return the next scripted action."""

        del observation
        self.action_masks_seen.append(action_masks)
        action_sequence = self._deterministic_actions if deterministic else self._stochastic_actions
        index_attr = "_deterministic_index" if deterministic else "_stochastic_index"
        action_index = int(getattr(self, index_attr))
        if action_index < len(action_sequence):
            action = int(action_sequence[action_index])
            setattr(self, index_attr, int(action_index + 1))
            return action, None
        return int(action_sequence[-1]), None

    def current_deterministic_action_probabilities(self) -> np.ndarray | None:
        """Return the deterministic probability vector for the current decision index."""

        if not self._distribution_probabilities_available:
            return None
        if self._deterministic_action_probabilities is not None:
            if self._deterministic_index < len(self._deterministic_action_probabilities):
                values = self._deterministic_action_probabilities[self._deterministic_index]
            else:
                values = self._deterministic_action_probabilities[-1]
            return np.asarray(values, dtype=np.float32)
        action_index = min(self._deterministic_index, len(self._deterministic_actions) - 1)
        action = int(self._deterministic_actions[action_index])
        probabilities = np.full(shape=(4,), fill_value=0.05, dtype=np.float32)
        probabilities[action] = 0.85
        return probabilities


class _FakePredictPolicy:
    """Minimal policy surface for deterministic ranking diagnostics."""

    def __init__(self, *, owner: FakePredictModel) -> None:
        self._owner = owner

    def obs_to_tensor(self, observation: Any) -> tuple[Any, bool]:
        """Return the observation unchanged for diagnostics tests."""

        return observation, False

    def get_distribution(self, observation: Any, action_masks: Any | None = None) -> Any:
        """Return a fake distribution object exposing deterministic probabilities."""

        del observation
        self._owner.distribution_action_masks_seen.append(action_masks)
        probabilities = self._owner.current_deterministic_action_probabilities()
        if probabilities is None:
            return object()
        return _FakeDistributionWrapper(probabilities=probabilities)


class _FakeDistributionWrapper:
    """Wrapper that mimics the nested distribution.probs surface."""

    def __init__(self, *, probabilities: np.ndarray) -> None:
        self.distribution = _FakeDistributionTensor(probabilities=probabilities)


class _FakeDistributionTensor:
    """Tensor-like probability carrier for diagnostics tests."""

    def __init__(self, *, probabilities: np.ndarray) -> None:
        self.probs = np.asarray(probabilities, dtype=np.float32).reshape(1, -1)


def _write_split_report(
    *,
    tmp_path: Path,
    run_id: str,
    entries: list[dict[str, Any]],
    split_mode: str,
) -> Path:
    """Write a minimal split report aligned to the seeded episode refs."""

    reports_root = tmp_path / "runs" / run_id / "data_features" / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    file_reports: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for item in entries:
        source_rel = str(item["source_rel"])
        partition = str(item["partition"])
        input_file = str((tmp_path / "runs" / run_id / "data_features" / "parquet" / source_rel).resolve())
        report_payload = next((entry for entry in file_reports if entry["input_file"] == input_file), None)
        if report_payload is None:
            report_payload = {
                "input_file": input_file,
                "status": "success",
                "train_range": None,
                "val_range": None,
                "test_range": None,
                "fold_count": 0,
                "failed_fold_count": 0,
            }
            file_reports.append(report_payload)
        report_payload[f"{partition}_range"] = {"row_count": int(item.get("row_count", 8))}
        if str(item.get("scope", "partition")) == "fold":
            fold_reports.append(
                {
                    "fold_id": int(item["fold_id"]),
                    "input_file": input_file,
                    "train_range": {"row_count": int(item.get("row_count", 8))} if partition == "train" else None,
                    "val_range": {"row_count": int(item.get("row_count", 8))} if partition == "val" else None,
                    "test_range": {"row_count": int(item.get("row_count", 8))} if partition == "test" else None,
                }
            )

    payload = {
        "generated_at_utc": "2026-03-08T00:00:00+00:00",
        "run_id": run_id,
        "split_mode": split_mode,
        "split_validation_overall": True,
        "file_reports": file_reports,
        "fold_reports": fold_reports,
    }
    split_report_path = reports_root / "split_validation_report.json"
    split_report_path.write_text(json.dumps(payload), encoding="utf-8")
    return split_report_path


def _write_dummy_model_zip(tmp_path: Path, run_id: str) -> Path:
    """Write a dummy zip file that satisfies the canonical artifact stance."""

    path = tmp_path / "runs" / run_id / "models" / "ppo_policy.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("model.txt", "dummy")
    return path
