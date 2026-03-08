"""Shared fixtures for Milestone 4.8 evaluation/backtest tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

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


class FakePredictModel:
    """Simple scripted model stub for deterministic evaluation tests."""

    def __init__(self, actions: list[int] | None = None) -> None:
        self._actions = actions or [1, 0, 3]
        self._index = 0
        self.random_seed: int | None = None

    def set_random_seed(self, seed: int) -> None:
        """Record the provided seed."""

        self.random_seed = int(seed)

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, None]:
        """Return the next scripted action."""

        del observation, deterministic
        if self._index < len(self._actions):
            action = int(self._actions[self._index])
            self._index += 1
            return action, None
        return int(self._actions[-1]), None


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
