"""Tests for canonical PPO artifact production."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.ppo_artifact_production import (
    ARTIFACT_PRODUCTION_COLAB_RUNTIME_DEPENDENCY_FAILED,
    ARTIFACT_PRODUCTION_COLAB_STAGE_INVALID,
    ARTIFACT_PRODUCTION_COMPILE_UNSUPPORTED,
    ARTIFACT_PRODUCTION_OUTPUT_CONFLICT,
    CANONICAL_ARTIFACT_FILENAME,
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    execute_ppo_artifact_production,
)
from tests.test_colab_runtime import _load_json, _patch_runtime_dependencies, _seed_closure_source, _write_json
from rl.colab_staging_closure import CLOSURE_REPORT_FILENAME, RUNTIME_DEPENDENCY_REPORT_FILENAME
from tests.ppo_artifact_production_fixtures import (
    FakeArtifactPpo,
    FakeArtifactTrainingEnv,
    seed_artifact_production_run,
    write_artifact_training_config,
)


def _run_production(
    *,
    run_id: str,
    env_config_path: Path,
    training_config_path: Path,
    state_manifest_path: Path,
    env_contract_report_path: Path,
    readiness_report_path: Path,
    episode_catalog_path: Path,
    split_report_path: Path,
    output_dir: Path,
):
    return execute_ppo_artifact_production(
        run_id=run_id,
        env_config_path=env_config_path,
        training_config_path=training_config_path,
        state_manifest_path=state_manifest_path,
        env_contract_report_path=env_contract_report_path,
        readiness_report_path=readiness_report_path,
        episode_catalog_path=episode_catalog_path,
        split_report_path=split_report_path,
        output_dir=output_dir,
    )


def test_success_writes_canonical_artifact_manifest_and_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "artifact_production_success"
    seeded = seed_artifact_production_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_artifact_training_config(tmp_path, run_id)

    monkeypatch.setattr("rl.ppo_artifact_production.TradingEnvGym", FakeArtifactTrainingEnv)
    monkeypatch.setattr("rl.ppo_artifact_production._import_ppo_class", lambda: FakeArtifactPpo)

    result = _run_production(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "artifact_production_out",
    )

    assert result.exit_code == 0
    assert result.report_paths.artifact_path.name == CANONICAL_ARTIFACT_FILENAME
    assert result.report_paths.artifact_path.exists()
    assert result.report_paths.manifest_path.name == MANIFEST_FILENAME
    assert result.report_paths.report_path.name == REPORT_FILENAME

    manifest = json.loads(result.report_paths.manifest_path.read_text(encoding="utf-8"))
    report = json.loads(result.report_paths.report_path.read_text(encoding="utf-8"))

    assert manifest["artifact"]["filename"] == CANONICAL_ARTIFACT_FILENAME
    assert manifest["artifact"]["load_back_succeeded"] is True
    assert manifest["lineages"]["semantic_hashes"]["split_report_hash"] is not None
    assert report["canonical_artifact_ready"] is True
    assert report["save_succeeded"] is True
    assert report["artifact_exists"] is True
    assert report["artifact_zip_valid"] is True
    assert report["load_back_succeeded"] is True
    assert report["load_back_model_class"] == "FakeArtifactPpo"
    assert report["production_summary"]["canonicality_checks"]["split_report_hash_recorded"] is True
    assert report["runtime"]["progress"]["active_mode"] == "disabled"
    assert report["runtime"]["execution_bounds"]["n_envs"] == 1
    assert report["runtime"]["optimizations"]["amp"]["requested"] is False


def test_missing_split_report_fails_closed_and_writes_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "artifact_production_missing_split"
    seeded = seed_artifact_production_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_artifact_training_config(tmp_path, run_id)
    missing_split_report_path = tmp_path / "missing_split_report.json"

    result = _run_production(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=missing_split_report_path,
        output_dir=tmp_path / "artifact_production_missing_split_out",
    )

    assert result.exit_code == 2
    assert result.report_paths.report_path.exists()
    report = json.loads(result.report_paths.report_path.read_text(encoding="utf-8"))
    assert report["canonical_artifact_ready"] is False
    assert "ARTIFACT_PRODUCTION_INPUT_MISSING" in report["failure_codes"]


def test_output_conflict_returns_two_without_writing_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "artifact_production_output_conflict"
    seeded = seed_artifact_production_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_artifact_training_config(tmp_path, run_id)
    output_dir = tmp_path / "existing_artifact_output"
    output_dir.mkdir()

    result = _run_production(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=output_dir,
    )

    assert result.exit_code == 2
    assert result.reports_written is False
    assert ARTIFACT_PRODUCTION_OUTPUT_CONFLICT in result.report_payload["failure_codes"]


def test_compile_request_fails_closed_and_records_runtime_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "artifact_production_compile_unsupported"
    seeded = seed_artifact_production_run(monkeypatch, tmp_path, run_id)
    training_config_path = write_artifact_training_config(tmp_path, run_id)

    result = execute_ppo_artifact_production(
        run_id=run_id,
        env_config_path=seeded["env_config_path"],
        training_config_path=training_config_path,
        state_manifest_path=seeded["state_manifest_path"],
        env_contract_report_path=seeded["env_contract_report_path"],
        readiness_report_path=seeded["readiness_report_path"],
        episode_catalog_path=seeded["episode_catalog_path"],
        split_report_path=seeded["split_report_path"],
        output_dir=tmp_path / "artifact_compile_unsupported_out",
        enable_torch_compile=True,
    )

    assert result.exit_code == 2
    report = json.loads(result.report_paths.report_path.read_text(encoding="utf-8"))
    assert ARTIFACT_PRODUCTION_COMPILE_UNSUPPORTED in report["failure_codes"]
    assert report["runtime"]["optimizations"]["torch_compile"]["requested"] is True
    assert report["runtime"]["optimizations"]["torch_compile"]["enabled"] is False


def test_colab_stage_invalid_report_blocks_canonical_entrypoint_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "artifact_production_colab_stage_invalid"
    _patch_runtime_dependencies(monkeypatch)
    source_paths = _seed_closure_source(monkeypatch, tmp_path, run_id=run_id)
    staging_root = tmp_path / "stage_invalid_closure"
    from rl.colab_runtime import stage_explicit_inputs

    stage_explicit_inputs(staging_root=staging_root, source_paths=source_paths)
    closure_report_path = staging_root / CLOSURE_REPORT_FILENAME
    closure_report = _load_json(closure_report_path)
    closure_report["overall_closure_valid"] = False
    _write_json(closure_report_path, closure_report)

    result = execute_ppo_artifact_production(
        run_id=run_id,
        env_config_path=staging_root / "env_contract" / "tmp" / source_paths["env_config"].name,
        training_config_path=staging_root / "configs" / "training_config.json",
        state_manifest_path=staging_root / "data_states" / "reports" / "state_manifest.json",
        env_contract_report_path=staging_root / "env_contract" / "reports" / "env_contract_report.json",
        readiness_report_path=staging_root / "env_readiness" / "reports" / "training_env_readiness_report.json",
        episode_catalog_path=staging_root / "env_readiness" / "reports" / "episode_catalog.json",
        split_report_path=staging_root / "data_features" / "reports" / "split_validation_report.json",
        output_dir=tmp_path / "artifact_stage_invalid_out",
    )

    assert result.exit_code == 2
    assert ARTIFACT_PRODUCTION_COLAB_STAGE_INVALID in result.report_payload["failure_codes"]


def test_colab_runtime_dependency_invalid_report_blocks_canonical_entrypoint_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "artifact_production_colab_runtime_invalid"
    _patch_runtime_dependencies(monkeypatch)
    source_paths = _seed_closure_source(monkeypatch, tmp_path, run_id=run_id)
    staging_root = tmp_path / "stage_invalid_runtime"
    from rl.colab_runtime import stage_explicit_inputs

    stage_explicit_inputs(staging_root=staging_root, source_paths=source_paths)
    runtime_report_path = staging_root / RUNTIME_DEPENDENCY_REPORT_FILENAME
    runtime_report = _load_json(runtime_report_path)
    runtime_report["runtime_dependency_overall"] = False
    _write_json(runtime_report_path, runtime_report)

    result = execute_ppo_artifact_production(
        run_id=run_id,
        env_config_path=staging_root / "env_contract" / "tmp" / source_paths["env_config"].name,
        training_config_path=staging_root / "configs" / "training_config.json",
        state_manifest_path=staging_root / "data_states" / "reports" / "state_manifest.json",
        env_contract_report_path=staging_root / "env_contract" / "reports" / "env_contract_report.json",
        readiness_report_path=staging_root / "env_readiness" / "reports" / "training_env_readiness_report.json",
        episode_catalog_path=staging_root / "env_readiness" / "reports" / "episode_catalog.json",
        split_report_path=staging_root / "data_features" / "reports" / "split_validation_report.json",
        output_dir=tmp_path / "artifact_runtime_invalid_out",
    )

    assert result.exit_code == 2
    assert ARTIFACT_PRODUCTION_COLAB_RUNTIME_DEPENDENCY_FAILED in result.report_payload["failure_codes"]
