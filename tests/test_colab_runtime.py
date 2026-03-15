"""Tests for closure-aware Colab staging and preflight validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest

from rl.colab_runtime import stage_explicit_inputs, validate_staged_preflight
from rl.colab_staging_closure import (
    CLOSURE_REPORT_FILENAME,
    RUNTIME_DEPENDENCY_REPORT_FILENAME,
    STAGING_MANIFEST_FILENAME,
)
from tests.evaluation_backtest_fixtures import seed_evaluation_run
from tests.ppo_artifact_production_fixtures import write_artifact_training_config


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write one JSON file for tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON payload from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    """Return SHA256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _patch_runtime_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch runtime dependency probes so staging tests are deterministic."""

    fake_torch = SimpleNamespace(
        __version__="2.1.0",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    available = {
        "torch": fake_torch,
        "gymnasium": object(),
        "stable_baselines3": object(),
        "pandas": object(),
        "pyarrow": object(),
    }

    def _fake_optional_import(module_name: str) -> tuple[object | None, str | None]:
        module = available.get(module_name)
        if module is None:
            return None, f"{module_name} missing"
        return module, None

    monkeypatch.setattr("rl.colab_staging_closure._optional_import", _fake_optional_import)


def _seed_closure_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, run_id: str) -> dict[str, Path]:
    """Seed a minimal but closure-complete source tree for Colab staging tests."""

    seeded = seed_evaluation_run(monkeypatch, tmp_path, run_id)
    run_root = tmp_path / "runs" / run_id
    state_root = run_root / "data_states"
    state_reports_root = state_root / "reports"
    env_contract_tmp_root = run_root / "env_contract" / "tmp"
    feature_root = run_root / "data_features"
    feature_reports_root = feature_root / "reports"
    feature_parquet_root = feature_root / "parquet"
    dataset_root = run_root / "data_datasets"
    dataset_reports_root = dataset_root / "reports"
    dataset_parquet_root = dataset_root / "parquet" / "partitions"

    split_report_path = seeded["split_report_path"]
    split_payload = _load_json(split_report_path)
    for file_report in split_payload["file_reports"]:
        input_file_path = Path(file_report["input_file"])
        input_file_path.parent.mkdir(parents=True, exist_ok=True)
        input_file_path.write_text("feature-placeholder", encoding="utf-8")

    feature_manifest_path = _write_json(
        feature_reports_root / "feature_manifest.json",
        {
            "run_id": run_id,
            "manifest_version": "features.manifest.v1",
            "feature_groups": ["core"],
        },
    )
    train_input_report_path = _write_json(
        feature_reports_root / "train_input_validation_report.json",
        {
            "run_id": run_id,
            "train_input_validation_overall": True,
            "manifest_path": str(feature_manifest_path.resolve()),
            "invocation_args": {
                "input_root": str(feature_parquet_root.resolve()),
                "reports_root": str(feature_reports_root.resolve()),
            },
            "file_reports": [{"input_file": item["input_file"]} for item in split_payload["file_reports"]],
        },
    )
    split_payload.update(
        {
            "manifest_path": str(feature_manifest_path.resolve()),
            "train_input_validation_report_path": str(train_input_report_path.resolve()),
            "invocation_args": {
                "input_root": str(feature_parquet_root.resolve()),
                "reports_root": str(feature_reports_root.resolve()),
            },
        }
    )
    _write_json(split_report_path, split_payload)

    state_manifest_path = seeded["state_manifest_path"]
    state_build_report_path = state_reports_root / "state_build_report.json"
    state_manifest_payload = _load_json(state_manifest_path)
    state_build_report_payload = _load_json(state_build_report_path)

    dataset_partition_metadata: list[dict[str, Any]] = []
    for item in state_manifest_payload["partition_metadata"]:
        parquet_path = dataset_parquet_root / str(item["partition"]) / str(item["source_rel"])
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_text("dataset-placeholder", encoding="utf-8")
        dataset_partition_metadata.append(
            {
                "scope": item["scope"],
                "partition": item["partition"],
                "source_rel": item["source_rel"],
                "fold_id": item["fold_id"],
                "output_path": str(parquet_path.resolve()),
            }
        )

    dataset_manifest_path = _write_json(
        dataset_reports_root / "dataset_manifest.json",
        {
            "run_id": run_id,
            "manifest_version": "datasets.manifest.v1",
            "output_completeness_ok": True,
            "source_lineage": {
                "feature_manifest_path": str(feature_manifest_path.resolve()),
                "train_input_validation_report_path": str(train_input_report_path.resolve()),
                "split_validation_report_path": str(split_report_path.resolve()),
            },
            "source_hashes": {
                "feature_manifest_hash": _sha256(feature_manifest_path),
                "train_input_report_hash": _sha256(train_input_report_path),
                "split_report_hash": _sha256(split_report_path),
                "source_file_inventory_hash": "dataset-inventory-hash",
            },
            "partition_metadata": dataset_partition_metadata,
        },
    )
    dataset_build_report_path = _write_json(
        dataset_reports_root / "dataset_build_report.json",
        {
            "run_id": run_id,
            "dataset_build_overall": True,
            "output_completeness_ok": True,
            "input_root": str(feature_parquet_root.resolve()),
            "output_root": str(dataset_root.resolve()),
            "dataset_build_report_path": str((dataset_reports_root / "dataset_build_report.json").resolve()),
            "dataset_manifest_path": str(dataset_manifest_path.resolve()),
            "source_paths": {
                "feature_manifest_path": str(feature_manifest_path.resolve()),
                "train_input_validation_report_path": str(train_input_report_path.resolve()),
                "split_validation_report_path": str(split_report_path.resolve()),
            },
            "source_hashes": {
                "feature_manifest_hash": _sha256(feature_manifest_path),
                "train_input_report_hash": _sha256(train_input_report_path),
                "split_report_hash": _sha256(split_report_path),
                "source_file_inventory_hash": "dataset-inventory-hash",
            },
            "invocation_args": {
                "input_root": str(feature_parquet_root.resolve()),
                "reports_root": str(feature_reports_root.resolve()),
                "output_root": str(dataset_root.resolve()),
                "feature_manifest_path": str(feature_manifest_path.resolve()),
                "train_input_report_path": str(train_input_report_path.resolve()),
                "split_report_path": str(split_report_path.resolve()),
            },
        },
    )

    state_manifest_payload["source_lineage"] = {
        "dataset_manifest_path": str(dataset_manifest_path.resolve()),
        "dataset_build_report_path": str(dataset_build_report_path.resolve()),
    }
    state_manifest_payload["source_hashes"] = {
        "dataset_manifest_hash": _sha256(dataset_manifest_path),
        "dataset_build_report_hash": _sha256(dataset_build_report_path),
        "source_file_inventory_hash": "state-inventory-hash",
    }
    _write_json(state_manifest_path, state_manifest_payload)

    state_build_report_payload.update(
        {
            "run_id": run_id,
            "state_build_overall": True,
            "output_completeness_ok": True,
            "input_root": str(dataset_root.resolve()),
            "output_root": str(state_root.resolve()),
            "state_build_report_path": str(state_build_report_path.resolve()),
            "state_manifest_path": str(state_manifest_path.resolve()),
            "scaler_stats_path": str((state_reports_root / "scaler_stats.json").resolve()),
            "source_paths": {
                "dataset_manifest_path": str(dataset_manifest_path.resolve()),
                "dataset_build_report_path": str(dataset_build_report_path.resolve()),
            },
            "source_hashes": {
                "dataset_manifest_hash": _sha256(dataset_manifest_path),
                "dataset_build_report_hash": _sha256(dataset_build_report_path),
                "source_file_inventory_hash": "state-inventory-hash",
            },
            "invocation_args": {
                "input_root": str(dataset_root.resolve()),
                "output_root": str(state_root.resolve()),
                "dataset_manifest_path": str(dataset_manifest_path.resolve()),
                "dataset_build_report_path": str(dataset_build_report_path.resolve()),
            },
        }
    )
    _write_json(state_build_report_path, state_build_report_payload)

    state_manifest_hash = _sha256(state_manifest_path)
    state_build_report_hash = _sha256(state_build_report_path)

    canonical_env_config_path = env_contract_tmp_root / seeded["env_config_path"].name
    canonical_env_config_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_env_config_path.write_text(seeded["env_config_path"].read_text(encoding="utf-8"), encoding="utf-8")

    env_contract_report_path = seeded["env_contract_report_path"]
    env_contract_payload = _load_json(env_contract_report_path)
    env_contract_payload.update(
        {
            "run_id": run_id,
            "env_contract_overall": True,
            "state_root": str(state_root.resolve()),
            "invocation_args": {
                **dict(env_contract_payload.get("invocation_args", {})),
                "state_root": str(state_root.resolve()),
                "env_config": str(canonical_env_config_path.resolve()),
                "report_path": str(env_contract_report_path.resolve()),
            },
            "source_lineage": {
                "state_manifest_path": str(state_manifest_path.resolve()),
                "state_build_report_path": str(state_build_report_path.resolve()),
                "state_manifest_hash": state_manifest_hash,
                "state_build_report_hash": state_build_report_hash,
            },
        }
    )
    _write_json(env_contract_report_path, env_contract_payload)

    episode_catalog_path = seeded["episode_catalog_path"]
    episode_catalog_payload = _load_json(episode_catalog_path)
    episode_catalog_payload.update(
        {
            "run_id": run_id,
            "state_root": str(state_root.resolve()),
            "catalog_path": str(episode_catalog_path.resolve()),
            "source_lineage": {
                "state_manifest_path": str(state_manifest_path.resolve()),
                "state_build_report_path": str(state_build_report_path.resolve()),
                "state_manifest_hash": state_manifest_hash,
                "state_build_report_hash": state_build_report_hash,
            },
        }
    )
    _write_json(episode_catalog_path, episode_catalog_payload)

    readiness_report_path = seeded["readiness_report_path"]
    readiness_payload = _load_json(readiness_report_path)
    readiness_payload.update(
        {
            "run_id": run_id,
            "state_root": str(state_root.resolve()),
            "catalog_path": str(episode_catalog_path.resolve()),
            "report_path": str(readiness_report_path.resolve()),
            "invocation_args": {
                **dict(readiness_payload.get("invocation_args", {})),
                "state_root": str(state_root.resolve()),
                "env_config": str(canonical_env_config_path.resolve()),
                "catalog_path": str(episode_catalog_path.resolve()),
                "report_path": str(readiness_report_path.resolve()),
            },
            "env_contract_reference": {
                "env_contract_overall": True,
                "source_lineage": {
                    "state_manifest_path": str(state_manifest_path.resolve()),
                    "state_build_report_path": str(state_build_report_path.resolve()),
                    "state_manifest_hash": state_manifest_hash,
                    "state_build_report_hash": state_build_report_hash,
                },
            },
        }
    )
    _write_json(readiness_report_path, readiness_payload)

    training_config_path = write_artifact_training_config(tmp_path, run_id)

    return {
        "env_config": canonical_env_config_path,
        "training_config": training_config_path,
        "state_manifest": state_manifest_path,
        "env_contract_report": env_contract_report_path,
        "readiness_report": readiness_report_path,
        "episode_catalog": episode_catalog_path,
        "split_report": split_report_path,
    }


def _copy_seeded_repo(source_repo_root: Path, extracted_repo_root: Path, source_paths: dict[str, Path]) -> dict[str, Path]:
    """Copy one seeded source repo tree into an extracted repo tree for cross-world tests."""

    shutil.copytree(source_repo_root, extracted_repo_root)
    extracted_paths: dict[str, Path] = {}
    for label, source_path in source_paths.items():
        extracted_paths[label] = extracted_repo_root / source_path.resolve().relative_to(source_repo_root.resolve())
    return extracted_paths


def test_stage_explicit_inputs_generates_closure_reports_and_normalized_dependency_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_dependencies(monkeypatch)
    source_paths = _seed_closure_source(monkeypatch, tmp_path, run_id="closure_stage_success")
    staging_root = tmp_path / "stage_success"

    manifest = stage_explicit_inputs(staging_root=staging_root, source_paths=source_paths)

    assert manifest["status"] == "success"
    assert (staging_root / STAGING_MANIFEST_FILENAME).exists()
    closure_report = _load_json(staging_root / CLOSURE_REPORT_FILENAME)
    runtime_report = _load_json(staging_root / RUNTIME_DEPENDENCY_REPORT_FILENAME)
    assert closure_report["overall_closure_valid"] is True
    assert runtime_report["runtime_dependency_overall"] is True
    assert sorted(closure_report["dependency_spec"].keys()) == [
        "optional_references",
        "projection_targets",
        "runtime_required",
        "seed_inputs",
    ]
    seed_labels = {item["label"] for item in closure_report["dependency_spec"]["seed_inputs"]}
    assert seed_labels == {
        "env_config",
        "training_config",
        "state_manifest",
        "env_contract_report",
        "readiness_report",
        "episode_catalog",
        "split_report",
    }
    runtime_labels = {item["label"] for item in closure_report["dependency_spec"]["runtime_required"]}
    assert {"state_build_report", "dataset_manifest", "dataset_build_report", "feature_manifest", "train_input_report"} <= runtime_labels
    seed_relpaths = {
        item["label"]: item["staged_relative_path"] for item in closure_report["dependency_spec"]["seed_inputs"]
    }
    assert seed_relpaths["env_config"] == "env_contract/tmp/closure_stage_success_env_config.json"
    optional_labels = {item["label"] for item in closure_report["dependency_spec"]["optional_references"]}
    assert "scaler_stats" in optional_labels
    staged_state_manifest = staging_root / "data_states" / "reports" / "state_manifest.json"
    assert staged_state_manifest.exists()
    assert "/mnt/c/" not in staged_state_manifest.read_text(encoding="utf-8")

    validation = validate_staged_preflight(staging_root=staging_root)
    assert validation["overall_valid"] is True


def test_stage_explicit_inputs_accepts_cross_world_source_evidence_and_extracted_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_dependencies(monkeypatch)
    run_id = "closure_stage_cross_world"
    source_repo_root = tmp_path / "source_world" / "Codex_Deneme"
    extracted_repo_root = tmp_path / "content" / "Codex_Deneme"
    source_paths = _seed_closure_source(monkeypatch, source_repo_root, run_id=run_id)
    extracted_paths = _copy_seeded_repo(source_repo_root, extracted_repo_root, source_paths)
    staging_root = tmp_path / "stage_cross_world"

    manifest = stage_explicit_inputs(staging_root=staging_root, source_paths=extracted_paths)

    assert manifest["status"] == "success"
    closure_report = _load_json(staging_root / CLOSURE_REPORT_FILENAME)
    assert closure_report["source_run_root"] == str((source_repo_root / "runs" / run_id).resolve())
    assert closure_report["extracted_run_root"] == str((extracted_repo_root / "runs" / run_id).resolve())
    staged_env_config = _load_json(staging_root / "env_contract" / "tmp" / extracted_paths["env_config"].name)
    assert staged_env_config["state_root"] == str((staging_root / "data_states").resolve())
    validation = validate_staged_preflight(staging_root=staging_root)
    assert validation["overall_valid"] is True


def test_stage_explicit_inputs_fails_closed_when_extracted_state_manifest_topology_is_noncanonical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_dependencies(monkeypatch)
    run_id = "closure_stage_bad_extracted_topology"
    source_repo_root = tmp_path / "source_world_bad" / "Codex_Deneme"
    extracted_repo_root = tmp_path / "content_bad" / "Codex_Deneme"
    source_paths = _seed_closure_source(monkeypatch, source_repo_root, run_id=run_id)
    extracted_paths = _copy_seeded_repo(source_repo_root, extracted_repo_root, source_paths)
    bad_state_manifest_path = extracted_repo_root / "alt" / "state_manifest.json"
    bad_state_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(extracted_paths["state_manifest"], bad_state_manifest_path)
    extracted_paths["state_manifest"] = bad_state_manifest_path
    staging_root = tmp_path / "stage_bad_extracted_topology"

    with pytest.raises(RuntimeError):
        stage_explicit_inputs(staging_root=staging_root, source_paths=extracted_paths)

    manifest = _load_json(staging_root / STAGING_MANIFEST_FILENAME)
    assert manifest["status"] == "failed"
    assert "state_manifest explicit input must be under data_states/reports/" in manifest["error"]


def test_stage_explicit_inputs_fails_closed_when_transitive_required_artifact_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_dependencies(monkeypatch)
    source_paths = _seed_closure_source(monkeypatch, tmp_path, run_id="closure_stage_missing_dataset")
    missing_dataset_build_report = (
        tmp_path / "runs" / "closure_stage_missing_dataset" / "data_datasets" / "reports" / "dataset_build_report.json"
    )
    missing_dataset_build_report.unlink()
    staging_root = tmp_path / "stage_missing_required"

    with pytest.raises(RuntimeError):
        stage_explicit_inputs(staging_root=staging_root, source_paths=source_paths)

    manifest = _load_json(staging_root / STAGING_MANIFEST_FILENAME)
    assert manifest["status"] == "failed"
    assert "dataset_build_report.json" in manifest["error"]


def test_validate_staged_preflight_fails_closed_on_residual_local_path_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_dependencies(monkeypatch)
    source_paths = _seed_closure_source(monkeypatch, tmp_path, run_id="closure_stage_leak")
    staging_root = tmp_path / "stage_leak"
    stage_explicit_inputs(staging_root=staging_root, source_paths=source_paths)

    split_report_path = staging_root / "data_features" / "reports" / "split_validation_report.json"
    split_payload = _load_json(split_report_path)
    split_payload["manifest_path"] = "/mnt/c/bad/feature_manifest.json"
    _write_json(split_report_path, split_payload)

    validation = validate_staged_preflight(staging_root=staging_root)

    assert validation["overall_valid"] is False
    assert validation["residual_local_path_leaks"]
    assert any(item["reason_code"] == "CLOSURE_LOCAL_PATH_LEAK" for item in validation["checks"])


def test_validate_staged_preflight_fails_closed_on_projected_runtime_path_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime_dependencies(monkeypatch)
    source_paths = _seed_closure_source(monkeypatch, tmp_path, run_id="closure_stage_runtime_mismatch")
    staging_root = tmp_path / "stage_runtime_mismatch"
    stage_explicit_inputs(staging_root=staging_root, source_paths=source_paths)

    env_config_path = staging_root / "env_contract" / "tmp" / source_paths["env_config"].name
    env_payload = _load_json(env_config_path)
    env_payload["state_root"] = str((staging_root / "wrong_data_states").resolve())
    _write_json(env_config_path, env_payload)

    validation = validate_staged_preflight(staging_root=staging_root)

    assert validation["overall_valid"] is False
    assert any(item["reason_code"] == "CLOSURE_RUNTIME_PATH_MISMATCH" for item in validation["checks"])
    assert any(item["label"] == "env_config" and item["field_path"] == "state_root" for item in validation["runtime_path_mismatches"])
