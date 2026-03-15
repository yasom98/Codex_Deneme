"""Closure-aware Colab staging helpers for canonical PPO artifact production."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.io_atomic import atomic_write_json
from core.logging import get_logger

LOGGER = get_logger(__name__)

STAGING_MANIFEST_FILENAME = "colab_input_staging_manifest.json"
CLOSURE_REPORT_FILENAME = "colab_staging_closure_report.json"
RUNTIME_DEPENDENCY_REPORT_FILENAME = "colab_runtime_dependency_report.json"

SEED_INPUT_LABELS: tuple[str, ...] = (
    "env_config",
    "training_config",
    "state_manifest",
    "env_contract_report",
    "readiness_report",
    "episode_catalog",
    "split_report",
)

_OPTIONAL_INPUT_LABELS: tuple[str, ...] = ("eval_config",)
_PROVENANCE_COPIED = "copied_evidence"
_PROVENANCE_PROJECTION = "staged_projection"
_RUNTIME_REQUIRED = "runtime_required"
_OPTIONAL_REFERENCE = "optional_reference"
_CONTRACT_CRITICAL_LEAK_PREFIXES: tuple[str, ...] = ("/mnt/c/",)


@dataclass(frozen=True)
class ArtifactSpec:
    """Normalized dependency-spec entry for one staged artifact."""

    label: str
    source_path: Path
    staged_relative_path: Path
    classification: str
    provenance_class: str
    projection_rule: str | None


@dataclass(frozen=True)
class WorldContexts:
    """Authoritative source-world and extracted-world roots for staging."""

    source_run_root: Path
    source_repo_root: Path
    extracted_run_root: Path
    extracted_repo_root: Path


def stage_dependency_closure(*, staging_root: Path, source_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Stage a closure-complete Colab root from canonical explicit inputs."""

    normalized_sources = {str(key): Path(value).resolve() for key, value in source_paths.items()}
    unsupported = sorted(set(normalized_sources) - set(SEED_INPUT_LABELS) - set(_OPTIONAL_INPUT_LABELS))
    if unsupported:
        raise ValueError(f"Unsupported staging labels: {unsupported}")

    missing_required = sorted(label for label in SEED_INPUT_LABELS if label not in normalized_sources)
    if missing_required:
        raise ValueError(f"Missing required staging labels: {missing_required}")

    staging_root_resolved = staging_root.resolve()
    if staging_root_resolved.exists():
        raise ValueError(f"staging_root must not exist before staging: {staging_root_resolved}")

    staging_root_resolved.mkdir(parents=True, exist_ok=False)

    try:
        resolution = _resolve_dependency_spec(normalized_sources)
        staged_records = _materialize_stage(
            staging_root=staging_root_resolved,
            artifact_specs=resolution["artifact_specs"],
            source_run_root=Path(str(resolution["source_run_root"])).resolve(),
            source_repo_root=Path(str(resolution["source_repo_root"])).resolve(),
            extracted_run_root=Path(str(resolution["extracted_run_root"])).resolve(),
            extracted_repo_root=Path(str(resolution["extracted_repo_root"])).resolve(),
        )
        closure_payload = _build_closure_report(
            staging_root=staging_root_resolved,
            resolution=resolution,
            staged_records=staged_records,
        )
        runtime_payload = _build_runtime_dependency_report(staging_root=staging_root_resolved)
        _validate_closure_report_payload(closure_payload)
        _validate_runtime_dependency_report_payload(runtime_payload)
        manifest_payload = _build_staging_manifest(
            staging_root=staging_root_resolved,
            closure_payload=closure_payload,
            runtime_payload=runtime_payload,
        )
        atomic_write_json(closure_payload, staging_root_resolved / CLOSURE_REPORT_FILENAME)
        atomic_write_json(runtime_payload, staging_root_resolved / RUNTIME_DEPENDENCY_REPORT_FILENAME)
        atomic_write_json(manifest_payload, staging_root_resolved / STAGING_MANIFEST_FILENAME)
        if manifest_payload["status"] != "success":
            raise RuntimeError(
                "Stage preflight failed | "
                f"overall_closure_valid={closure_payload['overall_closure_valid']} "
                f"runtime_dependency_overall={runtime_payload['runtime_dependency_overall']}"
            )
        return manifest_payload
    except Exception as exc:  # noqa: BLE001
        failure_payload = _build_failure_manifest(
            staging_root=staging_root_resolved,
            error_message=str(exc),
        )
        atomic_write_json(failure_payload, staging_root_resolved / STAGING_MANIFEST_FILENAME)
        LOGGER.error("Colab staging closure failed | staging_root=%s error=%s", staging_root_resolved, exc)
        raise RuntimeError(f"Colab staging closure failed: {exc}") from exc


def validate_existing_stage(*, staging_root: Path) -> dict[str, Any]:
    """Revalidate an existing staged root using the generated report contract."""

    staging_root_resolved = staging_root.resolve()
    closure_path = staging_root_resolved / CLOSURE_REPORT_FILENAME
    runtime_path = staging_root_resolved / RUNTIME_DEPENDENCY_REPORT_FILENAME
    if not closure_path.exists():
        raise ValueError(f"Missing closure report: {closure_path}")
    if not runtime_path.exists():
        raise ValueError(f"Missing runtime dependency report: {runtime_path}")

    closure_payload = _load_json_object(closure_path)
    runtime_payload = _load_json_object(runtime_path)
    _validate_closure_report_payload(closure_payload)
    _validate_runtime_dependency_report_payload(runtime_payload)

    validation_payload = _revalidate_stage_from_reports(
        staging_root=staging_root_resolved,
        closure_payload=closure_payload,
        runtime_payload=runtime_payload,
    )
    return validation_payload


def find_stage_root_for_inputs(*, input_paths: Sequence[Path]) -> Path | None:
    """Return the nearest ancestor containing the staged preflight reports."""

    for path in input_paths:
        resolved = path.resolve()
        if not resolved.exists():
            continue
        for candidate in (resolved.parent, *resolved.parents):
            if (candidate / CLOSURE_REPORT_FILENAME).exists() and (candidate / RUNTIME_DEPENDENCY_REPORT_FILENAME).exists():
                return candidate
    return None


def _resolve_dependency_spec(source_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Resolve authoritative runtime-required closure from canonical explicit inputs."""

    env_config = _load_json_object(source_paths["env_config"])
    state_manifest = _load_json_object(source_paths["state_manifest"])
    env_contract_report = _load_json_object(source_paths["env_contract_report"])
    readiness_report = _load_json_object(source_paths["readiness_report"])
    episode_catalog = _load_json_object(source_paths["episode_catalog"])
    split_report = _load_json_object(source_paths["split_report"])

    run_id = _require_string(env_config, "run_id", label="env_config.run_id")
    for label in ("state_manifest", "env_contract_report", "readiness_report", "episode_catalog", "split_report"):
        payload = {
            "state_manifest": state_manifest,
            "env_contract_report": env_contract_report,
            "readiness_report": readiness_report,
            "episode_catalog": episode_catalog,
            "split_report": split_report,
        }[label]
        payload_run_id = _require_string(payload, "run_id", label=f"{label}.run_id")
        if payload_run_id != run_id:
            raise ValueError(f"{label}.run_id mismatch: expected {run_id}, got {payload_run_id}")

    worlds = _derive_world_contexts(source_paths=source_paths, env_config=env_config, run_id=run_id)
    _validate_extracted_seed_inputs(source_paths=source_paths, worlds=worlds)

    expected_extracted_state_root = worlds.extracted_run_root / "data_states"
    mapped_state_root = _map_source_path_to_extracted(
        raw_path=_require_string(env_config, "state_root", label="env_config.state_root"),
        worlds=worlds,
        field_label="env_config.state_root",
    )
    if mapped_state_root != expected_extracted_state_root.resolve():
        raise ValueError(
            "env_config.state_root does not map to the extracted runtime state root: "
            f"{mapped_state_root} != {expected_extracted_state_root.resolve()}"
        )

    source_state_build_report_path = worlds.source_run_root / "data_states" / "reports" / "state_build_report.json"
    state_build_report_path = worlds.extracted_run_root / "data_states" / "reports" / "state_build_report.json"
    state_build_report = _load_json_object(state_build_report_path)

    dataset_manifest_path = _map_source_path_to_extracted(
        raw_path=_require_nested_string(
            state_manifest,
            ("source_lineage", "dataset_manifest_path"),
            label="state_manifest.source_lineage.dataset_manifest_path",
        ),
        worlds=worlds,
        field_label="state_manifest.source_lineage.dataset_manifest_path",
    )
    dataset_build_report_path = _map_source_path_to_extracted(
        raw_path=_require_nested_string(
            state_manifest,
            ("source_lineage", "dataset_build_report_path"),
            label="state_manifest.source_lineage.dataset_build_report_path",
        ),
        worlds=worlds,
        field_label="state_manifest.source_lineage.dataset_build_report_path",
    )
    dataset_manifest = _load_json_object(dataset_manifest_path)
    dataset_build_report = _load_json_object(dataset_build_report_path)

    feature_manifest_path = _map_source_path_to_extracted(
        raw_path=_require_nested_string(
            dataset_manifest,
            ("source_lineage", "feature_manifest_path"),
            label="dataset_manifest.source_lineage.feature_manifest_path",
        ),
        worlds=worlds,
        field_label="dataset_manifest.source_lineage.feature_manifest_path",
    )
    train_input_report_path = _map_source_path_to_extracted(
        raw_path=_require_nested_string(
            dataset_manifest,
            ("source_lineage", "train_input_validation_report_path"),
            label="dataset_manifest.source_lineage.train_input_validation_report_path",
        ),
        worlds=worlds,
        field_label="dataset_manifest.source_lineage.train_input_validation_report_path",
    )
    feature_manifest = _load_json_object(feature_manifest_path)
    train_input_report = _load_json_object(train_input_report_path)

    artifact_specs: list[ArtifactSpec] = []
    projection_targets: list[str] = []
    optional_refs: list[dict[str, Any]] = []

    def add_spec(
        *,
        label: str,
        source_path: Path,
        classification: str,
        provenance_class: str,
        projection_rule: str | None,
        staged_relative_path: Path | None = None,
    ) -> None:
        resolved_source = source_path.resolve()
        artifact_specs.append(
            ArtifactSpec(
                label=label,
                source_path=resolved_source,
                staged_relative_path=staged_relative_path or _default_staged_relative_path(
                    source_path=resolved_source,
                    source_run_root=worlds.extracted_run_root,
                ),
                classification=classification,
                provenance_class=provenance_class,
                projection_rule=projection_rule,
            )
        )
        if provenance_class == _PROVENANCE_PROJECTION:
            projection_targets.append(label)

    add_spec(
        label="training_config",
        source_path=source_paths["training_config"],
        classification=_RUNTIME_REQUIRED,
        provenance_class=_PROVENANCE_COPIED,
        projection_rule=None,
        staged_relative_path=Path("configs") / "training_config.json",
    )
    add_spec(label="feature_manifest", source_path=feature_manifest_path, classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_COPIED, projection_rule=None)
    feature_parquet_paths = sorted(
        {
            *(
                str(
                    _map_source_path_to_extracted(
                        raw_path=raw_path,
                        worlds=worlds,
                        field_label="split_report.file_reports.input_file",
                    )
                )
                for raw_path in _extract_feature_parquet_paths(split_report)
            ),
            *(
                str(
                    _map_source_path_to_extracted(
                        raw_path=raw_path,
                        worlds=worlds,
                        field_label="train_input_report.file_reports.input_file",
                    )
                )
                for raw_path in _extract_feature_parquet_paths(train_input_report)
            ),
        }
    )
    for index, path_value in enumerate(feature_parquet_paths):
        add_spec(
            label=f"feature_artifact_{index}",
            source_path=Path(path_value),
            classification=_RUNTIME_REQUIRED,
            provenance_class=_PROVENANCE_COPIED,
            projection_rule=None,
        )
    add_spec(label="train_input_report", source_path=train_input_report_path, classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="train_input_report_runtime_paths")
    add_spec(label="split_report", source_path=source_paths["split_report"], classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="split_report_runtime_paths")
    dataset_artifact_paths = sorted(
        {
            str(
                _map_source_path_to_extracted(
                    raw_path=raw_path,
                    worlds=worlds,
                    field_label="dataset_manifest.partition_metadata.output_path",
                )
            )
            for raw_path in _extract_output_paths(dataset_manifest)
        }
    )
    for index, path_value in enumerate(dataset_artifact_paths):
        add_spec(
            label=f"dataset_artifact_{index}",
            source_path=Path(path_value),
            classification=_RUNTIME_REQUIRED,
            provenance_class=_PROVENANCE_COPIED,
            projection_rule=None,
        )
    add_spec(label="dataset_manifest", source_path=dataset_manifest_path, classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="dataset_manifest_runtime_paths")
    add_spec(label="dataset_build_report", source_path=dataset_build_report_path, classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="dataset_build_report_runtime_paths")
    state_artifact_paths = sorted(
        {
            str(
                _map_source_path_to_extracted(
                    raw_path=raw_path,
                    worlds=worlds,
                    field_label="state_manifest.partition_metadata.output_path",
                )
            )
            for raw_path in _extract_output_paths(state_manifest)
        }
    )
    for index, path_value in enumerate(state_artifact_paths):
        add_spec(
            label=f"state_artifact_{index}",
            source_path=Path(path_value),
            classification=_RUNTIME_REQUIRED,
            provenance_class=_PROVENANCE_COPIED,
            projection_rule=None,
        )
    add_spec(label="state_manifest", source_path=source_paths["state_manifest"], classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="state_manifest_runtime_paths")
    add_spec(label="state_build_report", source_path=state_build_report_path, classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="state_build_report_runtime_paths")
    add_spec(
        label="env_config",
        source_path=source_paths["env_config"],
        classification=_RUNTIME_REQUIRED,
        provenance_class=_PROVENANCE_PROJECTION,
        projection_rule="env_config_runtime_paths",
        staged_relative_path=Path("env_contract") / "tmp" / source_paths["env_config"].name,
    )
    add_spec(label="env_contract_report", source_path=source_paths["env_contract_report"], classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="env_contract_runtime_paths")
    add_spec(label="readiness_report", source_path=source_paths["readiness_report"], classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="readiness_runtime_paths")
    add_spec(label="episode_catalog", source_path=source_paths["episode_catalog"], classification=_RUNTIME_REQUIRED, provenance_class=_PROVENANCE_PROJECTION, projection_rule="episode_catalog_runtime_paths")

    if "eval_config" in source_paths:
        add_spec(
            label="eval_config",
            source_path=source_paths["eval_config"],
            classification=_OPTIONAL_REFERENCE,
            provenance_class=_PROVENANCE_COPIED,
            projection_rule=None,
            staged_relative_path=Path("configs") / "eval_config.json",
        )

    scaler_stats_path_raw = _optional_nested_string(
        state_build_report,
        ("scaler_stats_path",),
    ) or _optional_nested_string(state_manifest, ("scaler_stats_ref",))
    if scaler_stats_path_raw:
        scaler_stats_path = _map_source_path_to_extracted(
            raw_path=scaler_stats_path_raw,
            worlds=worlds,
            field_label="state_build_report.scaler_stats_path",
        )
        optional_refs.append(
            {
                "label": "scaler_stats",
                "source_path": str(scaler_stats_path),
                "staged_relative_path": str(
                    _default_staged_relative_path(source_path=scaler_stats_path, source_run_root=worlds.extracted_run_root)
                ),
                "exists_in_source": scaler_stats_path.exists(),
            }
        )
        if scaler_stats_path.exists():
            add_spec(
                label="scaler_stats",
                source_path=scaler_stats_path,
                classification=_OPTIONAL_REFERENCE,
                provenance_class=_PROVENANCE_COPIED,
                projection_rule=None,
            )

    resolution = {
        "run_id": run_id,
        "source_run_root": worlds.source_run_root,
        "source_repo_root": worlds.source_repo_root,
        "extracted_run_root": worlds.extracted_run_root,
        "extracted_repo_root": worlds.extracted_repo_root,
        "artifact_specs": _dedupe_specs(artifact_specs),
        "dependency_spec": {
            "seed_inputs": [],
            "runtime_required": [],
            "projection_targets": [],
            "optional_references": [],
        },
        "projection_targets": projection_targets,
    }

    for spec in resolution["artifact_specs"]:
        spec_dict = _spec_to_dict(spec)
        if spec.label in SEED_INPUT_LABELS:
            resolution["dependency_spec"]["seed_inputs"].append(spec_dict)
        if spec.classification == _RUNTIME_REQUIRED and spec.label not in SEED_INPUT_LABELS:
            resolution["dependency_spec"]["runtime_required"].append(spec_dict)
        if spec.provenance_class == _PROVENANCE_PROJECTION:
            resolution["dependency_spec"]["projection_targets"].append(spec_dict)
        if spec.classification == _OPTIONAL_REFERENCE:
            resolution["dependency_spec"]["optional_references"].append(spec_dict)

    for item in optional_refs:
        if item["label"] not in {entry["label"] for entry in resolution["dependency_spec"]["optional_references"]}:
            resolution["dependency_spec"]["optional_references"].append(item)

    _validate_source_lineage_consistency(
        worlds=worlds,
        env_config=env_config,
        state_build_report_path=state_build_report_path,
        state_manifest_path=source_paths["state_manifest"],
        state_manifest_payload=state_manifest,
        env_contract_report=env_contract_report,
        readiness_report=readiness_report,
        episode_catalog=episode_catalog,
        dataset_manifest_path=dataset_manifest_path,
        dataset_build_report_path=dataset_build_report_path,
        dataset_manifest_payload=dataset_manifest,
        dataset_build_report_payload=dataset_build_report,
        split_report_path=source_paths["split_report"],
        split_report_payload=split_report,
        train_input_report_path=train_input_report_path,
        train_input_report_payload=train_input_report,
        feature_manifest_path=feature_manifest_path,
        source_state_build_report_path=source_state_build_report_path,
    )
    return resolution


def _materialize_stage(
    *,
    staging_root: Path,
    artifact_specs: Sequence[ArtifactSpec],
    source_run_root: Path,
    source_repo_root: Path,
    extracted_run_root: Path,
    extracted_repo_root: Path,
) -> list[dict[str, Any]]:
    """Copy/projection materialization in deterministic dependency order."""

    source_to_stage = {
        str(spec.source_path.resolve()): staging_root / spec.staged_relative_path
        for spec in artifact_specs
    }
    label_to_stage = {spec.label: staging_root / spec.staged_relative_path for spec in artifact_specs}
    path_mapper = _PathMapper(
        source_run_root=source_run_root,
        source_repo_root=source_repo_root,
        extracted_run_root=extracted_run_root,
        extracted_repo_root=extracted_repo_root,
        staging_root=staging_root,
        source_to_stage=source_to_stage,
    )

    staged_records: list[dict[str, Any]] = []

    for spec in artifact_specs:
        destination_path = staging_root / spec.staged_relative_path
        if spec.provenance_class == _PROVENANCE_COPIED:
            record = _copy_artifact(spec=spec, destination_path=destination_path)
        else:
            record = _project_artifact(
                spec=spec,
                destination_path=destination_path,
                path_mapper=path_mapper,
                label_to_stage=label_to_stage,
            )
        staged_records.append(record)

    return staged_records


def _build_closure_report(
    *,
    staging_root: Path,
    resolution: Mapping[str, Any],
    staged_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the machine-readable closure report."""

    leak_findings: list[dict[str, Any]] = []
    missing_required: list[dict[str, Any]] = []
    for record in staged_records:
        staged_path = Path(str(record["staged_path"]))
        if record["classification"] == _RUNTIME_REQUIRED and not staged_path.exists():
            missing_required.append({"label": record["label"], "staged_path": str(staged_path)})
        if record["provenance_class"] != _PROVENANCE_PROJECTION:
            continue
        payload = _load_json_object(staged_path)
        leak_findings.extend(
            _find_residual_local_path_leaks(
                label=str(record["label"]),
                payload=payload,
                staged_path=staged_path,
            )
        )

    runtime_path_mismatches = _collect_runtime_path_mismatches(
        artifacts=staged_records,
        staging_root=staging_root,
    )
    overall = len(missing_required) == 0 and len(leak_findings) == 0 and len(runtime_path_mismatches) == 0
    return {
        "generated_at_utc": _generated_at(),
        "staging_root": str(staging_root),
        "source_run_root": str(resolution["source_run_root"]),
        "source_repo_root": str(resolution["source_repo_root"]),
        "extracted_run_root": str(resolution["extracted_run_root"]),
        "extracted_repo_root": str(resolution["extracted_repo_root"]),
        "dependency_spec": dict(resolution["dependency_spec"]),
        "artifacts": [dict(record) for record in staged_records],
        "copied_files": [record["staged_path"] for record in staged_records if record["provenance_class"] == _PROVENANCE_COPIED],
        "rewritten_files": [record["staged_path"] for record in staged_records if record["provenance_class"] == _PROVENANCE_PROJECTION],
        "missing_required_dependencies": missing_required,
        "residual_local_path_leaks": leak_findings,
        "runtime_path_mismatches": runtime_path_mismatches,
        "closure_checks": [
            {
                "check_name": "runtime_required_present",
                "pass": len(missing_required) == 0,
                "reason_code": None if len(missing_required) == 0 else "CLOSURE_REQUIRED_ARTIFACT_MISSING",
                "detail": {"missing_count": len(missing_required)},
            },
            {
                "check_name": "contract_critical_local_path_leaks_absent",
                "pass": len(leak_findings) == 0,
                "reason_code": None if len(leak_findings) == 0 else "CLOSURE_LOCAL_PATH_LEAK",
                "detail": {"leak_count": len(leak_findings)},
            },
            {
                "check_name": "runtime_path_contract_consistent",
                "pass": len(runtime_path_mismatches) == 0,
                "reason_code": None if len(runtime_path_mismatches) == 0 else "CLOSURE_RUNTIME_PATH_MISMATCH",
                "detail": {"mismatch_count": len(runtime_path_mismatches)},
            },
        ],
        "overall_closure_valid": overall,
    }


def _build_runtime_dependency_report(*, staging_root: Path) -> dict[str, Any]:
    """Build machine-readable runtime dependency preflight."""

    python_version = {
        "major": int(os.sys.version_info.major),
        "minor": int(os.sys.version_info.minor),
        "micro": int(os.sys.version_info.micro),
    }
    torch_module, torch_error = _optional_import("torch")
    gym_module, gym_error = _optional_import("gymnasium")
    sb3_module, sb3_error = _optional_import("stable_baselines3")
    pandas_module, pandas_error = _optional_import("pandas")
    pyarrow_module, pyarrow_error = _optional_import("pyarrow")

    cuda_available = bool(torch_module is not None and bool(torch_module.cuda.is_available()))
    dependency_probe = {
        "python_version": python_version,
        "python_supported": (python_version["major"], python_version["minor"]) >= (3, 10),
        "torch_available": torch_module is not None,
        "torch_error": torch_error,
        "torch_version": getattr(torch_module, "__version__", None) if torch_module is not None else None,
        "cuda_available": cuda_available,
        "gpu_name": _gpu_name(torch_module) if torch_module is not None and cuda_available else None,
        "stable_baselines3_available": sb3_module is not None,
        "stable_baselines3_error": sb3_error,
        "gymnasium_available": gym_module is not None,
        "gymnasium_error": gym_error,
        "pandas_available": pandas_module is not None,
        "pandas_error": pandas_error,
        "pyarrow_available": pyarrow_module is not None,
        "pyarrow_error": pyarrow_error,
    }
    checks = [
        _dependency_check("python_version_supported", dependency_probe["python_supported"], detail={"python_version": python_version}),
        _dependency_check("torch_importable", torch_module is not None, detail={"error": torch_error}),
        _dependency_check("stable_baselines3_importable", sb3_module is not None, detail={"error": sb3_error}),
        _dependency_check("gymnasium_importable", gym_module is not None, detail={"error": gym_error}),
        _dependency_check("pandas_importable", pandas_module is not None, detail={"error": pandas_error}),
        _dependency_check("pyarrow_importable", pyarrow_module is not None, detail={"error": pyarrow_error}),
    ]
    return {
        "generated_at_utc": _generated_at(),
        "staging_root": str(staging_root),
        "dependency_probe": dependency_probe,
        "checks": checks,
        "runtime_dependency_overall": all(bool(item["pass"]) for item in checks),
    }


def _build_staging_manifest(
    *,
    staging_root: Path,
    closure_payload: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the orchestration manifest for one staged root."""

    status = "success" if bool(closure_payload["overall_closure_valid"]) and bool(runtime_payload["runtime_dependency_overall"]) else "failed"
    return {
        "generated_at_utc": _generated_at(),
        "status": status,
        "staging_root": str(staging_root),
        "report_paths": {
            "closure_report_path": str(staging_root / CLOSURE_REPORT_FILENAME),
            "runtime_dependency_report_path": str(staging_root / RUNTIME_DEPENDENCY_REPORT_FILENAME),
        },
        "dependency_spec_sections": {
            "seed_inputs": len(closure_payload["dependency_spec"]["seed_inputs"]),
            "runtime_required": len(closure_payload["dependency_spec"]["runtime_required"]),
            "projection_targets": len(closure_payload["dependency_spec"]["projection_targets"]),
            "optional_references": len(closure_payload["dependency_spec"]["optional_references"]),
        },
        "overall_closure_valid": bool(closure_payload["overall_closure_valid"]),
        "runtime_dependency_overall": bool(runtime_payload["runtime_dependency_overall"]),
    }


def _build_failure_manifest(*, staging_root: Path, error_message: str) -> dict[str, Any]:
    """Build a narrow failure manifest when stage orchestration aborts."""

    return {
        "generated_at_utc": _generated_at(),
        "status": "failed",
        "staging_root": str(staging_root),
        "report_paths": {
            "closure_report_path": str(staging_root / CLOSURE_REPORT_FILENAME),
            "runtime_dependency_report_path": str(staging_root / RUNTIME_DEPENDENCY_REPORT_FILENAME),
        },
        "error": error_message,
    }


def _copy_artifact(*, spec: ArtifactSpec, destination_path: Path) -> dict[str, Any]:
    """Copy one non-projected artifact with byte-for-byte verification."""

    source_sha = _sha256_file(spec.source_path)
    payload = spec.source_path.read_bytes()
    _atomic_write_bytes(payload, destination_path)
    destination_sha = _sha256_file(destination_path)
    if source_sha != destination_sha:
        raise RuntimeError(f"Byte-for-byte copy verification failed for {spec.label}")
    return {
        "label": spec.label,
        "classification": spec.classification,
        "provenance_class": spec.provenance_class,
        "runtime_required": spec.classification == _RUNTIME_REQUIRED,
        "projection_rule": spec.projection_rule,
        "source_path": str(spec.source_path),
        "staged_path": str(destination_path),
        "source_sha256": source_sha,
        "staged_sha256": destination_sha,
        "rewrite_applied": False,
        "hash_updates": [],
    }


def _project_artifact(
    *,
    spec: ArtifactSpec,
    destination_path: Path,
    path_mapper: "_PathMapper",
    label_to_stage: Mapping[str, Path],
) -> dict[str, Any]:
    """Apply one file-specific staged projection and write atomically."""

    source_payload = _load_json_object(spec.source_path)
    projected_payload = _apply_projection_rule(
        label=spec.label,
        payload=source_payload,
        path_mapper=path_mapper,
    )
    hash_updates = _apply_hash_rewrites(
        label=spec.label,
        payload=projected_payload,
        label_to_stage=label_to_stage,
    )
    atomic_write_json(projected_payload, destination_path)
    return {
        "label": spec.label,
        "classification": spec.classification,
        "provenance_class": spec.provenance_class,
        "runtime_required": spec.classification == _RUNTIME_REQUIRED,
        "projection_rule": spec.projection_rule,
        "source_path": str(spec.source_path),
        "staged_path": str(destination_path),
        "source_sha256": _sha256_file(spec.source_path),
        "staged_sha256": _sha256_file(destination_path),
        "rewrite_applied": True,
        "hash_updates": hash_updates,
    }


def _apply_projection_rule(
    *,
    label: str,
    payload: dict[str, Any],
    path_mapper: "_PathMapper",
) -> dict[str, Any]:
    """Apply file-specific path rebasing rules for one contract-critical JSON."""

    cloned = json.loads(json.dumps(payload))

    def rewrite(keys: Sequence[str], *, allow_directory: bool = True) -> None:
        value = _get_nested(cloned, keys)
        if not isinstance(value, str) or not value.strip():
            return
        rebased = path_mapper.rebase(value, allow_directory=allow_directory)
        _set_nested(cloned, keys, rebased)

    if label == "env_config":
        rewrite(("state_root",))
        return cloned

    if label == "train_input_report":
        rewrite(("manifest_path",), allow_directory=False)
        rewrite(("invocation_args", "input_root"))
        rewrite(("invocation_args", "reports_root"))
        for index, item in enumerate(cloned.get("file_reports", [])):
            if isinstance(item, Mapping):
                rewrite(("file_reports", str(index), "input_file"), allow_directory=False)
        return cloned

    if label == "split_report":
        rewrite(("manifest_path",), allow_directory=False)
        rewrite(("train_input_validation_report_path",), allow_directory=False)
        rewrite(("invocation_args", "input_root"))
        rewrite(("invocation_args", "reports_root"))
        for index, item in enumerate(cloned.get("file_reports", [])):
            if isinstance(item, Mapping):
                rewrite(("file_reports", str(index), "input_file"), allow_directory=False)
        for index, item in enumerate(cloned.get("fold_reports", [])):
            if isinstance(item, Mapping):
                rewrite(("fold_reports", str(index), "input_file"), allow_directory=False)
        return cloned

    if label == "dataset_manifest":
        rewrite(("source_lineage", "feature_manifest_path"), allow_directory=False)
        rewrite(("source_lineage", "train_input_validation_report_path"), allow_directory=False)
        rewrite(("source_lineage", "split_validation_report_path"), allow_directory=False)
        for index, item in enumerate(cloned.get("partition_metadata", [])):
            if isinstance(item, Mapping):
                rewrite(("partition_metadata", str(index), "output_path"), allow_directory=False)
        return cloned

    if label == "dataset_build_report":
        for keys in (
            ("input_root",),
            ("output_root",),
            ("dataset_build_report_path",),
            ("dataset_manifest_path",),
            ("source_paths", "feature_manifest_path"),
            ("source_paths", "train_input_validation_report_path"),
            ("source_paths", "split_validation_report_path"),
            ("invocation_args", "input_root"),
            ("invocation_args", "reports_root"),
            ("invocation_args", "output_root"),
            ("invocation_args", "feature_manifest_path"),
            ("invocation_args", "train_input_report_path"),
            ("invocation_args", "split_report_path"),
        ):
            rewrite(keys, allow_directory=keys[-1] in {"input_root", "reports_root", "output_root"})
        return cloned

    if label == "state_manifest":
        rewrite(("source_lineage", "dataset_manifest_path"), allow_directory=False)
        rewrite(("source_lineage", "dataset_build_report_path"), allow_directory=False)
        for index, item in enumerate(cloned.get("partition_metadata", [])):
            if isinstance(item, Mapping):
                rewrite(("partition_metadata", str(index), "output_path"), allow_directory=False)
        scaler_ref = _optional_nested_string(cloned, ("scaler_stats_ref",))
        if scaler_ref:
            rewrite(("scaler_stats_ref",), allow_directory=False)
        return cloned

    if label == "state_build_report":
        for keys in (
            ("input_root",),
            ("output_root",),
            ("state_build_report_path",),
            ("state_manifest_path",),
            ("scaler_stats_path",),
            ("source_paths", "dataset_manifest_path"),
            ("source_paths", "dataset_build_report_path"),
            ("invocation_args", "input_root"),
            ("invocation_args", "output_root"),
            ("invocation_args", "dataset_manifest_path"),
            ("invocation_args", "dataset_build_report_path"),
        ):
            rewrite(keys, allow_directory=keys[-1] in {"input_root", "output_root"})
        return cloned

    if label == "env_contract_report":
        for keys in (
            ("state_root",),
            ("source_lineage", "state_manifest_path"),
            ("source_lineage", "state_build_report_path"),
            ("invocation_args", "state_root"),
            ("invocation_args", "env_config"),
            ("invocation_args", "report_path"),
        ):
            rewrite(keys, allow_directory=keys[-1] == "state_root")
        return cloned

    if label == "readiness_report":
        for keys in (
            ("state_root",),
            ("env_contract_reference", "source_lineage", "state_manifest_path"),
            ("env_contract_reference", "source_lineage", "state_build_report_path"),
            ("catalog_path",),
            ("report_path",),
            ("invocation_args", "state_root"),
            ("invocation_args", "env_config"),
            ("invocation_args", "catalog_path"),
            ("invocation_args", "report_path"),
        ):
            rewrite(keys, allow_directory=keys[-1] == "state_root")
        return cloned

    if label == "episode_catalog":
        for keys in (
            ("state_root",),
            ("source_lineage", "state_manifest_path"),
            ("source_lineage", "state_build_report_path"),
            ("catalog_path",),
        ):
            rewrite(keys, allow_directory=keys[-1] == "state_root")
        return cloned

    raise ValueError(f"Unsupported projection target: {label}")


def _apply_hash_rewrites(
    *,
    label: str,
    payload: dict[str, Any],
    label_to_stage: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Update lineage hashes after staged projections have been written."""

    hash_updates: list[dict[str, Any]] = []

    def update(keys: Sequence[str], *, target_label: str) -> None:
        target_path = _find_staged_path_by_label(label=target_label, label_to_stage=label_to_stage)
        if target_path is None:
            raise ValueError(f"Unable to resolve staged hash target for {target_label}")
        old_value = _get_nested(payload, keys)
        new_value = _sha256_file(target_path)
        _set_nested(payload, keys, new_value)
        hash_updates.append({"field_path": ".".join(keys), "old_value": old_value, "new_value": new_value, "target_label": target_label})

    if label == "dataset_manifest":
        update(("source_hashes", "feature_manifest_hash"), target_label="feature_manifest")
        update(("source_hashes", "train_input_report_hash"), target_label="train_input_report")
        update(("source_hashes", "split_report_hash"), target_label="split_report")
    elif label == "dataset_build_report":
        update(("source_hashes", "feature_manifest_hash"), target_label="feature_manifest")
        update(("source_hashes", "train_input_report_hash"), target_label="train_input_report")
        update(("source_hashes", "split_report_hash"), target_label="split_report")
    elif label == "state_manifest":
        update(("source_hashes", "dataset_manifest_hash"), target_label="dataset_manifest")
        update(("source_hashes", "dataset_build_report_hash"), target_label="dataset_build_report")
    elif label == "state_build_report":
        update(("source_hashes", "dataset_manifest_hash"), target_label="dataset_manifest")
        update(("source_hashes", "dataset_build_report_hash"), target_label="dataset_build_report")
    elif label == "env_contract_report":
        update(("source_lineage", "state_manifest_hash"), target_label="state_manifest")
        update(("source_lineage", "state_build_report_hash"), target_label="state_build_report")
    elif label == "readiness_report":
        update(("env_contract_reference", "source_lineage", "state_manifest_hash"), target_label="state_manifest")
        update(("env_contract_reference", "source_lineage", "state_build_report_hash"), target_label="state_build_report")
    elif label == "episode_catalog":
        update(("source_lineage", "state_manifest_hash"), target_label="state_manifest")
        update(("source_lineage", "state_build_report_hash"), target_label="state_build_report")
    return hash_updates


def _revalidate_stage_from_reports(
    *,
    staging_root: Path,
    closure_payload: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate staged root against persisted reports to catch drift."""

    missing_required: list[dict[str, Any]] = []
    staged_path_mismatches: list[dict[str, Any]] = []
    stale_hash_mismatches: list[dict[str, Any]] = []
    residual_local_path_leaks: list[dict[str, Any]] = []
    runtime_path_mismatches: list[dict[str, Any]] = []

    artifacts = closure_payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("closure report artifacts must be list")

    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError("closure report artifacts entries must be objects")
        label = str(item.get("label", ""))
        staged_path_raw = item.get("staged_path")
        expected_sha = item.get("staged_sha256")
        runtime_required = bool(item.get("runtime_required", False))
        rewrite_applied = bool(item.get("rewrite_applied", False))
        if not isinstance(staged_path_raw, str) or not staged_path_raw.strip():
            raise ValueError("closure report artifact staged_path must be non-empty string")
        staged_path = Path(staged_path_raw)
        if runtime_required and not staged_path.exists():
            missing_required.append({"label": label, "staged_path": str(staged_path)})
            continue
        if not str(staged_path.resolve()).startswith(str(staging_root.resolve())):
            staged_path_mismatches.append({"label": label, "staged_path": str(staged_path)})
        if staged_path.exists() and isinstance(expected_sha, str):
            actual_sha = _sha256_file(staged_path)
            if actual_sha != expected_sha:
                stale_hash_mismatches.append(
                    {
                        "label": label,
                        "staged_path": str(staged_path),
                        "expected_sha256": expected_sha,
                        "actual_sha256": actual_sha,
                    }
                )
        if rewrite_applied and staged_path.exists():
            residual_local_path_leaks.extend(
                _find_residual_local_path_leaks(label=label, payload=_load_json_object(staged_path), staged_path=staged_path)
            )

    runtime_path_mismatches = _collect_runtime_path_mismatches(
        artifacts=artifacts,
        staging_root=staging_root,
    )
    current_runtime_report = _build_runtime_dependency_report(staging_root=staging_root)
    checks = [
        {
            "check_name": "persisted_closure_report_valid",
            "pass": bool(closure_payload.get("overall_closure_valid", False)),
            "reason_code": None if bool(closure_payload.get("overall_closure_valid", False)) else "CLOSURE_REPORT_INVALID",
            "detail": {},
        },
        {
            "check_name": "persisted_runtime_dependency_report_valid",
            "pass": bool(runtime_payload.get("runtime_dependency_overall", False)),
            "reason_code": None if bool(runtime_payload.get("runtime_dependency_overall", False)) else "RUNTIME_DEPENDENCY_REPORT_INVALID",
            "detail": {},
        },
        {
            "check_name": "required_artifacts_present",
            "pass": len(missing_required) == 0,
            "reason_code": None if len(missing_required) == 0 else "CLOSURE_REQUIRED_ARTIFACT_MISSING",
            "detail": {"missing_count": len(missing_required)},
        },
        {
            "check_name": "artifact_hashes_unchanged",
            "pass": len(stale_hash_mismatches) == 0,
            "reason_code": None if len(stale_hash_mismatches) == 0 else "CLOSURE_STAGE_DRIFT_DETECTED",
            "detail": {"mismatch_count": len(stale_hash_mismatches)},
        },
        {
            "check_name": "contract_critical_local_path_leaks_absent",
            "pass": len(residual_local_path_leaks) == 0,
            "reason_code": None if len(residual_local_path_leaks) == 0 else "CLOSURE_LOCAL_PATH_LEAK",
            "detail": {"leak_count": len(residual_local_path_leaks)},
        },
        {
            "check_name": "runtime_path_contract_consistent",
            "pass": len(runtime_path_mismatches) == 0,
            "reason_code": None if len(runtime_path_mismatches) == 0 else "CLOSURE_RUNTIME_PATH_MISMATCH",
            "detail": {"mismatch_count": len(runtime_path_mismatches)},
        },
        {
            "check_name": "runtime_dependency_probe_revalidated",
            "pass": bool(current_runtime_report["runtime_dependency_overall"]),
            "reason_code": None if bool(current_runtime_report["runtime_dependency_overall"]) else "RUNTIME_DEPENDENCY_PROBE_FAILED",
            "detail": {},
        },
    ]

    return {
        "generated_at_utc": _generated_at(),
        "staging_root": str(staging_root),
        "checks": checks,
        "missing_required_dependencies": missing_required,
        "staged_path_mismatches": staged_path_mismatches,
        "stale_hash_mismatches": stale_hash_mismatches,
        "residual_local_path_leaks": residual_local_path_leaks,
        "runtime_path_mismatches": runtime_path_mismatches,
        "runtime_dependency_report_revalidated": current_runtime_report,
        "overall_valid": all(bool(item["pass"]) for item in checks),
    }


def _collect_runtime_path_mismatches(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    staging_root: Path,
) -> list[dict[str, Any]]:
    """Validate projection-time runtime path contracts against staged truth."""

    label_to_stage = {
        str(item.get("label")): Path(str(item.get("staged_path"))).resolve()
        for item in artifacts
        if isinstance(item, Mapping) and isinstance(item.get("staged_path"), str)
    }
    mismatches: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("provenance_class")) != _PROVENANCE_PROJECTION:
            continue
        label = str(item.get("label"))
        staged_path_raw = item.get("staged_path")
        if not isinstance(staged_path_raw, str) or not staged_path_raw.strip():
            continue
        staged_path = Path(staged_path_raw).resolve()
        if not staged_path.exists():
            continue
        payload = _load_json_object(staged_path)
        mismatches.extend(
            _validate_projected_runtime_paths(
                label=label,
                payload=payload,
                staged_path=staged_path,
                label_to_stage=label_to_stage,
                staging_root=staging_root.resolve(),
            )
        )
    return mismatches


def _validate_projected_runtime_paths(
    *,
    label: str,
    payload: Mapping[str, Any],
    staged_path: Path,
    label_to_stage: Mapping[str, Path],
    staging_root: Path,
) -> list[dict[str, Any]]:
    """Validate one projected JSON payload against staged runtime expectations."""

    mismatches: list[dict[str, Any]] = []

    def expect(keys: Sequence[str], expected_path: Path, *, allow_directory: bool = False) -> None:
        actual_raw = _get_nested(payload, keys)
        if not isinstance(actual_raw, str) or not actual_raw.strip():
            mismatches.append(
                {
                    "label": label,
                    "staged_path": str(staged_path),
                    "field_path": ".".join(keys),
                    "reason": "missing_path",
                    "expected_path": str(expected_path.resolve()),
                    "actual_path": actual_raw,
                }
            )
            return
        actual_path = Path(actual_raw).resolve()
        if actual_path != expected_path.resolve():
            mismatches.append(
                {
                    "label": label,
                    "staged_path": str(staged_path),
                    "field_path": ".".join(keys),
                    "reason": "path_mismatch",
                    "expected_path": str(expected_path.resolve()),
                    "actual_path": str(actual_path),
                }
            )
            return
        if not actual_path.exists():
            mismatches.append(
                {
                    "label": label,
                    "staged_path": str(staged_path),
                    "field_path": ".".join(keys),
                    "reason": "path_missing",
                    "expected_path": str(expected_path.resolve()),
                    "actual_path": str(actual_path),
                }
            )
            return
        if allow_directory:
            if not actual_path.is_dir():
                mismatches.append(
                    {
                        "label": label,
                        "staged_path": str(staged_path),
                        "field_path": ".".join(keys),
                        "reason": "expected_directory",
                        "expected_path": str(expected_path.resolve()),
                        "actual_path": str(actual_path),
                    }
                )
        elif not actual_path.is_file():
            mismatches.append(
                {
                    "label": label,
                    "staged_path": str(staged_path),
                    "field_path": ".".join(keys),
                    "reason": "expected_file",
                    "expected_path": str(expected_path.resolve()),
                    "actual_path": str(actual_path),
                }
            )

    def expect_within_stage(keys: Sequence[str], *, allow_directory: bool = False) -> None:
        actual_raw = _get_nested(payload, keys)
        if not isinstance(actual_raw, str) or not actual_raw.strip():
            mismatches.append(
                {
                    "label": label,
                    "staged_path": str(staged_path),
                    "field_path": ".".join(keys),
                    "reason": "missing_path",
                    "expected_path": None,
                    "actual_path": actual_raw,
                }
            )
            return
        actual_path = Path(actual_raw).resolve()
        if not _is_relative_to(actual_path, staging_root):
            mismatches.append(
                {
                    "label": label,
                    "staged_path": str(staged_path),
                    "field_path": ".".join(keys),
                    "reason": "outside_staging_root",
                    "expected_path": str(staging_root),
                    "actual_path": str(actual_path),
                }
            )
            return
        if not actual_path.exists():
            mismatches.append(
                {
                    "label": label,
                    "staged_path": str(staged_path),
                    "field_path": ".".join(keys),
                    "reason": "path_missing",
                    "expected_path": str(staging_root),
                    "actual_path": str(actual_path),
                }
            )
            return
        if allow_directory:
            if not actual_path.is_dir():
                mismatches.append(
                    {
                        "label": label,
                        "staged_path": str(staged_path),
                        "field_path": ".".join(keys),
                        "reason": "expected_directory",
                        "expected_path": str(staging_root),
                        "actual_path": str(actual_path),
                    }
                )
        elif not actual_path.is_file():
            mismatches.append(
                {
                    "label": label,
                    "staged_path": str(staged_path),
                    "field_path": ".".join(keys),
                    "reason": "expected_file",
                    "expected_path": str(staging_root),
                    "actual_path": str(actual_path),
                }
            )

    feature_reports_root = staging_root / "data_features" / "reports"
    feature_parquet_root = staging_root / "data_features" / "parquet"
    dataset_root = staging_root / "data_datasets"
    state_root = staging_root / "data_states"

    if label == "env_config":
        expect(("state_root",), state_root, allow_directory=True)
        return mismatches
    if label == "train_input_report":
        expect(("manifest_path",), label_to_stage["feature_manifest"])
        expect(("invocation_args", "input_root"), feature_parquet_root, allow_directory=True)
        expect(("invocation_args", "reports_root"), feature_reports_root, allow_directory=True)
        for index, item in enumerate(payload.get("file_reports", [])):
            if isinstance(item, Mapping):
                expect_within_stage(("file_reports", str(index), "input_file"))
        return mismatches
    if label == "split_report":
        expect(("manifest_path",), label_to_stage["feature_manifest"])
        expect(("train_input_validation_report_path",), label_to_stage["train_input_report"])
        expect(("invocation_args", "input_root"), feature_parquet_root, allow_directory=True)
        expect(("invocation_args", "reports_root"), feature_reports_root, allow_directory=True)
        for index, item in enumerate(payload.get("file_reports", [])):
            if isinstance(item, Mapping):
                expect_within_stage(("file_reports", str(index), "input_file"))
        for index, item in enumerate(payload.get("fold_reports", [])):
            if isinstance(item, Mapping):
                expect_within_stage(("fold_reports", str(index), "input_file"))
        return mismatches
    if label == "dataset_manifest":
        expect(("source_lineage", "feature_manifest_path"), label_to_stage["feature_manifest"])
        expect(("source_lineage", "train_input_validation_report_path"), label_to_stage["train_input_report"])
        expect(("source_lineage", "split_validation_report_path"), label_to_stage["split_report"])
        for index, item in enumerate(payload.get("partition_metadata", [])):
            if isinstance(item, Mapping):
                expect_within_stage(("partition_metadata", str(index), "output_path"))
        return mismatches
    if label == "dataset_build_report":
        expect(("input_root",), feature_parquet_root, allow_directory=True)
        expect(("output_root",), dataset_root, allow_directory=True)
        expect(("dataset_build_report_path",), label_to_stage["dataset_build_report"])
        expect(("dataset_manifest_path",), label_to_stage["dataset_manifest"])
        expect(("source_paths", "feature_manifest_path"), label_to_stage["feature_manifest"])
        expect(("source_paths", "train_input_validation_report_path"), label_to_stage["train_input_report"])
        expect(("source_paths", "split_validation_report_path"), label_to_stage["split_report"])
        expect(("invocation_args", "input_root"), feature_parquet_root, allow_directory=True)
        expect(("invocation_args", "reports_root"), feature_reports_root, allow_directory=True)
        expect(("invocation_args", "output_root"), dataset_root, allow_directory=True)
        expect(("invocation_args", "feature_manifest_path"), label_to_stage["feature_manifest"])
        expect(("invocation_args", "train_input_report_path"), label_to_stage["train_input_report"])
        expect(("invocation_args", "split_report_path"), label_to_stage["split_report"])
        return mismatches
    if label == "state_manifest":
        expect(("source_lineage", "dataset_manifest_path"), label_to_stage["dataset_manifest"])
        expect(("source_lineage", "dataset_build_report_path"), label_to_stage["dataset_build_report"])
        for index, item in enumerate(payload.get("partition_metadata", [])):
            if isinstance(item, Mapping):
                expect_within_stage(("partition_metadata", str(index), "output_path"))
        if "scaler_stats" in label_to_stage and _optional_nested_string(payload, ("scaler_stats_ref",)):
            expect(("scaler_stats_ref",), label_to_stage["scaler_stats"])
        return mismatches
    if label == "state_build_report":
        expect(("input_root",), dataset_root, allow_directory=True)
        expect(("output_root",), state_root, allow_directory=True)
        expect(("state_build_report_path",), label_to_stage["state_build_report"])
        expect(("state_manifest_path",), label_to_stage["state_manifest"])
        if "scaler_stats" in label_to_stage and _optional_nested_string(payload, ("scaler_stats_path",)):
            expect(("scaler_stats_path",), label_to_stage["scaler_stats"])
        expect(("source_paths", "dataset_manifest_path"), label_to_stage["dataset_manifest"])
        expect(("source_paths", "dataset_build_report_path"), label_to_stage["dataset_build_report"])
        expect(("invocation_args", "input_root"), dataset_root, allow_directory=True)
        expect(("invocation_args", "output_root"), state_root, allow_directory=True)
        expect(("invocation_args", "dataset_manifest_path"), label_to_stage["dataset_manifest"])
        expect(("invocation_args", "dataset_build_report_path"), label_to_stage["dataset_build_report"])
        return mismatches
    if label == "env_contract_report":
        expect(("state_root",), state_root, allow_directory=True)
        expect(("source_lineage", "state_manifest_path"), label_to_stage["state_manifest"])
        expect(("source_lineage", "state_build_report_path"), label_to_stage["state_build_report"])
        expect(("invocation_args", "state_root"), state_root, allow_directory=True)
        expect(("invocation_args", "env_config"), label_to_stage["env_config"])
        expect(("invocation_args", "report_path"), label_to_stage["env_contract_report"])
        return mismatches
    if label == "readiness_report":
        expect(("state_root",), state_root, allow_directory=True)
        expect(("catalog_path",), label_to_stage["episode_catalog"])
        expect(("report_path",), label_to_stage["readiness_report"])
        expect(("env_contract_reference", "source_lineage", "state_manifest_path"), label_to_stage["state_manifest"])
        expect(("env_contract_reference", "source_lineage", "state_build_report_path"), label_to_stage["state_build_report"])
        expect(("invocation_args", "state_root"), state_root, allow_directory=True)
        expect(("invocation_args", "env_config"), label_to_stage["env_config"])
        expect(("invocation_args", "catalog_path"), label_to_stage["episode_catalog"])
        expect(("invocation_args", "report_path"), label_to_stage["readiness_report"])
        return mismatches
    if label == "episode_catalog":
        expect(("state_root",), state_root, allow_directory=True)
        expect(("catalog_path",), label_to_stage["episode_catalog"])
        expect(("source_lineage", "state_manifest_path"), label_to_stage["state_manifest"])
        expect(("source_lineage", "state_build_report_path"), label_to_stage["state_build_report"])
        return mismatches
    return mismatches


def _validate_closure_report_payload(payload: Mapping[str, Any]) -> None:
    """Validate closure report contract structurally and semantically."""

    if not isinstance(payload.get("dependency_spec"), Mapping):
        raise ValueError("closure report dependency_spec is required")
    dependency_spec = payload["dependency_spec"]
    for section in ("seed_inputs", "runtime_required", "projection_targets", "optional_references"):
        if not isinstance(dependency_spec.get(section), list):
            raise ValueError(f"closure report dependency_spec.{section} must be list")
    if not isinstance(payload.get("artifacts"), list):
        raise ValueError("closure report artifacts must be list")
    if "overall_closure_valid" not in payload:
        raise ValueError("closure report overall_closure_valid is required")


def _validate_runtime_dependency_report_payload(payload: Mapping[str, Any]) -> None:
    """Validate runtime dependency report contract structurally."""

    if not isinstance(payload.get("checks"), list):
        raise ValueError("runtime dependency report checks must be list")
    if not isinstance(payload.get("dependency_probe"), Mapping):
        raise ValueError("runtime dependency report dependency_probe is required")
    if "runtime_dependency_overall" not in payload:
        raise ValueError("runtime dependency report runtime_dependency_overall is required")


def _derive_world_contexts(*, source_paths: Mapping[str, Path], env_config: Mapping[str, Any], run_id: str) -> WorldContexts:
    """Derive authoritative source-world and extracted-world roots."""

    source_state_root = Path(_require_string(env_config, "state_root", label="env_config.state_root")).resolve()
    source_run_root = source_state_root.parent.resolve()
    source_repo_root = source_run_root.parent.parent.resolve()

    extracted_state_manifest_path = source_paths["state_manifest"].resolve()
    if extracted_state_manifest_path.name != "state_manifest.json":
        raise ValueError("state_manifest explicit input must point to state_manifest.json")
    if extracted_state_manifest_path.parent.name != "reports" or extracted_state_manifest_path.parent.parent.name != "data_states":
        raise ValueError("state_manifest explicit input must be under data_states/reports/")
    extracted_run_root = extracted_state_manifest_path.parents[2]
    if extracted_run_root.name != run_id:
        raise ValueError(
            "state_manifest explicit input run root does not match run_id: "
            f"{extracted_run_root.name} != {run_id}"
        )
    if extracted_run_root.parent.name != "runs":
        raise ValueError("state_manifest explicit input must live under runs/<run_id>/")
    extracted_repo_root = extracted_run_root.parent.parent.resolve()
    return WorldContexts(
        source_run_root=source_run_root,
        source_repo_root=source_repo_root,
        extracted_run_root=extracted_run_root.resolve(),
        extracted_repo_root=extracted_repo_root,
    )


def _validate_extracted_seed_inputs(*, source_paths: Mapping[str, Path], worlds: WorldContexts) -> None:
    """Validate explicit extracted inputs against canonical extracted topology."""

    expected_paths = {
        "state_manifest": worlds.extracted_run_root / "data_states" / "reports" / "state_manifest.json",
        "env_contract_report": worlds.extracted_run_root / "env_contract" / "reports" / "env_contract_report.json",
        "readiness_report": worlds.extracted_run_root / "env_readiness" / "reports" / "training_env_readiness_report.json",
        "episode_catalog": worlds.extracted_run_root / "env_readiness" / "reports" / "episode_catalog.json",
        "split_report": worlds.extracted_run_root / "data_features" / "reports" / "split_validation_report.json",
    }
    for label, expected_path in expected_paths.items():
        actual_path = source_paths[label].resolve()
        if actual_path != expected_path.resolve():
            raise ValueError(f"{label} explicit input must match canonical extracted path: {actual_path} != {expected_path.resolve()}")

    env_config_path = source_paths["env_config"].resolve()
    expected_env_tmp_root = (worlds.extracted_run_root / "env_contract" / "tmp").resolve()
    if not _is_relative_to(env_config_path, expected_env_tmp_root):
        raise ValueError(
            "env_config explicit input must live under extracted env_contract/tmp/: "
            f"{env_config_path} not under {expected_env_tmp_root}"
        )


def _map_source_path_to_extracted(
    *,
    raw_path: str,
    worlds: WorldContexts,
    field_label: str,
) -> Path:
    """Map one authoritative source-world path into the extracted-source world."""

    source_path = Path(raw_path).resolve()
    if _is_relative_to(source_path, worlds.source_run_root):
        return (worlds.extracted_run_root / source_path.relative_to(worlds.source_run_root)).resolve()
    if _is_relative_to(source_path, worlds.source_repo_root):
        return (worlds.extracted_repo_root / source_path.relative_to(worlds.source_repo_root)).resolve()
    raise ValueError(
        f"{field_label} cannot be mapped from source world into extracted world: "
        f"{source_path} is outside {worlds.source_run_root} and {worlds.source_repo_root}"
    )


def _validate_source_lineage_consistency(
    *,
    worlds: WorldContexts,
    env_config: Mapping[str, Any],
    source_state_build_report_path: Path,
    state_build_report_path: Path,
    state_manifest_path: Path,
    state_manifest_payload: Mapping[str, Any],
    env_contract_report: Mapping[str, Any],
    readiness_report: Mapping[str, Any],
    episode_catalog: Mapping[str, Any],
    dataset_manifest_path: Path,
    dataset_build_report_path: Path,
    dataset_manifest_payload: Mapping[str, Any],
    dataset_build_report_payload: Mapping[str, Any],
    split_report_path: Path,
    split_report_payload: Mapping[str, Any],
    train_input_report_path: Path,
    train_input_report_payload: Mapping[str, Any],
    feature_manifest_path: Path,
) -> None:
    """Validate authoritative source lineage before stage projection."""

    mapped_state_root = _map_source_path_to_extracted(
        raw_path=_require_string(env_config, "state_root", label="env_config.state_root"),
        worlds=worlds,
        field_label="env_config.state_root",
    )
    if mapped_state_root != (worlds.extracted_run_root / "data_states").resolve():
        raise ValueError("env_config.state_root extracted mapping mismatch")

    source_lineage = _require_mapping(env_contract_report, "source_lineage", label="env_contract_report.source_lineage")
    if _map_source_path_to_extracted(
        raw_path=_require_string(source_lineage, "state_manifest_path", label="env_contract_report.source_lineage.state_manifest_path"),
        worlds=worlds,
        field_label="env_contract_report.source_lineage.state_manifest_path",
    ) != state_manifest_path.resolve():
        raise ValueError("env_contract_report.source_lineage.state_manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(source_lineage, "state_build_report_path", label="env_contract_report.source_lineage.state_build_report_path"),
        worlds=worlds,
        field_label="env_contract_report.source_lineage.state_build_report_path",
    ) != state_build_report_path.resolve():
        raise ValueError("env_contract_report.source_lineage.state_build_report_path mismatch")
    if _require_string(source_lineage, "state_manifest_hash", label="env_contract_report.source_lineage.state_manifest_hash") != _sha256_file(state_manifest_path):
        raise ValueError("env_contract_report.source_lineage.state_manifest_hash mismatch")
    if _require_string(source_lineage, "state_build_report_hash", label="env_contract_report.source_lineage.state_build_report_hash") != _sha256_file(state_build_report_path):
        raise ValueError("env_contract_report.source_lineage.state_build_report_hash mismatch")

    readiness_source_lineage = _require_mapping(
        _require_mapping(readiness_report, "env_contract_reference", label="readiness_report.env_contract_reference"),
        "source_lineage",
        label="readiness_report.env_contract_reference.source_lineage",
    )
    if _map_source_path_to_extracted(
        raw_path=_require_string(readiness_source_lineage, "state_manifest_path", label="readiness_report.env_contract_reference.source_lineage.state_manifest_path"),
        worlds=worlds,
        field_label="readiness_report.env_contract_reference.source_lineage.state_manifest_path",
    ) != state_manifest_path.resolve():
        raise ValueError("readiness_report.source_lineage.state_manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(readiness_source_lineage, "state_build_report_path", label="readiness_report.env_contract_reference.source_lineage.state_build_report_path"),
        worlds=worlds,
        field_label="readiness_report.env_contract_reference.source_lineage.state_build_report_path",
    ) != state_build_report_path.resolve():
        raise ValueError("readiness_report.source_lineage.state_build_report_path mismatch")

    catalog_source_lineage = _require_mapping(episode_catalog, "source_lineage", label="episode_catalog.source_lineage")
    if _map_source_path_to_extracted(
        raw_path=_require_string(catalog_source_lineage, "state_manifest_path", label="episode_catalog.source_lineage.state_manifest_path"),
        worlds=worlds,
        field_label="episode_catalog.source_lineage.state_manifest_path",
    ) != state_manifest_path.resolve():
        raise ValueError("episode_catalog.source_lineage.state_manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(catalog_source_lineage, "state_build_report_path", label="episode_catalog.source_lineage.state_build_report_path"),
        worlds=worlds,
        field_label="episode_catalog.source_lineage.state_build_report_path",
    ) != state_build_report_path.resolve():
        raise ValueError("episode_catalog.source_lineage.state_build_report_path mismatch")

    state_build_payload = _load_json_object(state_build_report_path)
    state_build_source = _require_mapping(state_build_payload, "source_paths", label="state_build_report.source_paths")
    if source_state_build_report_path.resolve() != (worlds.source_run_root / "data_states" / "reports" / "state_build_report.json").resolve():
        raise ValueError("source_state_build_report_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(state_build_source, "dataset_manifest_path", label="state_build_report.source_paths.dataset_manifest_path"),
        worlds=worlds,
        field_label="state_build_report.source_paths.dataset_manifest_path",
    ) != dataset_manifest_path.resolve():
        raise ValueError("state_build_report.source_paths.dataset_manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(state_build_source, "dataset_build_report_path", label="state_build_report.source_paths.dataset_build_report_path"),
        worlds=worlds,
        field_label="state_build_report.source_paths.dataset_build_report_path",
    ) != dataset_build_report_path.resolve():
        raise ValueError("state_build_report.source_paths.dataset_build_report_path mismatch")

    state_manifest_source = _require_mapping(state_manifest_payload, "source_lineage", label="state_manifest.source_lineage")
    if _map_source_path_to_extracted(
        raw_path=_require_string(state_manifest_source, "dataset_manifest_path", label="state_manifest.source_lineage.dataset_manifest_path"),
        worlds=worlds,
        field_label="state_manifest.source_lineage.dataset_manifest_path",
    ) != dataset_manifest_path.resolve():
        raise ValueError("state_manifest.source_lineage.dataset_manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(state_manifest_source, "dataset_build_report_path", label="state_manifest.source_lineage.dataset_build_report_path"),
        worlds=worlds,
        field_label="state_manifest.source_lineage.dataset_build_report_path",
    ) != dataset_build_report_path.resolve():
        raise ValueError("state_manifest.source_lineage.dataset_build_report_path mismatch")

    dataset_manifest_source = _require_mapping(dataset_manifest_payload, "source_lineage", label="dataset_manifest.source_lineage")
    if _map_source_path_to_extracted(
        raw_path=_require_string(dataset_manifest_source, "feature_manifest_path", label="dataset_manifest.source_lineage.feature_manifest_path"),
        worlds=worlds,
        field_label="dataset_manifest.source_lineage.feature_manifest_path",
    ) != feature_manifest_path.resolve():
        raise ValueError("dataset_manifest.source_lineage.feature_manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(dataset_manifest_source, "train_input_validation_report_path", label="dataset_manifest.source_lineage.train_input_validation_report_path"),
        worlds=worlds,
        field_label="dataset_manifest.source_lineage.train_input_validation_report_path",
    ) != train_input_report_path.resolve():
        raise ValueError("dataset_manifest.source_lineage.train_input_validation_report_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(dataset_manifest_source, "split_validation_report_path", label="dataset_manifest.source_lineage.split_validation_report_path"),
        worlds=worlds,
        field_label="dataset_manifest.source_lineage.split_validation_report_path",
    ) != split_report_path.resolve():
        raise ValueError("dataset_manifest.source_lineage.split_validation_report_path mismatch")

    dataset_build_source = _require_mapping(dataset_build_report_payload, "source_paths", label="dataset_build_report.source_paths")
    if _map_source_path_to_extracted(
        raw_path=_require_string(dataset_build_source, "feature_manifest_path", label="dataset_build_report.source_paths.feature_manifest_path"),
        worlds=worlds,
        field_label="dataset_build_report.source_paths.feature_manifest_path",
    ) != feature_manifest_path.resolve():
        raise ValueError("dataset_build_report.source_paths.feature_manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(dataset_build_source, "train_input_validation_report_path", label="dataset_build_report.source_paths.train_input_validation_report_path"),
        worlds=worlds,
        field_label="dataset_build_report.source_paths.train_input_validation_report_path",
    ) != train_input_report_path.resolve():
        raise ValueError("dataset_build_report.source_paths.train_input_validation_report_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(dataset_build_source, "split_validation_report_path", label="dataset_build_report.source_paths.split_validation_report_path"),
        worlds=worlds,
        field_label="dataset_build_report.source_paths.split_validation_report_path",
    ) != split_report_path.resolve():
        raise ValueError("dataset_build_report.source_paths.split_validation_report_path mismatch")

    if _map_source_path_to_extracted(
        raw_path=_require_string(train_input_report_payload, "manifest_path", label="train_input_report.manifest_path"),
        worlds=worlds,
        field_label="train_input_report.manifest_path",
    ) != feature_manifest_path.resolve():
        raise ValueError("train_input_report.manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(split_report_payload, "manifest_path", label="split_report.manifest_path"),
        worlds=worlds,
        field_label="split_report.manifest_path",
    ) != feature_manifest_path.resolve():
        raise ValueError("split_report.manifest_path mismatch")
    if _map_source_path_to_extracted(
        raw_path=_require_string(split_report_payload, "train_input_validation_report_path", label="split_report.train_input_validation_report_path"),
        worlds=worlds,
        field_label="split_report.train_input_validation_report_path",
    ) != train_input_report_path.resolve():
        raise ValueError("split_report.train_input_validation_report_path mismatch")


def _extract_output_paths(payload: Mapping[str, Any]) -> list[str]:
    """Extract manifest output_path entries in stable order."""

    entries = payload.get("partition_metadata")
    if not isinstance(entries, list):
        raise ValueError("partition_metadata must be list")
    output_paths: list[str] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError("partition_metadata entries must be objects")
        output_path = _require_string(item, "output_path", label="partition_metadata.output_path")
        output_paths.append(output_path)
    return output_paths


def _extract_feature_parquet_paths(payload: Mapping[str, Any]) -> list[str]:
    """Extract feature parquet input paths from validation reports."""

    file_reports = payload.get("file_reports")
    if not isinstance(file_reports, list):
        raise ValueError("file_reports must be list")
    paths: list[str] = []
    for item in file_reports:
        if not isinstance(item, Mapping):
            raise ValueError("file_reports entries must be objects")
        input_file = _require_string(item, "input_file", label="file_reports.input_file")
        paths.append(input_file)
    return paths


def _find_residual_local_path_leaks(*, label: str, payload: Mapping[str, Any], staged_path: Path) -> list[dict[str, Any]]:
    """Detect leaked local-machine absolute paths after projection."""

    findings: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            for prefix in _CONTRACT_CRITICAL_LEAK_PREFIXES:
                if prefix in value:
                    findings.append(
                        {
                            "label": label,
                            "staged_path": str(staged_path),
                            "field_path": path,
                            "value": value,
                        }
                    )
                    break

    walk(payload, "")
    return findings


def _default_staged_relative_path(*, source_path: Path, source_run_root: Path) -> Path:
    """Return deterministic staged relative path for a source artifact."""

    resolved_source = source_path.resolve()
    if _is_relative_to(resolved_source, source_run_root):
        return resolved_source.relative_to(source_run_root)
    if resolved_source.name == "training_config.json":
        return Path("configs") / "training_config.json"
    return Path("external") / resolved_source.name


def _dedupe_specs(specs: Sequence[ArtifactSpec]) -> list[ArtifactSpec]:
    """Deduplicate specs while preserving first-write order."""

    seen: set[str] = set()
    ordered: list[ArtifactSpec] = []
    for spec in specs:
        key = f"{spec.source_path.resolve()}|{spec.staged_relative_path}"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(spec)
    return ordered


def _spec_to_dict(spec: ArtifactSpec) -> dict[str, Any]:
    """Convert one spec to stable machine-readable form."""

    return {
        "label": spec.label,
        "source_path": str(spec.source_path),
        "staged_relative_path": str(spec.staged_relative_path),
        "classification": spec.classification,
        "provenance_class": spec.provenance_class,
        "projection_rule": spec.projection_rule,
    }


def _resolve_source_run_root_from_specs(specs: Sequence[ArtifactSpec]) -> Path:
    """Resolve source run root from the staged state-manifest spec."""

    for spec in specs:
        if spec.label == "state_manifest":
            return spec.source_path.resolve().parents[2]
    raise ValueError("state_manifest spec is required to resolve source run root")


def _find_staged_path_by_label(*, label: str, label_to_stage: Mapping[str, Path]) -> Path | None:
    """Return staged path by logical label from staged-path mapping."""

    return label_to_stage.get(label)


@dataclass(frozen=True)
class _PathMapper:
    """Deterministic path rebasing for staged projections."""

    source_run_root: Path
    source_repo_root: Path
    extracted_run_root: Path
    extracted_repo_root: Path
    staging_root: Path
    source_to_stage: Mapping[str, Path]

    def rebase(self, raw_path: str, *, allow_directory: bool) -> str:
        """Rebase one exact source path or source-run-root-relative path."""

        resolved = Path(raw_path).resolve()
        candidates: list[Path] = [resolved]
        if _is_relative_to(resolved, self.source_run_root):
            candidates.append((self.extracted_run_root / resolved.relative_to(self.source_run_root)).resolve())
        elif _is_relative_to(resolved, self.source_repo_root):
            candidates.append((self.extracted_repo_root / resolved.relative_to(self.source_repo_root)).resolve())

        for candidate in candidates:
            exact = self.source_to_stage.get(str(candidate))
            if exact is not None:
                return str(exact.resolve())
            if _is_relative_to(candidate, self.extracted_run_root):
                rebased = self.staging_root / candidate.relative_to(self.extracted_run_root)
                if allow_directory or rebased.suffix:
                    return str(rebased.resolve())
        return raw_path


def _dependency_check(check_name: str, passed: bool, *, detail: Mapping[str, Any]) -> dict[str, Any]:
    """Build one runtime dependency check entry."""

    return {
        "check_name": check_name,
        "pass": bool(passed),
        "reason_code": None if bool(passed) else check_name.upper(),
        "detail": dict(detail),
    }


def _optional_import(module_name: str) -> tuple[Any | None, str | None]:
    """Import one optional runtime dependency."""

    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _gpu_name(torch_module: Any) -> str | None:
    """Resolve GPU name when torch CUDA is available."""

    try:
        if not bool(torch_module.cuda.is_available()):
            return None
        device_index = int(torch_module.cuda.current_device())
        return str(torch_module.cuda.get_device_properties(device_index).name)
    except Exception:  # noqa: BLE001
        return None


def _sha256_file(path: Path) -> str:
    """Return file-bytes SHA256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65_536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object strictly."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Unreadable JSON path: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be object: {path}")
    return payload


def _atomic_write_bytes(payload: bytes, destination_path: Path) -> None:
    """Atomically write bytes to destination."""

    tmp_path = destination_path.with_suffix(f"{destination_path.suffix}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, destination_path)
    except Exception as exc:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to atomically write bytes: {destination_path}") from exc


def _generated_at() -> str:
    """Return stable UTC ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _require_string(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    """Require a non-empty string field."""

    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty string")
    return value.strip()


def _require_mapping(payload: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    """Require mapping field."""

    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be object")
    return value


def _require_nested_string(payload: Mapping[str, Any], keys: Sequence[str], *, label: str) -> str:
    """Require a non-empty nested string."""

    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            raise ValueError(f"{label} must be non-empty string")
        current = current.get(key)
    if not isinstance(current, str) or not current.strip():
        raise ValueError(f"{label} must be non-empty string")
    return current.strip()


def _optional_nested_string(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    """Return nested string if present and non-empty."""

    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if isinstance(current, str) and current.strip():
        return current.strip()
    return None


def _get_nested(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Return nested payload value supporting list indexes encoded as strings."""

    current: Any = payload
    for key in keys:
        if isinstance(current, list):
            current = current[int(key)]
        elif isinstance(current, Mapping):
            current = current.get(key)
        else:
            return None
    return current


def _set_nested(payload: dict[str, Any], keys: Sequence[str], value: Any) -> None:
    """Set nested payload value supporting list indexes encoded as strings."""

    current: Any = payload
    for key in keys[:-1]:
        if isinstance(current, list):
            current = current[int(key)]
        else:
            current = current[key]
    leaf = keys[-1]
    if isinstance(current, list):
        current[int(leaf)] = value
    else:
        current[leaf] = value


def _is_relative_to(path: Path, other: Path) -> bool:
    """Compatibility helper for Path.is_relative_to on 3.10."""

    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True
