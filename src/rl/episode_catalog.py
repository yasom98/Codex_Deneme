"""Training-ready episode catalog builder for Milestone 4.6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl.env_core import EpisodeSource, EpisodeSpec
from rl.env_core import EpisodeRef

EPISODE_CATALOG_VERSION = "episode_catalog.v1"

EPISODE_CATALOG_STATE_MANIFEST_MISSING = "EPISODE_CATALOG_STATE_MANIFEST_MISSING"
EPISODE_CATALOG_STATE_MANIFEST_INVALID = "EPISODE_CATALOG_STATE_MANIFEST_INVALID"
EPISODE_CATALOG_STATE_BUILD_REPORT_MISSING = "EPISODE_CATALOG_STATE_BUILD_REPORT_MISSING"
EPISODE_CATALOG_STATE_BUILD_REPORT_INVALID = "EPISODE_CATALOG_STATE_BUILD_REPORT_INVALID"
EPISODE_CATALOG_RUN_ID_MISMATCH = "EPISODE_CATALOG_RUN_ID_MISMATCH"
EPISODE_CATALOG_STATE_BUILD_NOT_PASSED = "EPISODE_CATALOG_STATE_BUILD_NOT_PASSED"
EPISODE_CATALOG_OUTPUT_COMPLETENESS_FAILED = "EPISODE_CATALOG_OUTPUT_COMPLETENESS_FAILED"
EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID = "EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID"
EPISODE_CATALOG_RUNTIME_PRICE_CONTRACT_INVALID = "EPISODE_CATALOG_RUNTIME_PRICE_CONTRACT_INVALID"
EPISODE_CATALOG_ENTRY_INVALID = "EPISODE_CATALOG_ENTRY_INVALID"

PARTITION_ORDER = {"train": 0, "val": 1, "test": 2}
SCOPE_ORDER = {"partition": 0, "fold": 1, "aggregate": 2}


@dataclass
class ValidationIssue:
    """Machine-readable catalog issue."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeCatalogEntry:
    """Selection-ready episode inventory entry."""

    episode_ref: EpisodeRef
    row_count: int
    timestamp_min_utc: str | None
    timestamp_max_utc: str | None
    warmup_enabled: bool
    valid_from_row: int
    valid_from_timestamp: str | None
    usable_start_row: int
    usable_start_timestamp: str | None
    usable_row_count_after_warmup: int
    usable_step_count_after_warmup: int
    runtime_price_columns_present: bool
    observation_contract_hash: str
    state_feature_columns_hash: str
    timestamp_unique_ok: bool
    row_order_ok: bool
    eligible_for_readiness: bool
    eligible_for_training: bool
    readiness_eligibility_reasons: tuple[str, ...]
    training_eligibility_reasons: tuple[str, ...]
    episode_sort_key: tuple[int, int, int, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry into a JSON-ready dictionary."""

        payload = asdict(self)
        payload["episode_ref"] = _episode_ref_to_dict(self.episode_ref)
        payload["readiness_eligibility_reasons"] = list(self.readiness_eligibility_reasons)
        payload["training_eligibility_reasons"] = list(self.training_eligibility_reasons)
        payload["episode_sort_key"] = list(self.episode_sort_key)
        return payload


@dataclass(frozen=True)
class EpisodeCatalogResult:
    """Catalog build output."""

    payload: dict[str, Any]
    entries: tuple[EpisodeCatalogEntry, ...]
    entries_by_key: dict[tuple[str, str, str, int | None], EpisodeCatalogEntry]


def build_episode_catalog(*, run_id: str, state_root: Path) -> EpisodeCatalogResult:
    """Build a deterministic episode catalog from state artifacts."""

    state_root = state_root.resolve()
    manifest_path = state_root / "reports" / "state_manifest.json"
    report_path = state_root / "reports" / "state_build_report.json"

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    manifest = _load_json_object(
        path=manifest_path,
        missing_code=EPISODE_CATALOG_STATE_MANIFEST_MISSING,
        invalid_code=EPISODE_CATALOG_STATE_MANIFEST_INVALID,
        errors=errors,
    )
    report = _load_json_object(
        path=report_path,
        missing_code=EPISODE_CATALOG_STATE_BUILD_REPORT_MISSING,
        invalid_code=EPISODE_CATALOG_STATE_BUILD_REPORT_INVALID,
        errors=errors,
    )

    if manifest is not None:
        _validate_run_id("state_manifest.run_id", manifest, run_id, errors)
    if report is not None:
        _validate_run_id("state_build_report.run_id", report, run_id, errors)
        if report.get("state_build_overall") is not True:
            errors.append(
                ValidationIssue(
                    code=EPISODE_CATALOG_STATE_BUILD_NOT_PASSED,
                    message="state_build_overall must be true before episode catalog build.",
                    context={"state_build_overall": report.get("state_build_overall")},
                )
            )
        if report.get("output_completeness_ok") is not True:
            errors.append(
                ValidationIssue(
                    code=EPISODE_CATALOG_OUTPUT_COMPLETENESS_FAILED,
                    message="state_build_report.output_completeness_ok must be true.",
                    context={"output_completeness_ok": report.get("output_completeness_ok")},
                )
            )

    if manifest is not None and manifest.get("output_completeness_ok") is not True:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OUTPUT_COMPLETENESS_FAILED,
                message="state_manifest.output_completeness_ok must be true.",
                context={"output_completeness_ok": manifest.get("output_completeness_ok")},
            )
        )

    observation_contract_hash = ""
    state_feature_columns_hash = ""
    state_feature_columns: tuple[str, ...] = ()
    runtime_price_columns_present = False
    contract_runtime_valid = False
    contract_linkage_valid = False
    row_order_policy_ok = False
    resolved_contract: dict[str, Any] | None = None

    if manifest is not None:
        (
            observation_contract_hash,
            state_feature_columns_hash,
            state_feature_columns,
            contract_linkage_valid,
            row_order_policy_ok,
            observation_contract_metadata,
        ) = _resolve_observation_contract_metadata(manifest=manifest, errors=errors)
        runtime_price_columns_present, contract_runtime_valid, runtime_price_contract_metadata = _resolve_runtime_price_contract_metadata(
            manifest=manifest,
            state_feature_columns=state_feature_columns,
            errors=errors,
        )
        if contract_linkage_valid and contract_runtime_valid:
            resolved_contract = {
                **observation_contract_metadata,
                **runtime_price_contract_metadata,
            }
            resolved_contract["artifact_dtypes"] = {
                str(key): str(value)
                for key, value in observation_contract_metadata["selected_dtypes"].items()
                if str(key) in {
                    observation_contract_metadata["timestamp_column"],
                    *tuple(observation_contract_metadata["state_feature_columns"]),
                }
            }
            for column in runtime_price_contract_metadata["required_runtime_columns"]:
                resolved_contract["artifact_dtypes"][str(column)] = str(
                    runtime_price_contract_metadata["runtime_price_dtypes"][str(column)]
                )

    entries: list[EpisodeCatalogEntry] = []
    if manifest is not None:
        raw_entries = manifest.get("partition_metadata")
        if not isinstance(raw_entries, list):
            errors.append(
                ValidationIssue(
                    code=EPISODE_CATALOG_STATE_MANIFEST_INVALID,
                    message="state_manifest.partition_metadata must be list.",
                    context={},
                )
            )
            raw_entries = []

        for index, item in enumerate(raw_entries):
            if not isinstance(item, Mapping):
                errors.append(
                    ValidationIssue(
                        code=EPISODE_CATALOG_ENTRY_INVALID,
                        message="partition_metadata entry must be object.",
                        context={"index": index},
                    )
                )
                continue
            entry, entry_errors = _build_catalog_entry(
                item=item,
                observation_contract_hash=observation_contract_hash,
                state_feature_columns_hash=state_feature_columns_hash,
                runtime_price_columns_present=runtime_price_columns_present,
                contract_runtime_valid=contract_runtime_valid,
                contract_linkage_valid=contract_linkage_valid,
                row_order_policy_ok=row_order_policy_ok,
                resolved_contract=resolved_contract,
            )
            errors.extend(entry_errors)
            if entry is not None:
                entries.append(entry)

    entries_sorted = tuple(sorted(entries, key=lambda item: item.episode_sort_key))
    entries_by_key = {
        _episode_ref_key(item.episode_ref): item
        for item in entries_sorted
    }

    eligible_episode_refs_sorted_by_domain = {
        "readiness": [_episode_ref_to_dict(item.episode_ref) for item in entries_sorted if item.eligible_for_readiness],
        "training": [_episode_ref_to_dict(item.episode_ref) for item in entries_sorted if item.eligible_for_training],
    }
    eligible_episode_refs_sorted_hash_by_domain = {
        key: _hash_canonical_json(value)
        for key, value in eligible_episode_refs_sorted_by_domain.items()
    }
    eligible_episode_count_by_domain = {
        key: int(len(value))
        for key, value in eligible_episode_refs_sorted_by_domain.items()
    }

    payload = {
        "episode_catalog_version": EPISODE_CATALOG_VERSION,
        "run_id": run_id,
        "state_root": str(state_root),
        "source_lineage": {
            "state_manifest_path": str(manifest_path),
            "state_build_report_path": str(report_path),
            "state_manifest_hash": _sha256_file_optional(manifest_path),
            "state_build_report_hash": _sha256_file_optional(report_path),
        },
        "selection_order_policy": {
            "partition_order": dict(PARTITION_ORDER),
            "scope_order": dict(SCOPE_ORDER),
            "fold_order_policy": "null_as_minus_one_then_ascending",
            "source_rel_order_policy": "lexicographic_ascending",
            "final_episode_sort_key_schema": [
                "partition_order",
                "scope_order",
                "fold_id_normalized",
                "source_rel",
            ],
        },
        "eligibility_policy": {
            "readiness": {
                "minimum_usable_rows_after_warmup": 2,
                "minimum_usable_steps_after_warmup": 1,
                "requires_runtime_price_contract_usable": True,
                "requires_warmup_contract_resolved": True,
                "requires_observation_state_linkage_valid": True,
                "requires_timestamp_unique_ok": True,
                "requires_row_order_ok": True,
                "requires_output_path_exists": True,
            },
            "training": {
                "base_domain": "readiness",
                "required_partition": "train",
                "allowed_scopes": ["partition", "fold"],
            },
        },
        "eligible_episode_refs_sorted_by_domain": eligible_episode_refs_sorted_by_domain,
        "eligible_episode_refs_sorted_hash_by_domain": eligible_episode_refs_sorted_hash_by_domain,
        "episodes_total": int(len(entries_sorted)),
        "eligible_episode_count_by_domain": eligible_episode_count_by_domain,
        "episodes": [item.to_dict() for item in entries_sorted],
        "warnings": [asdict(item) for item in warnings],
        "errors": [asdict(item) for item in errors],
        "episode_catalog_overall": len(errors) == 0,
    }
    return EpisodeCatalogResult(payload=payload, entries=entries_sorted, entries_by_key=entries_by_key)


def _build_catalog_entry(
    *,
    item: Mapping[str, Any],
    observation_contract_hash: str,
    state_feature_columns_hash: str,
    runtime_price_columns_present: bool,
    contract_runtime_valid: bool,
    contract_linkage_valid: bool,
    row_order_policy_ok: bool,
    resolved_contract: Mapping[str, Any] | None,
) -> tuple[EpisodeCatalogEntry | None, list[ValidationIssue]]:
    """Build one catalog entry from manifest metadata."""

    errors: list[ValidationIssue] = []

    scope = str(item.get("scope", "")).strip()
    partition = str(item.get("partition", "")).strip()
    source_rel = str(item.get("source_rel", "")).strip()
    fold_id_raw = item.get("fold_id")
    row_count_raw = item.get("row_count")
    timestamp_min_utc = item.get("timestamp_min_utc")
    timestamp_max_utc = item.get("timestamp_max_utc")
    output_path_raw = item.get("output_path")

    if scope not in SCOPE_ORDER:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_ENTRY_INVALID,
                message="partition_metadata.scope is invalid.",
                context={"scope": scope},
            )
        )
        return None, errors
    if partition not in PARTITION_ORDER:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_ENTRY_INVALID,
                message="partition_metadata.partition is invalid.",
                context={"partition": partition},
            )
        )
        return None, errors
    if not source_rel:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_ENTRY_INVALID,
                message="partition_metadata.source_rel is required.",
                context={},
            )
        )
        return None, errors
    if not isinstance(row_count_raw, int) or row_count_raw < 0:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_ENTRY_INVALID,
                message="partition_metadata.row_count must be non-negative integer.",
                context={"row_count": row_count_raw},
            )
        )
        return None, errors

    fold_id: int | None
    if scope == "fold":
        if not isinstance(fold_id_raw, int) or fold_id_raw < 0:
            errors.append(
                ValidationIssue(
                    code=EPISODE_CATALOG_ENTRY_INVALID,
                    message="fold scope requires non-negative fold_id.",
                    context={"fold_id": fold_id_raw},
                )
            )
            return None, errors
        fold_id = int(fold_id_raw)
    else:
        fold_id = None

    if not isinstance(output_path_raw, str) or not output_path_raw.strip():
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_ENTRY_INVALID,
                message="partition_metadata.output_path is required.",
                context={"episode_ref": {"scope": scope, "partition": partition, "source_rel": source_rel, "fold_id": fold_id}},
            )
        )
        return None, errors
    output_path = Path(output_path_raw).resolve()

    warmup_contract_raw = item.get("warmup_contract")
    warmup_contract = warmup_contract_raw if isinstance(warmup_contract_raw, Mapping) else {}
    warmup_enabled = bool(warmup_contract.get("enabled", False))
    valid_from_row_raw = warmup_contract.get("valid_from_row", 0)
    valid_from_timestamp_raw = warmup_contract.get("valid_from_timestamp")
    if not isinstance(valid_from_row_raw, int) or valid_from_row_raw < 0:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_ENTRY_INVALID,
                message="warmup_contract.valid_from_row must be non-negative integer.",
                context={"episode_ref": {"scope": scope, "partition": partition, "source_rel": source_rel, "fold_id": fold_id}},
            )
        )
        return None, errors
    valid_from_row = int(valid_from_row_raw)
    valid_from_timestamp = str(valid_from_timestamp_raw).strip() if isinstance(valid_from_timestamp_raw, str) and valid_from_timestamp_raw.strip() else None

    usable_start_row = valid_from_row
    usable_start_timestamp = valid_from_timestamp
    usable_row_count_after_warmup = max(int(row_count_raw) - usable_start_row, 0)
    usable_step_count_after_warmup = max(usable_row_count_after_warmup - 1, 0)

    timestamp_unique_ok = bool(item.get("timestamp_unique_ok") is True and int(item.get("duplicate_timestamp_count", 0)) == 0)
    row_order_ok = bool(row_order_policy_ok and timestamp_unique_ok and int(row_count_raw) >= 0)
    output_path_exists = output_path.exists()
    warmup_contract_resolved = usable_start_row < int(row_count_raw)
    readiness_reasons: list[str] = []
    if not output_path_exists:
        readiness_reasons.append("output_path_missing")
    if not contract_runtime_valid:
        readiness_reasons.append("runtime_price_contract_invalid")
    if not runtime_price_columns_present:
        readiness_reasons.append("runtime_price_columns_missing")
    if not contract_linkage_valid:
        readiness_reasons.append("observation_state_linkage_invalid")
    if not warmup_contract_resolved:
        readiness_reasons.append("warmup_contract_unusable")
    if usable_row_count_after_warmup < 2:
        readiness_reasons.append("usable_row_count_below_minimum")
    if usable_step_count_after_warmup < 1:
        readiness_reasons.append("usable_step_count_below_minimum")
    if not timestamp_unique_ok:
        readiness_reasons.append("timestamp_uniqueness_unhealthy")
    if not row_order_ok:
        readiness_reasons.append("row_order_unhealthy")
    if resolved_contract is not None and output_path_exists and contract_runtime_valid and contract_linkage_valid:
        source = EpisodeSource()
        spec = EpisodeSpec(
            scope=scope,
            partition=partition,
            source_rel=source_rel,
            fold_id=fold_id,
            output_path=output_path,
            row_count=int(row_count_raw),
        )
        try:
            source.load_episode(
                spec=spec,
                expected_columns=resolved_contract["artifact_columns"],
                observation_columns=resolved_contract["state_feature_columns"],
                strict_post_valid_numeric_columns=resolved_contract["strict_post_valid_numeric_columns"],
                expected_dtypes=resolved_contract["artifact_dtypes"],
                timestamp_column=resolved_contract["timestamp_column"],
                execution_price_column=resolved_contract["execution_price_column"],
                mark_to_market_column=resolved_contract["mark_to_market_column"],
                include_timestamp_in_observation=False,
                observation_output_dtype="float32",
                allowed_safe_casts={"uint8->float32"},
                valid_observation_start_row=usable_start_row,
                valid_observation_start_timestamp=usable_start_timestamp,
                warmup_head_nan_profile=resolved_contract["warmup_head_nan_profile"](warmup_contract),
            )
        except ValueError as exc:
            source_error = str(exc)
            if source_error.startswith("TIMESTAMP_ORDERING_VIOLATION"):
                row_order_ok = False
                if "row_order_unhealthy" not in readiness_reasons:
                    readiness_reasons.append("row_order_unhealthy")
            elif source_error.startswith("TIMESTAMP_DUPLICATES"):
                timestamp_unique_ok = False
                row_order_ok = False
                if "timestamp_uniqueness_unhealthy" not in readiness_reasons:
                    readiness_reasons.append("timestamp_uniqueness_unhealthy")
                if "row_order_unhealthy" not in readiness_reasons:
                    readiness_reasons.append("row_order_unhealthy")
            elif source_error.startswith("POST_VALID_OBSERVATION_NAN"):
                readiness_reasons.append("post_valid_observation_non_finite")
            elif source_error.startswith("EPISODE_TOO_SHORT_AFTER_WARMUP") or source_error.startswith("EFFECTIVE_START_INVALID"):
                warmup_contract_resolved = False
                if "warmup_contract_unusable" not in readiness_reasons:
                    readiness_reasons.append("warmup_contract_unusable")
            elif source_error.startswith("EXECUTION_PRICE_COLUMN_MISSING") or source_error.startswith("MARK_TO_MARKET_COLUMN_MISSING"):
                if "runtime_price_columns_missing" not in readiness_reasons:
                    readiness_reasons.append("runtime_price_columns_missing")
            else:
                readiness_reasons.append("episode_source_validation_failed")

    training_reasons = list(readiness_reasons)
    if partition != "train":
        training_reasons.append("partition_not_train")
    if scope not in {"partition", "fold"}:
        training_reasons.append("scope_not_training_supported")

    episode_ref = EpisodeRef(scope=scope, partition=partition, source_rel=source_rel, fold_id=fold_id)
    entry = EpisodeCatalogEntry(
        episode_ref=episode_ref,
        row_count=int(row_count_raw),
        timestamp_min_utc=str(timestamp_min_utc) if isinstance(timestamp_min_utc, str) else None,
        timestamp_max_utc=str(timestamp_max_utc) if isinstance(timestamp_max_utc, str) else None,
        warmup_enabled=warmup_enabled,
        valid_from_row=valid_from_row,
        valid_from_timestamp=valid_from_timestamp,
        usable_start_row=usable_start_row,
        usable_start_timestamp=usable_start_timestamp,
        usable_row_count_after_warmup=usable_row_count_after_warmup,
        usable_step_count_after_warmup=usable_step_count_after_warmup,
        runtime_price_columns_present=bool(runtime_price_columns_present),
        observation_contract_hash=observation_contract_hash,
        state_feature_columns_hash=state_feature_columns_hash,
        timestamp_unique_ok=timestamp_unique_ok,
        row_order_ok=row_order_ok,
        eligible_for_readiness=len(readiness_reasons) == 0,
        eligible_for_training=len(training_reasons) == 0,
        readiness_eligibility_reasons=tuple(readiness_reasons),
        training_eligibility_reasons=tuple(training_reasons),
        episode_sort_key=(
            PARTITION_ORDER[partition],
            SCOPE_ORDER[scope],
            fold_id if fold_id is not None else -1,
            source_rel,
        ),
    )
    return entry, errors


def _resolve_observation_contract_metadata(
    *,
    manifest: Mapping[str, Any],
    errors: list[ValidationIssue],
) -> tuple[str, str, tuple[str, ...], bool, bool, dict[str, Any]]:
    """Resolve observation-contract hashes and health flags."""

    observation_contract = manifest.get("observation_contract")
    if not isinstance(observation_contract, Mapping):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="observation_contract missing or invalid in state manifest.",
                context={},
            )
        )
        return "", "", (), False, False, {}

    timestamp_policy = observation_contract.get("timestamp_policy")
    selected_input_columns = observation_contract.get("selected_input_columns")
    state_feature_columns = observation_contract.get("state_feature_columns")
    strict_post_valid_numeric_columns = observation_contract.get("strict_post_valid_numeric_columns")
    row_order_policy = observation_contract.get("row_order_policy")
    geometry_feature_version = observation_contract.get("geometry_feature_version")
    geometry_feature_formulas = observation_contract.get("geometry_feature_formulas")
    conditional_raw_columns = observation_contract.get("conditional_raw_columns")
    conditional_column_policy = observation_contract.get("conditional_column_policy")
    conditional_column_replacements = observation_contract.get("conditional_column_replacements")
    event_columns = observation_contract.get("event_columns")
    regime_columns = observation_contract.get("regime_columns")
    geometry_columns = observation_contract.get("geometry_columns")

    if not isinstance(timestamp_policy, Mapping):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="observation_contract.timestamp_policy must be object.",
                context={},
            )
        )
        return "", "", (), False, False, {}
    timestamp_column = timestamp_policy.get("timestamp_column")
    if not isinstance(timestamp_column, str) or not timestamp_column.strip():
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="observation_contract.timestamp_policy.timestamp_column is required.",
                context={},
            )
        )
        return "", "", (), False, False, {}
    if not isinstance(selected_input_columns, Sequence) or not all(isinstance(item, str) for item in selected_input_columns):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="observation_contract.selected_input_columns must be list[str].",
                context={},
            )
        )
        return "", "", (), False, False, {}
    if not isinstance(state_feature_columns, Sequence) or not all(isinstance(item, str) for item in state_feature_columns):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="observation_contract.state_feature_columns must be list[str].",
                context={},
            )
        )
        return "", "", (), False, False, {}
    if not isinstance(strict_post_valid_numeric_columns, Sequence) or not all(
        isinstance(item, str) for item in strict_post_valid_numeric_columns
    ):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="observation_contract.strict_post_valid_numeric_columns must be list[str].",
                context={},
            )
        )
        return "", "", (), False, False, {}
    dtype_policy = observation_contract.get("dtype_policy")
    selected_dtypes = dtype_policy.get("selected_dtypes") if isinstance(dtype_policy, Mapping) else None
    if not isinstance(selected_dtypes, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in selected_dtypes.items()):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="observation_contract.dtype_policy.selected_dtypes must be dict[str, str].",
                context={},
            )
        )
        return "", "", (), False, False, {}

    state_feature_columns_tuple = tuple(str(item) for item in state_feature_columns)
    expected_selected = [timestamp_column, *state_feature_columns_tuple]
    if list(selected_input_columns) != expected_selected:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="selected_input_columns must equal timestamp + state_feature_columns.",
                context={"expected": expected_selected, "actual": list(selected_input_columns)},
            )
        )
        return "", "", (), False, False, {}
    if list(strict_post_valid_numeric_columns or []) != list(state_feature_columns_tuple):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="strict_post_valid_numeric_columns must equal state_feature_columns.",
                context={},
            )
        )
        return "", "", (), False, False, {}
    row_order_ok = bool(
        isinstance(row_order_policy, Mapping)
        and row_order_policy.get("name") == "timestamp_ascending"
        and row_order_policy.get("stable_tie_breaker") == "source_row_position"
    )
    if not row_order_ok:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_OBSERVATION_CONTRACT_INVALID,
                message="row_order_policy must be timestamp_ascending with source_row_position tie-breaker.",
                context={"row_order_policy": row_order_policy},
            )
        )
        return "", "", (), False, False, {}

    observation_contract_hash = _hash_canonical_json(
        {
            "state_feature_columns": list(state_feature_columns_tuple),
            "event_columns": list(event_columns or []),
            "regime_columns": list(regime_columns or []),
            "geometry_columns": list(geometry_columns or []),
            "strict_post_valid_numeric_columns": list(strict_post_valid_numeric_columns or []),
            "conditional_raw_columns": list(conditional_raw_columns or []),
            "conditional_column_policy": conditional_column_policy,
            "conditional_column_replacements": conditional_column_replacements,
            "geometry_feature_version": geometry_feature_version,
            "geometry_feature_formulas": geometry_feature_formulas,
        }
    )
    state_feature_columns_hash = _hash_canonical_json({"state_feature_columns": list(state_feature_columns_tuple)})
    return (
        observation_contract_hash,
        state_feature_columns_hash,
        state_feature_columns_tuple,
        True,
        True,
        {
            "timestamp_column": timestamp_column,
            "state_feature_columns": tuple(state_feature_columns_tuple),
            "strict_post_valid_numeric_columns": tuple(str(item) for item in strict_post_valid_numeric_columns),
            "selected_dtypes": {str(k): str(v) for k, v in selected_dtypes.items()},
        },
    )


def _resolve_runtime_price_contract_metadata(
    *,
    manifest: Mapping[str, Any],
    state_feature_columns: Sequence[str],
    errors: list[ValidationIssue],
) -> tuple[bool, bool, dict[str, Any]]:
    """Resolve runtime-price contract health."""

    runtime_price_contract = manifest.get("runtime_price_contract")
    observation_contract = manifest.get("observation_contract")
    timestamp_column = None
    if isinstance(observation_contract, Mapping):
        timestamp_policy = observation_contract.get("timestamp_policy")
        if isinstance(timestamp_policy, Mapping):
            timestamp_column = timestamp_policy.get("timestamp_column")

    if not isinstance(runtime_price_contract, Mapping):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract missing or invalid in state manifest.",
                context={},
            )
        )
        return False, False, {}

    execution_price_column = runtime_price_contract.get("execution_price_column")
    mark_to_market_column = runtime_price_contract.get("mark_to_market_column")
    required_runtime_columns = runtime_price_contract.get("required_runtime_columns")
    artifact_columns = runtime_price_contract.get("artifact_columns")
    runtime_price_dtypes = runtime_price_contract.get("runtime_price_dtypes")
    if (
        not isinstance(execution_price_column, str)
        or not execution_price_column.strip()
        or not isinstance(mark_to_market_column, str)
        or not mark_to_market_column.strip()
        or not isinstance(required_runtime_columns, Sequence)
        or not all(isinstance(item, str) for item in required_runtime_columns)
        or not isinstance(artifact_columns, Sequence)
        or not all(isinstance(item, str) for item in artifact_columns)
        or not isinstance(runtime_price_dtypes, Mapping)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in runtime_price_dtypes.items())
        or not isinstance(timestamp_column, str)
        or not timestamp_column.strip()
    ):
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract fields are incomplete.",
                context={},
            )
        )
        return False, False, {}

    required_runtime_columns_list = [str(item) for item in required_runtime_columns]
    artifact_columns_list = [str(item) for item in artifact_columns]
    missing_runtime_dtype_columns = [column for column in required_runtime_columns_list if column not in runtime_price_dtypes]
    if missing_runtime_dtype_columns:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_dtypes must cover all required runtime columns.",
                context={"missing_runtime_dtype_columns": missing_runtime_dtype_columns},
            )
        )
        return False, False, {}
    expected_artifact_columns = [timestamp_column, *list(state_feature_columns), *required_runtime_columns_list]
    runtime_price_columns_present = (
        execution_price_column in required_runtime_columns_list
        and mark_to_market_column in required_runtime_columns_list
        and artifact_columns_list == expected_artifact_columns
    )
    if not runtime_price_columns_present:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_RUNTIME_PRICE_CONTRACT_INVALID,
                message="runtime_price_contract artifact ordering or required columns are invalid.",
                context={
                    "expected_artifact_columns": expected_artifact_columns,
                    "actual_artifact_columns": artifact_columns_list,
                    "required_runtime_columns": required_runtime_columns_list,
                },
            )
        )
        return False, False, {}
    return (
        True,
        True,
        {
            "execution_price_column": execution_price_column,
            "mark_to_market_column": mark_to_market_column,
            "required_runtime_columns": tuple(required_runtime_columns_list),
            "artifact_columns": tuple(artifact_columns_list),
            "runtime_price_dtypes": {
                str(key): str(value)
                for key, value in runtime_price_dtypes.items()
                if isinstance(key, str) and isinstance(value, str)
            },
            "warmup_head_nan_profile": lambda warmup_contract: {
                str(key): int(value)
                for key, value in dict(warmup_contract.get("head_nan_profile", {})).items()
                if isinstance(key, str) and isinstance(value, int) and value > 0
            },
        },
    )


def _load_json_object(
    *,
    path: Path,
    missing_code: str,
    invalid_code: str,
    errors: list[ValidationIssue],
) -> dict[str, Any] | None:
    """Load a JSON object payload."""

    if not path.exists():
        errors.append(ValidationIssue(code=missing_code, message="JSON artifact not found.", context={"path": str(path)}))
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(ValidationIssue(code=invalid_code, message="JSON artifact is invalid.", context={"path": str(path), "error": str(exc)}))
        return None
    if not isinstance(payload, dict):
        errors.append(ValidationIssue(code=invalid_code, message="JSON payload must be object.", context={"path": str(path)}))
        return None
    return payload


def _validate_run_id(field: str, payload: Mapping[str, Any], expected_run_id: str, errors: list[ValidationIssue]) -> None:
    """Validate run_id consistency."""

    seen = payload.get("run_id")
    if seen != expected_run_id:
        errors.append(
            ValidationIssue(
                code=EPISODE_CATALOG_RUN_ID_MISMATCH,
                message="run_id mismatch across episode catalog lineage.",
                context={"field": field, "expected_run_id": expected_run_id, "seen_run_id": seen},
            )
        )


def _episode_ref_key(ref: EpisodeRef) -> tuple[str, str, str, int | None]:
    """Return deterministic episode-ref key."""

    return (ref.scope, ref.partition, ref.source_rel, ref.fold_id)


def _episode_ref_to_dict(ref: EpisodeRef) -> dict[str, Any]:
    """Serialize episode ref."""

    return {
        "scope": ref.scope,
        "partition": ref.partition,
        "source_rel": ref.source_rel,
        "fold_id": ref.fold_id,
    }


def _hash_canonical_json(payload: Any) -> str:
    """Hash payload with deterministic JSON canonicalization."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file_optional(path: Path) -> str | None:
    """Return sha256 hash when file exists."""

    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
