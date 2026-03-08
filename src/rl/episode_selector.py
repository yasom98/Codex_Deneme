"""Deterministic episode selector for Milestone 4.6 orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from rl.episode_catalog import EpisodeCatalogResult, EpisodeCatalogEntry, _episode_ref_key, _episode_ref_to_dict, _hash_canonical_json
from rl.env_core import EpisodeRef

READINESS_DOMAIN = "readiness"
TRAINING_DOMAIN = "training"

SELECTION_POLICY_FIXED = "fixed_episode"
SELECTION_POLICY_SEEDED_RANDOM = "seeded_random_episode"

EPISODE_SELECTION_UNSUPPORTED_POLICY = "EPISODE_SELECTION_UNSUPPORTED_POLICY"
EPISODE_SELECTION_FIXED_INPUT_REQUIRED = "EPISODE_SELECTION_FIXED_INPUT_REQUIRED"
EPISODE_SELECTION_FIXED_NOT_FOUND = "EPISODE_SELECTION_FIXED_NOT_FOUND"
EPISODE_SELECTION_FIXED_NOT_ELIGIBLE = "EPISODE_SELECTION_FIXED_NOT_ELIGIBLE"
EPISODE_SELECTION_EMPTY_DOMAIN = "EPISODE_SELECTION_EMPTY_DOMAIN"


@dataclass
class ValidationIssue:
    """Machine-readable selector issue."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeSelectionResult:
    """Selected episode and deterministic trace."""

    selected_episode_ref: EpisodeRef | None
    selected_entry: EpisodeCatalogEntry | None
    eligible_domain_used: str
    candidate_refs_sorted: tuple[dict[str, Any], ...]
    trace: dict[str, Any]
    fixed_episode_input_source: str | None
    fixed_episode_input_value: dict[str, Any] | None
    errors: tuple[ValidationIssue, ...]


def select_episode(
    *,
    catalog: EpisodeCatalogResult,
    selection_policy: str,
    seed: int,
    fixed_episode_ref: EpisodeRef | None = None,
) -> EpisodeSelectionResult:
    """Select an episode deterministically from the catalog."""

    errors: list[ValidationIssue] = []
    if selection_policy == SELECTION_POLICY_FIXED:
        domain = READINESS_DOMAIN
    elif selection_policy == SELECTION_POLICY_SEEDED_RANDOM:
        domain = TRAINING_DOMAIN
    else:
        errors.append(
            ValidationIssue(
                code=EPISODE_SELECTION_UNSUPPORTED_POLICY,
                message="Unsupported selection policy.",
                context={"selection_policy": selection_policy},
            )
        )
        return EpisodeSelectionResult(
            selected_episode_ref=None,
            selected_entry=None,
            eligible_domain_used="unknown",
            candidate_refs_sorted=(),
            trace={
                "selection_policy": selection_policy,
                "seed": seed,
                "candidate_count": 0,
                "eligible_domain_used": "unknown",
                "eligible_episode_refs_sorted_hash": _hash_canonical_json([]),
                "selected_index": None,
                "selected_episode_ref": None,
            },
            fixed_episode_input_source=None,
            fixed_episode_input_value=None,
            errors=tuple(errors),
        )

    candidate_entries = _candidate_entries(catalog=catalog, domain=domain)
    candidate_refs_sorted = tuple(_episode_ref_to_dict(item.episode_ref) for item in candidate_entries)
    candidate_hash = _hash_canonical_json(candidate_refs_sorted)
    selected_index: int | None = None
    selected_entry: EpisodeCatalogEntry | None = None
    fixed_episode_input_source: str | None = None
    fixed_episode_input_value: dict[str, Any] | None = None

    if selection_policy == SELECTION_POLICY_FIXED:
        fixed_episode_input_source = "env_config.episode_ref"
        fixed_episode_input_value = _episode_ref_to_dict(fixed_episode_ref) if fixed_episode_ref is not None else None
        if fixed_episode_ref is None:
            errors.append(
                ValidationIssue(
                    code=EPISODE_SELECTION_FIXED_INPUT_REQUIRED,
                    message="fixed_episode requires explicit episode_ref input.",
                    context={},
                )
            )
        else:
            candidate_index = { _episode_ref_key(item.episode_ref): idx for idx, item in enumerate(candidate_entries) }
            entry = catalog.entries_by_key.get(_episode_ref_key(fixed_episode_ref))
            if entry is None:
                errors.append(
                    ValidationIssue(
                        code=EPISODE_SELECTION_FIXED_NOT_FOUND,
                        message="fixed_episode input was not found in episode catalog.",
                        context={"episode_ref": _episode_ref_to_dict(fixed_episode_ref)},
                    )
                )
            elif not entry.eligible_for_readiness:
                errors.append(
                    ValidationIssue(
                        code=EPISODE_SELECTION_FIXED_NOT_ELIGIBLE,
                        message="fixed_episode input is not readiness-eligible in 4.6.",
                        context={
                            "episode_ref": _episode_ref_to_dict(fixed_episode_ref),
                            "readiness_eligibility_reasons": list(entry.readiness_eligibility_reasons),
                        },
                    )
                )
            else:
                selected_entry = entry
                selected_index = candidate_index.get(_episode_ref_key(entry.episode_ref))
    else:
        if not candidate_entries:
            errors.append(
                ValidationIssue(
                    code=EPISODE_SELECTION_EMPTY_DOMAIN,
                    message="No training-eligible episodes are available for seeded random selection.",
                    context={},
                )
            )
        else:
            rng = random.Random(seed)
            selected_index = int(rng.randrange(len(candidate_entries)))
            selected_entry = candidate_entries[selected_index]

    trace = {
        "selection_policy": selection_policy,
        "seed": int(seed),
        "candidate_count": int(len(candidate_entries)),
        "eligible_domain_used": domain,
        "eligible_episode_refs_sorted_hash": candidate_hash,
        "selected_index": selected_index,
        "selected_episode_ref": _episode_ref_to_dict(selected_entry.episode_ref) if selected_entry is not None else None,
    }
    return EpisodeSelectionResult(
        selected_episode_ref=selected_entry.episode_ref if selected_entry is not None else None,
        selected_entry=selected_entry,
        eligible_domain_used=domain,
        candidate_refs_sorted=candidate_refs_sorted,
        trace=trace,
        fixed_episode_input_source=fixed_episode_input_source,
        fixed_episode_input_value=fixed_episode_input_value,
        errors=tuple(errors),
    )


def _candidate_entries(*, catalog: EpisodeCatalogResult, domain: str) -> tuple[EpisodeCatalogEntry, ...]:
    """Return domain-specific candidate entries in deterministic order."""

    if domain == READINESS_DOMAIN:
        return tuple(item for item in catalog.entries if item.eligible_for_readiness)
    if domain == TRAINING_DOMAIN:
        return tuple(item for item in catalog.entries if item.eligible_for_training)
    return ()
