"""Compact evaluation summary reader/writer for Colab evaluation bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from core.io_atomic import atomic_write_json
from rl.passivity_diagnostics import PASSIVITY_DIAGNOSTICS_REPORT_FILENAME

EVALUATION_SUMMARY_FILENAME = "evaluation_summary.json"


def build_evaluation_summary(*, output_dir: Path) -> dict[str, Any]:
    """Build a compact evaluation summary from authoritative evaluation reports."""

    output_dir_resolved = output_dir.resolve()
    validation_payload = _load_required_json_object(
        output_dir_resolved / "evaluation_validation_report.json",
        label="evaluation_validation_report.json",
    )
    manifest_payload = _load_required_json_object(
        output_dir_resolved / "evaluation_manifest.json",
        label="evaluation_manifest.json",
    )
    backtest_payload = _load_required_json_object(
        output_dir_resolved / "evaluation_backtest_report.json",
        label="evaluation_backtest_report.json",
    )
    passivity_path = output_dir_resolved / PASSIVITY_DIAGNOSTICS_REPORT_FILENAME
    passivity_payload = _load_optional_json_object(passivity_path, label=PASSIVITY_DIAGNOSTICS_REPORT_FILENAME)

    model_load_detail = _extract_phase_detail(backtest_payload.get("startup_phase_trace"), phase_name="model_load")
    strategy_metrics = _as_mapping(backtest_payload.get("strategy_metrics"))
    deterministic_eval = _as_mapping(_mapping_get(passivity_payload, "deterministic_eval"))
    stochastic_eval = _as_mapping(_mapping_get(passivity_payload, "stochastic_eval"))
    comparison = _as_mapping(_mapping_get(passivity_payload, "deterministic_vs_stochastic"))
    hold_summary = _as_mapping(_mapping_get(deterministic_eval, "hold_dominance_summary"))
    ranking_summary = _as_mapping(_mapping_get(deterministic_eval, "action_ranking_summary"))

    return {
        "evaluation_success": _first_present_bool(
            backtest_payload.get("evaluation_success"),
        ),
        "validation_overall_pass": _first_present_bool(
            validation_payload.get("overall_pass"),
        ),
        "model_class": _optional_string(_mapping_get(model_load_detail, "model_class")),
        "detection_source": _optional_string(_mapping_get(model_load_detail, "detection_source")),
        "detected_maskable": _optional_bool(_mapping_get(model_load_detail, "detected_maskable")),
        "action_masking_enabled": _first_present_bool(
            _mapping_get(model_load_detail, "action_masking_enabled"),
            backtest_payload.get("action_masking_enabled"),
            manifest_payload.get("action_masking_enabled"),
            validation_payload.get("action_masking_enabled"),
        ),
        "passivity_diagnostics_enabled": _first_present_bool(
            backtest_payload.get("passivity_diagnostics_enabled"),
            manifest_payload.get("passivity_diagnostics_enabled"),
            validation_payload.get("passivity_diagnostics_enabled"),
        ),
        "final_equity": _optional_float(_mapping_get(strategy_metrics, "final_equity")),
        "total_return": _optional_float(_mapping_get(strategy_metrics, "total_return")),
        "num_trades": _optional_int_like(_mapping_get(strategy_metrics, "num_trades")),
        "invalid_action_ratio": _optional_float(_mapping_get(deterministic_eval, "invalid_action_ratio")),
        "action_semantic_counts": _optional_int_mapping(_mapping_get(deterministic_eval, "action_semantic_counts")),
        "position_transition_counts": _optional_int_mapping(
            _mapping_get(deterministic_eval, "position_transition_counts")
        ),
        "profit_factor": None,
        "max_drawdown_pct": _ratio_to_pct(_mapping_get(strategy_metrics, "max_drawdown")),
        "passivity_report_exists": bool(passivity_payload is not None),
        "deterministic_hold_share": _optional_float(_mapping_get(hold_summary, "hold_share")),
        "deterministic_hold_extreme": _optional_bool(_mapping_get(comparison, "deterministic_hold_extreme")),
        "deterministic_hold_dominance_margin_band": _optional_string(
            _mapping_get(ranking_summary, "hold_dominance_margin_band")
        ),
        "stochastic_num_trades": _optional_int_like(_mapping_get(stochastic_eval, "num_trades")),
        "stochastic_final_equity": _optional_float(_mapping_get(stochastic_eval, "final_equity")),
        "num_trades_delta": _optional_number(_mapping_get(comparison, "num_trades_delta")),
        "hold_share_delta": _optional_float(_mapping_get(comparison, "hold_share_delta")),
    }


def write_evaluation_summary(*, output_dir: Path) -> dict[str, Any]:
    """Build and atomically persist the compact evaluation summary."""

    output_dir_resolved = output_dir.resolve()
    summary_payload = build_evaluation_summary(output_dir=output_dir_resolved)
    atomic_write_json(summary_payload, output_dir_resolved / EVALUATION_SUMMARY_FILENAME)
    return summary_payload


def _load_required_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required JSON object or fail closed."""

    if not path.exists():
        raise ValueError(f"Missing required evaluation summary input: {label} ({path})")
    if not path.is_file():
        raise ValueError(f"Evaluation summary input must be a readable file: {label} ({path})")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation summary input contains invalid JSON: {label} ({path})") from exc
    except OSError as exc:
        raise ValueError(f"Evaluation summary input could not be read: {label} ({path})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation summary input JSON must decode to an object: {label} ({path})")
    return payload


def _load_optional_json_object(path: Path, *, label: str) -> dict[str, Any] | None:
    """Load one optional JSON object when present."""

    if not path.exists():
        return None
    return _load_required_json_object(path, label=label)


def _extract_phase_detail(startup_phase_trace: Any, *, phase_name: str) -> Mapping[str, Any] | None:
    """Return the matching phase detail mapping without relying on list indexes."""

    if not isinstance(startup_phase_trace, list):
        return None
    for item in startup_phase_trace:
        if not isinstance(item, Mapping):
            continue
        if item.get("phase") != phase_name:
            continue
        detail = item.get("detail")
        if isinstance(detail, Mapping):
            return detail
        return None
    return None


def _ratio_to_pct(value: Any) -> float | None:
    """Convert a drawdown ratio into percentage points when finite."""

    ratio_value = _optional_float(value)
    if ratio_value is None:
        return None
    return ratio_value * 100.0


def _first_present_bool(*values: Any) -> bool | None:
    """Return the first value that is explicitly boolean."""

    for value in values:
        boolean_value = _optional_bool(value)
        if boolean_value is not None:
            return boolean_value
    return None


def _mapping_get(mapping: Mapping[str, Any] | None, key: str) -> Any:
    """Read one mapping key when the mapping exists."""

    if not isinstance(mapping, Mapping):
        return None
    return mapping.get(key)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return a mapping-like value when present."""

    if isinstance(value, Mapping):
        return value
    return None


def _optional_string(value: Any) -> str | None:
    """Return a non-empty string when available."""

    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _optional_bool(value: Any) -> bool | None:
    """Return an explicit boolean when available."""

    if isinstance(value, bool):
        return value
    return None


def _optional_float(value: Any) -> float | None:
    """Return a finite float when safely available."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _optional_int_like(value: Any) -> int | None:
    """Return an integer when the numeric value is safely integral."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _optional_number(value: Any) -> int | float | None:
    """Return an int/float when the numeric value is finite."""

    int_value = _optional_int_like(value)
    if int_value is not None:
        return int_value
    float_value = _optional_float(value)
    if float_value is not None:
        return float_value
    return None


def _optional_int_mapping(value: Any) -> dict[str, int] | None:
    """Return a shallow string-keyed integer mapping when safe."""

    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, int] = {}
    for key, item in value.items():
        int_value = _optional_int_like(item)
        if int_value is None:
            return None
        normalized[str(key)] = int_value
    return normalized
