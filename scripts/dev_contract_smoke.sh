#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/dev_contract_smoke.sh <run_id>
# Developer-only smoke/regression wrapper for the validated contract chain.

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash scripts/dev_contract_smoke.sh <run_id>\n' >&2
    exit 64
fi

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
    printf 'Usage: bash scripts/dev_contract_smoke.sh <run_id>\n' >&2
    exit 64
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="python3"

RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
STANDARDIZED_ROOT="${RUN_ROOT}/data_standardized/parquet"
FEATURE_ROOT="${RUN_ROOT}/data_features"
FEATURE_PARQUET_ROOT="${FEATURE_ROOT}/parquet"
FEATURE_REPORTS_ROOT="${FEATURE_ROOT}/reports"
DATASET_ROOT="${RUN_ROOT}/data_datasets"
DATASET_REPORT_PATH="${DATASET_ROOT}/reports/dataset_build_report.json"
STATE_ROOT="${RUN_ROOT}/data_states"
STATE_REPORTS_ROOT="${STATE_ROOT}/reports"
STATE_MANIFEST_PATH="${STATE_REPORTS_ROOT}/state_manifest.json"
STATE_BUILD_REPORT_PATH="${STATE_REPORTS_ROOT}/state_build_report.json"
ENV_ROOT="${RUN_ROOT}/env_contract"
ENV_TMP_DIR="${ENV_ROOT}/tmp"
ENV_CONFIG_PATH="${ENV_TMP_DIR}/dev_contract_smoke_env_config.json"
ENV_REPORT_PATH="${ENV_ROOT}/reports/env_contract_report.json"

FEATURE_SUMMARY_PATH="${FEATURE_REPORTS_ROOT}/summary.json"
TRAIN_INPUT_REPORT_PATH="${FEATURE_REPORTS_ROOT}/train_input_validation_report.json"
SPLIT_REPORT_PATH="${FEATURE_REPORTS_ROOT}/split_validation_report.json"

CURRENT_STEP="init"

print_command() {
    printf '+'
    for arg in "$@"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
}

print_summary() {
    local exit_code="$?"

    trap - EXIT

    if [[ "${exit_code}" -eq 0 ]]; then
        printf '[PASS] run_id=%s\n' "${RUN_ID}"
    else
        printf '[FAIL] run_id=%s step=%s exit_code=%s\n' "${RUN_ID}" "${CURRENT_STEP}" "${exit_code}"
    fi

    printf 'Report paths:\n'
    printf '%s\n' \
        "${FEATURE_SUMMARY_PATH}" \
        "${TRAIN_INPUT_REPORT_PATH}" \
        "${SPLIT_REPORT_PATH}" \
        "${DATASET_REPORT_PATH}" \
        "${STATE_BUILD_REPORT_PATH}" \
        "${ENV_REPORT_PATH}"

    exit "${exit_code}"
}

trap print_summary EXIT

run_command() {
    CURRENT_STEP="$1"
    shift
    print_command "$@"
    "$@"
}

generate_env_config() {
    CURRENT_STEP="generate_env_config"

    local -a cmd=(
        "${PYTHON_BIN}"
        -
        "${STATE_MANIFEST_PATH}"
        "${ENV_CONFIG_PATH}"
        "${RUN_ID}"
        "${STATE_ROOT}"
    )

    print_command "${cmd[@]}"
    "${cmd[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def main() -> int:
    manifest_path = Path(sys.argv[1]).resolve()
    env_config_path = Path(sys.argv[2]).resolve()
    run_id = sys.argv[3]
    state_root = str(Path(sys.argv[4]).resolve())

    if not manifest_path.exists():
        raise FileNotFoundError(f"state_manifest missing: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _require_object(payload, name="state_manifest")
    raw_entries = manifest.get("partition_metadata")
    if not isinstance(raw_entries, list):
        raise ValueError("state_manifest.partition_metadata must be a list")

    selected_entry: dict[str, Any] | None = None
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        if item.get("scope") == "partition" and item.get("partition") == "train":
            selected_entry = item
            break

    if selected_entry is None:
        raise ValueError("state_manifest does not contain a train partition entry")

    source_rel = selected_entry.get("source_rel")
    if not isinstance(source_rel, str) or not source_rel.strip():
        raise ValueError("selected train partition entry must define source_rel")

    fold_id = selected_entry.get("fold_id")
    if fold_id is not None and not isinstance(fold_id, int):
        raise ValueError("selected train partition entry fold_id must be int or null")

    env_config = {
        "run_id": run_id,
        "state_root": state_root,
        "episode_ref": {
            "scope": "partition",
            "partition": "train",
            "source_rel": source_rel,
            "fold_id": fold_id,
        },
        "execution_price_column": "close",
        "mark_to_market_column": "close",
        "include_timestamp_in_observation": False,
        "observation_output_dtype": "float32",
        "observation_dtype_policy": "strict",
        "allowed_safe_casts": ["uint8->float32"],
        "initial_cash": 1000.0,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "max_steps": None,
        "seed": 42,
        "execution_timing_contract": {
            "observation_timestamp_policy": "row_t",
            "execution_price_policy": "close_t",
            "reward_accrual_interval_policy": "post_action_t_to_t_plus_1",
            "mark_to_market_policy": "next_row_close",
        },
        "action_semantics_contract": {
            "action_space_type": "Discrete",
            "action_space_n": 4,
            "invalid_action_policy": "noop_with_info_flag",
            "reversal_policy": "disallow_same_step",
            "position_model": "single_position_unit",
        },
        "reward_contract": {
            "reward_version": "reward.v1",
            "reward_formula_summary": "pnl_delta - fees - slippage_cost",
            "included_components": ["pnl_delta", "fees", "slippage_cost"],
            "reward_scale": 1.0,
            "reward_clip_min": None,
            "reward_clip_max": None,
        },
        "termination_contract": {
            "data_end_terminated": True,
            "max_steps_truncated": True,
        },
    }

    env_config_path.parent.mkdir(parents=True, exist_ok=True)
    env_config_path.write_text(json.dumps(env_config, ensure_ascii=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
}

run_command \
    "pytest_subset" \
    "${PYTHON_BIN}" \
    -m \
    pytest \
    -q \
    "${PROJECT_ROOT}/tests/test_features_engine.py::test_make_features_cli_honors_input_root_override" \
    "${PROJECT_ROOT}/tests/test_validate_train_inputs_cli.py::test_cli_pass_exit_zero_and_writes_report" \
    "${PROJECT_ROOT}/tests/test_validate_splits_cli.py::test_cli_pass_exit_zero_and_writes_report" \
    "${PROJECT_ROOT}/tests/test_build_datasets_cli.py::test_cli_runtime_price_contract_enabled_close_close" \
    "${PROJECT_ROOT}/tests/test_build_states_cli.py::test_cli_scaling_default_explicitly_recorded" \
    "${PROJECT_ROOT}/tests/test_validate_env_contract_cli.py::test_cli_success_exit_zero_and_report"

run_command \
    "make_features" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/make_features.py" \
    --config \
    "${PROJECT_ROOT}/configs/features.yaml" \
    --run-id \
    "${RUN_ID}" \
    --input-root \
    "${STANDARDIZED_ROOT}"

run_command \
    "validate_train_inputs" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/validate_train_inputs.py" \
    --run-id \
    "${RUN_ID}" \
    --input-root \
    "${FEATURE_PARQUET_ROOT}"

run_command \
    "validate_splits" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/validate_splits.py" \
    --run-id \
    "${RUN_ID}" \
    --input-root \
    "${FEATURE_PARQUET_ROOT}" \
    --split-mode \
    ratio_chrono \
    --train-ratio \
    0.70 \
    --val-ratio \
    0.15 \
    --test-ratio \
    0.15

run_command \
    "build_datasets" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/build_datasets.py" \
    --run-id \
    "${RUN_ID}" \
    --input-root \
    "${FEATURE_PARQUET_ROOT}" \
    --overwrite \
    true \
    --execution-price-column \
    close \
    --mark-to-market-column \
    close

run_command \
    "build_states" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/build_states.py" \
    --run-id \
    "${RUN_ID}" \
    --input-root \
    "${DATASET_ROOT}" \
    --overwrite \
    true \
    --execution-price-column \
    close \
    --mark-to-market-column \
    close

generate_env_config

run_command \
    "validate_env_contract" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/validate_env_contract.py" \
    --run-id \
    "${RUN_ID}" \
    --state-root \
    "${STATE_ROOT}" \
    --env-config \
    "${ENV_CONFIG_PATH}" \
    --smoke-step \
    true
