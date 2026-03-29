#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical bounded Colab launch package for first main training path.
# Deterministic micro-patching is out of scope; this package only stages, gates, trains, and evaluates.

export RUN_ID="${RUN_ID:-20260314Tbinance_perp_hist_full_003}"
export REPO_ROOT="${REPO_ROOT:-/content/Codex_Deneme}"
export DRIVE_ROOT="/content/drive/MyDrive/Codex_Deneme/Codex_Deneme_Assets"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "BLOCK: REPO_ROOT missing: ${REPO_ROOT}" >&2
  exit 2
fi

if [[ ! -d "${DRIVE_ROOT}" ]]; then
  echo "BLOCK: DRIVE_ROOT missing: ${DRIVE_ROOT}" >&2
  exit 2
fi

required_drive_assets=(
  "${DRIVE_ROOT}/runs/${RUN_ID}/env_contract/tmp/bounded_training_preparation_env_config.json"
  "${DRIVE_ROOT}/runs/${RUN_ID}/env_contract/reports/env_contract_report.json"
  "${DRIVE_ROOT}/runs/${RUN_ID}/env_readiness/reports/training_env_readiness_report.json"
  "${DRIVE_ROOT}/runs/${RUN_ID}/env_readiness/reports/episode_catalog.json"
  "${DRIVE_ROOT}/runs/${RUN_ID}/data_states/reports/state_manifest.json"
  "${DRIVE_ROOT}/runs/${RUN_ID}/data_features/reports/split_validation_report.json"
)

for path in "${required_drive_assets[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "BLOCK: required asset missing: ${path}" >&2
    exit 2
  fi
done

if [[ ! -f "${REPO_ROOT}/configs/training_config.colab_first_real.example.json" ]]; then
  echo "BLOCK: missing training config under repo root" >&2
  exit 2
fi

if [[ ! -f "${REPO_ROOT}/configs/eval_config.episodic.example.json" ]]; then
  echo "BLOCK: missing evaluation config under repo root" >&2
  exit 2
fi

utcstamp="$(date -u +%Y%m%dT%H%M%SZ)"
export STAGE_ID="stage_${utcstamp}"
export ARTIFACT_ID="artifact_${utcstamp}"
export EVAL_ID="eval_${utcstamp}"
export STAGE_ROOT="/content/codex_stage/${RUN_ID}/${STAGE_ID}"
export ARTIFACT_OUT="/content/codex_runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ID}"
export EVAL_OUT="/content/codex_runs/${RUN_ID}/evaluation/${EVAL_ID}"

for out_path in "${STAGE_ROOT}" "${ARTIFACT_OUT}" "${EVAL_OUT}"; do
  if [[ -e "${out_path}" ]]; then
    echo "BLOCK: expected fresh output path but found existing: ${out_path}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"

python scripts/stage_colab_inputs.py \
  --staging-root "${STAGE_ROOT}" \
  --env-config "${DRIVE_ROOT}/runs/${RUN_ID}/env_contract/tmp/bounded_training_preparation_env_config.json" \
  --training-config "${REPO_ROOT}/configs/training_config.colab_first_real.example.json" \
  --state-manifest "${DRIVE_ROOT}/runs/${RUN_ID}/data_states/reports/state_manifest.json" \
  --env-contract-report "${DRIVE_ROOT}/runs/${RUN_ID}/env_contract/reports/env_contract_report.json" \
  --readiness-report "${DRIVE_ROOT}/runs/${RUN_ID}/env_readiness/reports/training_env_readiness_report.json" \
  --episode-catalog "${DRIVE_ROOT}/runs/${RUN_ID}/env_readiness/reports/episode_catalog.json" \
  --split-report "${DRIVE_ROOT}/runs/${RUN_ID}/data_features/reports/split_validation_report.json" \
  --eval-config "${REPO_ROOT}/configs/eval_config.episodic.example.json" \
  --log-level INFO

python - <<'PY'
import json
import os
from pathlib import Path

stage_root = Path(os.environ["STAGE_ROOT"])
closure_path = stage_root / "colab_staging_closure_report.json"
runtime_path = stage_root / "colab_runtime_dependency_report.json"
if not closure_path.exists():
    raise SystemExit(f"BLOCK: missing closure report: {closure_path}")
if not runtime_path.exists():
    raise SystemExit(f"BLOCK: missing runtime dependency report: {runtime_path}")

closure = json.loads(closure_path.read_text(encoding="utf-8"))
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
if closure.get("overall_closure_valid") is not True:
    raise SystemExit("BLOCK: overall_closure_valid != true")
if runtime.get("runtime_dependency_overall") is not True:
    raise SystemExit("BLOCK: runtime_dependency_overall != true")
PY

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("BLOCK: CUDA is not available; aborting before artifact production.")
PY

python scripts/produce_canonical_ppo_artifact.py \
  --run-id "${RUN_ID}" \
  --env-config "${STAGE_ROOT}/env_contract/tmp/bounded_training_preparation_env_config.json" \
  --training-config "${STAGE_ROOT}/configs/training_config.json" \
  --state-manifest "${STAGE_ROOT}/data_states/reports/state_manifest.json" \
  --env-contract-report "${STAGE_ROOT}/env_contract/reports/env_contract_report.json" \
  --readiness-report "${STAGE_ROOT}/env_readiness/reports/training_env_readiness_report.json" \
  --episode-catalog "${STAGE_ROOT}/env_readiness/reports/episode_catalog.json" \
  --split-report "${STAGE_ROOT}/data_features/reports/split_validation_report.json" \
  --output-dir "${ARTIFACT_OUT}" \
  --progress-mode auto \
  --memory-log-interval-steps 2048 \
  --log-level INFO

python - <<'PY'
import json
import os
from pathlib import Path

artifact_out = Path(os.environ["ARTIFACT_OUT"])
model_path = artifact_out / "canonical_ppo_model.zip"
report_path = artifact_out / "artifact_production_report.json"
if not model_path.exists():
    raise SystemExit(f"BLOCK: canonical_ppo_model.zip missing: {model_path}")
if not report_path.exists():
    raise SystemExit(f"BLOCK: artifact_production_report.json missing: {report_path}")

report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("canonical_artifact_ready") is not True:
    raise SystemExit("BLOCK: artifact_production_report canonical_artifact_ready != true")
PY

python scripts/evaluate_policy.py \
  --run-id "${RUN_ID}" \
  --model-artifact "${ARTIFACT_OUT}/canonical_ppo_model.zip" \
  --env-config "${STAGE_ROOT}/env_contract/tmp/bounded_training_preparation_env_config.json" \
  --eval-config "${STAGE_ROOT}/configs/eval_config.json" \
  --state-manifest "${STAGE_ROOT}/data_states/reports/state_manifest.json" \
  --env-contract-report "${STAGE_ROOT}/env_contract/reports/env_contract_report.json" \
  --readiness-report "${STAGE_ROOT}/env_readiness/reports/training_env_readiness_report.json" \
  --episode-catalog "${STAGE_ROOT}/env_readiness/reports/episode_catalog.json" \
  --split-report "${STAGE_ROOT}/data_features/reports/split_validation_report.json" \
  --output-dir "${EVAL_OUT}" \
  --progress-mode auto \
  --log-level INFO

python - <<'PY'
import json
import os
from pathlib import Path

artifact_report_path = Path(os.environ["ARTIFACT_OUT"]) / "artifact_production_report.json"
eval_report_path = Path(os.environ["EVAL_OUT"]) / "evaluation_backtest_report.json"
if not artifact_report_path.exists():
    raise SystemExit(f"BLOCK: artifact report missing: {artifact_report_path}")
if not eval_report_path.exists():
    raise SystemExit(f"BLOCK: evaluation report missing: {eval_report_path}")

artifact = json.loads(artifact_report_path.read_text(encoding="utf-8"))
evaluation = json.loads(eval_report_path.read_text(encoding="utf-8"))

canonical_artifact_ready = bool(artifact.get("canonical_artifact_ready"))
evaluation_success = bool(evaluation.get("evaluation_success"))
if not canonical_artifact_ready:
    raise SystemExit("BLOCK: canonical_artifact_ready != true")
if not evaluation_success:
    raise SystemExit("BLOCK: evaluation_success != true")

strategy_metrics = evaluation.get("strategy_metrics") or {}
benchmark_metrics = evaluation.get("benchmark_metrics") or {}

summary = {
    "run_id": os.environ["RUN_ID"],
    "drive_root": os.environ["DRIVE_ROOT"],
    "stage_root": os.environ["STAGE_ROOT"],
    "artifact_out": os.environ["ARTIFACT_OUT"],
    "eval_out": os.environ["EVAL_OUT"],
    "canonical_artifact_ready": canonical_artifact_ready,
    "evaluation_success": evaluation_success,
    "strategy_total_return": strategy_metrics.get("total_return"),
    "benchmark_total_return": benchmark_metrics.get("total_return"),
}
print(json.dumps(summary, separators=(",", ":"), ensure_ascii=True))
PY
