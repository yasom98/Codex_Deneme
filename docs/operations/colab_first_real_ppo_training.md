# Purpose

Bu dokuman, ilk ciddi Colab tabanli PPO artifact-production ve evaluation akisini canonical GitHub + Drive + local staging modeliyle kilitler.

# Scope

- Proven lineage only: `20260314Tbinance_perp_hist_full_003`
- Canonical chain only:
  - `scripts/stage_colab_inputs.py`
  - `scripts/produce_canonical_ppo_artifact.py`
  - `scripts/evaluate_policy.py`
- `scripts/launch_training.py` smoke/prelaunch gate olarak kalir.
- `docs/handoffs/` klasoru legacy evidence'tir; canonical operator workflow degildir.

# Canonical Storage Policy

- GitHub = code / config / docs / tests / spec
- Google Drive = agir run-scoped data ve output
- Colab = disposable execution environment
- Colab full repo clone yapar
- Training ve evaluation local VM stage root'tan calisir
- Overwrite yasaktir
- Her stage / artifact / eval attempt icin benzersiz output klasoru kullanilir

# Hardware Policy

- Preferred Colab GPU: `A100`
- Fallback Colab GPU: `T4`
- `H100` bu ilk serious path icin zorunlu degildir

# Runtime Setup

Colab notebook'ta once Drive'i mount et:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Ardindan:

- `Runtime -> Change runtime type -> GPU`
- GPU secilmeden training baslatma

Canonical path degiskenleri:

```bash
export REPO_ROOT=/content/Codex_Deneme
export DRIVE_ROOT=/content/drive/MyDrive/Codex_Deneme/Codex_Deneme_Assets
export RUN_ID=20260314Tbinance_perp_hist_full_003
export STAGE_ID=stage_$(date -u +%Y%m%dT%H%M%SZ)
export ARTIFACT_ID=artifact_$(date -u +%Y%m%dT%H%M%SZ)
export EVAL_ID=eval_$(date -u +%Y%m%dT%H%M%SZ)
export STAGE_ROOT=/content/codex_stage/${RUN_ID}/${STAGE_ID}
export ARTIFACT_OUT=/content/codex_runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ID}
export EVAL_OUT=/content/codex_runs/${RUN_ID}/evaluation/${EVAL_ID}
```

# Canonical Main-Run Package

Main training icin tek canonical operator paketi (repo clone/pull ve dependency kurulumu tamamlandiktan sonra):

```bash
bash "${REPO_ROOT}/scripts/run_colab_main_training_package.sh"
```

Bu paket:

- fixed DRIVE_ROOT kullanir: `/content/drive/MyDrive/Codex_Deneme/Codex_Deneme_Assets`
- stage closure + runtime dependency gate'lerini fail-closed kontrol eder
- explicit CUDA gate'i artifact production oncesi fail-closed uygular
- final compact JSON summary'de canonical metrikleri okur:
  - `strategy_total_return`
  - `benchmark_total_return`

Repo'yu GitHub'dan clone veya pull et:

```bash
if [ ! -d "${REPO_ROOT}/.git" ]; then
  git clone https://github.com/yasom98/Codex_Deneme.git "${REPO_ROOT}"
else
  git -C "${REPO_ROOT}" pull --ff-only
fi
```

Bagimliliklari kur:

```bash
cd "${REPO_ROOT}"
python -m pip install -r requirements.colab.txt
```

# Fail-Closed Torch And Dependency Gate

Bu path fail-closed'dur:

- `torch` import olmadan training baslamaz
- `stable_baselines3`, `gymnasium`, `pandas`, `pyarrow` importlari gecmeden training baslamaz
- `scripts/stage_colab_inputs.py` tarafindan uretilen runtime dependency report `runtime_dependency_overall=true` degilse training veya evaluation baslatilmaz

Istege bagli degil, zorunlu gate:

```bash
python - <<'PY'
import importlib
import torch

required = ["stable_baselines3", "gymnasium", "pandas", "pyarrow", "numpy", "tqdm"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing runtime dependencies: {missing}")
print({
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; do not start first real training run.")
PY
```

# Required Drive Inputs

Ilk real training run icin Drive'da asagidaki set bulunmalidir:

- `${DRIVE_ROOT}/runs/${RUN_ID}/env_contract/tmp/bounded_training_preparation_env_config.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/env_contract/reports/env_contract_report.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/env_readiness/reports/training_env_readiness_report.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/env_readiness/reports/episode_catalog.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_features/reports/feature_manifest.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_features/reports/train_input_validation_report.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_features/reports/split_validation_report.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_features/parquet/`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_datasets/reports/dataset_manifest.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_datasets/reports/dataset_build_report.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_datasets/parquet/partitions/`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_states/reports/state_manifest.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_states/reports/state_build_report.json`
- `${DRIVE_ROOT}/runs/${RUN_ID}/data_states/parquet/partitions/`

Training config repo'dan gelir:

- `${REPO_ROOT}/configs/training_config.colab_first_real.example.json`

Evaluation-only icin ek olarak su model zip Drive'da bulunmalidir:

- `${DRIVE_ROOT}/runs/${RUN_ID}/ppo_artifact/<ARTIFACT_ATTEMPT_ID>/canonical_ppo_model.zip`

# Staging Workflow

Mounted Drive ustunden dogrudan training calistirma. Closure-complete local stage root kullan:

```bash
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
```

Staging sonrasi beklenen kritik dosyalar:

- `${STAGE_ROOT}/colab_input_staging_manifest.json`
- `${STAGE_ROOT}/colab_staging_closure_report.json`
- `${STAGE_ROOT}/colab_runtime_dependency_report.json`
- `${STAGE_ROOT}/configs/training_config.json`
- `${STAGE_ROOT}/configs/eval_config.json`

Training veya evaluation baslatmadan once bu gate'i zorunlu kontrol et:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

stage_root = Path(os.environ["STAGE_ROOT"])
closure = json.loads((stage_root / "colab_staging_closure_report.json").read_text(encoding="utf-8"))
runtime = json.loads((stage_root / "colab_runtime_dependency_report.json").read_text(encoding="utf-8"))
payload = {
    "overall_closure_valid": closure["overall_closure_valid"],
    "runtime_dependency_overall": runtime["runtime_dependency_overall"],
}
print(payload)
if payload["overall_closure_valid"] is not True:
    raise SystemExit("Closure validation failed; do not start training/evaluation.")
if payload["runtime_dependency_overall"] is not True:
    raise SystemExit("Runtime dependency validation failed; do not start training/evaluation.")
PY
```

# First Real Training Command

`startup_policy=fresh_only` politikasi nedeniyle `ARTIFACT_OUT` onceden var olmamalidir.

```bash
cd "${REPO_ROOT}"

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
```

Basarili training artifact seti:

- `${ARTIFACT_OUT}/canonical_ppo_model.zip`
- `${ARTIFACT_OUT}/artifact_production_manifest.json`
- `${ARTIFACT_OUT}/artifact_production_report.json`
- `${ARTIFACT_OUT}/checkpoint_artifacts/ppo_model_step_00025000.zip` *(istege bagli, checkpoint_export_steps yapilandirildiginda)*
- `${ARTIFACT_OUT}/checkpoint_artifacts/ppo_model_step_00050000.zip`
- `${ARTIFACT_OUT}/checkpoint_artifacts/ppo_model_step_00100000.zip`

# Evaluation Command

Training'den hemen sonra ayni stage root ile evaluation calisabilir:

```bash
cd "${REPO_ROOT}"

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
```

Evaluation-only run icin manuel model copy adimi zorunludur. `scripts/stage_colab_inputs.py` model zip'i otomatik stage ETMEZ.

```bash
export ARTIFACT_ATTEMPT_ID=artifact_20260315T000000Z
export MODEL_LOCAL_ROOT=/content/codex_models/${RUN_ID}/${ARTIFACT_ATTEMPT_ID}
mkdir -p "${MODEL_LOCAL_ROOT}"
cp "${DRIVE_ROOT}/runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ATTEMPT_ID}/canonical_ppo_model.zip" \
  "${MODEL_LOCAL_ROOT}/canonical_ppo_model.zip"
```

Evaluation-only komutu:

```bash
cd "${REPO_ROOT}"

python scripts/evaluate_policy.py \
  --run-id "${RUN_ID}" \
  --model-artifact "${MODEL_LOCAL_ROOT}/canonical_ppo_model.zip" \
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
```

# Sync-Back Contract

Drive authoritative output root'lari:

- stage provenance: `${DRIVE_ROOT}/runs/${RUN_ID}/colab_stage/${STAGE_ID}/`
- artifact production: `${DRIVE_ROOT}/runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ID}/`
- evaluation: `${DRIVE_ROOT}/runs/${RUN_ID}/evaluation/${EVAL_ID}/`

Success sonrasi yazilacak setler:

- stage:
  - `colab_input_staging_manifest.json`
  - `colab_staging_closure_report.json`
  - `colab_runtime_dependency_report.json`
- training:
  - `canonical_ppo_model.zip`
  - `artifact_production_manifest.json`
  - `artifact_production_report.json`
  - `checkpoint_artifacts/` *(istege bagli, checkpoint_export_steps yapilandirildiginda)*
- evaluation:
  - `evaluation_validation_report.json`
  - `evaluation_manifest.json`
  - `evaluation_backtest_report.json`
  - varsa `evaluation_step_trace.parquet`
  - varsa risk overlay loglari

Example sync commands:

```bash
mkdir -p "${DRIVE_ROOT}/runs/${RUN_ID}/colab_stage/${STAGE_ID}"
cp "${STAGE_ROOT}/colab_input_staging_manifest.json" "${DRIVE_ROOT}/runs/${RUN_ID}/colab_stage/${STAGE_ID}/"
cp "${STAGE_ROOT}/colab_staging_closure_report.json" "${DRIVE_ROOT}/runs/${RUN_ID}/colab_stage/${STAGE_ID}/"
cp "${STAGE_ROOT}/colab_runtime_dependency_report.json" "${DRIVE_ROOT}/runs/${RUN_ID}/colab_stage/${STAGE_ID}/"

mkdir -p "${DRIVE_ROOT}/runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ID}"
cp "${ARTIFACT_OUT}/canonical_ppo_model.zip" "${DRIVE_ROOT}/runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ID}/"
cp "${ARTIFACT_OUT}/artifact_production_manifest.json" "${DRIVE_ROOT}/runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ID}/"
cp "${ARTIFACT_OUT}/artifact_production_report.json" "${DRIVE_ROOT}/runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ID}/"
cp -r "${ARTIFACT_OUT}/checkpoint_artifacts" "${DRIVE_ROOT}/runs/${RUN_ID}/ppo_artifact/${ARTIFACT_ID}/" 2>/dev/null || true

mkdir -p "${DRIVE_ROOT}/runs/${RUN_ID}/evaluation/${EVAL_ID}"
cp "${EVAL_OUT}/evaluation_validation_report.json" "${DRIVE_ROOT}/runs/${RUN_ID}/evaluation/${EVAL_ID}/"
cp "${EVAL_OUT}/evaluation_manifest.json" "${DRIVE_ROOT}/runs/${RUN_ID}/evaluation/${EVAL_ID}/"
cp "${EVAL_OUT}/evaluation_backtest_report.json" "${DRIVE_ROOT}/runs/${RUN_ID}/evaluation/${EVAL_ID}/"
```

Failure policy:

- Stage failure: yalnizca stage report seti varsa onu sync et
- Training failure: final report/manifest yazildiysa unique artifact klasorunu sync et
- Evaluation failure: final report yazildiysa unique eval klasorunu sync et
- `.tmp` dosyalari sync edilmez
- Var olan Drive attempt klasoru UZERINE YAZILMAZ

# Operator Checklist

- repo GitHub'dan full clone veya pull edildi
- `requirements.colab.txt` kuruldu
- `torch` import edildi ve `cuda_available=true`
- staging manifest exists and `status=success`
- closure report exists and `overall_closure_valid=true`
- runtime dependency report exists and `runtime_dependency_overall=true`
- `ARTIFACT_OUT` benzersiz
- `EVAL_OUT` benzersiz
- evaluation-only ise model zip manuel local copy ile alindi

# Non-Goals

- schema redesign yok
- environment redesign yok
- feature redesign yok
- hyperparameter sweep redesign yok
- legacy tracked CSV migration bu passtta yok
