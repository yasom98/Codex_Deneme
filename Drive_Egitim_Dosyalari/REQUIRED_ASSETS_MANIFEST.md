# Required Assets Manifest

## File-Level Contract

Asagidaki tablo staging/training/evaluation icin her kritik girdinin tam sinifini kilitler.

| Asset | Classification |
|---|---|
| `env_contract/tmp/bounded_training_preparation_env_config.json` | `MUST BE UPLOADED TO DRIVE` |
| `configs/training_config.colab_first_real.example.json` | `COMES FROM GITHUB REPO` |
| `configs/eval_config.episodic.example.json` | `COMES FROM GITHUB REPO` |
| `data_states/reports/state_manifest.json` | `MUST BE UPLOADED TO DRIVE` |
| `env_contract/reports/env_contract_report.json` | `MUST BE UPLOADED TO DRIVE` |
| `env_readiness/reports/training_env_readiness_report.json` | `MUST BE UPLOADED TO DRIVE` |
| `env_readiness/reports/episode_catalog.json` | `MUST BE UPLOADED TO DRIVE` |
| `data_features/reports/split_validation_report.json` | `MUST BE UPLOADED TO DRIVE` |
| `data_features/reports/feature_manifest.json` | `MUST BE UPLOADED TO DRIVE` |
| `data_features/reports/train_input_validation_report.json` | `MUST BE UPLOADED TO DRIVE` |
| `data_features/parquet/` | `MUST BE UPLOADED TO DRIVE` |
| `data_datasets/reports/dataset_manifest.json` | `MUST BE UPLOADED TO DRIVE` |
| `data_datasets/reports/dataset_build_report.json` | `MUST BE UPLOADED TO DRIVE` |
| `data_datasets/parquet/partitions/` | `MUST BE UPLOADED TO DRIVE` |
| `data_states/reports/state_build_report.json` | `MUST BE UPLOADED TO DRIVE` |
| `data_states/parquet/partitions/` | `MUST BE UPLOADED TO DRIVE` |
| `colab_input_staging_manifest.json` | `GENERATED IN COLAB DURING STAGING` |
| `colab_staging_closure_report.json` | `GENERATED IN COLAB DURING STAGING` |
| `colab_runtime_dependency_report.json` | `GENERATED IN COLAB DURING STAGING` |
| `canonical_ppo_model.zip` | `GENERATED LATER DURING TRAINING/EVAL` |
| `artifact_production_manifest.json` | `GENERATED LATER DURING TRAINING/EVAL` |
| `artifact_production_report.json` | `GENERATED LATER DURING TRAINING/EVAL` |
| `evaluation_manifest.json` | `GENERATED LATER DURING TRAINING/EVAL` |
| `evaluation_validation_report.json` | `GENERATED LATER DURING TRAINING/EVAL` |
| `evaluation_backtest_report.json` | `GENERATED LATER DURING TRAINING/EVAL` |

## Minimum Required Set For First Real Training Run

Drive altinda zorunlu set:

- `runs/<RUN_ID>/env_contract/tmp/bounded_training_preparation_env_config.json`
- `runs/<RUN_ID>/env_contract/reports/env_contract_report.json`
- `runs/<RUN_ID>/env_readiness/reports/training_env_readiness_report.json`
- `runs/<RUN_ID>/env_readiness/reports/episode_catalog.json`
- `runs/<RUN_ID>/data_features/reports/feature_manifest.json`
- `runs/<RUN_ID>/data_features/reports/train_input_validation_report.json`
- `runs/<RUN_ID>/data_features/reports/split_validation_report.json`
- `runs/<RUN_ID>/data_features/parquet/`
- `runs/<RUN_ID>/data_datasets/reports/dataset_manifest.json`
- `runs/<RUN_ID>/data_datasets/reports/dataset_build_report.json`
- `runs/<RUN_ID>/data_datasets/parquet/partitions/`
- `runs/<RUN_ID>/data_states/reports/state_manifest.json`
- `runs/<RUN_ID>/data_states/reports/state_build_report.json`
- `runs/<RUN_ID>/data_states/parquet/partitions/`

Repo'dan zorunlu set:

- full repo clone
- `requirements.colab.txt`
- `configs/training_config.colab_first_real.example.json`
- `scripts/stage_colab_inputs.py`
- `scripts/produce_canonical_ppo_artifact.py`

## Minimum Required Set For First Evaluation-Only Run

Training setine ek olarak:

- `runs/<RUN_ID>/ppo_artifact/<ARTIFACT_ATTEMPT_ID>/canonical_ppo_model.zip`

Onemli:

- `scripts/stage_colab_inputs.py` model zip stage etmez
- operator model zip'i Drive'dan local VM'e manuel kopyalar

## Stage Output Contract

Training veya evaluation baslamadan once stage root'ta su dosyalar olusmus olmalidir:

- `colab_input_staging_manifest.json`
- `colab_staging_closure_report.json`
- `colab_runtime_dependency_report.json`

Bu iki alan `true` olmadan run baslatilmaz:

- `overall_closure_valid`
- `runtime_dependency_overall`
