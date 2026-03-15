# Codex_Deneme Repo Audit Report

## Summary

Bu audit, repo'yu GitHub + Google Drive + Colab hybrid execution modeli icin degerlendirir.

Karar:

- Yerel working tree code-complete olgunluga yakindir.
- GitHub'in canonical code source of truth olmasi icin local code/config/docs/tests yuzeyinin push edilmesi gerekir.
- Google Drive canonical heavy asset source of truth olmalidir.
- Colab canonical olarak full repo clone + local staging root ile calismalidir.

## Inspected Areas

- `src/`
- `scripts/`
- `tests/`
- `configs/`
- `docs/`
- `indicator_specs/`
- `runs/`
- Colab staging/runtime yardimcilari
- training/evaluation launchers
- path/config resolution code
- `.gitignore`

## Code Completeness Judgment

Repo'da asagidaki kritik execution yuzeyi mevcuttur:

- explicit training/eval launchers
- env/data/state/feature pipeline kodu
- readiness/contract validation kodu
- Colab staging/runtime helpers
- machine-readable report/manifests yazan bounded execution yolu

Eksik olan canonical repo unsurlari bu passtta eklenmistir:

- `requirements.colab.txt`
- GitHub + Drive + Colab operator workflow dokumani
- `Drive_Egitim_Dosyalari/` packaging contract scaffold'i
- machine-specific path contamination icin config duzeltmesi

## GitHub Classification

### MUST PUSH

GitHub'a push edilmesi gereken siniflar:

- `src/` altindaki code dosyalari
- `scripts/` altindaki code dosyalari
- `tests/`
- `configs/*.json`, `configs/*.yaml`
- `indicator_specs/`
- canonical workflow docs
- audit docs

Ozellikle local working tree'de bulunan ve Colab/training path icin kritik olan dosyalar MUST PUSH blocker'idir:

- `scripts/stage_colab_inputs.py`
- `src/rl/colab_runtime.py`
- `src/rl/colab_staging_closure.py`
- `src/rl/notebook_progress.py`
- `configs/training_config.colab_first_real.example.json`

### SAFE TO PUSH

- code-like docs/tests/config/spec dosyalari
- operator docs
- manifest template dosyalari
- `Drive_Egitim_Dosyalari/` scaffold'i

### MUST NOT PUSH

- `runs/`
- checkpoint/output/report artifactlari
- local runtime cache/noise
- yeni agir CSV/data files

## Google Drive Classification

### REQUIRED FOR FIRST REAL TRAINING RUN

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

### REQUIRED FOR EVALUATION-ONLY

Training setine ek olarak:

- `runs/<RUN_ID>/ppo_artifact/<ARTIFACT_ATTEMPT_ID>/canonical_ppo_model.zip`

### OPTIONAL

- `raw_market_data/legacy_csv/`
- `runs/<RUN_ID>/data_standardized/`
- `runs/<RUN_ID>/data_tail_refresh/`
- smoke output klasorleri
- checkpoint klasorleri

## Local-Only Path Blockers

Audit aninda tespit edilen machine-specific blocker:

- `configs/data.yaml` icinde absolute local path vardi

Bu passtta repo-relative hale getirilmistir.

## Legacy Tracked Migration Debt

Su iki dosya repo root'ta tracked agir asset olarak duruyor:

- `BTC_USDT_15m_price_data.csv`
- `BTC_USDT_5m_price_data.csv`

Karar:

- Bu dosyalar canonical repo asset'i degildir
- Bu passtta silinmeyecekler
- `legacy tracked migration debt` olarak acikca isaretlenecekler
- Yeni agir data dosyalari GitHub'a eklenmeyecektir

## Legacy Docs Policy

`docs/handoffs/` klasoru:

- repo'da kalir
- canonical operator workflow input'u sayilmaz
- legacy evidence/history olarak tutulur
- yeni workflow docs tarafindan `non-canonical legacy evidence` olarak isaretlenir

## Readiness Judgment

Bu passtan sonra repo:

- canonical workflow dokumani
- Drive packaging contract'i
- Colab dependency manifest'i
- repo-relative path hijyeni

bakimindan daha net hale gelir.

Yine de operatorun sonraki zorunlu adimi vardir:

- local code/config/docs/tests degisikliklerini GitHub'a push etmek
- heavy/run-scoped assetleri Drive'a tek seferlik canonical taxonomy ile yuklemek
