# GitHub + Drive + Colab Workflow

## Canonical Model

Bu repo icin tek canonical model sudur:

- GitHub = code / config / docs / tests / spec
- Google Drive = heavy training/evaluation inputs ve run outputs
- Colab = disposable execution environment

Kararlar:

- Colab full repo clone yapar
- Yalnizca training subset fetch edilmez
- Training ve evaluation local VM stage root'tan calisir
- Drive mounted path ustunden dogrudan cok-sayida kucuk read ile run baslatilmaz
- `tgz` day-to-day tasima modeli degildir

## Source Of Truth Rules

### GitHub'da kalacaklar

- `src/`
- `scripts/`
- `tests/`
- `configs/`
- `indicator_specs/`
- operator docs
- audit docs
- Drive packaging contract scaffold'i

### Drive'da kalacaklar

- raw market data
- feature/state/dataset parquet outputlari
- env/input reports
- stage provenance reports
- model artifacts
- evaluation reports
- checkpoints

## Colab First-Run Contract

1. Colab GPU runtime sec.
2. Drive'i `/content/drive` altina mount et.
3. Canonical asset root'u fixed kullan:
   - `/content/drive/MyDrive/Codex_Deneme/Codex_Deneme_Assets`
4. Asset root altinda required run dosyalari yoksa fail-closed dur.
5. Repo'yu GitHub'dan `/content/Codex_Deneme` altina clone veya pull et.
6. `python -m pip install -r requirements.colab.txt` calistir.
7. `torch` ve staging runtime dependency gate'i gecmeden training baslatma.
8. Drive'daki required run assetlerini `scripts/stage_colab_inputs.py` ile local stage root'a tas.
9. Training/evaluation local stage root'tan calissin.
10. Final report/artifact setini Drive'da unique attempt klasorlerine sync et.

Canonical operator package:

- `bash /content/Codex_Deneme/scripts/run_colab_main_training_package.sh`

## Fail-Closed Runtime Gate

Asagidaki sartlar saglanmadan training/evaluation baslatilmaz:

- `torch` import edilir
- `torch.cuda.is_available()` training icin `true`
- `stable_baselines3` import edilir
- `gymnasium` import edilir
- `pandas` import edilir
- `pyarrow` import edilir
- stage closure report `overall_closure_valid=true`
- runtime dependency report `runtime_dependency_overall=true`

## Minimum Required Drive Asset Set

### Training

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

### Evaluation-Only

Training setine ek olarak:

- `runs/<RUN_ID>/ppo_artifact/<ARTIFACT_ATTEMPT_ID>/canonical_ppo_model.zip`

Onemli not:

- `scripts/stage_colab_inputs.py` model zip stage etmez
- evaluation-only path'te operator model zip'i Drive'dan local VM'e MANUEL kopyalar

## RUN_ID Policy

- `RUN_ID` = lineage id
- `RUN_ID` = attempt id degildir
- Yeni serious training/eval attempt varsayilan olarak yeni `RUN_ID` olusturmaz
- Ayni lineage ile birden fazla unique attempt klasoru olusturulabilir
- Yeni `RUN_ID` ancak upstream lineage degistiginde olusur

Canonical child output klasorleri:

- stage: `runs/<RUN_ID>/colab_stage/stage_<UTCSTAMP>/`
- training artifact: `runs/<RUN_ID>/ppo_artifact/artifact_<UTCSTAMP>/`
- evaluation: `runs/<RUN_ID>/evaluation/eval_<UTCSTAMP>/`

Overwrite politikasi:

- mevcut attempt klasoru yeniden kullanilmaz
- mevcut Drive output root'u uzerine yazilmaz
- yeni unique attempt klasoru acilir

## Legacy Docs

`docs/handoffs/` klasoru:

- legacy evidence'tir
- canonical operator workflow degildir
- path authority kaynagi degildir
- tarihsel referans olarak repo'da kalabilir

## Migration Debt

Repo root'taki mevcut tracked CSV'ler:

- `BTC_USDT_15m_price_data.csv`
- `BTC_USDT_5m_price_data.csv`

Karar:

- bu dosyalar bu passtta silinmez
- canonical repo asset'i olarak kabul edilmez
- `legacy tracked migration debt` olarak degerlendirilir
