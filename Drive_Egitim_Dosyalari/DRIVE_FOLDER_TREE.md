# Canonical Drive Folder Tree

```text
MyDrive/
  Codex_Deneme/
    Codex_Deneme_Assets/
      raw_market_data/
        legacy_csv/
      runs/
        <RUN_ID>/
          env_contract/
            reports/
              env_contract_report.json
            tmp/
              bounded_training_preparation_env_config.json
          env_readiness/
            reports/
              training_env_readiness_report.json
              episode_catalog.json
          data_standardized/
          data_features/
            parquet/
            reports/
              feature_manifest.json
              train_input_validation_report.json
              split_validation_report.json
          data_datasets/
            parquet/
              partitions/
            reports/
              dataset_manifest.json
              dataset_build_report.json
          data_states/
            parquet/
              partitions/
            reports/
              state_manifest.json
              state_build_report.json
          colab_stage/
            stage_<UTCSTAMP>/
              colab_input_staging_manifest.json
              colab_staging_closure_report.json
              colab_runtime_dependency_report.json
          ppo_artifact/
            artifact_<UTCSTAMP>/
              canonical_ppo_model.zip
              artifact_production_manifest.json
              artifact_production_report.json
          evaluation/
            eval_<UTCSTAMP>/
              evaluation_validation_report.json
              evaluation_manifest.json
              evaluation_backtest_report.json
              evaluation_step_trace.parquet
          checkpoints/
            artifact_<UTCSTAMP>/
```

Kurallar:

- `RUN_ID` lineage id'dir
- `stage_<UTCSTAMP>`, `artifact_<UTCSTAMP>`, `eval_<UTCSTAMP>` unique attempt klasorleridir
- overwrite yasaktir
- repo clone bu Drive root altinda tutulmaz
- canonical asset root: `/content/drive/MyDrive/Codex_Deneme/Codex_Deneme_Assets`
