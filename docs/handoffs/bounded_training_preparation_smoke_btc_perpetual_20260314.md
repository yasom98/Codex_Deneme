# Bounded Training-Preparation Smoke: BTCUSDT Perpetual 2026-03-14

## 1. Purpose

This document records the bounded training-preparation smoke executed for the proven BTCUSDT perpetual run `20260314Tbinance_perp_hist_full_003`.

The goal of this task was to produce the missing launcher-prerequisite artifacts needed before any bounded training execution task can be opened. This was not a training launch.

## 2. Preflight status

### Verified fact

The downloaded BTCUSDT gap data were already processed before this task started.

Evidence files:

- `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/separate_parallel_lineage_processing_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/summary.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/feature_manifest.json`

Verified facts from those artifacts:

- `standardize.status = success`
- `feature_build.status = success`
- `feature_contract_compatibility.status = success`
- `feature_manifest.json` records `timestamp = datetime64[ns, UTC]`
- `summary.json` records `total_files = 3`, `succeeded_files = 3`, `failed_files = 0`, `manifest_generated = true`

### Verified fact

Indicators/features had already been computed before this task started.

Evidence files:

- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/summary.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/feature_manifest.json`

### Verified fact

The three BTCUSDT feature parquet outputs were present at task start:

- `runs/20260314Tbinance_perp_hist_full_003/data_features/parquet/binance_perpetual__BTC_USDT_1m_price_data.parquet`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/parquet/binance_perpetual__BTC_USDT_5m_price_data.parquet`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/parquet/binance_perpetual__BTC_USDT_15m_price_data.parquet`

### Verified fact

At task start, the run was prep-ready but not launch-ready.

Reason:

- the canonical feature artifacts existed and were contract-valid
- but the launcher-prerequisite chain had not yet been materialized for this run

Missing-at-start artifacts:

- `data_features/reports/train_input_validation_report.json`
- `data_features/reports/split_validation_report.json`
- `data_datasets/reports/dataset_manifest.json`
- `data_datasets/reports/dataset_build_report.json`
- `data_states/reports/state_manifest.json`
- `data_states/reports/state_build_report.json`
- `env_contract/reports/env_contract_report.json`
- `env_readiness/reports/training_env_readiness_report.json`
- `env_readiness/reports/episode_catalog.json`
- run-specific explicit env config JSON

## 3. Explicit inputs used

- Run id: `20260314Tbinance_perp_hist_full_003`
- Canonical feature parquet root:
  - `runs/20260314Tbinance_perp_hist_full_003/data_features/parquet`
- Feature manifest:
  - `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/feature_manifest.json`
- Feature summary:
  - `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/summary.json`
- Processing proof:
  - `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/separate_parallel_lineage_processing_report.json`
- Split policy used:
  - `ratio_chrono`
  - `train_ratio = 0.70`
  - `val_ratio = 0.15`
  - `test_ratio = 0.15`
- Runtime price columns used:
  - `execution_price_column = close`
  - `mark_to_market_column = close`
- Readiness invocation used:
  - `selection_policy = seeded_random_episode`
  - `start_policy = start_at_valid_from_row`
  - `min_remaining_steps = 2`
  - `seed = 42`

## 4. Steps executed

| Step | Status | Notes |
| --- | --- | --- |
| `validate_train_inputs.py` | `pass` | Completed successfully against the explicit feature parquet root. |
| `validate_splits.py` | `pass` | Initial parallel invocation failed because `train_input_validation_report.json` had not been written yet; sequential rerun passed. |
| `build_datasets.py` | `pass` | Completed successfully with explicit manifest/report inputs and runtime price columns. |
| `build_states.py` | `pass` | Initial parallel invocation failed because dataset reports were not yet written; sequential rerun passed. |
| Run-specific env config generation | `pass` | Generated from the produced `state_manifest.json` using the repo’s existing canonical train-partition selection pattern. |
| `validate_env_contract.py` | `pass` | Completed successfully with `--smoke-step true`. |
| `validate_training_env_readiness.py` | `pass` | Completed successfully and wrote both readiness and episode-catalog reports. |

## 5. Artifacts produced

Newly produced artifacts in this task:

- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/train_input_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/split_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_datasets/reports/dataset_build_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_datasets/reports/dataset_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_build_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_contract/tmp/bounded_training_preparation_env_config.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`

Verified result highlights:

- `train_input_validation_overall = true`
- `split_validation_overall = true`
- `dataset_build_overall = true`
- `state_build_overall = true`
- `state_manifest.output_completeness_ok = true`
- `env_contract_overall = true`
- `readiness_overall = true`
- `episode_catalog_overall = true`

## 6. Remaining blockers, if any

### Hard blockers

- None found during this bounded preparation task.

### Minor notes

- Two early parallel invocations failed closed because their upstream prerequisite reports had not been written yet:
  - `validate_splits.py`
  - `build_states.py`
- Both passed on immediate sequential rerun with the same explicit inputs.

## 7. Launch-readiness conclusion

### Verified fact

This run is now ready for a bounded training execution smoke task.

Basis for that conclusion:

- the canonical BTCUSDT perpetual feature artifacts are present and contract-valid
- the training-preparation prerequisite chain now exists for this run
- the required launch-side evidence artifacts now exist:
  - run-specific explicit env config
  - `state_manifest.json`
  - `env_contract_report.json`
  - `training_env_readiness_report.json`
  - `episode_catalog.json`

Fail-closed note:

- This conclusion means the run is ready to open a bounded training execution smoke task.
- It does not claim that full training has already been launched or completed.

## 8. Recommended next narrow task

Open one bounded training execution smoke task for `20260314Tbinance_perp_hist_full_003` using explicit inputs:

- env config:
  - `runs/20260314Tbinance_perp_hist_full_003/env_contract/tmp/bounded_training_preparation_env_config.json`
- training config:
  - `configs/training_config.launch_smoke.example.json`
- state manifest:
  - `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- env contract report:
  - `runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- readiness report:
  - `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- episode catalog:
  - `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`

That next task should remain bounded and should execute `launch_training.py` once with a fresh explicit output directory.
