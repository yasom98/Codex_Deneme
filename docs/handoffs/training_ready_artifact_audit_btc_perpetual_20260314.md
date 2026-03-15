# Training-Ready Artifact Audit: BTCUSDT Perpetual 2026-03-14

## 1. Purpose

This document audits whether the proven BTCUSDT Binance USD-M perpetual lineage is ready to serve as the canonical input basis for a bounded training smoke.

This is a readiness audit only. It does not start training, does not redesign the pipeline, and does not claim non-BTC operational readiness.

## 2. Canonical basis

- Repo: `Codex_Deneme`
- Symbol: `BTCUSDT`
- Canonical successful session id: `20260314Tbinance_perp_hist_full_003`
- Explicit cutoff: `2026-03-14T10:00:00+00:00`
- Prior state-capture reference: `docs/handoffs/post_gapfill_audit_btc_perpetual_20260314.md`

## 3. Training-relevant artifacts inspected

| Artifact / Surface | Why It Matters |
| --- | --- |
| `docs/handoffs/post_gapfill_audit_btc_perpetual_20260314.md` | Confirms the canonical successful BTCUSDT perpetual data operation and immutable baseline. |
| `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/historical_backfill_checkpoint.json` | Verifies the completed raw Binance perpetual backfill session, cutoff, and per-timeframe completion. |
| `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/separate_parallel_lineage_processing_report.json` | Verifies standardization success, feature build success, and feature contract compatibility success. |
| `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/feature_manifest.json` | Verifies the feature contract surface, including canonical dtype evidence such as `timestamp = datetime64[ns, UTC]`. |
| `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/summary.json` | Verifies the feature build output root, total files processed, parity/indicator checks, and manifest generation. |
| `scripts/validate_train_inputs.py` | Defines the canonical next feature-input validation entrypoint and its default input root. |
| `scripts/validate_splits.py` | Defines the canonical split validation entrypoint and confirms it also starts from the feature parquet root. |
| `scripts/build_datasets.py` | Defines the canonical dataset-build entrypoint from validated feature parquet inputs. |
| `scripts/build_states.py` | Defines the canonical state-build entrypoint from dataset artifacts and the expected state manifest path. |
| `scripts/validate_env_contract.py` | Defines the canonical env-contract gate and the required explicit env-config + state-root inputs. |
| `scripts/validate_training_env_readiness.py` | Defines the canonical readiness gate and the required state-root + env-config inputs. |
| `scripts/launch_training.py` | Defines the final bounded launch gate and the exact explicit artifacts it requires. |
| `src/rl/training_launcher.py` | Confirms the launch gate validates `state_manifest`, `env_contract_report`, `training_env_readiness_report`, and `episode_catalog` strictly. |
| `configs/training_config.launch_smoke.example.json` | Confirms a bounded training-launch config example exists for the next smoke stage. |

## 4. Current readiness assessment

### Verified ready

The following artifact set is verified and appears to be the canonical basis for the next training-preparation step:

- Feature parquet input root:
  - `runs/20260314Tbinance_perp_hist_full_003/data_features/parquet`
- Feature manifest:
  - `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/feature_manifest.json`
- Feature summary:
  - `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/summary.json`
- Supporting processing evidence:
  - `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/separate_parallel_lineage_processing_report.json`

Verified facts:

- `data_features/parquet` exists and contains the three BTCUSDT perpetual files for `1m`, `5m`, and `15m`.
- `feature_manifest.json` exists and records `timestamp = datetime64[ns, UTC]`.
- `summary.json` records `total_files = 3`, `succeeded_files = 3`, `failed_files = 0`, `manifest_generated = true`.
- `separate_parallel_lineage_processing_report.json` records:
  - `standardize.status = success`
  - `feature_build.status = success`
  - `feature_contract_compatibility.status = success`

Inference:

- Because `scripts/validate_train_inputs.py`, `scripts/validate_splits.py`, and `scripts/build_datasets.py` all default to `runs/<run_id>/data_features/parquet`, this feature parquet root is the canonical next-stage input basis for a bounded training-preparation smoke.

### Partially verified / ambiguous

- `configs/training_config.launch_smoke.example.json` exists and is the repo-local bounded training-launch config example.
  - Verified fact: the file exists.
  - Unresolved ambiguity: it is not sufficient on its own to launch training for this run because the launch gate also requires state, env contract, readiness, and episode-catalog artifacts that are not yet present for `20260314Tbinance_perp_hist_full_003`.

### Missing / blocking

The following launch-precondition artifacts are missing for `20260314Tbinance_perp_hist_full_003`:

- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/train_input_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/split_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_datasets/reports/dataset_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_datasets/reports/dataset_build_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_build_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`
- a run-specific explicit env config JSON for this run

Verified fact:

- `scripts/launch_training.py` requires all of the following explicit inputs:
  - `--env-config`
  - `--training-config`
  - `--state-manifest`
  - `--env-contract-report`
  - `--readiness-report`
  - `--episode-catalog`
  - `--output-dir`

Conclusion:

- The BTCUSDT perpetual lineage is verified as the canonical data basis for the next training-preparation step.
- It is not yet directly launch-training-ready because the downstream training-precondition artifacts have not yet been materialized for this run.

## 5. Explicit-path recommendation

The explicit canonical input path for the next step is:

- `runs/20260314Tbinance_perp_hist_full_003/data_features/parquet`

Supporting explicit evidence paths to carry into the next task:

- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/feature_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/summary.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/separate_parallel_lineage_processing_report.json`
- `configs/training_config.launch_smoke.example.json`

Fail-closed note:

- A single canonical explicit path for `launch_training.py` cannot yet be named truthfully, because the required `state_manifest`, `env_contract_report`, `training_env_readiness_report`, and `episode_catalog` artifacts do not yet exist for this run.

## 6. Blocking issues or ambiguities

### Hard blockers

- The training-launch prerequisite chain has not yet been materialized for `20260314Tbinance_perp_hist_full_003`.
- The run-specific launch gate inputs required by `scripts/launch_training.py` are therefore incomplete.

### Minor notes

- No evidence in this audit contradicts the BTCUSDT perpetual feature artifacts themselves.
- No non-BTC symbol is claimed as operationally proven by this audit.

## 7. Recommended next narrow task

Open one bounded training-preparation smoke task for run `20260314Tbinance_perp_hist_full_003` that starts explicitly from:

- `runs/20260314Tbinance_perp_hist_full_003/data_features/parquet`

That next task should do only the minimum contract chain needed to make a later launch-smoke possible:

1. run `validate_train_inputs.py`
2. run `validate_splits.py`
3. run `build_datasets.py`
4. run `build_states.py`
5. generate a run-specific env config from the resulting train partition/state manifest
6. run `validate_env_contract.py`
7. run `validate_training_env_readiness.py`

This is the single correct bounded next step because it materializes the missing launch prerequisites without broadening into full training execution.

## 8. Boundaries

- BTCUSDT proven route only
- This is a readiness audit, not a training execution
- No claim is made here about non-BTC operational readiness
