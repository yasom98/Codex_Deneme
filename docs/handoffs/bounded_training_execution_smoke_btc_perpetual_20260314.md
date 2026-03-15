# Bounded Training Execution Smoke: BTCUSDT Perpetual 2026-03-14

## 1. Purpose

This document records one bounded training execution smoke for the prepared BTCUSDT perpetual run `20260314Tbinance_perp_hist_full_003`.

## 2. Preflight launch-readiness

### Verified fact

The run was verified launch-ready before the smoke was started.

Exact prerequisite artifacts verified:

- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/train_input_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/split_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_datasets/reports/dataset_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_datasets/reports/dataset_build_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_build_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_contract/tmp/bounded_training_preparation_env_config.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`
- `configs/training_config.launch_smoke.example.json`

Verified readiness evidence:

- `state_manifest.output_completeness_ok = true`
- `env_contract_overall = true`
- `readiness_overall = true`
- `episode_catalog_overall = true`
- `training_config.launch_smoke.example.json` contains:
  - `smoke_mode = launch_smoke`
  - `total_timesteps = 32`
  - `smoke_learn_timesteps = 8`
  - `device = cpu`

Exact explicit input paths used:

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

Exact smoke bounds used:

- `smoke_mode = launch_smoke`
- `total_timesteps = 32`
- `smoke_learn_timesteps = 8`
- `device = cpu`

## 3. Launcher invocation

Exact launcher path used:

- `scripts/launch_training.py`

Exact explicit invocation inputs:

- `--run-id 20260314Tbinance_perp_hist_full_003`
- `--env-config runs/20260314Tbinance_perp_hist_full_003/env_contract/tmp/bounded_training_preparation_env_config.json`
- `--training-config configs/training_config.launch_smoke.example.json`
- `--state-manifest runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- `--env-contract-report runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- `--readiness-report runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- `--episode-catalog runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`

Exact explicit output directory used:

- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001`

## 4. Execution result

### Verified fact

The bounded training smoke started, progressed, and completed cleanly enough to validate the execution path.

Repo-local evidence:

- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_launch_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_launch_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_smoke_report.json`

Verified execution facts:

- launcher argument/config validation passed:
  - `training_launch_validation_report.json -> overall_pass = true`
- environment instantiation passed:
  - `training_smoke_report.json -> startup_phase_trace[env_init].status = completed`
- algorithm initialization passed:
  - `training_smoke_report.json -> startup_phase_trace[algo_init].status = completed`
- short learn step actually began:
  - `training_smoke_report.json -> startup_phase_trace[learn_start].status = completed`
  - `smoke_learn_timesteps = 8`
- short learn step completed:
  - `training_smoke_report.json -> startup_phase_trace[learn_finish].status = completed`
  - `num_timesteps = 8`
- smoke result:
  - `training_smoke_report.json -> smoke_success = true`
- launcher exit:
  - `scripts/launch_training.py` exited with code `0`

## 5. Output artifacts produced

- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_launch_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_launch_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_smoke_report.json`

## 6. Training-smoke conclusion

### Verified fact

Execution smoke passed.

## 7. Recommended next narrow task

Open one bounded canonical PPO artifact-production smoke using:

- `scripts/produce_canonical_ppo_artifact.py`
- the same explicit env/state/readiness inputs used here
- `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/split_validation_report.json`
- a fresh explicit output directory

That is the next small, sensible follow-up because the launcher execution path is now verified and the next remaining repo-native step is producing one canonical training artifact.
