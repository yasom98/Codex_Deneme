# Purpose

This file captures one bounded canonical PPO artifact-production smoke for the prepared BTCUSDT perpetual run `20260314Tbinance_perp_hist_full_003`.

# Preflight artifact-readiness

Verified ready before invocation: yes.

Verified prerequisite artifacts:
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
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_launch_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_launch_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_smoke_report.json`

Contract-reference surface used for this path:
- `configs/training_config.artifact_production.example.json`

Truthful correction:
- `configs/training_config.launch_smoke.example.json` was not used for artifact production.
- Repo-local contract evidence showed that `produce_canonical_ppo_artifact.py` requires the strict artifact-production config shape from `src/rl/ppo_artifact_production.py`.

Exact script/config/env/state inputs used:
- script: `scripts/produce_canonical_ppo_artifact.py`
- env config: `runs/20260314Tbinance_perp_hist_full_003/env_contract/tmp/bounded_training_preparation_env_config.json`
- training config: `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/tmp/bounded_artifact_training_config.json`
- state manifest: `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- env contract report: `runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- readiness report: `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- episode catalog: `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`
- split report: `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/split_validation_report.json`

Fields inherited from the artifact-production contract reference:
- `algorithm=ppo`
- `policy=MlpPolicy`
- `seed=42`
- `episode_selection_mode=seeded_random_episode`
- `startup_policy=fresh_only`
- `algo_params.learning_rate=0.0003`
- `algo_params.gamma=0.99`
- `algo_params.gae_lambda=0.95`
- `algo_params.clip_range=0.2`
- `algo_params.ent_coef=0.0`
- `algo_params.vf_coef=0.5`
- `algo_params.max_grad_norm=0.5`

Fields narrowed for smoke purposes:
- `total_timesteps: 100000 -> 16`
- `device: auto -> cpu`
- `algo_params.n_steps: 2048 -> 8`
- `algo_params.batch_size: 64 -> 4`
- `algo_params.n_epochs: 10 -> 2`

Why the bounds remain truthful:
- The bounded config preserved the strict required keys and the same canonical PPO policy/selection/startup semantics.
- Only compute scale and device were reduced to validate the artifact-production path without turning this task into a real training run.

Exact smoke bounds used:
- `total_timesteps=16`
- `device=cpu`
- `startup_policy=fresh_only`
- `episode_selection_mode=seeded_random_episode`

Exact success artifact definition used:
- `canonical_ppo_model.zip` exists at the explicit output path
- `artifact_production_manifest.json` exists
- `artifact_production_report.json` exists
- `artifact_production_report.json` records:
  - `canonical_artifact_ready=true`
  - `save_succeeded=true`
  - `artifact_exists=true`
  - `artifact_zip_valid=true`
  - `load_back_succeeded=true`

# Invocation

Script path used:
- `scripts/produce_canonical_ppo_artifact.py`

Exact explicit input paths used:
- `--run-id 20260314Tbinance_perp_hist_full_003`
- `--env-config runs/20260314Tbinance_perp_hist_full_003/env_contract/tmp/bounded_training_preparation_env_config.json`
- `--training-config runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/tmp/bounded_artifact_training_config.json`
- `--state-manifest runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- `--env-contract-report runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- `--readiness-report runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- `--episode-catalog runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`
- `--split-report runs/20260314Tbinance_perp_hist_full_003/data_features/reports/split_validation_report.json`
- `--output-dir runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001`
- `--log-level INFO`

Exact explicit output directory used:
- `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001`

# Execution result

Artifact-production smoke start: yes.

Artifact-production smoke progress: yes.

Repo-local execution evidence:
- process exit code: `0`
- `artifact_production_report.json` status: `success`
- `artifact_production_report.json` `canonical_artifact_ready=true`
- `startup_phase_trace` phases completed:
  - `validation`
  - `env_init`
  - `algo_init`
  - `learn_start`
  - `learn_finish`
  - `artifact_save`
  - `artifact_load`
  - `report_write`

Expected canonical artifact production: yes.

Produced canonical artifact evidence:
- artifact path exists
- artifact zip opened successfully
- report records `artifact_zip_valid=true`
- report records `load_back_succeeded=true`
- report records `load_back_model_class=PPO`

# Output artifacts produced

- `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/tmp/bounded_artifact_training_config.json`
- `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/canonical_ppo_model.zip`
- `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/artifact_production_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/artifact_production_report.json`

# Canonical-artifact conclusion

Canonical artifact smoke passed.

# Recommended next narrow task

Run one bounded explicit-path evaluation smoke against `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/canonical_ppo_model.zip` using the same BTCUSDT run-local state/env/readiness lineage.
