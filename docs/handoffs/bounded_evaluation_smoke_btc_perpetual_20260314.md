# Purpose

This file captures one bounded explicit-path evaluation smoke for the prepared BTCUSDT perpetual run `20260314Tbinance_perp_hist_full_003` and its canonical PPO artifact.

# Preflight evaluation-readiness

Verified ready before evaluation: yes.

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
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_launch_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_launch_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/training_launch/bounded_execution_smoke_001/training_smoke_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/canonical_ppo_model.zip`
- `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/artifact_production_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/artifact_production_report.json`

Canonical evaluation entrypoint verified:
- `scripts/evaluate_policy.py`

Contract behavior verified from:
- `src/rl/evaluation_backtest.py`

Exact script/config/model/env/state inputs used:
- script: `scripts/evaluate_policy.py`
- model artifact: `runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/canonical_ppo_model.zip`
- env config: `runs/20260314Tbinance_perp_hist_full_003/env_contract/tmp/bounded_training_preparation_env_config.json`
- eval config: `configs/eval_config.episodic.example.json`
- state manifest: `runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- env contract report: `runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- readiness report: `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- episode catalog: `runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`
- split report: `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/split_validation_report.json`

Exact smoke bounds used:
- `evaluation_mode=episodic_eval_backtest`
- `target_mode=explicit_partition`
- `target_partition=validation`
- `max_eval_episodes=3`
- `max_eval_steps=4096`
- `device=cpu`
- `deterministic=true`
- `write_step_trace=false`
- `risk_overlay_enabled=false`

Target partition resolution verified before launch:
- `eval_config.target_partition=validation` is contract-valid.
- The repo applies the single explicit alias `validation -> val`.
- `episode_catalog.json` contains concrete `val` episode refs for BTCUSDT `1m`, `5m`, `15m`.
- `split_validation_report.json` confirms `val_range` exists for all three selected files.

Exact success definition used:
- process exit code `0`
- `evaluation_validation_report.json` exists
- `evaluation_manifest.json` exists
- `evaluation_backtest_report.json` exists
- `evaluation_validation_report.json` shows `overall_pass=true`
- `evaluation_backtest_report.json` shows `evaluation_success=true`
- `startup_phase_trace` completes:
  - `validation`
  - `model_load`
  - `env_init`
  - `eval_start`
  - `eval_finish`
  - `report_write`
- `evaluation_manifest.json` records the canonical PPO artifact path exactly

# Invocation

Script path used:
- `scripts/evaluate_policy.py`

Exact explicit input paths used:
- `--run-id 20260314Tbinance_perp_hist_full_003`
- `--model-artifact runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/canonical_ppo_model.zip`
- `--env-config runs/20260314Tbinance_perp_hist_full_003/env_contract/tmp/bounded_training_preparation_env_config.json`
- `--eval-config configs/eval_config.episodic.example.json`
- `--state-manifest runs/20260314Tbinance_perp_hist_full_003/data_states/reports/state_manifest.json`
- `--env-contract-report runs/20260314Tbinance_perp_hist_full_003/env_contract/reports/env_contract_report.json`
- `--readiness-report runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/training_env_readiness_report.json`
- `--episode-catalog runs/20260314Tbinance_perp_hist_full_003/env_readiness/reports/episode_catalog.json`
- `--split-report runs/20260314Tbinance_perp_hist_full_003/data_features/reports/split_validation_report.json`
- `--output-dir runs/20260314Tbinance_perp_hist_full_003/evaluation_smoke/bounded_eval_smoke_001`
- `--log-level INFO`

Exact explicit output directory used:
- `runs/20260314Tbinance_perp_hist_full_003/evaluation_smoke/bounded_eval_smoke_001`

# Execution result

Evaluation smoke start: yes.

Evaluation smoke progress: yes.

Canonical PPO artifact consumed successfully: yes.

Repo-local execution evidence:
- process exit code: `0`
- `evaluation_validation_report.json` `overall_pass=true`
- `evaluation_backtest_report.json` `evaluation_success=true`
- `evaluation_manifest.json` records:
  - `model_artifact_path=/mnt/c/Users/YASİN/OneDrive/Desktop/Codex_Deneme/runs/20260314Tbinance_perp_hist_full_003/ppo_artifact_smoke/bounded_artifact_smoke_001/canonical_ppo_model.zip`
  - `selected_partition=validation`
  - `selected_episode_refs` resolved to the three `val` BTCUSDT parquet paths
  - `lineages.partition_alias_resolution` records `validation_to_val_v1_compatibility`

Phase-level result:
- argument/config validation passed
- model load passed
- environment initialization passed
- evaluation actually started
- evaluation finished
- repo-local evaluation evidence was written

Selected episode set consumed by evaluation:
- `binance_perpetual__BTC_USDT_15m_price_data.parquet` partition `val`
- `binance_perpetual__BTC_USDT_1m_price_data.parquet` partition `val`
- `binance_perpetual__BTC_USDT_5m_price_data.parquet` partition `val`

# Output artifacts produced

- `runs/20260314Tbinance_perp_hist_full_003/evaluation_smoke/bounded_eval_smoke_001/evaluation_validation_report.json`
- `runs/20260314Tbinance_perp_hist_full_003/evaluation_smoke/bounded_eval_smoke_001/evaluation_manifest.json`
- `runs/20260314Tbinance_perp_hist_full_003/evaluation_smoke/bounded_eval_smoke_001/evaluation_backtest_report.json`

Not produced by design:
- `runs/20260314Tbinance_perp_hist_full_003/evaluation_smoke/bounded_eval_smoke_001/evaluation_step_trace.parquet`
  - `write_step_trace=false`

# Evaluation-smoke conclusion

Evaluation smoke passed.

# Recommended next narrow task

Prepare one small Colab-oriented bounded handoff package that references the proven BTCUSDT run-local inputs, the canonical PPO artifact, and the bounded evaluation evidence for a remote longer-run training/evaluation step.
