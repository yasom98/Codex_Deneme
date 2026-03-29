"""Additive passivity diagnostics for PPO training and evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import io
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch as th

from rl.env_core import (
    ACTION_CLOSE_POSITION,
    ACTION_HOLD,
    ACTION_MAPPING,
    ACTION_OPEN_LONG,
    ACTION_OPEN_SHORT,
    ExecutionEngine,
)

PASSIVITY_DIAGNOSTICS_REPORT_FILENAME = "passivity_diagnostics_report.json"
TRAINING_PASSIVITY_DIAGNOSTICS_REPORT_FILENAME = "training_passivity_diagnostics_report.json"
TRAINING_ENTROPY_TIMESERIES_FILENAME = "training_entropy_timeseries.csv"
DETERMINISTIC_ACTION_RANKING_TRACE_FILENAME = "deterministic_action_ranking_trace.csv"
COMPACT_STEP_DIAGNOSTICS_REPORT_FILENAME = "compact_step_diagnostics_report.json"
TRAINING_PASSIVITY_DIAGNOSTICS_CONTRACT_VERSION = "training_passivity_diagnostics.v11"
EVALUATION_PASSIVITY_DIAGNOSTICS_CONTRACT_VERSION = "evaluation_passivity_diagnostics.v3"
DETERMINISTIC_HOLD_GAP_THRESHOLD = 0.10
TRAINING_AGE1_REPRESENTATIVE_STATE_LIMIT = 8
MINIBATCH_LIFECYCLE_PHASE_ORDER = ("flat", "age1_in_position", "age2plus_long", "age2plus_short")
SB3_TRAIN_LOGGER_FIELDS = (
    ("train_approx_kl", "train/approx_kl"),
    ("train_clip_fraction", "train/clip_fraction"),
    ("train_explained_variance", "train/explained_variance"),
    ("train_value_loss", "train/value_loss"),
    ("train_entropy_loss", "train/entropy_loss"),
    ("train_policy_gradient_loss", "train/policy_gradient_loss"),
    ("train_clip_range", "train/clip_range"),
    ("train_clip_range_vf", "train/clip_range_vf"),
)

CANONICAL_ACTION_ORDER = (
    ACTION_HOLD,
    ACTION_OPEN_LONG,
    ACTION_OPEN_SHORT,
    ACTION_CLOSE_POSITION,
)
CANONICAL_ACTION_SEMANTICS = tuple(ACTION_MAPPING[action_id] for action_id in CANONICAL_ACTION_ORDER)
CANONICAL_POSITION_VALUES = (-1, 0, 1)
CANONICAL_POSITION_KEYS = tuple(str(value) for value in CANONICAL_POSITION_VALUES)
CANONICAL_TRANSITION_KEYS = tuple(
    f"{before}->{after}" for before in CANONICAL_POSITION_VALUES for after in CANONICAL_POSITION_VALUES
)
POSITION_REGIME_BY_VALUE = {-1: "short", 0: "flat", 1: "long"}
POSITION_REGIME_ORDER = ("flat", "long", "short")


@dataclass(frozen=True)
class TrainingPassivityDiagnosticsArtifacts:
    """Serialized training diagnostics artifacts."""

    report_payload: dict[str, Any]
    entropy_timeseries_csv: str


def build_training_passivity_diagnostics_callback(
    *,
    age1_close_position_positive_advantage_clip: float | None = None,
    age1_close_position_pairwise_hold_mean_advantage_subtract: bool = False,
    representative_flat_state_probability_audit: bool = False,
    age2plus_long_state_audit: bool = False,
    ppo_minibatch_lifecycle_phase_trace: bool = False,
    phase_stratified_ppo_minibatches: bool = False,
    churn_family_optimizer_path_isolation: bool = False,
    flat_entry_pressure_optimizer_path_isolation: bool = False,
    age1_vs_age2plus_in_position_optimizer_path_isolation: bool = False,
    age2plus_long_vs_age2plus_short_optimizer_path_isolation: bool = False,
) -> Any:
    """Build an additive SB3 callback that records passivity telemetry."""

    from stable_baselines3.common.callbacks import BaseCallback

    class _TrainingPassivityDiagnosticsCallback(BaseCallback):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self._action_counts: Counter[str] = Counter()
            self._action_reward_sums = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
            self._action_reward_counts = {semantic: 0 for semantic in CANONICAL_ACTION_SEMANTICS}
            self._position_transition_counts: Counter[str] = Counter()
            self._rollout_rows: list[dict[str, Any]] = []
            self._last_action_counts = {semantic: 0 for semantic in CANONICAL_ACTION_SEMANTICS}
            self._last_action_reward_sums = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
            self._last_action_reward_counts = {semantic: 0 for semantic in CANONICAL_ACTION_SEMANTICS}
            self._current_rollout_age1_records: list[dict[str, Any]] = []
            self._current_rollout_age2plus_long_records: list[dict[str, Any]] = []
            self._current_rollout_minibatch_phase_records: list[dict[str, Any]] = []
            self._representative_age1_state_bank: list[dict[str, Any]] = []
            self._representative_flat_state_bank: list[dict[str, Any]] = []
            self._representative_long_state_bank: list[dict[str, Any]] = []
            self._current_rollout_pre_bank_probabilities: dict[str, float] | None = None
            self._current_rollout_pre_flat_bank_probabilities: dict[str, float] | None = None
            self._current_rollout_pre_long_bank_probabilities: dict[str, Any] | None = None
            self._pending_post_update_row_index: int | None = None
            self._current_rollout_step_index = 0
            self._previous_step_open_flags: list[bool] = []
            self._position_age_before_step: list[int] = []
            self._representative_long_state_update_trace: list[dict[str, Any]] = []
            self._ppo_minibatch_trace_rows: list[dict[str, Any]] = []
            self._age1_close_position_positive_advantage_clip = (
                float(age1_close_position_positive_advantage_clip)
                if age1_close_position_positive_advantage_clip is not None
                else None
            )
            self._age1_close_position_pairwise_hold_mean_advantage_subtract = bool(
                age1_close_position_pairwise_hold_mean_advantage_subtract
            )
            self._representative_flat_state_probability_audit = bool(representative_flat_state_probability_audit)
            self._age2plus_long_state_audit = bool(age2plus_long_state_audit)
            self._ppo_minibatch_lifecycle_phase_trace = bool(ppo_minibatch_lifecycle_phase_trace)
            self._phase_stratified_ppo_minibatches = bool(phase_stratified_ppo_minibatches)
            self._churn_family_optimizer_path_isolation = bool(churn_family_optimizer_path_isolation)
            self._flat_entry_pressure_optimizer_path_isolation = bool(flat_entry_pressure_optimizer_path_isolation)
            self._age1_vs_age2plus_in_position_optimizer_path_isolation = bool(
                age1_vs_age2plus_in_position_optimizer_path_isolation
            )
            self._age2plus_long_vs_age2plus_short_optimizer_path_isolation = bool(
                age2plus_long_vs_age2plus_short_optimizer_path_isolation
            )

        def _on_training_start(self) -> None:
            n_envs = int(getattr(self.training_env, "num_envs", 1))
            self._previous_step_open_flags = [False] * n_envs
            self._position_age_before_step = [0] * n_envs

        def _on_rollout_start(self) -> None:
            self._current_rollout_age1_records = []
            self._current_rollout_age2plus_long_records = []
            self._current_rollout_minibatch_phase_records = []
            self._current_rollout_step_index = 0
            if self._pending_post_update_row_index is not None:
                if self._representative_age1_state_bank:
                    representative_probs = _evaluate_representative_age1_state_bank(
                        model=self.model,
                        representative_states=self._representative_age1_state_bank,
                    )
                    _attach_age1_probability_drift(
                        row=self._rollout_rows[self._pending_post_update_row_index],
                        representative_probabilities=representative_probs,
                    )
                if self._representative_flat_state_probability_audit and self._representative_flat_state_bank:
                    representative_flat_probs = _evaluate_representative_flat_state_bank(
                        model=self.model,
                        representative_states=self._representative_flat_state_bank,
                    )
                    _attach_flat_probability_drift(
                        row=self._rollout_rows[self._pending_post_update_row_index],
                        representative_probabilities=representative_flat_probs,
                    )
                if self._age2plus_long_state_audit and self._representative_long_state_bank:
                    representative_long_probs = _evaluate_representative_long_state_bank(
                        model=self.model,
                        representative_states=self._representative_long_state_bank,
                    )
                    _attach_long_probability_drift(
                        row=self._rollout_rows[self._pending_post_update_row_index],
                        representative_probabilities=representative_long_probs,
                    )
                    _append_representative_long_state_update_trace(
                        trace_rows=self._representative_long_state_update_trace,
                        update_index=int(self._rollout_rows[self._pending_post_update_row_index]["update_index"]),
                        representative_pre_update=(
                            self._rollout_rows[self._pending_post_update_row_index].get(
                                "_representative_long_state_details_pre_update"
                            )
                        ),
                        representative_post_update=(
                            representative_long_probs.get("records")
                            if isinstance(representative_long_probs, Mapping)
                            else None
                        ),
                    )
                _attach_sb3_train_logger_metrics(
                    row=self._rollout_rows[self._pending_post_update_row_index],
                    model=self.model,
                )
                self._pending_post_update_row_index = None
            if self._representative_age1_state_bank:
                representative_probs = _evaluate_representative_age1_state_bank(
                    model=self.model,
                    representative_states=self._representative_age1_state_bank,
                )
                self._current_rollout_pre_bank_probabilities = representative_probs
            else:
                self._current_rollout_pre_bank_probabilities = None
            if self._representative_flat_state_probability_audit and self._representative_flat_state_bank:
                self._current_rollout_pre_flat_bank_probabilities = _evaluate_representative_flat_state_bank(
                    model=self.model,
                    representative_states=self._representative_flat_state_bank,
                )
            else:
                self._current_rollout_pre_flat_bank_probabilities = None
            if self._age2plus_long_state_audit and self._representative_long_state_bank:
                self._current_rollout_pre_long_bank_probabilities = _evaluate_representative_long_state_bank(
                    model=self.model,
                    representative_states=self._representative_long_state_bank,
                )
            else:
                self._current_rollout_pre_long_bank_probabilities = None

        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            rewards = self.locals.get("rewards")
            if infos is None or rewards is None:
                raise RuntimeError("passivity diagnostics require infos and rewards in callback locals")
            obs_tensor = self.locals.get("obs_tensor")
            action_masks = self.locals.get("action_masks")
            actions = self.locals.get("actions")
            values = self.locals.get("values")
            log_probs = self.locals.get("log_probs")
            dones = self.locals.get("dones")
            if any(value is None for value in (obs_tensor, action_masks, actions, values, log_probs, dones)):
                raise RuntimeError("passivity diagnostics require rollout tensors and masks in callback locals")

            reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
            if not np.isfinite(reward_values).all():
                raise RuntimeError("passivity diagnostics observed non-finite rewards during training")

            info_items = list(infos)
            if len(info_items) != int(reward_values.shape[0]):
                raise RuntimeError("passivity diagnostics reward/info cardinality mismatch during training")
            action_mask_array = np.asarray(action_masks, dtype=np.bool_)
            if action_mask_array.ndim == 1:
                action_mask_array = action_mask_array.reshape(1, -1)
            action_array = np.asarray(actions).reshape(-1)
            done_array = np.asarray(dones, dtype=np.bool_).reshape(-1)
            value_array = np.asarray(values.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            log_prob_array = np.asarray(log_probs.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            observation_array = np.asarray(obs_tensor.detach().cpu().numpy(), dtype=np.float32)
            if observation_array.ndim == 1:
                observation_array = observation_array.reshape(1, -1)
            with th.no_grad():
                distribution = self.model.policy.get_distribution(obs_tensor, action_masks=action_mask_array)
            probability_array = getattr(getattr(distribution, "distribution", distribution), "probs", None)
            if probability_array is None:
                raise RuntimeError("passivity diagnostics require action probabilities during training")
            probability_values = np.asarray(probability_array.detach().cpu().numpy(), dtype=np.float64)
            if probability_values.ndim == 1:
                probability_values = probability_values.reshape(1, -1)
            next_open_flags = list(self._previous_step_open_flags)
            next_position_ages = list(self._position_age_before_step)

            for env_index, (info, reward_value) in enumerate(zip(info_items, reward_values, strict=True)):
                if not isinstance(info, Mapping):
                    raise RuntimeError("passivity diagnostics require mapping-like env infos during training")
                semantic = str(info.get("action_semantic", ""))
                if semantic not in CANONICAL_ACTION_SEMANTICS:
                    raise RuntimeError(f"unsupported training action semantic for diagnostics: {semantic}")
                position_before = _coerce_position(info.get("position_before"), field_name="position_before")
                position_after = _coerce_position(info.get("position_after"), field_name="position_after")
                transition_key = f"{position_before}->{position_after}"
                if transition_key not in CANONICAL_TRANSITION_KEYS:
                    raise RuntimeError(f"unsupported training position transition for diagnostics: {transition_key}")
                position_age_before = (
                    int(self._position_age_before_step[env_index]) if env_index < len(self._position_age_before_step) else 0
                )

                self._action_counts[semantic] += 1
                self._action_reward_sums[semantic] += float(reward_value)
                self._action_reward_counts[semantic] += 1
                self._position_transition_counts[transition_key] += 1
                if (
                    env_index < len(self._previous_step_open_flags)
                    and self._previous_step_open_flags[env_index]
                    and position_before in {-1, 1}
                ):
                    record = {
                        "rollout_step_index": int(self._current_rollout_step_index),
                        "env_index": int(env_index),
                        "timestamp": str(info.get("timestamp")),
                        "position_before": int(position_before),
                        "chosen_action_semantic": semantic,
                        "reward_total": float(reward_value),
                        "value_pre": float(value_array[env_index]),
                        "log_prob_pre": float(log_prob_array[env_index]),
                        "prob_hold_pre": float(probability_values[env_index, ACTION_HOLD]),
                        "prob_close_pre": float(probability_values[env_index, ACTION_CLOSE_POSITION]),
                    }
                    self._current_rollout_age1_records.append(record)
                    if len(self._representative_age1_state_bank) < TRAINING_AGE1_REPRESENTATIVE_STATE_LIMIT:
                        self._representative_age1_state_bank.append(
                            {
                                "timestamp": str(info.get("timestamp")),
                                "position_before": int(position_before),
                                "observation": observation_array[env_index].astype(np.float32, copy=True),
                                "action_mask": action_mask_array[env_index].astype(np.bool_, copy=True),
                            }
                        )
                if self._age2plus_long_state_audit and position_before == 1 and position_age_before >= 2:
                    age2plus_record = {
                        "rollout_step_index": int(self._current_rollout_step_index),
                        "env_index": int(env_index),
                        "timestamp": str(info.get("timestamp")),
                        "position_before": int(position_before),
                        "position_age_before_step": int(position_age_before),
                        "chosen_action_semantic": semantic,
                        "reward_total": float(reward_value),
                        "value_pre": float(value_array[env_index]),
                        "log_prob_pre": float(log_prob_array[env_index]),
                        "prob_hold_pre": float(probability_values[env_index, ACTION_HOLD]),
                        "prob_close_pre": float(probability_values[env_index, ACTION_CLOSE_POSITION]),
                    }
                    self._current_rollout_age2plus_long_records.append(age2plus_record)
                    if len(self._representative_long_state_bank) < TRAINING_AGE1_REPRESENTATIVE_STATE_LIMIT:
                        self._representative_long_state_bank.append(
                            {
                                "timestamp": str(info.get("timestamp")),
                                "position_before": int(position_before),
                                "position_age_before_step": int(position_age_before),
                                "observation": observation_array[env_index].astype(np.float32, copy=True),
                                "action_mask": action_mask_array[env_index].astype(np.bool_, copy=True),
                            }
                        )
                if (
                    self._representative_flat_state_probability_audit
                    and position_before == 0
                    and len(self._representative_flat_state_bank) < TRAINING_AGE1_REPRESENTATIVE_STATE_LIMIT
                ):
                    self._representative_flat_state_bank.append(
                        {
                            "timestamp": str(info.get("timestamp")),
                            "position_before": int(position_before),
                            "observation": observation_array[env_index].astype(np.float32, copy=True),
                            "action_mask": action_mask_array[env_index].astype(np.bool_, copy=True),
                        }
                    )
                if (
                    self._ppo_minibatch_lifecycle_phase_trace
                    or self._phase_stratified_ppo_minibatches
                    or self._churn_family_optimizer_path_isolation
                    or self._flat_entry_pressure_optimizer_path_isolation
                    or self._age1_vs_age2plus_in_position_optimizer_path_isolation
                    or self._age2plus_long_vs_age2plus_short_optimizer_path_isolation
                ):
                    self._current_rollout_minibatch_phase_records.append(
                        {
                            "flat_index": int(self._current_rollout_step_index * len(info_items) + env_index),
                            "rollout_step_index": int(self._current_rollout_step_index),
                            "env_index": int(env_index),
                            "timestamp": str(info.get("timestamp")),
                            "position_before": int(position_before),
                            "position_age_before_step": int(position_age_before),
                            "chosen_action_semantic": semantic,
                            "lifecycle_phase": _classify_minibatch_lifecycle_phase(
                                position_before=position_before,
                                position_age_before_step=position_age_before,
                            ),
                        }
                    )
                next_open_flags[env_index] = bool(semantic in {"OPEN_LONG", "OPEN_SHORT"} and not bool(done_array[env_index]))
                next_position_ages[env_index] = _next_position_age(
                    position_before=position_before,
                    position_after=position_after,
                    age_before=position_age_before,
                    done=bool(done_array[env_index]),
                )
            self._previous_step_open_flags = next_open_flags
            self._position_age_before_step = next_position_ages
            self._current_rollout_step_index += 1
            return True

        def _on_rollout_end(self) -> None:
            entropy_mean, mean_action_probabilities = _compute_policy_statistics(self.model)
            row = {
                "update_index": int(len(self._rollout_rows) + 1),
                "num_timesteps": int(getattr(self.model, "num_timesteps", 0)),
                "mean_policy_entropy": float(entropy_mean),
            }
            for semantic in CANONICAL_ACTION_SEMANTICS:
                normalized = _action_field_name(semantic)
                current_count = int(self._action_counts[semantic])
                current_reward_sum = float(self._action_reward_sums[semantic])
                current_reward_count = int(self._action_reward_counts[semantic])
                row[f"selected_{normalized}_count"] = current_count - int(self._last_action_counts[semantic])
                row[f"reward_sum_{normalized}"] = current_reward_sum - float(self._last_action_reward_sums[semantic])
                row[f"reward_count_{normalized}"] = current_reward_count - int(self._last_action_reward_counts[semantic])
                self._last_action_counts[semantic] = current_count
                self._last_action_reward_sums[semantic] = current_reward_sum
                self._last_action_reward_counts[semantic] = current_reward_count

            if mean_action_probabilities is None:
                for semantic in CANONICAL_ACTION_SEMANTICS:
                    row[f"mean_{_action_field_name(semantic)}_probability"] = None
            else:
                for semantic in CANONICAL_ACTION_SEMANTICS:
                    row[f"mean_{_action_field_name(semantic)}_probability"] = float(mean_action_probabilities[semantic])
            age1_records = _annotate_age1_advantages(
                records=self._current_rollout_age1_records,
                rollout_buffer=getattr(self.model, "rollout_buffer", None),
            )
            age2plus_long_records = _annotate_rollout_records(
                records=self._current_rollout_age2plus_long_records,
                rollout_buffer=getattr(self.model, "rollout_buffer", None),
            )
            row.update(_build_age1_action_update_summary(age1_records))
            row.update(_build_age2plus_long_action_update_summary(age2plus_long_records))
            _attach_sb3_train_logger_metrics(row=row, model=None)
            row.update(
                _apply_age1_close_position_pairwise_hold_mean_advantage_subtract(
                    records=age1_records,
                    rollout_buffer=getattr(self.model, "rollout_buffer", None),
                    enabled=self._age1_close_position_pairwise_hold_mean_advantage_subtract,
                )
            )
            row.update(
                _apply_age1_close_position_positive_advantage_clip(
                    records=age1_records,
                    rollout_buffer=getattr(self.model, "rollout_buffer", None),
                    positive_advantage_clip=self._age1_close_position_positive_advantage_clip,
                )
            )
            row["representative_flat_state_count"] = int(len(self._representative_flat_state_bank))
            row["representative_age1_state_count"] = int(len(self._representative_age1_state_bank))
            row["representative_long_state_count"] = int(len(self._representative_long_state_bank))
            representative_pre_probs = self._current_rollout_pre_bank_probabilities
            if representative_pre_probs is None and self._representative_age1_state_bank:
                representative_pre_probs = _evaluate_representative_age1_state_bank(
                    model=self.model,
                    representative_states=self._representative_age1_state_bank,
                )
            _attach_age1_probability_pre_update(
                row=row,
                representative_probabilities=representative_pre_probs,
            )
            _attach_flat_probability_pre_update(
                row=row,
                representative_probabilities=self._current_rollout_pre_flat_bank_probabilities,
            )
            _attach_long_probability_pre_update(
                row=row,
                representative_probabilities=self._current_rollout_pre_long_bank_probabilities,
            )
            row["_representative_long_state_details_pre_update"] = (
                [dict(item) for item in self._current_rollout_pre_long_bank_probabilities.get("records", ())]
                if isinstance(self._current_rollout_pre_long_bank_probabilities, Mapping)
                else []
            )
            _attach_age1_probability_drift(row=row, representative_probabilities=None)
            _attach_flat_probability_drift(row=row, representative_probabilities=None)
            _attach_long_probability_drift(row=row, representative_probabilities=None)
            self._rollout_rows.append(row)
            self._pending_post_update_row_index = int(len(self._rollout_rows) - 1)

        def build_artifacts(
            self,
            *,
            run_id: str,
            production_session_id: str,
            selected_episode_ref: Mapping[str, Any] | None,
            action_masking_enabled: bool,
            total_timesteps_requested: int,
            num_timesteps_after_learn: int,
        ) -> TrainingPassivityDiagnosticsArtifacts:
            if not self._rollout_rows:
                raise RuntimeError("passivity diagnostics did not record any rollout telemetry during training")
            if self._pending_post_update_row_index is not None and (
                self._representative_age1_state_bank
                or self._representative_flat_state_bank
                or self._representative_long_state_bank
            ):
                representative_post_probs = (
                    _evaluate_representative_age1_state_bank(
                        model=self.model,
                        representative_states=self._representative_age1_state_bank,
                    )
                    if self._representative_age1_state_bank
                    else None
                )
                _attach_age1_probability_drift(
                    row=self._rollout_rows[self._pending_post_update_row_index],
                    representative_probabilities=representative_post_probs,
                )
                representative_flat_post_probs = (
                    _evaluate_representative_flat_state_bank(
                        model=self.model,
                        representative_states=self._representative_flat_state_bank,
                    )
                    if self._representative_flat_state_probability_audit and self._representative_flat_state_bank
                    else None
                )
                _attach_flat_probability_drift(
                    row=self._rollout_rows[self._pending_post_update_row_index],
                    representative_probabilities=representative_flat_post_probs,
                )
                representative_long_post_probs = (
                    _evaluate_representative_long_state_bank(
                        model=self.model,
                        representative_states=self._representative_long_state_bank,
                    )
                    if self._age2plus_long_state_audit and self._representative_long_state_bank
                    else None
                )
                _attach_long_probability_drift(
                    row=self._rollout_rows[self._pending_post_update_row_index],
                    representative_probabilities=representative_long_post_probs,
                )
                _append_representative_long_state_update_trace(
                    trace_rows=self._representative_long_state_update_trace,
                    update_index=int(self._rollout_rows[self._pending_post_update_row_index]["update_index"]),
                    representative_pre_update=(
                        self._rollout_rows[self._pending_post_update_row_index].get(
                            "_representative_long_state_details_pre_update"
                        )
                    ),
                    representative_post_update=(
                        representative_long_post_probs.get("records")
                        if isinstance(representative_long_post_probs, Mapping)
                        else None
                    ),
                )
                _attach_sb3_train_logger_metrics(
                    row=self._rollout_rows[self._pending_post_update_row_index],
                    model=self.model,
                )
                self._pending_post_update_row_index = None

            report_payload = {
                "contract_version": TRAINING_PASSIVITY_DIAGNOSTICS_CONTRACT_VERSION,
                "run_id": run_id,
                "production_session_id": production_session_id,
                "action_masking_enabled": bool(action_masking_enabled),
                "selected_episode_ref": dict(selected_episode_ref) if selected_episode_ref is not None else None,
                "total_timesteps_requested": int(total_timesteps_requested),
                "num_timesteps_after_learn": int(num_timesteps_after_learn),
                "age1_close_position_positive_advantage_clip": _coerce_optional_float(
                    self._age1_close_position_positive_advantage_clip
                ),
                "age1_close_position_pairwise_hold_mean_advantage_subtract": bool(
                    self._age1_close_position_pairwise_hold_mean_advantage_subtract
                ),
                "representative_flat_state_probability_audit": bool(
                    self._representative_flat_state_probability_audit
                ),
                "age2plus_long_state_audit": bool(self._age2plus_long_state_audit),
                "ppo_minibatch_lifecycle_phase_trace": bool(self._ppo_minibatch_lifecycle_phase_trace),
                "phase_stratified_ppo_minibatches": bool(self._phase_stratified_ppo_minibatches),
                "churn_family_optimizer_path_isolation": bool(self._churn_family_optimizer_path_isolation),
                "flat_entry_pressure_optimizer_path_isolation": bool(
                    self._flat_entry_pressure_optimizer_path_isolation
                ),
                "age1_vs_age2plus_in_position_optimizer_path_isolation": bool(
                    self._age1_vs_age2plus_in_position_optimizer_path_isolation
                ),
                "age2plus_long_vs_age2plus_short_optimizer_path_isolation": bool(
                    self._age2plus_long_vs_age2plus_short_optimizer_path_isolation
                ),
                "training_rollout_point_count": int(len(self._rollout_rows)),
                "training_action_semantic_counts": _ordered_action_counts(self._action_counts),
                "training_action_reward_summary": _build_action_reward_summary(
                    reward_sums=self._action_reward_sums,
                    reward_counts=self._action_reward_counts,
                ),
                "training_position_transition_counts": _ordered_transition_counts(self._position_transition_counts),
                "training_hold_dominance_summary": _build_hold_dominance_summary(
                    action_counts=_ordered_action_counts(self._action_counts)
                ),
                "age1_in_position_representative_states": [
                    {
                        "representative_index": int(index + 1),
                        "timestamp": str(item["timestamp"]),
                        "position_before": int(item["position_before"]),
                    }
                    for index, item in enumerate(self._representative_age1_state_bank)
                ],
                "flat_state_representative_states": [
                    {
                        "representative_index": int(index + 1),
                        "timestamp": str(item["timestamp"]),
                        "position_before": int(item["position_before"]),
                    }
                    for index, item in enumerate(self._representative_flat_state_bank)
                ],
                "age2plus_long_representative_states": [
                    {
                        "representative_index": int(index + 1),
                        "timestamp": str(item["timestamp"]),
                        "position_before": int(item["position_before"]),
                        "position_age_before_step": int(item["position_age_before_step"]),
                    }
                    for index, item in enumerate(self._representative_long_state_bank)
                ],
                "age2plus_long_representative_state_update_trace": [
                    dict(item) for item in self._representative_long_state_update_trace
                ],
                "ppo_minibatch_lifecycle_phase_trace_rows": [
                    dict(item) for item in self._ppo_minibatch_trace_rows
                ],
                "age1_in_position_update_summary": _build_age1_update_summary_rows(self._rollout_rows),
                "age2plus_long_update_summary": _build_age2plus_long_update_summary_rows(self._rollout_rows),
                "ppo_minibatch_lifecycle_phase_update_summary": _build_ppo_minibatch_update_summary_rows(
                    minibatch_rows=self._ppo_minibatch_trace_rows,
                    rollout_rows=self._rollout_rows,
                ),
                "ppo_minibatch_lifecycle_phase_trace_summary": _build_ppo_minibatch_trace_summary(
                    minibatch_rows=self._ppo_minibatch_trace_rows,
                    rollout_rows=self._rollout_rows,
                ),
                "policy_entropy_summary": _build_series_summary(self._rollout_rows, key="mean_policy_entropy"),
                "mean_action_probability_summary": _build_probability_summary(self._rollout_rows),
                "generated_at_utc": _generated_at(),
            }
            return TrainingPassivityDiagnosticsArtifacts(
                report_payload=report_payload,
                entropy_timeseries_csv=_build_training_entropy_timeseries_csv(self._rollout_rows),
            )

        def get_pending_update_index_for_minibatch_trace(self) -> int | None:
            """Return the rollout/update index currently awaiting PPO train() completion."""

            if not self._ppo_minibatch_lifecycle_phase_trace or self._pending_post_update_row_index is None:
                return None
            raw_update_index = self._rollout_rows[self._pending_post_update_row_index].get("update_index")
            return _coerce_optional_int(raw_update_index)

        def get_current_rollout_sample_metadata_for_minibatch_trace(self) -> list[dict[str, Any]]:
            """Return the current rollout sample metadata in flattened rollout-buffer order."""

            if not (
                self._ppo_minibatch_lifecycle_phase_trace
                or self._phase_stratified_ppo_minibatches
                or self._churn_family_optimizer_path_isolation
                or self._flat_entry_pressure_optimizer_path_isolation
                or self._age1_vs_age2plus_in_position_optimizer_path_isolation
                or self._age2plus_long_vs_age2plus_short_optimizer_path_isolation
            ):
                return []
            return [dict(item) for item in self._current_rollout_minibatch_phase_records]

        def record_ppo_minibatch_lifecycle_phase_trace(
            self,
            *,
            update_index: int | None,
            epoch_index: int,
            minibatch_index: int,
            batch_indices: Sequence[int],
            raw_advantages: Sequence[float],
            normalized_advantages: Sequence[float],
        ) -> None:
            """Append one audit-only PPO minibatch composition row."""

            if not self._ppo_minibatch_lifecycle_phase_trace or update_index is None:
                return
            row = _build_ppo_minibatch_composition_row(
                update_index=int(update_index),
                epoch_index=int(epoch_index),
                minibatch_index=int(minibatch_index),
                batch_indices=batch_indices,
                sample_metadata=self._current_rollout_minibatch_phase_records,
                raw_advantages=raw_advantages,
                normalized_advantages=normalized_advantages,
            )
            if row is not None:
                self._ppo_minibatch_trace_rows.append(row)

    return _TrainingPassivityDiagnosticsCallback()


def summarize_eval_policy_behavior(
    *,
    episode_runtimes: Sequence[Any],
    strategy_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize policy behavior across one deterministic or stochastic evaluation set."""

    action_counts: Counter[str] = Counter()
    position_after_counts: Counter[str] = Counter()
    position_transition_counts: Counter[str] = Counter()
    invalid_action_count = 0
    step_count = 0

    for runtime in episode_runtimes:
        for record in runtime.step_records:
            semantic = str(record["action_semantic"])
            if semantic in CANONICAL_ACTION_SEMANTICS:
                action_counts[semantic] += 1
            position_key = str(int(record["position_after"]))
            if position_key in CANONICAL_POSITION_KEYS:
                position_after_counts[position_key] += 1
            # Track position transitions: enables distinguishing HOLD-only vs open-but-no-close vs exposure changes
            position_before_val = int(record["position_before"])
            position_after_val = int(record["position_after"])
            transition_key = f"{position_before_val}->{position_after_val}"
            if transition_key in CANONICAL_TRANSITION_KEYS:
                position_transition_counts[transition_key] += 1
            invalid_action_count += int(bool(record["invalid_action"]))
            step_count += 1

    ordered_action_counts = _ordered_action_counts(action_counts)
    ordered_position_counts = _ordered_position_counts(position_after_counts)
    ordered_transition_counts = _ordered_transition_counts(position_transition_counts)
    hold_dominance = _build_hold_dominance_summary(action_counts=ordered_action_counts)
    metrics = dict(strategy_metrics or {})
    return {
        "episode_count": int(len(episode_runtimes)),
        "step_count": int(step_count),
        "action_semantic_counts": ordered_action_counts,
        "position_after_counts": ordered_position_counts,
        "position_transition_counts": ordered_transition_counts,
        "action_conditioned_reward_summary": build_action_conditioned_reward_summary(episode_runtimes),
        "num_trades": _coerce_optional_int(metrics.get("num_trades")),
        "final_equity": _coerce_optional_float(metrics.get("final_equity")),
        "total_return": _coerce_optional_float(metrics.get("total_return")),
        "avg_trade_return": _coerce_optional_float(metrics.get("avg_trade_return")),
        "invalid_action_ratio": (float(invalid_action_count) / float(step_count)) if step_count > 0 else 0.0,
        "hold_dominance_summary": hold_dominance,
    }


def build_eval_passivity_diagnostics_report(
    *,
    run_id: str,
    evaluation_session_id: str,
    action_masking_enabled: bool,
    evaluation_policy_mode: str | None,
    deterministic_summary: Mapping[str, Any],
    stochastic_summary: Mapping[str, Any],
    deterministic_action_ranking_summary: Mapping[str, Any] | None = None,
    deterministic_position_conditional_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable deterministic vs stochastic comparison report."""

    deterministic_payload = dict(deterministic_summary)
    if deterministic_action_ranking_summary is not None:
        deterministic_payload["action_ranking_summary"] = dict(deterministic_action_ranking_summary)
    if deterministic_position_conditional_summary is not None:
        deterministic_payload["position_conditional_action_ranking_summary"] = dict(
            deterministic_position_conditional_summary
        )
    deterministic_action_counts = deterministic_summary["action_semantic_counts"]
    stochastic_action_counts = stochastic_summary["action_semantic_counts"]
    deterministic_position_counts = deterministic_summary["position_after_counts"]
    stochastic_position_counts = stochastic_summary["position_after_counts"]
    deterministic_hold_share = float(deterministic_summary["hold_dominance_summary"]["hold_share"])
    stochastic_hold_share = float(stochastic_summary["hold_dominance_summary"]["hold_share"])
    action_count_delta = {
        semantic: int(stochastic_action_counts[semantic]) - int(deterministic_action_counts[semantic])
        for semantic in CANONICAL_ACTION_SEMANTICS
    }
    position_after_count_delta = {
        key: int(stochastic_position_counts[key]) - int(deterministic_position_counts[key]) for key in CANONICAL_POSITION_KEYS
    }
    # Transition delta: enables detecting open-but-no-close vs pure HOLD vs exposure changes
    deterministic_transition_counts = deterministic_summary.get("position_transition_counts", {})
    stochastic_transition_counts = stochastic_summary.get("position_transition_counts", {})
    position_transition_count_delta = {
        key: int(stochastic_transition_counts.get(key, 0)) - int(deterministic_transition_counts.get(key, 0))
        for key in CANONICAL_TRANSITION_KEYS
    }
    return {
        "contract_version": EVALUATION_PASSIVITY_DIAGNOSTICS_CONTRACT_VERSION,
        "run_id": run_id,
        "evaluation_session_id": evaluation_session_id,
        "action_masking_enabled": bool(action_masking_enabled),
        "evaluation_policy_mode": evaluation_policy_mode,
        "deterministic_eval": deterministic_payload,
        "stochastic_eval": dict(stochastic_summary),
        "deterministic_vs_stochastic": {
            "action_semantic_count_delta": action_count_delta,
            "position_after_count_delta": position_after_count_delta,
            "position_transition_count_delta": position_transition_count_delta,
            "num_trades_delta": _subtract_optional_numbers(
                stochastic_summary.get("num_trades"),
                deterministic_summary.get("num_trades"),
            ),
            "invalid_action_ratio_delta": float(stochastic_summary["invalid_action_ratio"])
            - float(deterministic_summary["invalid_action_ratio"]),
            "hold_share_delta": stochastic_hold_share - deterministic_hold_share,
            "stochastic_more_active_than_deterministic": bool(
                _coerce_optional_int(stochastic_summary.get("num_trades"), default=0)
                > _coerce_optional_int(deterministic_summary.get("num_trades"), default=0)
            ),
            "deterministic_hold_extreme": bool(deterministic_hold_share >= 0.95),
            "deterministic_vs_stochastic_action_discrepancy": bool(
                action_count_delta != {semantic: 0 for semantic in CANONICAL_ACTION_SEMANTICS}
            ),
        },
        "generated_at_utc": _generated_at(),
    }


def build_deterministic_action_ranking_row(
    *,
    episode_index: int,
    episode_ref: Mapping[str, Any],
    step_ordinal: int,
    step_index: int,
    timestamp: str,
    position_before: int,
    selected_action_semantic: str,
    action_probabilities: Mapping[str, float],
    valid_action_mask: Sequence[bool] | np.ndarray | None = None,
    gap_threshold: float = DETERMINISTIC_HOLD_GAP_THRESHOLD,
) -> dict[str, Any]:
    """Build one deterministic eval ranking row from canonical action probabilities."""

    if selected_action_semantic not in CANONICAL_ACTION_SEMANTICS:
        raise RuntimeError(f"unsupported selected_action_semantic for ranking diagnostics: {selected_action_semantic}")
    if not isinstance(timestamp, str) or not timestamp:
        raise RuntimeError("ranking diagnostics require non-empty timestamp strings")
    position_before_value = _coerce_position(position_before, field_name="position_before")
    ranking_snapshot = build_action_ranking_snapshot(
        position_before=position_before_value,
        action_probabilities=action_probabilities,
        valid_action_mask=valid_action_mask,
        gap_threshold=gap_threshold,
    )
    top_ranked_semantics = ranking_snapshot["top_ranked_semantics"]
    top_ranked_probabilities = ranking_snapshot["top_ranked_probabilities"]

    return {
        "evaluation_episode_index": int(episode_index),
        "episode_scope": str(episode_ref["scope"]),
        "episode_partition": str(episode_ref["partition"]),
        "episode_source_rel": str(episode_ref["source_rel"]),
        "episode_fold_id": episode_ref["fold_id"],
        "step_ordinal": int(step_ordinal),
        "step_index": int(step_index),
        "timestamp": timestamp,
        "position_before": position_before_value,
        "position_regime": ranking_snapshot["position_regime"],
        "selected_action_semantic": selected_action_semantic,
        "selected_action_probability": ranking_snapshot["canonical_probabilities"][selected_action_semantic],
        "top_1_action_semantic": top_ranked_semantics[0],
        "top_1_probability": top_ranked_probabilities[0],
        "top_2_action_semantic": top_ranked_semantics[1],
        "top_2_probability": top_ranked_probabilities[1],
        "top_3_action_semantic": top_ranked_semantics[2],
        "top_3_probability": top_ranked_probabilities[2],
        "top_1_minus_top_2_gap": ranking_snapshot["top_1_minus_top_2_gap"],
        "hold_gap_vs_next_best_valid_action": ranking_snapshot["hold_gap_vs_next_best_valid_action"],
        "hold_next_best_valid_action_semantic": ranking_snapshot["hold_next_best_valid_action_semantic"],
        "hold_next_best_valid_action_probability": ranking_snapshot["hold_next_best_valid_action_probability"],
        "hold_is_top_1": ranking_snapshot["hold_is_top_1"],
        "hold_gap_below_threshold": ranking_snapshot["hold_gap_below_threshold"],
        "hold_gap_band": ranking_snapshot["hold_gap_band"],
        "selected_matches_top_1": bool(selected_action_semantic == top_ranked_semantics[0]),
        "valid_action_count": ranking_snapshot["valid_action_count"],
        "valid_action_semantics": list(ranking_snapshot["valid_action_semantics"]),
        "prob_hold": ranking_snapshot["canonical_probabilities"]["HOLD"],
        "prob_open_long": ranking_snapshot["canonical_probabilities"]["OPEN_LONG"],
        "prob_open_short": ranking_snapshot["canonical_probabilities"]["OPEN_SHORT"],
        "prob_close_position": ranking_snapshot["canonical_probabilities"]["CLOSE_POSITION"],
    }


def build_deterministic_action_ranking_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    gap_threshold: float = DETERMINISTIC_HOLD_GAP_THRESHOLD,
) -> dict[str, Any]:
    """Summarize deterministic HOLD argmax margin behavior."""

    if not rows:
        raise RuntimeError("deterministic action ranking diagnostics require at least one row")
    gap_threshold_value = _coerce_probability_scalar(gap_threshold, field_name="gap_threshold")
    hold_top_1_rows = [row for row in rows if bool(row.get("hold_is_top_1"))]
    hold_top_1_gaps = [float(row["top_1_minus_top_2_gap"]) for row in hold_top_1_rows]
    runner_up_counts: Counter[str] = Counter(
        str(row["top_2_action_semantic"]) for row in hold_top_1_rows if row.get("top_2_action_semantic") is not None
    )
    step_count = int(len(rows))
    fraction_hold_is_top_1 = float(len(hold_top_1_rows) / step_count)
    fraction_gap_below_threshold = float(
        sum(1 for row in rows if bool(row.get("hold_gap_below_threshold"))) / step_count
    )
    return {
        "gap_threshold": gap_threshold_value,
        "step_count": step_count,
        "hold_top1_step_count": int(len(hold_top_1_rows)),
        "fraction_of_steps_hold_is_top1": fraction_hold_is_top_1,
        "mean_hold_top1_gap": float(sum(hold_top_1_gaps) / len(hold_top_1_gaps)) if hold_top_1_gaps else None,
        "median_hold_top1_gap": float(np.median(hold_top_1_gaps)) if hold_top_1_gaps else None,
        "p95_hold_top1_gap": float(np.quantile(hold_top_1_gaps, 0.95)) if hold_top_1_gaps else None,
        "fraction_of_steps_gap_below_threshold": fraction_gap_below_threshold,
        "fraction_of_hold_top1_steps_gap_below_threshold": (
            float(sum(1 for row in hold_top_1_rows if bool(row.get("hold_gap_below_threshold"))) / len(hold_top_1_rows))
            if hold_top_1_rows
            else None
        ),
        "top2_runner_up_counts": _ordered_action_counts(runner_up_counts),
        "hold_dominance_margin_band": _classify_hold_dominance_margin(
            fraction_of_steps_hold_is_top1=fraction_hold_is_top_1,
            fraction_of_steps_gap_below_threshold=fraction_gap_below_threshold,
        ),
        "position_conditional_action_ranking_summary": build_position_conditional_action_ranking_summary(rows),
    }


def build_deterministic_action_ranking_trace_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize deterministic action ranking diagnostics as CSV."""

    fieldnames = [
        "evaluation_episode_index",
        "episode_scope",
        "episode_partition",
        "episode_source_rel",
        "episode_fold_id",
        "step_ordinal",
        "step_index",
        "timestamp",
        "position_before",
        "position_regime",
        "selected_action_semantic",
        "selected_action_probability",
        "top_1_action_semantic",
        "top_1_probability",
        "top_2_action_semantic",
        "top_2_probability",
        "top_3_action_semantic",
        "top_3_probability",
        "top_1_minus_top_2_gap",
        "hold_gap_vs_next_best_valid_action",
        "hold_next_best_valid_action_semantic",
        "hold_next_best_valid_action_probability",
        "hold_is_top_1",
        "hold_gap_below_threshold",
        "hold_gap_band",
        "selected_matches_top_1",
        "valid_action_count",
        "valid_action_semantics",
        "prob_hold",
        "prob_open_long",
        "prob_open_short",
        "prob_close_position",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in fieldnames})
    return buffer.getvalue()


def _compute_policy_statistics(model: Any) -> tuple[float, dict[str, float] | None]:
    """Compute masked/unmasked mean policy entropy over the latest rollout buffer."""

    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is None:
        raise RuntimeError("passivity diagnostics require model.rollout_buffer")

    observations = getattr(rollout_buffer, "observations", None)
    if observations is None:
        raise RuntimeError("passivity diagnostics require rollout_buffer.observations")

    flattened_observations = _flatten_rollout_observations(observations)
    action_masks = getattr(rollout_buffer, "action_masks", None)
    flattened_action_masks = _flatten_action_masks(action_masks) if action_masks is not None else None

    with th.no_grad():
        observation_tensor, _ = model.policy.obs_to_tensor(flattened_observations)
        if flattened_action_masks is not None:
            try:
                distribution = model.policy.get_distribution(observation_tensor, action_masks=flattened_action_masks)
            except TypeError:
                distribution = model.policy.get_distribution(observation_tensor)
        else:
            distribution = model.policy.get_distribution(observation_tensor)
        entropy_tensor = distribution.entropy()
        entropy_mean = float(th.as_tensor(entropy_tensor).float().mean().cpu().item())

    if not math.isfinite(entropy_mean):
        raise RuntimeError("passivity diagnostics observed non-finite rollout entropy")

    action_probabilities = extract_action_probabilities(distribution)
    return entropy_mean, action_probabilities


def _flatten_rollout_observations(observations: Any) -> Any:
    """Flatten rollout-buffer observations into batch-major shape."""

    if isinstance(observations, Mapping):
        flattened: dict[str, np.ndarray] = {}
        for key, value in observations.items():
            array = np.asarray(value)
            flattened[str(key)] = _flatten_array_prefix(array)
        return flattened
    return _flatten_array_prefix(np.asarray(observations))


def _flatten_action_masks(action_masks: Any) -> np.ndarray:
    """Flatten rollout-buffer action masks into batch-major shape."""

    return _flatten_array_prefix(np.asarray(action_masks, dtype=np.float32))


def _flatten_array_prefix(array: np.ndarray) -> np.ndarray:
    """Flatten rollout buffer leading rollout/env dimensions into one batch dimension."""

    if array.ndim >= 3:
        return array.reshape((-1,) + array.shape[2:])
    if array.ndim == 2:
        return array
    if array.ndim == 1:
        return array.reshape(-1, 1)
    raise RuntimeError("passivity diagnostics require non-scalar rollout arrays")


def extract_action_probabilities(distribution: Any) -> dict[str, float] | None:
    """Extract canonical action probabilities when the distribution exposes them."""

    raw_distribution = getattr(distribution, "distribution", distribution)
    probs_tensor = getattr(raw_distribution, "probs", None)
    if probs_tensor is None:
        return None
    probs_array = probs_tensor.detach().cpu().numpy() if hasattr(probs_tensor, "detach") else np.asarray(probs_tensor)
    if probs_array.ndim == 1:
        probs_array = probs_array.reshape(1, -1)
    if probs_array.ndim != 2 or probs_array.shape[1] != len(CANONICAL_ACTION_SEMANTICS):
        return None
    if not np.isfinite(probs_array).all():
        raise RuntimeError("passivity diagnostics observed non-finite action probabilities")
    mean_probs = probs_array.mean(axis=0)
    return {
        ACTION_MAPPING[action_id]: float(mean_probs[index]) for index, action_id in enumerate(CANONICAL_ACTION_ORDER)
    }


def build_action_ranking_snapshot(
    *,
    position_before: int,
    action_probabilities: Mapping[str, float],
    valid_action_mask: Sequence[bool] | np.ndarray | None = None,
    gap_threshold: float = DETERMINISTIC_HOLD_GAP_THRESHOLD,
) -> dict[str, Any]:
    """Build a canonical ranking snapshot from policy probabilities."""

    position_before_value = _coerce_position(position_before, field_name="position_before")
    gap_threshold_value = _coerce_probability_scalar(gap_threshold, field_name="gap_threshold")
    canonical_probabilities = {
        semantic: _coerce_probability_scalar(action_probabilities.get(semantic), field_name=f"probability[{semantic}]")
        for semantic in CANONICAL_ACTION_SEMANTICS
    }
    ranked_semantics = sorted(
        CANONICAL_ACTION_SEMANTICS,
        key=lambda semantic: (-canonical_probabilities[semantic], CANONICAL_ACTION_SEMANTICS.index(semantic)),
    )
    ranking_triplet = tuple(ranked_semantics[:3])
    if len(ranking_triplet) != 3:
        raise RuntimeError("ranking snapshot requires at least three canonical actions")

    mask = _resolve_valid_action_mask(position_before=position_before_value, valid_action_mask=valid_action_mask)
    valid_semantics = tuple(
        ACTION_MAPPING[action_id]
        for action_id, is_valid in zip(CANONICAL_ACTION_ORDER, mask.tolist(), strict=True)
        if bool(is_valid)
    )
    valid_action_count = int(sum(1 for is_valid in mask.tolist() if bool(is_valid)))
    hold_valid_alternatives = [semantic for semantic in valid_semantics if semantic != "HOLD"]
    next_best_valid_semantic = (
        max(
            hold_valid_alternatives,
            key=lambda semantic: (canonical_probabilities[semantic], -CANONICAL_ACTION_SEMANTICS.index(semantic)),
        )
        if hold_valid_alternatives
        else None
    )
    hold_next_best_probability = (
        canonical_probabilities[next_best_valid_semantic] if next_best_valid_semantic is not None else None
    )
    hold_gap = (
        float(canonical_probabilities["HOLD"] - hold_next_best_probability)
        if hold_next_best_probability is not None
        else None
    )
    top_1_probability = canonical_probabilities[ranking_triplet[0]]
    top_2_probability = canonical_probabilities[ranking_triplet[1]]
    top_gap = float(top_1_probability - top_2_probability)
    hold_is_top_1 = ranking_triplet[0] == "HOLD"
    hold_gap_below_threshold = bool(hold_is_top_1 and top_gap < gap_threshold_value)
    hold_gap_band = "small" if hold_gap_below_threshold else "large" if hold_is_top_1 else "not_hold_top_1"
    return {
        "position_regime": POSITION_REGIME_BY_VALUE[position_before_value],
        "canonical_probabilities": canonical_probabilities,
        "top_ranked_semantics": ranking_triplet,
        "top_ranked_probabilities": (
            top_1_probability,
            top_2_probability,
            canonical_probabilities[ranking_triplet[2]],
        ),
        "top_1_minus_top_2_gap": top_gap,
        "hold_is_top_1": bool(hold_is_top_1),
        "hold_gap_below_threshold": hold_gap_below_threshold,
        "hold_gap_band": hold_gap_band,
        "hold_gap_vs_next_best_valid_action": hold_gap,
        "hold_next_best_valid_action_semantic": next_best_valid_semantic,
        "hold_next_best_valid_action_probability": hold_next_best_probability,
        "valid_action_count": valid_action_count,
        "valid_action_semantics": valid_semantics,
    }


def build_position_conditional_action_ranking_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize deterministic ranking behavior by position regime."""

    summary: dict[str, Any] = {}
    for regime in POSITION_REGIME_ORDER:
        regime_rows = [row for row in rows if str(row.get("position_regime")) == regime]
        top1_counts = Counter(str(row.get("top_1_action_semantic")) for row in regime_rows if row.get("top_1_action_semantic"))
        top2_counts = Counter(str(row.get("top_2_action_semantic")) for row in regime_rows if row.get("top_2_action_semantic"))
        runner_up_counts = Counter(
            str(row.get("top_2_action_semantic"))
            for row in regime_rows
            if bool(row.get("hold_is_top_1")) and row.get("top_2_action_semantic")
        )
        hold_gaps = [
            float(row["hold_gap_vs_next_best_valid_action"])
            for row in regime_rows
            if row.get("hold_gap_vs_next_best_valid_action") is not None
        ]
        valid_counts = [
            int(row["valid_action_count"])
            for row in regime_rows
            if row.get("valid_action_count") is not None
        ]
        step_count = int(len(regime_rows))
        hold_top1_count = int(sum(1 for row in regime_rows if bool(row.get("hold_is_top_1"))))
        summary[regime] = {
            "step_count": step_count,
            "top1_action_frequency": _ordered_action_counts(top1_counts),
            "top2_action_frequency": _ordered_action_counts(top2_counts),
            "hold_top1_rate": (float(hold_top1_count) / float(step_count)) if step_count > 0 else None,
            "runner_up_distribution": _ordered_action_counts(runner_up_counts),
            "mean_hold_gap": float(sum(hold_gaps) / len(hold_gaps)) if hold_gaps else None,
            "p50_hold_gap": float(np.quantile(hold_gaps, 0.50)) if hold_gaps else None,
            "p95_hold_gap": float(np.quantile(hold_gaps, 0.95)) if hold_gaps else None,
            "valid_action_count_summary": _build_valid_action_count_summary(valid_counts, step_count=step_count),
        }
    return summary


def build_action_conditioned_reward_summary(episode_runtimes: Sequence[Any]) -> dict[str, dict[str, float | int | None]]:
    """Aggregate reward decomposition by selected action semantic from evaluation step records."""

    reward_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    realized_pnl_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    mtm_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    fee_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    slippage_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    risk_penalty_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    inactivity_penalty_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    close_realized_pnl_bonus_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    duration_totals = {semantic: 0.0 for semantic in CANONICAL_ACTION_SEMANTICS}
    duration_counts = {semantic: 0 for semantic in CANONICAL_ACTION_SEMANTICS}
    counts = {semantic: 0 for semantic in CANONICAL_ACTION_SEMANTICS}

    for runtime in episode_runtimes:
        for record in runtime.step_records:
            semantic = str(record.get("action_semantic", ""))
            if semantic not in CANONICAL_ACTION_SEMANTICS:
                continue
            counts[semantic] += 1
            reward_totals[semantic] += _coerce_optional_float(record.get("reward_total")) or 0.0
            realized_pnl_totals[semantic] += _coerce_optional_float(record.get("realized_pnl_contribution")) or 0.0
            mtm_totals[semantic] += (
                _coerce_optional_float(record.get("position_mtm_contribution"))
                if record.get("position_mtm_contribution") is not None
                else _coerce_optional_float(record.get("pnl_delta"))
                or 0.0
            )
            fee_totals[semantic] += _coerce_optional_float(record.get("fees")) or 0.0
            slippage_totals[semantic] += _coerce_optional_float(record.get("slippage_cost")) or 0.0
            risk_penalty_totals[semantic] += _coerce_optional_float(record.get("risk_penalty")) or 0.0
            inactivity_penalty_totals[semantic] += _coerce_optional_float(record.get("inactivity_penalty")) or 0.0
            close_realized_pnl_bonus_totals[semantic] += (
                _coerce_optional_float(record.get("close_realized_pnl_bonus_contribution")) or 0.0
            )
            duration_value = record.get("holding_duration_after_entry_steps")
            if duration_value is not None:
                duration_counts[semantic] += 1
                duration_totals[semantic] += float(duration_value)

    return {
        semantic: {
            "count": counts[semantic],
            "avg_total_reward": _safe_mean(reward_totals[semantic], counts[semantic]),
            "avg_realized_pnl_contribution": _safe_mean(realized_pnl_totals[semantic], counts[semantic]),
            "avg_mtm_contribution": _safe_mean(mtm_totals[semantic], counts[semantic]),
            "avg_fee_cost": _safe_mean(fee_totals[semantic], counts[semantic]),
            "avg_slippage_cost": _safe_mean(slippage_totals[semantic], counts[semantic]),
            "avg_risk_penalty": _safe_mean(risk_penalty_totals[semantic], counts[semantic]),
            "avg_inactivity_penalty": _safe_mean(inactivity_penalty_totals[semantic], counts[semantic]),
            "avg_close_realized_pnl_bonus_contribution": _safe_mean(
                close_realized_pnl_bonus_totals[semantic],
                counts[semantic],
            ),
            "avg_holding_duration_after_entry_steps": _safe_mean(duration_totals[semantic], duration_counts[semantic]),
        }
        for semantic in CANONICAL_ACTION_SEMANTICS
    }


def build_compact_step_diagnostics_report(
    *,
    run_id: str,
    evaluation_session_id: str,
    evaluation_policy_mode: str,
    action_masking_enabled: bool,
    compact_step_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact per-step diagnostics artifact for bounded evaluation debugging."""

    return {
        "contract_version": "compact_step_diagnostics.v1",
        "run_id": run_id,
        "evaluation_session_id": evaluation_session_id,
        "evaluation_policy_mode": evaluation_policy_mode,
        "action_masking_enabled": bool(action_masking_enabled),
        "step_count": int(len(compact_step_rows)),
        "records": [dict(row) for row in compact_step_rows],
        "generated_at_utc": _generated_at(),
    }


def _classify_minibatch_lifecycle_phase(*, position_before: int, position_age_before_step: int) -> str:
    """Classify one rollout sample into the bounded lifecycle phases used by the PPO minibatch audit."""

    if int(position_before) == 0:
        return "flat"
    if int(position_age_before_step) <= 1:
        return "age1_in_position"
    if int(position_before) == 1:
        return "age2plus_long"
    return "age2plus_short"


def _empty_phase_count_map() -> dict[str, int]:
    """Return a zero-initialized lifecycle-phase count map."""

    return {phase: 0 for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER}


def _empty_phase_action_count_map() -> dict[str, dict[str, int]]:
    """Return zero-initialized lifecycle-phase/action counts."""

    return {
        phase: {semantic: 0 for semantic in CANONICAL_ACTION_SEMANTICS}
        for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER
    }


def _empty_phase_action_value_lists() -> dict[str, dict[str, list[float]]]:
    """Return lifecycle-phase/action containers for minibatch metric aggregation."""

    return {
        phase: {semantic: [] for semantic in CANONICAL_ACTION_SEMANTICS}
        for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER
    }


def _phase_percentages_from_counts(counts_by_phase: Mapping[str, int]) -> dict[str, float | None]:
    """Convert one lifecycle-phase count map into percentages."""

    total = int(sum(int(counts_by_phase.get(phase, 0)) for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER))
    if total <= 0:
        return {phase: None for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER}
    return {
        phase: float(int(counts_by_phase.get(phase, 0)) / total)
        for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER
    }


def _dominant_phase_from_counts(counts_by_phase: Mapping[str, int]) -> str | None:
    """Return the dominant lifecycle phase for one count map."""

    dominant_phase = None
    dominant_count = 0
    for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER:
        count = int(counts_by_phase.get(phase, 0))
        if count > dominant_count:
            dominant_phase = phase
            dominant_count = count
    return dominant_phase if dominant_count > 0 else None


def _phase_action_mean_map(values_by_phase_action: Mapping[str, Mapping[str, Sequence[float]]]) -> dict[str, dict[str, float | None]]:
    """Project lifecycle-phase/action value lists into mean maps."""

    projected: dict[str, dict[str, float | None]] = {}
    for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER:
        projected[phase] = {}
        phase_values = values_by_phase_action.get(phase, {})
        for semantic in CANONICAL_ACTION_SEMANTICS:
            raw_values = phase_values.get(semantic, ())
            numeric_values = [float(item) for item in raw_values if item is not None]
            projected[phase][semantic] = (
                float(sum(numeric_values) / len(numeric_values)) if numeric_values else None
            )
    return projected


def _build_ppo_minibatch_composition_row(
    *,
    update_index: int,
    epoch_index: int,
    minibatch_index: int,
    batch_indices: Sequence[int],
    sample_metadata: Sequence[Mapping[str, Any]],
    raw_advantages: Sequence[float],
    normalized_advantages: Sequence[float],
) -> dict[str, Any] | None:
    """Build one audit-only PPO minibatch lifecycle composition row."""

    batch_index_values = [int(item) for item in np.asarray(batch_indices, dtype=np.int64).reshape(-1)]
    if not batch_index_values:
        return None
    raw_advantage_values = [float(item) for item in np.asarray(raw_advantages, dtype=np.float64).reshape(-1)]
    normalized_advantage_values = [
        float(item) for item in np.asarray(normalized_advantages, dtype=np.float64).reshape(-1)
    ]
    if len(batch_index_values) != len(raw_advantage_values) or len(batch_index_values) != len(normalized_advantage_values):
        raise RuntimeError("minibatch lifecycle phase trace requires aligned batch indices and advantages")

    lifecycle_phase_counts = _empty_phase_count_map()
    phase_action_counts = _empty_phase_action_count_map()
    raw_values_by_phase_action = _empty_phase_action_value_lists()
    normalized_values_by_phase_action = _empty_phase_action_value_lists()
    close_sample_counts_by_phase = _empty_phase_count_map()
    positive_close_raw_advantage_counts_by_phase = _empty_phase_count_map()
    positive_close_normalized_advantage_counts_by_phase = _empty_phase_count_map()

    metadata_count = int(len(sample_metadata))
    for flat_index, raw_advantage, normalized_advantage in zip(
        batch_index_values,
        raw_advantage_values,
        normalized_advantage_values,
        strict=True,
    ):
        if flat_index < 0 or flat_index >= metadata_count:
            raise RuntimeError("minibatch lifecycle phase trace received a flat index outside rollout metadata bounds")
        sample_record = sample_metadata[flat_index]
        phase = str(sample_record.get("lifecycle_phase"))
        semantic = str(sample_record.get("chosen_action_semantic"))
        if phase not in MINIBATCH_LIFECYCLE_PHASE_ORDER:
            raise RuntimeError(f"unsupported lifecycle phase for minibatch trace: {phase}")
        if semantic not in CANONICAL_ACTION_SEMANTICS:
            raise RuntimeError(f"unsupported action semantic for minibatch trace: {semantic}")
        lifecycle_phase_counts[phase] += 1
        phase_action_counts[phase][semantic] += 1
        raw_values_by_phase_action[phase][semantic].append(float(raw_advantage))
        normalized_values_by_phase_action[phase][semantic].append(float(normalized_advantage))
        if semantic == "CLOSE_POSITION":
            close_sample_counts_by_phase[phase] += 1
            if float(raw_advantage) > 0.0:
                positive_close_raw_advantage_counts_by_phase[phase] += 1
            if float(normalized_advantage) > 0.0:
                positive_close_normalized_advantage_counts_by_phase[phase] += 1

    return {
        "update_index": int(update_index),
        "epoch_index": int(epoch_index),
        "minibatch_index": int(minibatch_index),
        "sample_count": int(len(batch_index_values)),
        "lifecycle_phase_counts": dict(lifecycle_phase_counts),
        "phase_action_counts": {
            phase: dict(phase_action_counts[phase]) for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER
        },
        "phase_action_mean_raw_advantage": _phase_action_mean_map(raw_values_by_phase_action),
        "phase_action_mean_normalized_advantage": _phase_action_mean_map(normalized_values_by_phase_action),
        "close_sample_counts_by_phase": dict(close_sample_counts_by_phase),
        "positive_close_raw_advantage_counts_by_phase": dict(positive_close_raw_advantage_counts_by_phase),
        "positive_close_normalized_advantage_counts_by_phase": dict(
            positive_close_normalized_advantage_counts_by_phase
        ),
        "close_sample_phase_percentages": _phase_percentages_from_counts(close_sample_counts_by_phase),
        "positive_close_raw_advantage_phase_percentages": _phase_percentages_from_counts(
            positive_close_raw_advantage_counts_by_phase
        ),
        "positive_close_normalized_advantage_phase_percentages": _phase_percentages_from_counts(
            positive_close_normalized_advantage_counts_by_phase
        ),
        "dominant_close_sample_phase": _dominant_phase_from_counts(close_sample_counts_by_phase),
        "dominant_positive_close_raw_advantage_phase": _dominant_phase_from_counts(
            positive_close_raw_advantage_counts_by_phase
        ),
        "dominant_positive_close_normalized_advantage_phase": _dominant_phase_from_counts(
            positive_close_normalized_advantage_counts_by_phase
        ),
    }


def _build_ppo_minibatch_update_summary_rows(
    *,
    minibatch_rows: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate minibatch lifecycle composition rows into one summary per PPO update."""

    rollout_rows_by_update = {
        _coerce_optional_int(row.get("update_index")): row
        for row in rollout_rows
        if _coerce_optional_int(row.get("update_index")) is not None
    }
    grouped_rows: dict[int, list[Mapping[str, Any]]] = {}
    for row in minibatch_rows:
        update_index = _coerce_optional_int(row.get("update_index"))
        if update_index is None:
            continue
        grouped_rows.setdefault(int(update_index), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for update_index in sorted(grouped_rows):
        rows_for_update = grouped_rows[update_index]
        lifecycle_phase_counts = _empty_phase_count_map()
        close_sample_counts_by_phase = _empty_phase_count_map()
        positive_close_raw_advantage_counts_by_phase = _empty_phase_count_map()
        positive_close_normalized_advantage_counts_by_phase = _empty_phase_count_map()
        dominant_close_phase_counts = _empty_phase_count_map()
        dominant_positive_close_raw_phase_counts = _empty_phase_count_map()
        dominant_positive_close_normalized_phase_counts = _empty_phase_count_map()

        for row in rows_for_update:
            for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER:
                lifecycle_phase_counts[phase] += int(_coerce_optional_int(row.get("lifecycle_phase_counts", {}).get(phase)) or 0)
                close_sample_counts_by_phase[phase] += int(
                    _coerce_optional_int(row.get("close_sample_counts_by_phase", {}).get(phase)) or 0
                )
                positive_close_raw_advantage_counts_by_phase[phase] += int(
                    _coerce_optional_int(row.get("positive_close_raw_advantage_counts_by_phase", {}).get(phase)) or 0
                )
                positive_close_normalized_advantage_counts_by_phase[phase] += int(
                    _coerce_optional_int(row.get("positive_close_normalized_advantage_counts_by_phase", {}).get(phase))
                    or 0
                )
            dominant_close_phase = row.get("dominant_close_sample_phase")
            if dominant_close_phase in dominant_close_phase_counts:
                dominant_close_phase_counts[str(dominant_close_phase)] += 1
            dominant_positive_close_raw_phase = row.get("dominant_positive_close_raw_advantage_phase")
            if dominant_positive_close_raw_phase in dominant_positive_close_raw_phase_counts:
                dominant_positive_close_raw_phase_counts[str(dominant_positive_close_raw_phase)] += 1
            dominant_positive_close_normalized_phase = row.get("dominant_positive_close_normalized_advantage_phase")
            if dominant_positive_close_normalized_phase in dominant_positive_close_normalized_phase_counts:
                dominant_positive_close_normalized_phase_counts[str(dominant_positive_close_normalized_phase)] += 1

        rollout_row = rollout_rows_by_update.get(update_index, {})
        close_drift = _coerce_optional_float(
            rollout_row.get("representative_long_mean_close_position_probability_drift")
            if isinstance(rollout_row, Mapping)
            else None
        )
        summary_rows.append(
            {
                "update_index": int(update_index),
                "minibatch_count": int(len(rows_for_update)),
                "lifecycle_phase_counts": dict(lifecycle_phase_counts),
                "close_sample_counts_by_phase": dict(close_sample_counts_by_phase),
                "positive_close_raw_advantage_counts_by_phase": dict(
                    positive_close_raw_advantage_counts_by_phase
                ),
                "positive_close_normalized_advantage_counts_by_phase": dict(
                    positive_close_normalized_advantage_counts_by_phase
                ),
                "close_sample_phase_percentages": _phase_percentages_from_counts(close_sample_counts_by_phase),
                "positive_close_raw_advantage_phase_percentages": _phase_percentages_from_counts(
                    positive_close_raw_advantage_counts_by_phase
                ),
                "positive_close_normalized_advantage_phase_percentages": _phase_percentages_from_counts(
                    positive_close_normalized_advantage_counts_by_phase
                ),
                "dominant_close_sample_phase": _dominant_phase_from_counts(close_sample_counts_by_phase),
                "dominant_positive_close_raw_advantage_phase": _dominant_phase_from_counts(
                    positive_close_raw_advantage_counts_by_phase
                ),
                "dominant_positive_close_normalized_advantage_phase": _dominant_phase_from_counts(
                    positive_close_normalized_advantage_counts_by_phase
                ),
                "dominant_close_sample_phase_minibatch_counts": dict(dominant_close_phase_counts),
                "dominant_positive_close_raw_advantage_phase_minibatch_counts": dict(
                    dominant_positive_close_raw_phase_counts
                ),
                "dominant_positive_close_normalized_advantage_phase_minibatch_counts": dict(
                    dominant_positive_close_normalized_phase_counts
                ),
                "representative_long_mean_close_position_probability_drift": close_drift,
                "positive_representative_long_close_drift": (
                    bool(close_drift > 0.0) if close_drift is not None else None
                ),
            }
        )
    return summary_rows


def _build_ppo_minibatch_trace_summary(
    *,
    minibatch_rows: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one compact audit summary for PPO minibatch lifecycle composition."""

    update_summary_rows = _build_ppo_minibatch_update_summary_rows(
        minibatch_rows=minibatch_rows,
        rollout_rows=rollout_rows,
    )
    all_close_counts = _empty_phase_count_map()
    all_positive_raw_counts = _empty_phase_count_map()
    all_positive_normalized_counts = _empty_phase_count_map()
    positive_drift_close_counts = _empty_phase_count_map()
    positive_drift_positive_raw_counts = _empty_phase_count_map()
    positive_drift_positive_normalized_counts = _empty_phase_count_map()
    positive_drift_update_indices: list[int] = []

    for row in update_summary_rows:
        for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER:
            all_close_counts[phase] += int(_coerce_optional_int(row.get("close_sample_counts_by_phase", {}).get(phase)) or 0)
            all_positive_raw_counts[phase] += int(
                _coerce_optional_int(row.get("positive_close_raw_advantage_counts_by_phase", {}).get(phase)) or 0
            )
            all_positive_normalized_counts[phase] += int(
                _coerce_optional_int(row.get("positive_close_normalized_advantage_counts_by_phase", {}).get(phase)) or 0
            )
        if row.get("positive_representative_long_close_drift") is True:
            positive_drift_update_indices.append(int(row["update_index"]))
            for phase in MINIBATCH_LIFECYCLE_PHASE_ORDER:
                positive_drift_close_counts[phase] += int(
                    _coerce_optional_int(row.get("close_sample_counts_by_phase", {}).get(phase)) or 0
                )
                positive_drift_positive_raw_counts[phase] += int(
                    _coerce_optional_int(row.get("positive_close_raw_advantage_counts_by_phase", {}).get(phase)) or 0
                )
                positive_drift_positive_normalized_counts[phase] += int(
                    _coerce_optional_int(row.get("positive_close_normalized_advantage_counts_by_phase", {}).get(phase))
                    or 0
                )

    return {
        "update_count": int(len(update_summary_rows)),
        "minibatch_count": int(len(minibatch_rows)),
        "all_updates_close_sample_phase_counts": dict(all_close_counts),
        "all_updates_positive_close_raw_advantage_phase_counts": dict(all_positive_raw_counts),
        "all_updates_positive_close_normalized_advantage_phase_counts": dict(all_positive_normalized_counts),
        "all_updates_close_sample_phase_percentages": _phase_percentages_from_counts(all_close_counts),
        "all_updates_positive_close_raw_advantage_phase_percentages": _phase_percentages_from_counts(
            all_positive_raw_counts
        ),
        "all_updates_positive_close_normalized_advantage_phase_percentages": _phase_percentages_from_counts(
            all_positive_normalized_counts
        ),
        "all_updates_dominant_close_sample_phase": _dominant_phase_from_counts(all_close_counts),
        "all_updates_dominant_positive_close_raw_advantage_phase": _dominant_phase_from_counts(
            all_positive_raw_counts
        ),
        "all_updates_dominant_positive_close_normalized_advantage_phase": _dominant_phase_from_counts(
            all_positive_normalized_counts
        ),
        "positive_representative_long_close_drift_update_indices": [int(item) for item in positive_drift_update_indices],
        "positive_drift_updates_close_sample_phase_counts": dict(positive_drift_close_counts),
        "positive_drift_updates_positive_close_raw_advantage_phase_counts": dict(
            positive_drift_positive_raw_counts
        ),
        "positive_drift_updates_positive_close_normalized_advantage_phase_counts": dict(
            positive_drift_positive_normalized_counts
        ),
        "positive_drift_updates_close_sample_phase_percentages": _phase_percentages_from_counts(
            positive_drift_close_counts
        ),
        "positive_drift_updates_positive_close_raw_advantage_phase_percentages": _phase_percentages_from_counts(
            positive_drift_positive_raw_counts
        ),
        "positive_drift_updates_positive_close_normalized_advantage_phase_percentages": _phase_percentages_from_counts(
            positive_drift_positive_normalized_counts
        ),
        "positive_drift_updates_dominant_close_sample_phase": _dominant_phase_from_counts(
            positive_drift_close_counts
        ),
        "positive_drift_updates_dominant_positive_close_raw_advantage_phase": _dominant_phase_from_counts(
            positive_drift_positive_raw_counts
        ),
        "positive_drift_updates_dominant_positive_close_normalized_advantage_phase": _dominant_phase_from_counts(
            positive_drift_positive_normalized_counts
        ),
    }


def _build_training_entropy_timeseries_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize rollout-point entropy telemetry as CSV."""

    fieldnames = [
        "update_index",
        "num_timesteps",
        "mean_policy_entropy",
        "mean_hold_probability",
        "mean_open_long_probability",
        "mean_open_short_probability",
        "mean_close_position_probability",
        "selected_hold_count",
        "selected_open_long_count",
        "selected_open_short_count",
        "selected_close_position_count",
        "reward_sum_hold",
        "reward_sum_open_long",
        "reward_sum_open_short",
        "reward_sum_close_position",
        "reward_count_hold",
        "reward_count_open_long",
        "reward_count_open_short",
        "reward_count_close_position",
        "sampled_age1_hold_count",
        "sampled_age1_close_position_count",
        "sampled_age1_hold_mean_raw_advantage",
        "sampled_age1_close_position_mean_raw_advantage",
        "sampled_age1_hold_mean_normalized_advantage",
        "sampled_age1_close_position_mean_normalized_advantage",
        "age1_close_position_pairwise_hold_mean_advantage_subtract",
        "age1_close_position_pairwise_hold_mean_raw_advantage_baseline",
        "age1_close_position_pairwise_hold_mean_advantage_subtract_applied_count",
        "age1_close_position_pairwise_hold_mean_advantage_subtract_fallback_count",
        "age1_close_position_mean_raw_advantage_after_pairwise_correction",
        "age1_close_position_mean_normalized_advantage_after_pairwise_correction",
        "age1_close_position_positive_advantage_clip",
        "age1_close_position_positive_advantage_clip_applied_count",
        "age1_close_position_mean_raw_advantage_after_clip",
        "age1_close_position_mean_normalized_advantage_after_clip",
        "sampled_age2plus_long_hold_count",
        "sampled_age2plus_long_close_position_count",
        "sampled_age2plus_long_hold_mean_raw_advantage",
        "sampled_age2plus_long_close_position_mean_raw_advantage",
        "sampled_age2plus_long_hold_mean_normalized_advantage",
        "sampled_age2plus_long_close_position_mean_normalized_advantage",
        "sampled_age2plus_long_hold_mean_value_pre",
        "sampled_age2plus_long_close_position_mean_value_pre",
        "sampled_age2plus_long_hold_mean_return_target",
        "sampled_age2plus_long_close_position_mean_return_target",
        "sampled_age2plus_long_hold_mean_value_error",
        "sampled_age2plus_long_close_position_mean_value_error",
        "sampled_age2plus_long_hold_mean_abs_value_error",
        "sampled_age2plus_long_close_position_mean_abs_value_error",
        "sampled_age2plus_long_hold_mean_squared_value_error",
        "sampled_age2plus_long_close_position_mean_squared_value_error",
        "representative_flat_state_count",
        "representative_flat_mean_hold_probability_pre_update",
        "representative_flat_mean_open_long_probability_pre_update",
        "representative_flat_mean_open_short_probability_pre_update",
        "representative_flat_mean_open_long_minus_hold_pre_update",
        "representative_flat_mean_hold_probability_post_update",
        "representative_flat_mean_open_long_probability_post_update",
        "representative_flat_mean_open_short_probability_post_update",
        "representative_flat_mean_open_long_minus_hold_post_update",
        "representative_flat_mean_open_long_minus_hold_drift",
        "representative_age1_mean_hold_probability_pre_update",
        "representative_age1_mean_close_position_probability_pre_update",
        "representative_age1_mean_hold_probability_post_update",
        "representative_age1_mean_close_position_probability_post_update",
        "representative_age1_mean_hold_probability_drift",
        "representative_age1_mean_close_position_probability_drift",
        "representative_long_state_count",
        "representative_long_mean_hold_probability_pre_update",
        "representative_long_mean_close_position_probability_pre_update",
        "representative_long_mean_value_estimate_pre_update",
        "representative_long_mean_hold_probability_post_update",
        "representative_long_mean_close_position_probability_post_update",
        "representative_long_mean_value_estimate_post_update",
        "representative_long_mean_hold_probability_drift",
        "representative_long_mean_close_position_probability_drift",
        "representative_long_mean_value_estimate_drift",
        "train_approx_kl",
        "train_clip_fraction",
        "train_explained_variance",
        "train_value_loss",
        "train_entropy_loss",
        "train_policy_gradient_loss",
        "train_clip_range",
        "train_clip_range_vf",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in fieldnames})
    return buffer.getvalue()


def _annotate_rollout_records(*, records: Sequence[Mapping[str, Any]], rollout_buffer: Any) -> list[dict[str, Any]]:
    """Attach raw/normalized advantages plus return targets to sampled rollout records."""

    if rollout_buffer is None:
        raise RuntimeError("passivity diagnostics require model.rollout_buffer")
    advantages = np.asarray(getattr(rollout_buffer, "advantages", None), dtype=np.float64)
    if advantages.ndim != 2:
        raise RuntimeError("passivity diagnostics require rollout_buffer.advantages with shape [n_steps, n_envs]")
    returns_raw = getattr(rollout_buffer, "returns", None)
    returns = None if returns_raw is None else np.asarray(returns_raw, dtype=np.float64)
    if returns is not None and returns.ndim != 2:
        raise RuntimeError("passivity diagnostics require rollout_buffer.returns with shape [n_steps, n_envs]")
    flat_advantages = advantages.reshape(-1)
    if flat_advantages.size == 0:
        raise RuntimeError("passivity diagnostics require non-empty rollout advantages")
    mean_advantage = float(flat_advantages.mean())
    std_advantage = float(flat_advantages.std())
    denom = std_advantage + 1e-8
    annotated: list[dict[str, Any]] = []
    for record in records:
        step_index = int(record["rollout_step_index"])
        env_index = int(record["env_index"])
        raw_advantage = float(advantages[step_index, env_index])
        annotated_record = dict(record)
        annotated_record["raw_advantage"] = raw_advantage
        annotated_record["normalized_advantage"] = float((raw_advantage - mean_advantage) / denom)
        return_target = float(returns[step_index, env_index]) if returns is not None else None
        value_pre = _coerce_optional_float(record.get("value_pre"))
        annotated_record["return_target"] = return_target
        if return_target is not None and value_pre is not None:
            value_error = float(return_target - value_pre)
            annotated_record["value_error"] = value_error
            annotated_record["abs_value_error"] = float(abs(value_error))
            annotated_record["squared_value_error"] = float(value_error * value_error)
        else:
            annotated_record["value_error"] = None
            annotated_record["abs_value_error"] = None
            annotated_record["squared_value_error"] = None
        annotated.append(annotated_record)
    return annotated


def _annotate_age1_advantages(*, records: Sequence[Mapping[str, Any]], rollout_buffer: Any) -> list[dict[str, Any]]:
    """Attach raw and rollout-normalized advantages to sampled age1 records."""

    return _annotate_rollout_records(records=records, rollout_buffer=rollout_buffer)


def _build_age1_action_update_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize sampled age1 HOLD vs CLOSE behavior for one PPO rollout/update."""

    hold_records = [record for record in records if str(record.get("chosen_action_semantic")) == "HOLD"]
    close_records = [record for record in records if str(record.get("chosen_action_semantic")) == "CLOSE_POSITION"]
    return {
        "sampled_age1_total_count": int(len(records)),
        "sampled_age1_hold_count": int(len(hold_records)),
        "sampled_age1_close_position_count": int(len(close_records)),
        "sampled_age1_hold_mean_raw_advantage": _mean_from_records(hold_records, key="raw_advantage"),
        "sampled_age1_close_position_mean_raw_advantage": _mean_from_records(close_records, key="raw_advantage"),
        "sampled_age1_hold_mean_normalized_advantage": _mean_from_records(hold_records, key="normalized_advantage"),
        "sampled_age1_close_position_mean_normalized_advantage": _mean_from_records(
            close_records,
            key="normalized_advantage",
        ),
    }


def _build_age2plus_long_action_update_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize sampled age2+ long HOLD vs CLOSE behavior for one PPO rollout/update."""

    hold_records = [record for record in records if str(record.get("chosen_action_semantic")) == "HOLD"]
    close_records = [record for record in records if str(record.get("chosen_action_semantic")) == "CLOSE_POSITION"]
    return {
        "sampled_age2plus_long_total_count": int(len(records)),
        "sampled_age2plus_long_hold_count": int(len(hold_records)),
        "sampled_age2plus_long_close_position_count": int(len(close_records)),
        "sampled_age2plus_long_hold_mean_raw_advantage": _mean_from_records(hold_records, key="raw_advantage"),
        "sampled_age2plus_long_close_position_mean_raw_advantage": _mean_from_records(
            close_records,
            key="raw_advantage",
        ),
        "sampled_age2plus_long_hold_mean_normalized_advantage": _mean_from_records(
            hold_records,
            key="normalized_advantage",
        ),
        "sampled_age2plus_long_close_position_mean_normalized_advantage": _mean_from_records(
            close_records,
            key="normalized_advantage",
        ),
        "sampled_age2plus_long_hold_mean_value_pre": _mean_from_records(hold_records, key="value_pre"),
        "sampled_age2plus_long_close_position_mean_value_pre": _mean_from_records(close_records, key="value_pre"),
        "sampled_age2plus_long_hold_mean_return_target": _mean_from_records(hold_records, key="return_target"),
        "sampled_age2plus_long_close_position_mean_return_target": _mean_from_records(
            close_records,
            key="return_target",
        ),
        "sampled_age2plus_long_hold_mean_value_error": _mean_from_records(hold_records, key="value_error"),
        "sampled_age2plus_long_close_position_mean_value_error": _mean_from_records(
            close_records,
            key="value_error",
        ),
        "sampled_age2plus_long_hold_mean_abs_value_error": _mean_from_records(
            hold_records,
            key="abs_value_error",
        ),
        "sampled_age2plus_long_close_position_mean_abs_value_error": _mean_from_records(
            close_records,
            key="abs_value_error",
        ),
        "sampled_age2plus_long_hold_mean_squared_value_error": _mean_from_records(
            hold_records,
            key="squared_value_error",
        ),
        "sampled_age2plus_long_close_position_mean_squared_value_error": _mean_from_records(
            close_records,
            key="squared_value_error",
        ),
    }


def _apply_age1_close_position_positive_advantage_clip(
    *,
    records: Sequence[Mapping[str, Any]],
    rollout_buffer: Any,
    positive_advantage_clip: float | None,
) -> dict[str, Any]:
    """Clip sampled age1 CLOSE_POSITION policy advantages before PPO train."""

    summary = {
        "age1_close_position_positive_advantage_clip": _coerce_optional_float(positive_advantage_clip),
        "age1_close_position_positive_advantage_clip_applied_count": 0,
        "age1_close_position_mean_raw_advantage_after_clip": None,
        "age1_close_position_mean_normalized_advantage_after_clip": None,
    }
    if positive_advantage_clip is None:
        return summary
    if rollout_buffer is None:
        raise RuntimeError("passivity diagnostics require model.rollout_buffer")
    advantage_array = np.asarray(getattr(rollout_buffer, "advantages", None))
    if advantage_array.ndim != 2:
        raise RuntimeError("passivity diagnostics require rollout_buffer.advantages with shape [n_steps, n_envs]")

    clip_value = float(positive_advantage_clip)
    applied_count = 0
    for record in records:
        if str(record.get("chosen_action_semantic")) != "CLOSE_POSITION":
            continue
        step_index = int(record["rollout_step_index"])
        env_index = int(record["env_index"])
        if float(advantage_array[step_index, env_index]) > clip_value:
            advantage_array[step_index, env_index] = clip_value
            applied_count += 1

    close_records_after_clip = [
        record
        for record in _annotate_age1_advantages(records=records, rollout_buffer=rollout_buffer)
        if str(record.get("chosen_action_semantic")) == "CLOSE_POSITION"
    ]
    summary["age1_close_position_positive_advantage_clip_applied_count"] = int(applied_count)
    summary["age1_close_position_mean_raw_advantage_after_clip"] = _mean_from_records(
        close_records_after_clip,
        key="raw_advantage",
    )
    summary["age1_close_position_mean_normalized_advantage_after_clip"] = _mean_from_records(
        close_records_after_clip,
        key="normalized_advantage",
    )
    return summary


def _apply_age1_close_position_pairwise_hold_mean_advantage_subtract(
    *,
    records: Sequence[Mapping[str, Any]],
    rollout_buffer: Any,
    enabled: bool,
) -> dict[str, Any]:
    """Subtract the sampled age1 HOLD mean raw advantage from sampled age1 CLOSE_POSITION entries."""

    summary = {
        "age1_close_position_pairwise_hold_mean_advantage_subtract": bool(enabled),
        "age1_close_position_pairwise_hold_mean_raw_advantage_baseline": None,
        "age1_close_position_pairwise_hold_mean_advantage_subtract_applied_count": 0,
        "age1_close_position_pairwise_hold_mean_advantage_subtract_fallback_count": 0,
        "age1_close_position_mean_raw_advantage_after_pairwise_correction": None,
        "age1_close_position_mean_normalized_advantage_after_pairwise_correction": None,
    }
    if not enabled:
        return summary
    if rollout_buffer is None:
        raise RuntimeError("passivity diagnostics require model.rollout_buffer")
    advantage_array = np.asarray(getattr(rollout_buffer, "advantages", None))
    if advantage_array.ndim != 2:
        raise RuntimeError("passivity diagnostics require rollout_buffer.advantages with shape [n_steps, n_envs]")

    hold_records = [record for record in records if str(record.get("chosen_action_semantic")) == "HOLD"]
    if not hold_records:
        summary["age1_close_position_pairwise_hold_mean_advantage_subtract_fallback_count"] = 1
        close_records_after_fallback = [
            record
            for record in _annotate_age1_advantages(records=records, rollout_buffer=rollout_buffer)
            if str(record.get("chosen_action_semantic")) == "CLOSE_POSITION"
        ]
        summary["age1_close_position_mean_raw_advantage_after_pairwise_correction"] = _mean_from_records(
            close_records_after_fallback,
            key="raw_advantage",
        )
        summary["age1_close_position_mean_normalized_advantage_after_pairwise_correction"] = _mean_from_records(
            close_records_after_fallback,
            key="normalized_advantage",
        )
        return summary

    hold_mean_raw_advantage = _mean_from_records(hold_records, key="raw_advantage")
    summary["age1_close_position_pairwise_hold_mean_raw_advantage_baseline"] = hold_mean_raw_advantage
    if hold_mean_raw_advantage is None:
        summary["age1_close_position_pairwise_hold_mean_advantage_subtract_fallback_count"] = 1
        return summary

    applied_count = 0
    hold_mean = float(hold_mean_raw_advantage)
    for record in records:
        if str(record.get("chosen_action_semantic")) != "CLOSE_POSITION":
            continue
        step_index = int(record["rollout_step_index"])
        env_index = int(record["env_index"])
        advantage_array[step_index, env_index] = float(advantage_array[step_index, env_index]) - hold_mean
        applied_count += 1

    close_records_after_correction = [
        record
        for record in _annotate_age1_advantages(records=records, rollout_buffer=rollout_buffer)
        if str(record.get("chosen_action_semantic")) == "CLOSE_POSITION"
    ]
    summary["age1_close_position_pairwise_hold_mean_advantage_subtract_applied_count"] = int(applied_count)
    summary["age1_close_position_mean_raw_advantage_after_pairwise_correction"] = _mean_from_records(
        close_records_after_correction,
        key="raw_advantage",
    )
    summary["age1_close_position_mean_normalized_advantage_after_pairwise_correction"] = _mean_from_records(
        close_records_after_correction,
        key="normalized_advantage",
    )
    return summary


def _build_age1_update_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project stable age1 per-update fields for machine-readable audit review."""

    fieldnames = (
        "update_index",
        "num_timesteps",
        "sampled_age1_total_count",
        "sampled_age1_hold_count",
        "sampled_age1_close_position_count",
        "sampled_age1_hold_mean_raw_advantage",
        "sampled_age1_close_position_mean_raw_advantage",
        "sampled_age1_hold_mean_normalized_advantage",
        "sampled_age1_close_position_mean_normalized_advantage",
        "age1_close_position_pairwise_hold_mean_advantage_subtract",
        "age1_close_position_pairwise_hold_mean_raw_advantage_baseline",
        "age1_close_position_pairwise_hold_mean_advantage_subtract_applied_count",
        "age1_close_position_pairwise_hold_mean_advantage_subtract_fallback_count",
        "age1_close_position_mean_raw_advantage_after_pairwise_correction",
        "age1_close_position_mean_normalized_advantage_after_pairwise_correction",
        "age1_close_position_positive_advantage_clip",
        "age1_close_position_positive_advantage_clip_applied_count",
        "age1_close_position_mean_raw_advantage_after_clip",
        "age1_close_position_mean_normalized_advantage_after_clip",
        "representative_flat_state_count",
        "representative_flat_mean_hold_probability_pre_update",
        "representative_flat_mean_open_long_probability_pre_update",
        "representative_flat_mean_open_short_probability_pre_update",
        "representative_flat_mean_open_long_minus_hold_pre_update",
        "representative_flat_mean_hold_probability_post_update",
        "representative_flat_mean_open_long_probability_post_update",
        "representative_flat_mean_open_short_probability_post_update",
        "representative_flat_mean_open_long_minus_hold_post_update",
        "representative_flat_mean_open_long_minus_hold_drift",
        "representative_age1_state_count",
        "representative_age1_mean_hold_probability_pre_update",
        "representative_age1_mean_close_position_probability_pre_update",
        "representative_age1_mean_hold_probability_post_update",
        "representative_age1_mean_close_position_probability_post_update",
        "representative_age1_mean_hold_probability_drift",
        "representative_age1_mean_close_position_probability_drift",
    )
    return [{field: row.get(field) for field in fieldnames} for row in rows]


def _build_age2plus_long_update_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project stable age2+ long per-update fields for machine-readable audit review."""

    fieldnames = (
        "update_index",
        "num_timesteps",
        "sampled_age2plus_long_total_count",
        "sampled_age2plus_long_hold_count",
        "sampled_age2plus_long_close_position_count",
        "sampled_age2plus_long_hold_mean_raw_advantage",
        "sampled_age2plus_long_close_position_mean_raw_advantage",
        "sampled_age2plus_long_hold_mean_normalized_advantage",
        "sampled_age2plus_long_close_position_mean_normalized_advantage",
        "sampled_age2plus_long_hold_mean_value_pre",
        "sampled_age2plus_long_close_position_mean_value_pre",
        "sampled_age2plus_long_hold_mean_return_target",
        "sampled_age2plus_long_close_position_mean_return_target",
        "sampled_age2plus_long_hold_mean_value_error",
        "sampled_age2plus_long_close_position_mean_value_error",
        "sampled_age2plus_long_hold_mean_abs_value_error",
        "sampled_age2plus_long_close_position_mean_abs_value_error",
        "sampled_age2plus_long_hold_mean_squared_value_error",
        "sampled_age2plus_long_close_position_mean_squared_value_error",
        "representative_long_state_count",
        "representative_long_mean_hold_probability_pre_update",
        "representative_long_mean_close_position_probability_pre_update",
        "representative_long_mean_value_estimate_pre_update",
        "representative_long_mean_hold_probability_post_update",
        "representative_long_mean_close_position_probability_post_update",
        "representative_long_mean_value_estimate_post_update",
        "representative_long_mean_hold_probability_drift",
        "representative_long_mean_close_position_probability_drift",
        "representative_long_mean_value_estimate_drift",
        "train_approx_kl",
        "train_clip_fraction",
        "train_explained_variance",
        "train_value_loss",
        "train_entropy_loss",
        "train_policy_gradient_loss",
        "train_clip_range",
        "train_clip_range_vf",
    )
    return [{field: row.get(field) for field in fieldnames} for row in rows]


def _evaluate_representative_age1_state_bank(
    *,
    model: Any,
    representative_states: Sequence[Mapping[str, Any]],
) -> dict[str, float] | None:
    """Evaluate mean HOLD/CLOSE probabilities on a fixed representative age1 state bank."""

    if not representative_states:
        return None
    observations = np.stack(
        [np.asarray(item["observation"], dtype=np.float32) for item in representative_states],
        axis=0,
    )
    action_masks = np.stack(
        [np.asarray(item["action_mask"], dtype=np.bool_) for item in representative_states],
        axis=0,
    )
    with th.no_grad():
        observation_tensor, _ = model.policy.obs_to_tensor(observations)
        try:
            distribution = model.policy.get_distribution(observation_tensor, action_masks=action_masks)
        except TypeError:
            distribution = model.policy.get_distribution(observation_tensor)
    probabilities = extract_action_probabilities(distribution)
    if probabilities is None:
        return None
    return {
        "mean_hold_probability": float(probabilities["HOLD"]),
        "mean_close_position_probability": float(probabilities["CLOSE_POSITION"]),
    }


def _evaluate_representative_flat_state_bank(
    *,
    model: Any,
    representative_states: Sequence[Mapping[str, Any]],
) -> dict[str, float] | None:
    """Evaluate mean HOLD/OPEN probabilities on a fixed representative flat-state bank."""

    if not representative_states:
        return None
    observations = np.stack(
        [np.asarray(item["observation"], dtype=np.float32) for item in representative_states],
        axis=0,
    )
    action_masks = np.stack(
        [np.asarray(item["action_mask"], dtype=np.bool_) for item in representative_states],
        axis=0,
    )
    with th.no_grad():
        observation_tensor, _ = model.policy.obs_to_tensor(observations)
        try:
            distribution = model.policy.get_distribution(observation_tensor, action_masks=action_masks)
        except TypeError:
            distribution = model.policy.get_distribution(observation_tensor)
    probabilities = extract_action_probabilities(distribution)
    if probabilities is None:
        return None
    return {
        "mean_hold_probability": float(probabilities["HOLD"]),
        "mean_open_long_probability": float(probabilities["OPEN_LONG"]),
        "mean_open_short_probability": float(probabilities["OPEN_SHORT"]),
    }


def _evaluate_representative_long_state_bank(
    *,
    model: Any,
    representative_states: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Evaluate representative age2+ long-state probabilities and values on a fixed bank."""

    if not representative_states:
        return None
    observations = np.stack(
        [np.asarray(item["observation"], dtype=np.float32) for item in representative_states],
        axis=0,
    )
    action_masks = np.stack(
        [np.asarray(item["action_mask"], dtype=np.bool_) for item in representative_states],
        axis=0,
    )
    with th.no_grad():
        observation_tensor, _ = model.policy.obs_to_tensor(observations)
        try:
            distribution = model.policy.get_distribution(observation_tensor, action_masks=action_masks)
        except TypeError:
            distribution = model.policy.get_distribution(observation_tensor)
        value_predictions = getattr(model.policy, "predict_values", None)
        if value_predictions is None:
            value_estimates = np.full((len(representative_states),), np.nan, dtype=np.float64)
        else:
            value_tensor = value_predictions(observation_tensor)
            value_estimates = np.asarray(value_tensor.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    probabilities = extract_action_probabilities(distribution)
    if probabilities is None:
        return None
    distribution_tensor = getattr(getattr(distribution, "distribution", distribution), "probs", None)
    if distribution_tensor is None:
        return None
    probability_matrix = np.asarray(distribution_tensor.detach().cpu().numpy(), dtype=np.float64)
    if probability_matrix.ndim != 2:
        return None
    records = []
    for index, item in enumerate(representative_states):
        hold_probability = float(probability_matrix[index, ACTION_HOLD])
        close_probability = float(probability_matrix[index, ACTION_CLOSE_POSITION])
        value_estimate = float(value_estimates[index]) if np.isfinite(value_estimates[index]) else None
        records.append(
            {
                "representative_index": int(index + 1),
                "timestamp": str(item["timestamp"]),
                "position_before": int(item["position_before"]),
                "position_age_before_step": _coerce_optional_int(item.get("position_age_before_step")),
                "hold_probability": hold_probability,
                "close_position_probability": close_probability,
                "hold_minus_close_gap": float(hold_probability - close_probability),
                "value_estimate": value_estimate,
            }
        )
    return {
        "mean_hold_probability": float(probabilities["HOLD"]),
        "mean_close_position_probability": float(probabilities["CLOSE_POSITION"]),
        "mean_value_estimate": _mean_from_records(records, key="value_estimate"),
        "records": records,
    }


def _attach_age1_probability_pre_update(
    *,
    row: dict[str, Any],
    representative_probabilities: Mapping[str, Any] | None,
) -> None:
    """Attach representative pre-update HOLD/CLOSE probabilities to one rollout row."""

    if representative_probabilities is None:
        row["representative_age1_mean_hold_probability_pre_update"] = None
        row["representative_age1_mean_close_position_probability_pre_update"] = None
        return
    row["representative_age1_mean_hold_probability_pre_update"] = _coerce_optional_float(
        representative_probabilities.get("mean_hold_probability")
    )
    row["representative_age1_mean_close_position_probability_pre_update"] = _coerce_optional_float(
        representative_probabilities.get("mean_close_position_probability")
    )


def _attach_age1_probability_drift(
    *,
    row: dict[str, Any],
    representative_probabilities: Mapping[str, Any] | None,
) -> None:
    """Attach representative post-update HOLD/CLOSE probabilities and drift to one rollout row."""

    if representative_probabilities is None:
        row["representative_age1_mean_hold_probability_post_update"] = None
        row["representative_age1_mean_close_position_probability_post_update"] = None
        row["representative_age1_mean_hold_probability_drift"] = None
        row["representative_age1_mean_close_position_probability_drift"] = None
        return
    hold_post = _coerce_optional_float(representative_probabilities.get("mean_hold_probability"))
    close_post = _coerce_optional_float(representative_probabilities.get("mean_close_position_probability"))
    row["representative_age1_mean_hold_probability_post_update"] = hold_post
    row["representative_age1_mean_close_position_probability_post_update"] = close_post
    hold_pre = _coerce_optional_float(row.get("representative_age1_mean_hold_probability_pre_update"))
    close_pre = _coerce_optional_float(row.get("representative_age1_mean_close_position_probability_pre_update"))
    row["representative_age1_mean_hold_probability_drift"] = (
        float(hold_post - hold_pre) if hold_post is not None and hold_pre is not None else None
    )
    row["representative_age1_mean_close_position_probability_drift"] = (
        float(close_post - close_pre) if close_post is not None and close_pre is not None else None
    )


def _attach_flat_probability_pre_update(
    *,
    row: dict[str, Any],
    representative_probabilities: Mapping[str, Any] | None,
) -> None:
    """Attach representative flat-state HOLD/OPEN probabilities to one rollout row."""

    if representative_probabilities is None:
        row["representative_flat_mean_hold_probability_pre_update"] = None
        row["representative_flat_mean_open_long_probability_pre_update"] = None
        row["representative_flat_mean_open_short_probability_pre_update"] = None
        row["representative_flat_mean_open_long_minus_hold_pre_update"] = None
        return
    hold_pre = _coerce_optional_float(representative_probabilities.get("mean_hold_probability"))
    open_long_pre = _coerce_optional_float(representative_probabilities.get("mean_open_long_probability"))
    open_short_pre = _coerce_optional_float(representative_probabilities.get("mean_open_short_probability"))
    row["representative_flat_mean_hold_probability_pre_update"] = hold_pre
    row["representative_flat_mean_open_long_probability_pre_update"] = open_long_pre
    row["representative_flat_mean_open_short_probability_pre_update"] = open_short_pre
    row["representative_flat_mean_open_long_minus_hold_pre_update"] = (
        float(open_long_pre - hold_pre) if open_long_pre is not None and hold_pre is not None else None
    )


def _attach_flat_probability_drift(
    *,
    row: dict[str, Any],
    representative_probabilities: Mapping[str, Any] | None,
) -> None:
    """Attach representative flat-state post-update probabilities and gap drift."""

    if representative_probabilities is None:
        row["representative_flat_mean_hold_probability_post_update"] = None
        row["representative_flat_mean_open_long_probability_post_update"] = None
        row["representative_flat_mean_open_short_probability_post_update"] = None
        row["representative_flat_mean_open_long_minus_hold_post_update"] = None
        row["representative_flat_mean_open_long_minus_hold_drift"] = None
        return
    hold_post = _coerce_optional_float(representative_probabilities.get("mean_hold_probability"))
    open_long_post = _coerce_optional_float(representative_probabilities.get("mean_open_long_probability"))
    open_short_post = _coerce_optional_float(representative_probabilities.get("mean_open_short_probability"))
    row["representative_flat_mean_hold_probability_post_update"] = hold_post
    row["representative_flat_mean_open_long_probability_post_update"] = open_long_post
    row["representative_flat_mean_open_short_probability_post_update"] = open_short_post
    gap_post = float(open_long_post - hold_post) if open_long_post is not None and hold_post is not None else None
    row["representative_flat_mean_open_long_minus_hold_post_update"] = gap_post
    gap_pre = _coerce_optional_float(row.get("representative_flat_mean_open_long_minus_hold_pre_update"))
    row["representative_flat_mean_open_long_minus_hold_drift"] = (
        float(gap_post - gap_pre) if gap_post is not None and gap_pre is not None else None
    )


def _attach_long_probability_pre_update(
    *,
    row: dict[str, Any],
    representative_probabilities: Mapping[str, Any] | None,
) -> None:
    """Attach representative long-state HOLD/CLOSE probabilities to one rollout row."""

    if representative_probabilities is None:
        row["representative_long_mean_hold_probability_pre_update"] = None
        row["representative_long_mean_close_position_probability_pre_update"] = None
        row["representative_long_mean_value_estimate_pre_update"] = None
        return
    row["representative_long_mean_hold_probability_pre_update"] = _coerce_optional_float(
        representative_probabilities.get("mean_hold_probability")
    )
    row["representative_long_mean_close_position_probability_pre_update"] = _coerce_optional_float(
        representative_probabilities.get("mean_close_position_probability")
    )
    row["representative_long_mean_value_estimate_pre_update"] = _coerce_optional_float(
        representative_probabilities.get("mean_value_estimate")
    )


def _attach_long_probability_drift(
    *,
    row: dict[str, Any],
    representative_probabilities: Mapping[str, Any] | None,
) -> None:
    """Attach representative long-state post-update probabilities and drift to one rollout row."""

    if representative_probabilities is None:
        row["representative_long_mean_hold_probability_post_update"] = None
        row["representative_long_mean_close_position_probability_post_update"] = None
        row["representative_long_mean_value_estimate_post_update"] = None
        row["representative_long_mean_hold_probability_drift"] = None
        row["representative_long_mean_close_position_probability_drift"] = None
        row["representative_long_mean_value_estimate_drift"] = None
        return
    hold_post = _coerce_optional_float(representative_probabilities.get("mean_hold_probability"))
    close_post = _coerce_optional_float(representative_probabilities.get("mean_close_position_probability"))
    value_post = _coerce_optional_float(representative_probabilities.get("mean_value_estimate"))
    row["representative_long_mean_hold_probability_post_update"] = hold_post
    row["representative_long_mean_close_position_probability_post_update"] = close_post
    row["representative_long_mean_value_estimate_post_update"] = value_post
    hold_pre = _coerce_optional_float(row.get("representative_long_mean_hold_probability_pre_update"))
    close_pre = _coerce_optional_float(row.get("representative_long_mean_close_position_probability_pre_update"))
    value_pre = _coerce_optional_float(row.get("representative_long_mean_value_estimate_pre_update"))
    row["representative_long_mean_hold_probability_drift"] = (
        float(hold_post - hold_pre) if hold_post is not None and hold_pre is not None else None
    )
    row["representative_long_mean_close_position_probability_drift"] = (
        float(close_post - close_pre) if close_post is not None and close_pre is not None else None
    )
    row["representative_long_mean_value_estimate_drift"] = (
        float(value_post - value_pre) if value_post is not None and value_pre is not None else None
    )


def _attach_sb3_train_logger_metrics(*, row: dict[str, Any], model: Any | None) -> None:
    """Attach the latest SB3 PPO train() logger scalars to one rollout row."""

    logger_values = getattr(getattr(model, "logger", None), "name_to_value", None)
    for field_name, logger_key in SB3_TRAIN_LOGGER_FIELDS:
        metric_value = None
        if isinstance(logger_values, Mapping):
            raw_value = logger_values.get(logger_key)
            metric_value = _coerce_optional_float(raw_value) if raw_value is not None else None
        row[field_name] = metric_value


def _append_representative_long_state_update_trace(
    *,
    trace_rows: list[dict[str, Any]],
    update_index: int,
    representative_pre_update: Any,
    representative_post_update: Any,
) -> None:
    """Append per-state representative age2+ long probability/value drift rows."""

    pre_records = [
        dict(item)
        for item in (representative_pre_update or [])
        if isinstance(item, Mapping) and "representative_index" in item
    ]
    post_records = [
        dict(item)
        for item in (representative_post_update or [])
        if isinstance(item, Mapping) and "representative_index" in item
    ]
    if not pre_records and not post_records:
        return
    pre_by_index = {int(item["representative_index"]): item for item in pre_records}
    post_by_index = {int(item["representative_index"]): item for item in post_records}
    for representative_index in sorted(set(pre_by_index) | set(post_by_index)):
        pre_item = pre_by_index.get(representative_index)
        post_item = post_by_index.get(representative_index)
        hold_pre = _coerce_optional_float(pre_item.get("hold_probability")) if pre_item is not None else None
        close_pre = _coerce_optional_float(pre_item.get("close_position_probability")) if pre_item is not None else None
        value_pre = _coerce_optional_float(pre_item.get("value_estimate")) if pre_item is not None else None
        hold_post = _coerce_optional_float(post_item.get("hold_probability")) if post_item is not None else None
        close_post = _coerce_optional_float(post_item.get("close_position_probability")) if post_item is not None else None
        value_post = _coerce_optional_float(post_item.get("value_estimate")) if post_item is not None else None
        trace_rows.append(
            {
                "update_index": int(update_index),
                "representative_index": int(representative_index),
                "timestamp": (
                    str(pre_item.get("timestamp"))
                    if pre_item is not None
                    else str(post_item.get("timestamp")) if post_item is not None else None
                ),
                "position_age_before_step": (
                    _coerce_optional_int(pre_item.get("position_age_before_step"))
                    if pre_item is not None
                    else _coerce_optional_int(post_item.get("position_age_before_step"))
                    if post_item is not None
                    else None
                ),
                "hold_probability_pre_update": hold_pre,
                "close_position_probability_pre_update": close_pre,
                "hold_minus_close_gap_pre_update": (
                    float(hold_pre - close_pre) if hold_pre is not None and close_pre is not None else None
                ),
                "value_estimate_pre_update": value_pre,
                "hold_probability_post_update": hold_post,
                "close_position_probability_post_update": close_post,
                "hold_minus_close_gap_post_update": (
                    float(hold_post - close_post) if hold_post is not None and close_post is not None else None
                ),
                "value_estimate_post_update": value_post,
                "hold_probability_drift": (
                    float(hold_post - hold_pre) if hold_post is not None and hold_pre is not None else None
                ),
                "close_position_probability_drift": (
                    float(close_post - close_pre) if close_post is not None and close_pre is not None else None
                ),
                "value_estimate_drift": (
                    float(value_post - value_pre) if value_post is not None and value_pre is not None else None
                ),
            }
        )


def _next_position_age(*, position_before: int, position_after: int, age_before: int, done: bool) -> int:
    """Advance per-env position age for the next training decision step."""

    if done or position_after == 0:
        return 0
    if position_before == 0 and position_after in {-1, 1}:
        return 1
    if position_before in {-1, 1} and position_after == position_before:
        return max(int(age_before), 1) + 1
    if position_after in {-1, 1}:
        return 1
    return 0


def _mean_from_records(records: Sequence[Mapping[str, Any]], *, key: str) -> float | None:
    """Compute a stable mean for one numeric key across sampled records."""

    values = [_coerce_optional_float(record.get(key)) for record in records]
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


def _ordered_action_counts(action_counts: Mapping[str, int]) -> dict[str, int]:
    """Return canonical action counts with zero-fill."""

    return {semantic: int(action_counts.get(semantic, 0)) for semantic in CANONICAL_ACTION_SEMANTICS}


def _ordered_position_counts(position_counts: Mapping[str, int]) -> dict[str, int]:
    """Return canonical position-after counts with zero-fill."""

    return {key: int(position_counts.get(key, 0)) for key in CANONICAL_POSITION_KEYS}


def _ordered_transition_counts(transition_counts: Mapping[str, int]) -> dict[str, int]:
    """Return canonical position transition counts with zero-fill."""

    return {key: int(transition_counts.get(key, 0)) for key in CANONICAL_TRANSITION_KEYS}


def _build_action_reward_summary(
    *,
    reward_sums: Mapping[str, float],
    reward_counts: Mapping[str, int],
) -> dict[str, dict[str, float | int | None]]:
    """Build action-conditioned reward aggregation with stable keys."""

    summary: dict[str, dict[str, float | int | None]] = {}
    for semantic in CANONICAL_ACTION_SEMANTICS:
        count = int(reward_counts.get(semantic, 0))
        reward_sum = float(reward_sums.get(semantic, 0.0))
        reward_mean = (reward_sum / float(count)) if count > 0 else None
        summary[semantic] = {
            "count": count,
            "reward_sum": reward_sum,
            "reward_mean": reward_mean,
        }
    return summary


def _resolve_valid_action_mask(
    *,
    position_before: int,
    valid_action_mask: Sequence[bool] | np.ndarray | None,
) -> np.ndarray:
    """Resolve a canonical valid-action mask, falling back to runtime action semantics."""

    if valid_action_mask is None:
        return ExecutionEngine.valid_action_mask(position_before=position_before)
    mask = np.asarray(valid_action_mask, dtype=np.bool_).reshape(-1)
    expected = (len(CANONICAL_ACTION_ORDER),)
    if mask.shape != expected:
        raise RuntimeError(f"passivity diagnostics expected valid_action_mask shape {expected}, got {mask.shape}")
    if not bool(mask.any()):
        raise RuntimeError("passivity diagnostics require at least one valid action")
    return mask


def _build_valid_action_count_summary(valid_counts: Sequence[int], *, step_count: int) -> dict[str, Any] | None:
    """Build a stable valid-action-count summary when the data is available."""

    if step_count == 0:
        return None
    if not valid_counts:
        return {"available": False, "counts": {}, "min": None, "mean": None, "max": None}
    counts = Counter(int(value) for value in valid_counts)
    return {
        "available": True,
        "counts": {str(key): int(counts[key]) for key in sorted(counts)},
        "min": int(min(valid_counts)),
        "mean": float(sum(valid_counts) / len(valid_counts)),
        "max": int(max(valid_counts)),
    }


def _build_hold_dominance_summary(*, action_counts: Mapping[str, int]) -> dict[str, Any]:
    """Build a compact HOLD dominance summary."""

    ordered_counts = _ordered_action_counts(action_counts)
    total_count = int(sum(ordered_counts.values()))
    hold_count = int(ordered_counts["HOLD"])
    hold_share = (float(hold_count) / float(total_count)) if total_count > 0 else 0.0
    dominant_action, dominant_count = max(ordered_counts.items(), key=lambda item: (item[1], item[0]))
    dominant_share = (float(dominant_count) / float(total_count)) if total_count > 0 else 0.0
    return {
        "total_count": total_count,
        "hold_count": hold_count,
        "hold_share": hold_share,
        "non_hold_count": int(total_count - hold_count),
        "non_hold_share": (1.0 - hold_share) if total_count > 0 else 0.0,
        "dominant_action_semantic": dominant_action,
        "dominant_action_share": dominant_share,
        "all_hold": bool(total_count > 0 and hold_count == total_count),
    }


def _build_series_summary(rows: Sequence[Mapping[str, Any]], *, key: str) -> dict[str, Any]:
    """Build min/mean/max summary for a numeric rollout-point series."""

    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        raise RuntimeError(f"passivity diagnostics require non-empty series for {key}")
    if not all(math.isfinite(item) for item in values):
        raise RuntimeError(f"passivity diagnostics observed non-finite values for {key}")
    return {
        "count": int(len(values)),
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


def _build_probability_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build additive probability-dominance summary when available."""

    probability_keys = {
        semantic: f"mean_{_action_field_name(semantic)}_probability" for semantic in CANONICAL_ACTION_SEMANTICS
    }
    if any(row.get(key) is None for row in rows for key in probability_keys.values()):
        return {
            "available": False,
            "reason": "distribution_probabilities_unavailable",
            "mean_action_probabilities": None,
            "mean_hold_probability": None,
        }
    mean_probabilities = {
        semantic: float(sum(float(row[probability_keys[semantic]]) for row in rows) / len(rows))
        for semantic in CANONICAL_ACTION_SEMANTICS
    }
    return {
        "available": True,
        "reason": None,
        "mean_action_probabilities": mean_probabilities,
        "mean_hold_probability": float(mean_probabilities["HOLD"]),
    }


def _safe_mean(total: float, count: int) -> float | None:
    """Return a finite arithmetic mean or None when no observations exist."""

    if count <= 0:
        return None
    return float(total / float(count))


def _classify_hold_dominance_margin(
    *,
    fraction_of_steps_hold_is_top1: float,
    fraction_of_steps_gap_below_threshold: float,
) -> str:
    """Classify deterministic HOLD argmax dominance as narrow, wide, or inconclusive."""

    if fraction_of_steps_hold_is_top1 < 0.95:
        return "inconclusive"
    if fraction_of_steps_gap_below_threshold >= 0.50:
        return "narrow"
    return "wide"


def _action_field_name(semantic: str) -> str:
    """Normalize canonical action semantic into snake-case field suffix."""

    return semantic.lower()


def _coerce_position(value: Any, *, field_name: str) -> int:
    """Coerce a position value into the canonical {-1,0,1} domain."""

    if not isinstance(value, int) or isinstance(value, bool) or value not in CANONICAL_POSITION_VALUES:
        raise RuntimeError(f"passivity diagnostics require {field_name} in {-1,0,1}; got {value!r}")
    return int(value)


def _coerce_optional_int(value: Any, *, default: int | None = None) -> int | None:
    """Coerce optional integer-valued metric fields."""

    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"passivity diagnostics expected optional int, got {value!r}")
    return int(value)


def _coerce_optional_float(value: Any) -> float | None:
    """Coerce optional float-valued metric fields."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RuntimeError(f"passivity diagnostics expected optional float, got {value!r}")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise RuntimeError(f"passivity diagnostics expected finite float, got {value!r}")
    return value_float


def _coerce_probability_scalar(value: Any, *, field_name: str) -> float:
    """Coerce one probability-like scalar into a finite [0,1] float."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RuntimeError(f"passivity diagnostics expected numeric {field_name}, got {value!r}")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise RuntimeError(f"passivity diagnostics expected finite {field_name}, got {value!r}")
    if value_float < 0.0 or value_float > 1.0:
        raise RuntimeError(f"passivity diagnostics expected {field_name} in [0,1], got {value!r}")
    return value_float


def _subtract_optional_numbers(left: Any, right: Any) -> float | int | None:
    """Subtract optional scalar diagnostics values when both sides are present."""

    left_value = _coerce_optional_float(left)
    right_value = _coerce_optional_float(right)
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _generated_at() -> str:
    """Return an RFC3339 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()
