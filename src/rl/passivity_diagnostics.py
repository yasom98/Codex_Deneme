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
)

PASSIVITY_DIAGNOSTICS_REPORT_FILENAME = "passivity_diagnostics_report.json"
TRAINING_ENTROPY_TIMESERIES_FILENAME = "training_entropy_timeseries.csv"
DETERMINISTIC_ACTION_RANKING_TRACE_FILENAME = "deterministic_action_ranking_trace.csv"
TRAINING_PASSIVITY_DIAGNOSTICS_CONTRACT_VERSION = "training_passivity_diagnostics.v1"
EVALUATION_PASSIVITY_DIAGNOSTICS_CONTRACT_VERSION = "evaluation_passivity_diagnostics.v2"
DETERMINISTIC_HOLD_GAP_THRESHOLD = 0.10

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


@dataclass(frozen=True)
class TrainingPassivityDiagnosticsArtifacts:
    """Serialized training diagnostics artifacts."""

    report_payload: dict[str, Any]
    entropy_timeseries_csv: str


def build_training_passivity_diagnostics_callback() -> Any:
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

        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            rewards = self.locals.get("rewards")
            if infos is None or rewards is None:
                raise RuntimeError("passivity diagnostics require infos and rewards in callback locals")

            reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
            if not np.isfinite(reward_values).all():
                raise RuntimeError("passivity diagnostics observed non-finite rewards during training")

            info_items = list(infos)
            if len(info_items) != int(reward_values.shape[0]):
                raise RuntimeError("passivity diagnostics reward/info cardinality mismatch during training")

            for info, reward_value in zip(info_items, reward_values, strict=True):
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

                self._action_counts[semantic] += 1
                self._action_reward_sums[semantic] += float(reward_value)
                self._action_reward_counts[semantic] += 1
                self._position_transition_counts[transition_key] += 1
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
            self._rollout_rows.append(row)

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

            report_payload = {
                "contract_version": TRAINING_PASSIVITY_DIAGNOSTICS_CONTRACT_VERSION,
                "run_id": run_id,
                "production_session_id": production_session_id,
                "action_masking_enabled": bool(action_masking_enabled),
                "selected_episode_ref": dict(selected_episode_ref) if selected_episode_ref is not None else None,
                "total_timesteps_requested": int(total_timesteps_requested),
                "num_timesteps_after_learn": int(num_timesteps_after_learn),
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
                "policy_entropy_summary": _build_series_summary(self._rollout_rows, key="mean_policy_entropy"),
                "mean_action_probability_summary": _build_probability_summary(self._rollout_rows),
                "generated_at_utc": _generated_at(),
            }
            return TrainingPassivityDiagnosticsArtifacts(
                report_payload=report_payload,
                entropy_timeseries_csv=_build_training_entropy_timeseries_csv(self._rollout_rows),
            )

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
    deterministic_summary: Mapping[str, Any],
    stochastic_summary: Mapping[str, Any],
    deterministic_action_ranking_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable deterministic vs stochastic comparison report."""

    deterministic_payload = dict(deterministic_summary)
    if deterministic_action_ranking_summary is not None:
        deterministic_payload["action_ranking_summary"] = dict(deterministic_action_ranking_summary)
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
    gap_threshold: float = DETERMINISTIC_HOLD_GAP_THRESHOLD,
) -> dict[str, Any]:
    """Build one deterministic eval ranking row from canonical action probabilities."""

    if selected_action_semantic not in CANONICAL_ACTION_SEMANTICS:
        raise RuntimeError(f"unsupported selected_action_semantic for ranking diagnostics: {selected_action_semantic}")
    if not isinstance(timestamp, str) or not timestamp:
        raise RuntimeError("ranking diagnostics require non-empty timestamp strings")
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
    top_1_semantic = ranked_semantics[0]
    top_2_semantic = ranked_semantics[1]
    top_1_probability = canonical_probabilities[top_1_semantic]
    top_2_probability = canonical_probabilities[top_2_semantic]
    gap = top_1_probability - top_2_probability
    hold_is_top_1 = top_1_semantic == "HOLD"
    hold_gap_below_threshold = bool(hold_is_top_1 and gap < gap_threshold_value)
    hold_gap_band = "small" if hold_gap_below_threshold else "large" if hold_is_top_1 else "not_hold_top_1"

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
        "selected_action_semantic": selected_action_semantic,
        "top_1_action_semantic": top_1_semantic,
        "top_1_probability": top_1_probability,
        "top_2_action_semantic": top_2_semantic,
        "top_2_probability": top_2_probability,
        "top_1_minus_top_2_gap": gap,
        "hold_is_top_1": bool(hold_is_top_1),
        "hold_gap_below_threshold": hold_gap_below_threshold,
        "hold_gap_band": hold_gap_band,
        "selected_matches_top_1": bool(selected_action_semantic == top_1_semantic),
        "prob_hold": canonical_probabilities["HOLD"],
        "prob_open_long": canonical_probabilities["OPEN_LONG"],
        "prob_open_short": canonical_probabilities["OPEN_SHORT"],
        "prob_close_position": canonical_probabilities["CLOSE_POSITION"],
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
        "selected_action_semantic",
        "top_1_action_semantic",
        "top_1_probability",
        "top_2_action_semantic",
        "top_2_probability",
        "top_1_minus_top_2_gap",
        "hold_is_top_1",
        "hold_gap_below_threshold",
        "hold_gap_band",
        "selected_matches_top_1",
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
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in fieldnames})
    return buffer.getvalue()


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
