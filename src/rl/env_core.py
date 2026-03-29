"""Framework-agnostic RL environment core for Milestone 4.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

ACTION_HOLD = 0
ACTION_OPEN_LONG = 1
ACTION_OPEN_SHORT = 2
ACTION_CLOSE_POSITION = 3

ACTION_MAPPING: dict[int, str] = {
    ACTION_HOLD: "HOLD",
    ACTION_OPEN_LONG: "OPEN_LONG",
    ACTION_OPEN_SHORT: "OPEN_SHORT",
    ACTION_CLOSE_POSITION: "CLOSE_POSITION",
}

POSITION_PLACEHOLDER_COLUMN = "position_placeholder"
FLOATING_PNL_PLACEHOLDER_COLUMN = "floating_pnl_placeholder"
DRAWDOWN_PLACEHOLDER_COLUMN = "drawdown_placeholder"
HOLDING_AGE_IN_POSITION_COLUMN = "holding_age_in_position"
ENTRY_RETURN_BPS_COLUMN = "entry_return_bps"


@dataclass(frozen=True)
class EpisodeRef:
    """Stable episode selector for env reset."""

    scope: str
    partition: str
    source_rel: str
    fold_id: int | None

    def __post_init__(self) -> None:
        if self.scope not in {"partition", "fold"}:
            raise ValueError("episode_ref.scope must be one of {partition, fold}")
        if self.partition not in {"train", "val", "test"}:
            raise ValueError("episode_ref.partition must be one of {train, val, test}")
        if not self.source_rel.strip():
            raise ValueError("episode_ref.source_rel must be non-empty")
        if self.scope == "partition" and self.fold_id is not None:
            raise ValueError("episode_ref.fold_id must be null when scope=partition")
        if self.scope == "fold":
            if not isinstance(self.fold_id, int) or self.fold_id < 0:
                raise ValueError("episode_ref.fold_id must be >=0 integer when scope=fold")


@dataclass(frozen=True)
class EpisodeSpec:
    """Resolved episode artifact metadata."""

    scope: str
    partition: str
    source_rel: str
    fold_id: int | None
    output_path: Path
    row_count: int

    def key(self) -> tuple[str, str, str, int | None]:
        """Return deterministic key for catalog lookup."""

        return (self.scope, self.partition, self.source_rel, self.fold_id)


class EpisodeCatalog:
    """In-memory catalog of available episodes."""

    def __init__(self, entries: Sequence[EpisodeSpec]) -> None:
        self._entries = list(entries)
        self._index = {item.key(): item for item in self._entries}

    @property
    def entries(self) -> list[EpisodeSpec]:
        """Return all episode entries."""

        return list(self._entries)

    def find_episode(self, ref: EpisodeRef) -> EpisodeSpec | None:
        """Find episode matching the supplied selector."""

        return self._index.get((ref.scope, ref.partition, ref.source_rel, ref.fold_id))

    @classmethod
    def from_manifest(cls, manifest_payload: Mapping[str, Any]) -> EpisodeCatalog:
        """Construct catalog from state manifest payload."""

        entries_raw = manifest_payload.get("partition_metadata")
        if not isinstance(entries_raw, list):
            raise ValueError("state_manifest.partition_metadata must be list")

        entries: list[EpisodeSpec] = []
        for item in entries_raw:
            if not isinstance(item, dict):
                raise ValueError("state_manifest.partition_metadata entries must be objects")
            scope = str(item.get("scope", "")).strip()
            partition = str(item.get("partition", "")).strip()
            source_rel = str(item.get("source_rel", "")).strip()
            fold_id_raw = item.get("fold_id")
            output_path_raw = item.get("output_path")
            row_count_raw = item.get("row_count")

            if scope not in {"partition", "fold", "aggregate"}:
                raise ValueError("state_manifest.partition_metadata.scope is invalid")
            if partition not in {"train", "val", "test"}:
                raise ValueError("state_manifest.partition_metadata.partition is invalid")
            if not source_rel:
                raise ValueError("state_manifest.partition_metadata.source_rel is required")
            if not isinstance(output_path_raw, str) or not output_path_raw.strip():
                raise ValueError("state_manifest.partition_metadata.output_path is required")
            if not isinstance(row_count_raw, int) or row_count_raw < 0:
                raise ValueError("state_manifest.partition_metadata.row_count must be non-negative integer")

            fold_id: int | None
            if scope == "fold":
                if not isinstance(fold_id_raw, int) or fold_id_raw < 0:
                    raise ValueError("fold scope requires fold_id >= 0")
                fold_id = int(fold_id_raw)
            else:
                fold_id = None

            entries.append(
                EpisodeSpec(
                    scope=scope,
                    partition=partition,
                    source_rel=source_rel,
                    fold_id=fold_id,
                    output_path=Path(output_path_raw).resolve(),
                    row_count=int(row_count_raw),
                )
            )
        return cls(entries)


@dataclass(frozen=True)
class EpisodeData:
    """In-memory episode arrays for deterministic stepping."""

    observation_matrix: np.ndarray
    execution_price_vector: np.ndarray
    mark_to_market_price_vector: np.ndarray
    timestamp_vector: tuple[str, ...]
    observation_columns: tuple[str, ...]
    execution_price_column: str
    mark_to_market_column: str
    coercions_applied: tuple[dict[str, str], ...]
    episode_valid_start_row: int = 0
    warmup_applied: bool = False

    def __post_init__(self) -> None:
        if self.observation_matrix.ndim != 2:
            raise ValueError("observation_matrix must be 2-dimensional")
        if self.execution_price_vector.ndim != 1:
            raise ValueError("execution_price_vector must be 1-dimensional")
        if self.mark_to_market_price_vector.ndim != 1:
            raise ValueError("mark_to_market_price_vector must be 1-dimensional")
        if len(self.timestamp_vector) != int(self.observation_matrix.shape[0]):
            raise ValueError("timestamp_vector length must match observation rows")
        if int(self.observation_matrix.shape[0]) != int(self.execution_price_vector.shape[0]):
            raise ValueError("execution_price_vector length must match observation rows")
        if int(self.observation_matrix.shape[0]) != int(self.mark_to_market_price_vector.shape[0]):
            raise ValueError("mark_to_market_price_vector length must match observation rows")
        if int(self.observation_matrix.shape[0]) < 2:
            raise ValueError("episode must contain at least 2 rows")
        if int(self.observation_matrix.shape[1]) < 1:
            raise ValueError("episode must contain at least 1 observation feature")
        if int(self.episode_valid_start_row) < 0:
            raise ValueError("episode_valid_start_row must be >= 0")
        if int(self.episode_valid_start_row) >= int(self.observation_matrix.shape[0]):
            raise ValueError("episode_valid_start_row must be within observation rows")
        if int(self.observation_matrix.shape[0]) - int(self.episode_valid_start_row) < 2:
            raise ValueError("episode must contain at least 2 rows after warmup")


class EpisodeSource:
    """Episode loader that performs strict column and dtype validation."""

    def __init__(self, read_parquet_fn: Callable[[Path], pd.DataFrame] | None = None) -> None:
        self._read_parquet = read_parquet_fn if read_parquet_fn is not None else pd.read_parquet

    def load_episode(
        self,
        *,
        spec: EpisodeSpec,
        expected_columns: Sequence[str],
        observation_columns: Sequence[str],
        strict_post_valid_numeric_columns: Sequence[str],
        expected_dtypes: Mapping[str, str],
        timestamp_column: str,
        execution_price_column: str,
        mark_to_market_column: str,
        include_timestamp_in_observation: bool,
        observation_output_dtype: str,
        allowed_safe_casts: set[str],
        valid_observation_start_row: int,
        valid_observation_start_timestamp: str | None,
        warmup_head_nan_profile: Mapping[str, int],
    ) -> EpisodeData:
        """Load one episode and return validated in-memory arrays."""

        frame = self._read_parquet(spec.output_path)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("EPISODE_LOAD_FAILED: parquet payload is not dataframe")

        actual_columns = list(frame.columns)
        expected_columns_list = list(expected_columns)
        if actual_columns != expected_columns_list:
            raise ValueError("OBSERVATION_COLUMN_ORDER_MISMATCH")

        for col in expected_columns_list:
            expected_dtype = expected_dtypes.get(col)
            actual_dtype = str(frame[col].dtype)
            if expected_dtype is None or expected_dtype != actual_dtype:
                raise ValueError(f"OBSERVATION_DTYPE_MISMATCH:{col}:{expected_dtype}:{actual_dtype}")

        if execution_price_column not in frame.columns:
            raise ValueError("EXECUTION_PRICE_COLUMN_MISSING")
        if mark_to_market_column not in frame.columns:
            raise ValueError("MARK_TO_MARKET_COLUMN_MISSING")
        if timestamp_column not in frame.columns:
            raise ValueError("TIMESTAMP_COLUMN_MISSING")

        parsed_ts = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        if parsed_ts.isna().any():
            raise ValueError("TIMESTAMP_PARSE_FAILED")
        if not parsed_ts.is_monotonic_increasing:
            raise ValueError("TIMESTAMP_ORDERING_VIOLATION")
        duplicate_count = int(parsed_ts.duplicated().sum())
        if duplicate_count > 0:
            raise ValueError("TIMESTAMP_DUPLICATES")

        if include_timestamp_in_observation:
            raise ValueError("TIMESTAMP_IN_OBSERVATION_UNSUPPORTED")

        obs_columns = [str(col) for col in observation_columns]
        strict_columns = [str(col) for col in strict_post_valid_numeric_columns]
        if not obs_columns:
            raise ValueError("OBSERVATION_COLUMNS_EMPTY")
        if any(col not in frame.columns for col in obs_columns):
            raise ValueError("OBSERVATION_COLUMN_ORDER_MISMATCH")
        if timestamp_column in obs_columns:
            raise ValueError("TIMESTAMP_IN_OBSERVATION_UNSUPPORTED")
        if not strict_columns or strict_columns != list(dict.fromkeys(strict_columns)):
            raise ValueError("OBSERVATION_STRICT_COLUMNS_MISMATCH")
        if any(col not in obs_columns for col in strict_columns):
            raise ValueError("OBSERVATION_STRICT_COLUMNS_MISMATCH")

        if int(valid_observation_start_row) < 0 or int(valid_observation_start_row) > len(frame):
            raise ValueError(f"EFFECTIVE_START_INVALID:{valid_observation_start_row}:{len(frame)}")
        if int(valid_observation_start_row) < len(frame):
            derived_valid_start_timestamp = pd.Timestamp(parsed_ts.iloc[int(valid_observation_start_row)]).isoformat()
        else:
            derived_valid_start_timestamp = None
        if derived_valid_start_timestamp != valid_observation_start_timestamp:
            raise ValueError(
                f"EFFECTIVE_START_INVALID:{valid_observation_start_row}:{valid_observation_start_timestamp}:{derived_valid_start_timestamp}"
            )
        if len(frame) - int(valid_observation_start_row) < 2:
            raise ValueError(f"EPISODE_TOO_SHORT_AFTER_WARMUP:{valid_observation_start_row}:{len(frame)}")

        output_dtype = np.dtype(observation_output_dtype)
        obs_vectors: list[np.ndarray] = []
        coercions: list[dict[str, str]] = []
        strict_columns_set = set(strict_columns)

        for col in obs_columns:
            source_dtype = str(frame[col].dtype)
            target_dtype = output_dtype.name
            if source_dtype != target_dtype:
                cast_tag = f"{source_dtype}->{target_dtype}"
                if cast_tag not in allowed_safe_casts:
                    raise ValueError(f"OBS_CAST_NOT_ALLOWED:{col}:{cast_tag}")
                coercions.append(
                    {
                        "column": col,
                        "from_dtype": source_dtype,
                        "to_dtype": target_dtype,
                        "cast_tag": cast_tag,
                    }
                )
            numeric = pd.to_numeric(frame[col], errors="coerce")
            if col in strict_columns_set:
                finite_mask = np.isfinite(numeric.astype("float64", copy=False).to_numpy())
                allowed_head_non_finite = int(warmup_head_nan_profile.get(col, 0))
                if allowed_head_non_finite < 0 or allowed_head_non_finite > len(frame):
                    raise ValueError(f"EFFECTIVE_START_INVALID:{col}:{allowed_head_non_finite}:{len(frame)}")
                if allowed_head_non_finite > 0 and not bool((~finite_mask[:allowed_head_non_finite]).all()):
                    raise ValueError(f"EFFECTIVE_START_INVALID:{col}:{allowed_head_non_finite}")
                invalid_non_finite = np.flatnonzero(~finite_mask & (np.arange(len(frame)) >= allowed_head_non_finite))
                if invalid_non_finite.size > 0:
                    raise ValueError(f"POST_VALID_OBSERVATION_NAN:{col}:{int(invalid_non_finite[0])}")
            obs_vectors.append(numeric.to_numpy(dtype=output_dtype, copy=True))

        observation_matrix = np.column_stack(obs_vectors).astype(output_dtype, copy=False)
        execution_price_numeric = pd.to_numeric(frame[execution_price_column], errors="coerce")
        if execution_price_numeric.isna().any():
            raise ValueError("EXECUTION_PRICE_PARSE_FAILED")
        mark_to_market_numeric = pd.to_numeric(frame[mark_to_market_column], errors="coerce")
        if mark_to_market_numeric.isna().any():
            raise ValueError("MARK_TO_MARKET_PRICE_PARSE_FAILED")

        timestamps = tuple(pd.Timestamp(item).isoformat() for item in parsed_ts)
        return EpisodeData(
            observation_matrix=observation_matrix,
            execution_price_vector=execution_price_numeric.to_numpy(dtype=np.float64, copy=True),
            mark_to_market_price_vector=mark_to_market_numeric.to_numpy(dtype=np.float64, copy=True),
            timestamp_vector=timestamps,
            observation_columns=tuple(obs_columns),
            execution_price_column=execution_price_column,
            mark_to_market_column=mark_to_market_column,
            coercions_applied=tuple(coercions),
            episode_valid_start_row=int(valid_observation_start_row),
            warmup_applied=bool(int(valid_observation_start_row) > 0),
        )


@dataclass(frozen=True)
class PositionState:
    """Position exposure state."""

    exposure: int

    def __post_init__(self) -> None:
        if self.exposure not in {-1, 0, 1}:
            raise ValueError("exposure must be one of {-1, 0, +1}")


@dataclass(frozen=True)
class PortfolioState:
    """Portfolio accounting state."""

    portfolio_value: float


@dataclass(frozen=True)
class ExecutionResult:
    """Deterministic execution outcome for a single action."""

    action_raw: int
    action_semantic: str
    position_before: int
    position_after: int
    trade_units: int
    invalid_action: bool
    invalid_action_reason: str | None


class ExecutionEngine:
    """Deterministic V1 action-to-position transition engine."""

    @staticmethod
    def valid_action_mask(*, position_before: int) -> np.ndarray:
        """Return the current valid-action mask under existing runtime rules."""

        if position_before == 0:
            mask = np.asarray([True, True, True, False], dtype=np.bool_)
        elif position_before in {-1, 1}:
            mask = np.asarray([True, False, False, True], dtype=np.bool_)
        else:
            raise ValueError("position_before must be one of {-1, 0, +1}")

        if mask.shape != (len(ACTION_MAPPING),) or not bool(mask.any()):
            raise ValueError("valid_action_mask must expose at least one valid action in canonical action order")
        return mask

    @staticmethod
    def apply_action(*, position_before: int, action_raw: int) -> ExecutionResult:
        """Apply one discrete action under V1 no-reversal policy."""

        if action_raw not in ACTION_MAPPING:
            raise ValueError(f"Unsupported action id: {action_raw}")

        semantic = ACTION_MAPPING[action_raw]
        position_after = position_before
        trade_units = 0
        invalid_action = False
        invalid_reason: str | None = None

        if action_raw == ACTION_HOLD:
            pass
        elif action_raw == ACTION_OPEN_LONG:
            if position_before == 0:
                position_after = 1
                trade_units = 1
            elif position_before == 1:
                invalid_action = True
                invalid_reason = "already_long"
            else:
                invalid_action = True
                invalid_reason = "reversal_disallowed"
        elif action_raw == ACTION_OPEN_SHORT:
            if position_before == 0:
                position_after = -1
                trade_units = 1
            elif position_before == -1:
                invalid_action = True
                invalid_reason = "already_short"
            else:
                invalid_action = True
                invalid_reason = "reversal_disallowed"
        elif action_raw == ACTION_CLOSE_POSITION:
            if position_before == 0:
                invalid_action = True
                invalid_reason = "already_flat"
            else:
                position_after = 0
                trade_units = 1

        return ExecutionResult(
            action_raw=action_raw,
            action_semantic=semantic,
            position_before=position_before,
            position_after=position_after,
            trade_units=trade_units,
            invalid_action=invalid_action,
            invalid_action_reason=invalid_reason,
        )


@dataclass(frozen=True)
class RewardBreakdown:
    """Reward components for deterministic accounting."""

    pnl_delta: float
    position_mtm_contribution: float
    realized_pnl_contribution: float
    close_realized_pnl_bonus_contribution: float
    benchmark_relative_contribution: float
    fees: float
    slippage_cost: float
    risk_penalty: float
    inactivity_penalty: float
    invalid_close_flat_penalty: float
    reward_raw: float
    reward_total: float


@dataclass(frozen=True)
class DensePositionRewardConfig:
    """Explicit additive config for dense position-based reward shaping."""

    position_mtm_coefficient: float
    fee_coefficient: float
    slippage_coefficient: float
    risk_penalty_coefficient: float
    inactivity_penalty: float
    close_position_mtm_coefficient: float = 1.0
    close_position_mtm_holding_ramp_steps: int = 0
    close_position_mtm_unlock_after_steps: int = 0
    close_realized_pnl_coefficient: float = 0.0
    close_realized_pnl_bonus_unlock_after_steps: int = 0
    min_holding_steps_for_close_bonus: int = 0
    close_bonus_holding_ramp_steps: int = 0
    close_bonus_pnl_cap_abs: float | None = None
    benchmark_mode: str = "none"
    benchmark_relative_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if float(self.position_mtm_coefficient) <= 0.0:
            raise ValueError("position_mtm_coefficient must be > 0")
        if float(self.close_position_mtm_coefficient) < 0.0:
            raise ValueError("close_position_mtm_coefficient must be >= 0")
        if float(self.fee_coefficient) < 0.0:
            raise ValueError("fee_coefficient must be >= 0")
        if float(self.slippage_coefficient) < 0.0:
            raise ValueError("slippage_coefficient must be >= 0")
        if float(self.risk_penalty_coefficient) < 0.0:
            raise ValueError("risk_penalty_coefficient must be >= 0")
        if float(self.inactivity_penalty) < 0.0:
            raise ValueError("inactivity_penalty must be >= 0")
        if isinstance(self.close_position_mtm_holding_ramp_steps, bool) or not isinstance(
            self.close_position_mtm_holding_ramp_steps,
            int,
        ):
            raise ValueError("close_position_mtm_holding_ramp_steps must be int")
        if int(self.close_position_mtm_holding_ramp_steps) < 0:
            raise ValueError("close_position_mtm_holding_ramp_steps must be >= 0")
        if isinstance(self.close_position_mtm_unlock_after_steps, bool) or not isinstance(
            self.close_position_mtm_unlock_after_steps,
            int,
        ):
            raise ValueError("close_position_mtm_unlock_after_steps must be int")
        if int(self.close_position_mtm_unlock_after_steps) < 0:
            raise ValueError("close_position_mtm_unlock_after_steps must be >= 0")
        if float(self.close_realized_pnl_coefficient) < 0.0:
            raise ValueError("close_realized_pnl_coefficient must be >= 0")
        if isinstance(self.close_realized_pnl_bonus_unlock_after_steps, bool) or not isinstance(
            self.close_realized_pnl_bonus_unlock_after_steps,
            int,
        ):
            raise ValueError("close_realized_pnl_bonus_unlock_after_steps must be int")
        if int(self.close_realized_pnl_bonus_unlock_after_steps) < 0:
            raise ValueError("close_realized_pnl_bonus_unlock_after_steps must be >= 0")
        if isinstance(self.min_holding_steps_for_close_bonus, bool) or not isinstance(
            self.min_holding_steps_for_close_bonus,
            int,
        ):
            raise ValueError("min_holding_steps_for_close_bonus must be int")
        if int(self.min_holding_steps_for_close_bonus) < 0:
            raise ValueError("min_holding_steps_for_close_bonus must be >= 0")
        if isinstance(self.close_bonus_holding_ramp_steps, bool) or not isinstance(
            self.close_bonus_holding_ramp_steps,
            int,
        ):
            raise ValueError("close_bonus_holding_ramp_steps must be int")
        if int(self.close_bonus_holding_ramp_steps) < 0:
            raise ValueError("close_bonus_holding_ramp_steps must be >= 0")
        if self.close_bonus_pnl_cap_abs is not None and float(self.close_bonus_pnl_cap_abs) <= 0.0:
            raise ValueError("close_bonus_pnl_cap_abs must be > 0 when provided")
        if self.benchmark_mode not in {"none", "buy_and_hold"}:
            raise ValueError("benchmark_mode must be one of {none, buy_and_hold}")
        if float(self.benchmark_relative_coefficient) < 0.0:
            raise ValueError("benchmark_relative_coefficient must be >= 0")


class RewardEngine:
    """Reward decomposition contract for Milestone 4.5 v1/v2."""

    @staticmethod
    def compute_reward(
        *,
        reward_version: str,
        position_before: int,
        action_raw: int,
        invalid_action: bool,
        invalid_action_reason: str | None,
        position_after: int,
        price_exec: float,
        price_next: float,
        trade_units: int,
        fee_bps: float,
        slippage_bps: float,
        invalid_close_flat_penalty: float,
        reward_scale: float,
        reward_clip_min: float | None,
        reward_clip_max: float | None,
        dense_pbr_config: DensePositionRewardConfig | None = None,
        entry_price_exec: float | None = None,
        entry_fees: float = 0.0,
        entry_slippage_cost: float = 0.0,
        holding_steps_before_close: int | None = None,
        close_accounting_position_after: int | None = None,
    ) -> RewardBreakdown:
        """Compute deterministic reward components."""

        fees = abs(int(trade_units)) * float(price_exec) * (float(fee_bps) / 10_000.0)
        slippage_cost = abs(int(trade_units)) * float(price_exec) * (float(slippage_bps) / 10_000.0)
        effective_close_accounting_position_after = (
            int(position_after)
            if close_accounting_position_after is None
            else int(close_accounting_position_after)
        )
        realized_pnl_contribution = 0.0
        applied_invalid_close_flat_penalty = 0.0
        if (
            int(action_raw) == ACTION_CLOSE_POSITION
            and bool(invalid_action)
            and invalid_action_reason == "already_flat"
        ):
            applied_invalid_close_flat_penalty = float(invalid_close_flat_penalty)
        elif (
            int(action_raw) == ACTION_CLOSE_POSITION
            and int(position_before) in {-1, 1}
            and effective_close_accounting_position_after == 0
        ):
            if entry_price_exec is None:
                raise ValueError("entry_price_exec is required when closing a non-flat position")
            realized_pnl_contribution = (
                float(position_before) * (float(price_exec) - float(entry_price_exec))
                - float(entry_fees)
                - float(entry_slippage_cost)
                - float(fees)
                - float(slippage_cost)
            )

        if reward_version == "reward.v2_dense_pbr":
            if dense_pbr_config is None:
                raise ValueError("dense_pbr_config is required when reward_version=reward.v2_dense_pbr")
            effective_close_mtm_coefficient = float(dense_pbr_config.close_position_mtm_coefficient)
            close_step_is_active = (
                int(action_raw) == ACTION_CLOSE_POSITION
                and int(position_before) in {-1, 1}
                and int(position_after) == 0
            )
            if not close_step_is_active:
                effective_close_mtm_coefficient = 1.0
            else:
                effective_holding_steps = 0 if holding_steps_before_close is None else int(holding_steps_before_close)
                effective_holding_steps = max(
                    effective_holding_steps - int(dense_pbr_config.close_position_mtm_unlock_after_steps),
                    0,
                )
            if close_step_is_active and int(dense_pbr_config.close_position_mtm_holding_ramp_steps) > 0:
                mtm_ramp_ratio = min(
                    float(effective_holding_steps) / float(dense_pbr_config.close_position_mtm_holding_ramp_steps),
                    1.0,
                )
                effective_close_mtm_coefficient *= max(mtm_ramp_ratio, 0.0)
            position_mtm_contribution = (
                float(position_before)
                * (float(price_next) - float(price_exec))
                * float(dense_pbr_config.position_mtm_coefficient)
                * effective_close_mtm_coefficient
            )
            risk_penalty = abs(int(position_after)) * float(dense_pbr_config.risk_penalty_coefficient)
            inactivity_penalty = (
                float(dense_pbr_config.inactivity_penalty)
                if int(position_before) == 0 and int(position_after) == 0
                else 0.0
            )
            close_realized_pnl_bonus_contribution = 0.0
            if (
                int(action_raw) == ACTION_CLOSE_POSITION
                and int(position_before) in {-1, 1}
                and effective_close_accounting_position_after == 0
            ):
                close_bonus_gate_enabled = int(dense_pbr_config.min_holding_steps_for_close_bonus) > 0
                close_bonus_is_eligible = True
                effective_holding_steps = 0 if holding_steps_before_close is None else int(holding_steps_before_close)
                realized_bonus_unlock_after = int(dense_pbr_config.close_realized_pnl_bonus_unlock_after_steps)
                adjusted_bonus_holding_steps = max(effective_holding_steps - realized_bonus_unlock_after, 0)
                if realized_bonus_unlock_after > 0 and effective_holding_steps <= realized_bonus_unlock_after:
                    close_bonus_is_eligible = False
                if close_bonus_gate_enabled:
                    close_bonus_is_eligible = (
                        close_bonus_is_eligible
                        and
                        float(realized_pnl_contribution) > 0.0
                        and effective_holding_steps >= int(dense_pbr_config.min_holding_steps_for_close_bonus)
                    )
                if close_bonus_is_eligible:
                    bonus_basis = float(realized_pnl_contribution)
                    if dense_pbr_config.close_bonus_pnl_cap_abs is not None:
                        bonus_basis = float(
                            np.clip(
                                bonus_basis,
                                -float(dense_pbr_config.close_bonus_pnl_cap_abs),
                                float(dense_pbr_config.close_bonus_pnl_cap_abs),
                            )
                        )
                    if int(dense_pbr_config.close_bonus_holding_ramp_steps) > 0:
                        ramp_ratio = min(
                            float(adjusted_bonus_holding_steps)
                            / float(dense_pbr_config.close_bonus_holding_ramp_steps),
                            1.0,
                        )
                        bonus_basis = float(bonus_basis) * max(ramp_ratio, 0.0)
                    close_realized_pnl_bonus_contribution = (
                        float(bonus_basis) * float(dense_pbr_config.close_realized_pnl_coefficient)
                    )
            fee_penalty = fees * float(dense_pbr_config.fee_coefficient)
            slippage_penalty = slippage_cost * float(dense_pbr_config.slippage_coefficient)
            benchmark_relative_contribution = 0.0
            if (
                str(dense_pbr_config.benchmark_mode) == "buy_and_hold"
                and float(dense_pbr_config.benchmark_relative_coefficient) > 0.0
            ):
                benchmark_delta = float(price_next) - float(price_exec)
                benchmark_relative_contribution = (
                    -float(benchmark_delta) * float(dense_pbr_config.benchmark_relative_coefficient)
                )
            pnl_delta = float(position_mtm_contribution)
            reward_raw = (
                pnl_delta
                - fee_penalty
                - slippage_penalty
                - risk_penalty
                - inactivity_penalty
                - applied_invalid_close_flat_penalty
                + close_realized_pnl_bonus_contribution
                + benchmark_relative_contribution
            )
        else:
            pnl_delta = float(position_after) * (float(price_next) - float(price_exec))
            position_mtm_contribution = float(pnl_delta)
            risk_penalty = 0.0
            inactivity_penalty = 0.0
            close_realized_pnl_bonus_contribution = 0.0
            benchmark_relative_contribution = 0.0
            reward_raw = pnl_delta - fees - slippage_cost - applied_invalid_close_flat_penalty

        reward_total = reward_raw * float(reward_scale)
        if reward_clip_min is not None or reward_clip_max is not None:
            reward_total = float(np.clip(reward_total, reward_clip_min, reward_clip_max))

        return RewardBreakdown(
            pnl_delta=float(pnl_delta),
            position_mtm_contribution=float(position_mtm_contribution),
            realized_pnl_contribution=float(realized_pnl_contribution),
            close_realized_pnl_bonus_contribution=float(close_realized_pnl_bonus_contribution),
            benchmark_relative_contribution=float(benchmark_relative_contribution),
            fees=float(fees),
            slippage_cost=float(slippage_cost),
            risk_penalty=float(risk_penalty),
            inactivity_penalty=float(inactivity_penalty),
            invalid_close_flat_penalty=float(applied_invalid_close_flat_penalty),
            reward_raw=float(reward_raw),
            reward_total=float(reward_total),
        )


@dataclass(frozen=True)
class EpisodeRunnerConfig:
    """Config for deterministic episode stepping."""

    initial_cash: float
    fee_bps: float
    slippage_bps: float
    max_steps: int | None
    reward_scale: float
    reward_clip_min: float | None
    reward_clip_max: float | None
    invalid_close_flat_penalty: float
    seed: int | None
    reward_version: str = "reward.v1"
    dense_pbr_config: DensePositionRewardConfig | None = None
    close_position_transition_timing_policy: str = "flatten_before_interval"

    def __post_init__(self) -> None:
        if float(self.initial_cash) <= 0.0:
            raise ValueError("initial_cash must be > 0")
        if float(self.fee_bps) < 0.0:
            raise ValueError("fee_bps must be >= 0")
        if float(self.slippage_bps) < 0.0:
            raise ValueError("slippage_bps must be >= 0")
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError("max_steps must be > 0 when provided")
        if self.reward_clip_min is not None and self.reward_clip_max is not None:
            if float(self.reward_clip_min) > float(self.reward_clip_max):
                raise ValueError("reward_clip_min cannot exceed reward_clip_max")
        if float(self.invalid_close_flat_penalty) < 0.0:
            raise ValueError("invalid_close_flat_penalty must be >= 0")
        if self.reward_version not in {"reward.v1", "reward.v2", "reward.v2_dense_pbr"}:
            raise ValueError("reward_version must be one of {reward.v1, reward.v2, reward.v2_dense_pbr}")
        if self.reward_version == "reward.v2_dense_pbr":
            if self.dense_pbr_config is None:
                raise ValueError("dense_pbr_config must be provided for reward.v2_dense_pbr")
        elif self.dense_pbr_config is not None:
            raise ValueError("dense_pbr_config is only supported for reward.v2_dense_pbr")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be int or null")
        if self.close_position_transition_timing_policy not in {
            "flatten_before_interval",
            "flatten_after_interval",
        }:
            raise ValueError(
                "close_position_transition_timing_policy must be one of {flatten_before_interval, flatten_after_interval}"
            )


class EpisodeRunnerCore:
    """Framework-agnostic episode runner independent of Gymnasium."""

    def __init__(self, *, episode_ref: EpisodeRef, episode_data: EpisodeData, config: EpisodeRunnerConfig) -> None:
        self._episode_ref = episode_ref
        self._episode_data = episode_data
        self._config = config

        self._current_index: int | None = None
        self._steps_taken = 0
        self._position = PositionState(exposure=0)
        self._portfolio = PortfolioState(portfolio_value=float(config.initial_cash))
        self._open_position_entry_price_exec: float | None = None
        self._open_position_entry_fees = 0.0
        self._open_position_entry_slippage_cost = 0.0
        self._open_position_entry_step_ordinal: int | None = None
        self._peak_portfolio_value = float(config.initial_cash)
        self._seed: int | None = None
        self._terminated = False
        self._truncated = False
        self._position_placeholder_index = self._resolve_observation_column_index(POSITION_PLACEHOLDER_COLUMN)
        self._floating_pnl_placeholder_index = self._resolve_observation_column_index(FLOATING_PNL_PLACEHOLDER_COLUMN)
        self._drawdown_placeholder_index = self._resolve_observation_column_index(DRAWDOWN_PLACEHOLDER_COLUMN)
        self._holding_age_in_position_index = self._resolve_observation_column_index(HOLDING_AGE_IN_POSITION_COLUMN)
        self._entry_return_bps_index = self._resolve_observation_column_index(ENTRY_RETURN_BPS_COLUMN)

    @property
    def observation_dim(self) -> int:
        """Return observation feature dimension."""

        return int(self._episode_data.observation_matrix.shape[1])

    @property
    def action_mapping(self) -> dict[int, str]:
        """Return stable action mapping table."""

        return dict(ACTION_MAPPING)

    def current_action_mask(self) -> np.ndarray:
        """Return the valid discrete action mask for the current position state."""

        return ExecutionEngine.valid_action_mask(position_before=self._position.exposure).copy()

    def _resolve_observation_column_index(self, column_name: str) -> int | None:
        try:
            return int(self._episode_data.observation_columns.index(column_name))
        except ValueError:
            return None

    def _current_floating_pnl(self) -> float:
        if self._current_index is None:
            raise RuntimeError("EpisodeRunnerCore.reset() must be called before accessing runtime observation state.")
        if self._position.exposure == 0 or self._open_position_entry_price_exec is None:
            return 0.0
        current_mark_to_market = float(self._episode_data.mark_to_market_price_vector[int(self._current_index)])
        return (
            float(self._position.exposure) * (current_mark_to_market - float(self._open_position_entry_price_exec))
            - float(self._open_position_entry_fees)
            - float(self._open_position_entry_slippage_cost)
        )

    def _current_drawdown(self) -> float:
        peak_value = max(float(self._peak_portfolio_value), 1e-12)
        return max((peak_value - float(self._portfolio.portfolio_value)) / peak_value, 0.0)

    def _current_holding_age_in_position(self) -> float:
        if self._position.exposure == 0 or self._open_position_entry_step_ordinal is None:
            return 0.0
        return max(float(self._steps_taken - self._open_position_entry_step_ordinal), 0.0)

    def _current_entry_return_bps(self) -> float:
        if self._current_index is None:
            raise RuntimeError("EpisodeRunnerCore.reset() must be called before accessing runtime observation state.")
        if self._position.exposure == 0 or self._open_position_entry_price_exec is None:
            return 0.0
        entry_price = float(self._open_position_entry_price_exec)
        current_mark_to_market = float(self._episode_data.mark_to_market_price_vector[int(self._current_index)])
        gross_return = (current_mark_to_market - entry_price) / max(abs(entry_price), 1e-12)
        return float(self._position.exposure) * gross_return * 10000.0

    def _runtime_observation(self) -> np.ndarray:
        if self._current_index is None:
            raise RuntimeError("EpisodeRunnerCore.reset() must be called before accessing runtime observation state.")
        obs = self._episode_data.observation_matrix[self._current_index].copy()
        if self._position_placeholder_index is not None:
            obs[self._position_placeholder_index] = np.float32(self._position.exposure)
        if self._floating_pnl_placeholder_index is not None:
            obs[self._floating_pnl_placeholder_index] = np.float32(self._current_floating_pnl())
        if self._drawdown_placeholder_index is not None:
            obs[self._drawdown_placeholder_index] = np.float32(self._current_drawdown())
        if self._holding_age_in_position_index is not None:
            obs[self._holding_age_in_position_index] = np.float32(self._current_holding_age_in_position())
        if self._entry_return_bps_index is not None:
            obs[self._entry_return_bps_index] = np.float32(self._current_entry_return_bps())
        return obs

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset episode and return first observation and info payload."""

        self._current_index = int(self._episode_data.episode_valid_start_row)
        self._steps_taken = 0
        self._position = PositionState(exposure=0)
        self._portfolio = PortfolioState(portfolio_value=float(self._config.initial_cash))
        self._open_position_entry_price_exec = None
        self._open_position_entry_fees = 0.0
        self._open_position_entry_slippage_cost = 0.0
        self._open_position_entry_step_ordinal = None
        self._peak_portfolio_value = float(self._config.initial_cash)
        self._terminated = False
        self._truncated = False
        self._seed = seed if seed is not None else self._config.seed

        obs = self._runtime_observation()
        info = {
            "seed": self._seed,
            "episode_ref": {
                "scope": self._episode_ref.scope,
                "partition": self._episode_ref.partition,
                "source_rel": self._episode_ref.source_rel,
                "fold_id": self._episode_ref.fold_id,
            },
            "episode_length_rows": int(self._episode_data.observation_matrix.shape[0]),
            "episode_transitions": int(self._episode_data.observation_matrix.shape[0] - 1 - self._current_index),
            "episode_valid_start_row": int(self._episode_data.episode_valid_start_row),
            "effective_episode_start_row": int(self._episode_data.episode_valid_start_row),
            "warmup_applied": bool(self._episode_data.warmup_applied),
            "action_mapping": dict(ACTION_MAPPING),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one deterministic transition and return Gym-compatible tuple."""

        if self._current_index is None:
            raise RuntimeError("EpisodeRunnerCore.reset() must be called before step().")
        if self._terminated or self._truncated:
            raise RuntimeError("Cannot step after episode is finished; call reset().")

        current_index = int(self._current_index)
        exec_result = ExecutionEngine.apply_action(position_before=self._position.exposure, action_raw=int(action))

        price_exec = float(self._episode_data.execution_price_vector[current_index])
        price_next = float(self._episode_data.mark_to_market_price_vector[current_index + 1])
        holding_steps_before_close: int | None = None
        if (
            exec_result.position_before in {-1, 1}
            and exec_result.position_after == 0
            and exec_result.trade_units > 0
            and self._open_position_entry_step_ordinal is not None
        ):
            holding_steps_before_close = int(self._steps_taken - self._open_position_entry_step_ordinal)
        reward_position_after = exec_result.position_after
        reward_trade_units = exec_result.trade_units
        delayed_close_interval_accrual = (
            self._config.close_position_transition_timing_policy == "flatten_after_interval"
            and exec_result.position_before in {-1, 1}
            and exec_result.position_after == 0
            and exec_result.trade_units > 0
        )
        if delayed_close_interval_accrual:
            # Carry the existing exposure through the current accrual interval and flatten only for the next state.
            reward_position_after = exec_result.position_before
        reward = RewardEngine.compute_reward(
            reward_version=self._config.reward_version,
            position_before=exec_result.position_before,
            action_raw=exec_result.action_raw,
            invalid_action=exec_result.invalid_action,
            invalid_action_reason=exec_result.invalid_action_reason,
            position_after=reward_position_after,
            price_exec=price_exec,
            price_next=price_next,
            trade_units=reward_trade_units,
            fee_bps=self._config.fee_bps,
            slippage_bps=self._config.slippage_bps,
            invalid_close_flat_penalty=self._config.invalid_close_flat_penalty,
            reward_scale=self._config.reward_scale,
            reward_clip_min=self._config.reward_clip_min,
            reward_clip_max=self._config.reward_clip_max,
            dense_pbr_config=self._config.dense_pbr_config,
            entry_price_exec=self._open_position_entry_price_exec,
            entry_fees=self._open_position_entry_fees,
            entry_slippage_cost=self._open_position_entry_slippage_cost,
            holding_steps_before_close=holding_steps_before_close,
            close_accounting_position_after=exec_result.position_after,
        )

        self._position = PositionState(exposure=exec_result.position_after)
        if exec_result.position_before == 0 and exec_result.position_after in {-1, 1} and exec_result.trade_units > 0:
            self._open_position_entry_price_exec = float(price_exec)
            self._open_position_entry_fees = float(reward.fees)
            self._open_position_entry_slippage_cost = float(reward.slippage_cost)
            self._open_position_entry_step_ordinal = int(self._steps_taken)
        elif exec_result.position_before in {-1, 1} and exec_result.position_after == 0 and exec_result.trade_units > 0:
            self._open_position_entry_price_exec = None
            self._open_position_entry_fees = 0.0
            self._open_position_entry_slippage_cost = 0.0
            self._open_position_entry_step_ordinal = None
        self._portfolio = PortfolioState(portfolio_value=self._portfolio.portfolio_value + reward.reward_total)
        self._peak_portfolio_value = max(self._peak_portfolio_value, float(self._portfolio.portfolio_value))
        self._current_index = current_index + 1
        self._steps_taken += 1

        terminated = bool(self._current_index >= (len(self._episode_data.execution_price_vector) - 1))
        truncated = False
        if not terminated and self._config.max_steps is not None and self._steps_taken >= self._config.max_steps:
            truncated = True

        self._terminated = terminated
        self._truncated = truncated

        info: dict[str, Any] = {
            "timestamp": self._episode_data.timestamp_vector[current_index],
            "step_index": current_index,
            "position_before": exec_result.position_before,
            "position_after": exec_result.position_after,
            "action_raw": exec_result.action_raw,
            "action_semantic": exec_result.action_semantic,
            "invalid_action": exec_result.invalid_action,
            "invalid_action_reason": exec_result.invalid_action_reason,
            "price_exec": price_exec,
            "reward_total": reward.reward_total,
            "reward_components": {
                "pnl_delta": reward.pnl_delta,
                "position_mtm_contribution": reward.position_mtm_contribution,
                "realized_pnl_contribution": reward.realized_pnl_contribution,
                "close_realized_pnl_bonus_contribution": reward.close_realized_pnl_bonus_contribution,
                "benchmark_relative_contribution": reward.benchmark_relative_contribution,
                "risk_penalty": reward.risk_penalty,
                "inactivity_penalty": reward.inactivity_penalty,
                "invalid_close_flat_penalty": reward.invalid_close_flat_penalty,
                "reward_raw": reward.reward_raw,
                "reward_scaled": reward.reward_total,
            },
            "cost_components": {
                "fees": reward.fees,
                "slippage_cost": reward.slippage_cost,
                "trade_units": exec_result.trade_units,
            },
            "portfolio_value": self._portfolio.portfolio_value,
        }
        if terminated:
            info["termination_reason"] = "data_exhausted"
        if truncated:
            info["truncation_reason"] = "max_steps"

        obs = self._runtime_observation()
        return obs, float(reward.reward_total), terminated, truncated, info
