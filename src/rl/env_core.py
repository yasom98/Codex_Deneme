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
    fees: float
    slippage_cost: float
    invalid_close_flat_penalty: float
    reward_raw: float
    reward_total: float


class RewardEngine:
    """Reward decomposition contract for Milestone 4.5 v1/v2."""

    @staticmethod
    def compute_reward(
        *,
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
    ) -> RewardBreakdown:
        """Compute deterministic reward components."""

        pnl_delta = float(position_after) * (float(price_next) - float(price_exec))
        fees = abs(int(trade_units)) * float(price_exec) * (float(fee_bps) / 10_000.0)
        slippage_cost = abs(int(trade_units)) * float(price_exec) * (float(slippage_bps) / 10_000.0)
        applied_invalid_close_flat_penalty = 0.0
        if (
            int(action_raw) == ACTION_CLOSE_POSITION
            and bool(invalid_action)
            and invalid_action_reason == "already_flat"
        ):
            applied_invalid_close_flat_penalty = float(invalid_close_flat_penalty)
        reward_raw = pnl_delta - fees - slippage_cost - applied_invalid_close_flat_penalty
        reward_total = reward_raw * float(reward_scale)
        if reward_clip_min is not None or reward_clip_max is not None:
            reward_total = float(np.clip(reward_total, reward_clip_min, reward_clip_max))

        return RewardBreakdown(
            pnl_delta=float(pnl_delta),
            fees=float(fees),
            slippage_cost=float(slippage_cost),
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
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be int or null")


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
        self._seed: int | None = None
        self._terminated = False
        self._truncated = False

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

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset episode and return first observation and info payload."""

        self._current_index = int(self._episode_data.episode_valid_start_row)
        self._steps_taken = 0
        self._position = PositionState(exposure=0)
        self._portfolio = PortfolioState(portfolio_value=float(self._config.initial_cash))
        self._terminated = False
        self._truncated = False
        self._seed = seed if seed is not None else self._config.seed

        obs = self._episode_data.observation_matrix[self._current_index].copy()
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
        reward = RewardEngine.compute_reward(
            action_raw=exec_result.action_raw,
            invalid_action=exec_result.invalid_action,
            invalid_action_reason=exec_result.invalid_action_reason,
            position_after=exec_result.position_after,
            price_exec=price_exec,
            price_next=price_next,
            trade_units=exec_result.trade_units,
            fee_bps=self._config.fee_bps,
            slippage_bps=self._config.slippage_bps,
            invalid_close_flat_penalty=self._config.invalid_close_flat_penalty,
            reward_scale=self._config.reward_scale,
            reward_clip_min=self._config.reward_clip_min,
            reward_clip_max=self._config.reward_clip_max,
        )

        self._position = PositionState(exposure=exec_result.position_after)
        self._portfolio = PortfolioState(portfolio_value=self._portfolio.portfolio_value + reward.reward_total)
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

        obs = self._episode_data.observation_matrix[self._current_index].copy()
        return obs, float(reward.reward_total), terminated, truncated, info
