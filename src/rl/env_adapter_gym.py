"""Gymnasium adapter layer for Milestone 4.5 env core."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl.env_contract import EnvConfig, validate_env_contract
from rl.env_core import EpisodeRunnerCore, EpisodeRunnerConfig

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None


class TradingEnvGym(gym.Env if gym is not None else object):
    """Gymnasium-compatible wrapper over the framework-agnostic env core."""

    metadata = {"render_modes": []}

    def __init__(self, *, config: EnvConfig, validate_on_init: bool = True) -> None:
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required for TradingEnvGym.")

        self._config = config
        validation = validate_env_contract(config=config, smoke_step=False, invocation_args={"validate_on_init": validate_on_init})
        if validate_on_init and not bool(validation.report_payload.get("env_contract_overall", False)):
            first_error = validation.report_payload.get("errors", [{}])[0]
            code = first_error.get("code", "ENV_CONTRACT_PRECONDITION_FAILED")
            raise ValueError(str(code))

        if validation.episode_data is None:
            raise ValueError("ENV_CONTRACT_PRECONDITION_FAILED")

        self._validation = validation
        self._runner = EpisodeRunnerCore(
            episode_ref=config.episode_ref,
            episode_data=validation.episode_data,
            config=EpisodeRunnerConfig(
                initial_cash=config.initial_cash,
                fee_bps=config.fee_bps,
                slippage_bps=config.slippage_bps,
                max_steps=config.max_steps,
                reward_scale=config.reward_contract.reward_scale,
                reward_clip_min=config.reward_contract.reward_clip_min,
                reward_clip_max=config.reward_contract.reward_clip_max,
                seed=config.seed,
            ),
        )

        obs_dim = self._runner.observation_dim
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Gymnasium reset API."""

        del options
        super().reset(seed=seed)
        obs, info = self._runner.reset(seed=seed)
        return obs.astype(np.float32, copy=False), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Gymnasium step API."""

        obs, reward, terminated, truncated, info = self._runner.step(int(action))
        return obs.astype(np.float32, copy=False), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        """No-op render for v1."""

        return None

    def close(self) -> None:
        """No-op close for v1."""

        return None
