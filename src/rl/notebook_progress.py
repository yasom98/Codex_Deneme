"""Notebook-safe progress helpers for bounded training and evaluation flows."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import logging
import time
from typing import Any

from core.logging import get_logger
from rl.colab_runtime import capture_memory_snapshot

LOGGER = get_logger(__name__)

_VALID_PROGRESS_MODES = {"off", "auto", "notebook", "text"}


@dataclass(frozen=True)
class ProgressModeResolution:
    """Resolved live progress mode."""

    requested_mode: str
    active_mode: str
    colab_detected: bool
    notebook_detected: bool


def resolve_progress_mode(requested_mode: str) -> ProgressModeResolution:
    """Resolve the requested progress mode into a safe active mode."""

    normalized = requested_mode.strip().lower()
    if normalized not in _VALID_PROGRESS_MODES:
        raise ValueError(f"Unsupported progress mode: {requested_mode}")

    try:
        colab_detected = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        colab_detected = False
    notebook_detected = _notebook_detected()
    if normalized == "off":
        active_mode = "disabled"
    elif normalized == "text":
        active_mode = "text_tqdm"
    elif normalized == "notebook":
        active_mode = "notebook_tqdm" if (colab_detected or notebook_detected) else "text_tqdm"
    else:
        active_mode = "notebook_tqdm" if (colab_detected or notebook_detected) else "text_tqdm"

    return ProgressModeResolution(
        requested_mode=normalized,
        active_mode=active_mode,
        colab_detected=colab_detected,
        notebook_detected=notebook_detected,
    )


def log_progress_mode(*, logger: logging.Logger, resolution: ProgressModeResolution, scope: str) -> None:
    """Emit one concise operator-visible progress mode line."""

    logger.info(
        "Progress mode active | scope=%s requested_mode=%s active_mode=%s colab_detected=%s notebook_detected=%s",
        scope,
        resolution.requested_mode,
        resolution.active_mode,
        resolution.colab_detected,
        resolution.notebook_detected,
    )


class EvaluationProgressBar:
    """Single-region tqdm progress for evaluation/backtest execution."""

    def __init__(
        self,
        *,
        resolution: ProgressModeResolution,
        total_episodes: int,
        max_eval_steps: int,
        run_id: str,
        evaluation_mode: str,
        partition_label: str | None,
    ) -> None:
        self._resolution = resolution
        self._total_episodes = int(total_episodes)
        self._max_eval_steps = int(max_eval_steps)
        self._run_id = run_id
        self._evaluation_mode = evaluation_mode
        self._partition_label = partition_label
        self._bar = None
        self._last_refresh = 0.0
        if self._resolution.active_mode != "disabled":
            self._bar = _make_tqdm(
                total=self._total_episodes,
                desc=f"Eval {self._evaluation_mode}",
                unit="episode",
            )
            self._bar.set_postfix(
                {
                    "run": self._run_id,
                    "partition": self._partition_label or "-",
                    "step": f"0/{self._max_eval_steps}",
                },
                refresh=False,
            )

    def on_episode_start(self, *, episode_index: int, episode_ref: dict[str, Any]) -> None:
        """Update the live bar when one episode starts."""

        if self._bar is None:
            return
        self._bar.set_postfix(
            {
                "run": self._run_id,
                "partition": episode_ref.get("partition"),
                "episode": f"{int(episode_index) + 1}/{self._total_episodes}",
                "step": f"0/{self._max_eval_steps}",
            },
            refresh=False,
        )
        self._bar.refresh()

    def on_step(self, *, episode_index: int, step_ordinal: int) -> None:
        """Refresh the single live progress region at a bounded cadence."""

        if self._bar is None:
            return
        now = time.monotonic()
        if int(step_ordinal) != 0 and int(step_ordinal) % 64 != 0 and (now - self._last_refresh) < 1.0:
            return
        self._last_refresh = now
        self._bar.set_postfix(
            {
                "run": self._run_id,
                "episode": f"{int(episode_index) + 1}/{self._total_episodes}",
                "step": f"{int(step_ordinal)}/{self._max_eval_steps}",
            },
            refresh=False,
        )
        self._bar.refresh()

    def on_episode_finish(self, *, episode_index: int, step_count: int) -> None:
        """Advance the live bar when one episode finishes."""

        if self._bar is None:
            return
        self._bar.update(1)
        self._bar.set_postfix(
            {
                "run": self._run_id,
                "episode": f"{int(episode_index) + 1}/{self._total_episodes}",
                "step": f"{int(step_count)}/{self._max_eval_steps}",
            },
            refresh=False,
        )
        self._bar.refresh()

    def close(self) -> None:
        """Close the live evaluation bar."""

        if self._bar is not None:
            self._bar.close()


def build_training_progress_callback(
    *,
    resolution: ProgressModeResolution,
    total_timesteps: int,
    run_id: str,
    resolved_device: str | None,
    logger: logging.Logger | None = None,
    memory_log_interval_steps: int = 0,
    memory_snapshots: list[dict[str, Any]] | None = None,
) -> Any | None:
    """Build an optional SB3 callback for notebook-safe training progress."""

    if resolution.active_mode == "disabled" and int(memory_log_interval_steps) <= 0:
        return None

    module = importlib.import_module("stable_baselines3.common.callbacks")
    base_callback_class = getattr(module, "BaseCallback")
    active_logger = logger or LOGGER
    snapshots = memory_snapshots if memory_snapshots is not None else []

    class _TrainingProgressCallback(base_callback_class):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self._bar = None
            self._start_time = 0.0
            self._last_memory_step = 0
            self._total_timesteps = int(total_timesteps)

        def _on_training_start(self) -> None:
            self._start_time = time.monotonic()
            if resolution.active_mode != "disabled":
                self._bar = _make_tqdm(
                    total=self._total_timesteps,
                    desc="Train PPO",
                    unit="step",
                )
                self._bar.set_postfix(
                    {
                        "run": run_id,
                        "device": resolved_device or "-",
                        "eta_s": "?",
                    },
                    refresh=False,
                )

        def _on_step(self) -> bool:
            current_timesteps = int(getattr(self.model, "num_timesteps", 0))
            if self._bar is not None and current_timesteps > self._bar.n:
                self._bar.update(current_timesteps - self._bar.n)
                elapsed = max(time.monotonic() - self._start_time, 1e-6)
                fps = current_timesteps / elapsed
                remaining = max(self._total_timesteps - current_timesteps, 0)
                eta_seconds = remaining / fps if fps > 0 else None
                self._bar.set_postfix(
                    {
                        "run": run_id,
                        "device": resolved_device or "-",
                        "fps": f"{fps:.1f}",
                        "eta_s": f"{eta_seconds:.1f}" if eta_seconds is not None else "?",
                    },
                    refresh=False,
                )
                self._bar.refresh()

            if int(memory_log_interval_steps) > 0 and current_timesteps - self._last_memory_step >= int(memory_log_interval_steps):
                snapshot = capture_memory_snapshot(label="learn_progress", step=current_timesteps)
                snapshots.append(snapshot)
                active_logger.info(
                    "Training memory snapshot | run_id=%s step=%d process_rss_mib=%s gpu_allocated_mib=%s gpu_reserved_mib=%s",
                    run_id,
                    current_timesteps,
                    snapshot.get("process_rss_mib"),
                    snapshot.get("gpu_allocated_mib"),
                    snapshot.get("gpu_reserved_mib"),
                )
                self._last_memory_step = current_timesteps
            return True

        def _on_training_end(self) -> None:
            if self._bar is not None:
                if self._bar.n < self._total_timesteps:
                    self._bar.update(self._total_timesteps - self._bar.n)
                self._bar.close()

    return _TrainingProgressCallback()


def _make_tqdm(*, total: int, desc: str, unit: str) -> Any:
    """Return a tqdm instance via tqdm.auto for terminal/notebook compatibility."""

    module = importlib.import_module("tqdm.auto")
    tqdm_class = getattr(module, "tqdm")
    return tqdm_class(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)


def _notebook_detected() -> bool:
    """Return whether an IPython kernel is active."""

    try:
        from IPython import get_ipython
    except Exception:  # noqa: BLE001
        return False
    shell = get_ipython()
    return shell is not None and getattr(shell, "kernel", None) is not None
