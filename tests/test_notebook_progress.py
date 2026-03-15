"""Tests for notebook-safe progress mode resolution."""

from __future__ import annotations

import pytest

from rl.notebook_progress import resolve_progress_mode


def test_resolve_progress_mode_off_disables_progress() -> None:
    resolution = resolve_progress_mode("off")

    assert resolution.requested_mode == "off"
    assert resolution.active_mode == "disabled"


def test_resolve_progress_mode_text_uses_text_tqdm() -> None:
    resolution = resolve_progress_mode("text")

    assert resolution.requested_mode == "text"
    assert resolution.active_mode == "text_tqdm"


def test_resolve_progress_mode_notebook_falls_back_to_text_when_kernel_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rl.notebook_progress._notebook_detected", lambda: False)
    monkeypatch.setattr("rl.notebook_progress.importlib.util.find_spec", lambda module_name: None)

    resolution = resolve_progress_mode("notebook")

    assert resolution.requested_mode == "notebook"
    assert resolution.active_mode == "text_tqdm"
