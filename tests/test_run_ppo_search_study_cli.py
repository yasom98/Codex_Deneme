"""CLI tests for Milestone 4.9 PPO search study runner."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from tests.ppo_search_fixtures import build_search_study

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_ppo_search_study.py"


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_success_returns_zero_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seeded = build_search_study(monkeypatch, tmp_path, "ppo_search_cli_success")
    main = _load_main()
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "run_ppo_search_study.py",
            "--study-config",
            str(seeded["study_config_path"]),
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    summary = json.loads((seeded["output_root"] / "study_summary.json").read_text(encoding="utf-8"))
    assert summary["study_id"] == "ppo_search_cli_success_study"


def test_cli_runtime_failure_returns_three(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seeded = build_search_study(monkeypatch, tmp_path, "ppo_search_cli_runtime_failure")
    main = _load_main()
    monkeypatch.setitem(
        main.__globals__,
        "execute_ppo_search_study",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("runtime-boom")),
    )
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "run_ppo_search_study.py",
            "--study-config",
            str(seeded["study_config_path"]),
        ],
    )

    exit_code = int(main())
    assert exit_code == 3
