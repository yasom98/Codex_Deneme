"""Integration-style tests for Milestone 4.6 readiness CLI."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from tests.rl_readiness_fixtures import patch_read_parquet, seed_state_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_training_env_readiness.py"


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_cli_success_writes_catalog_and_readiness_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "readiness_cli_success"
    state_root, config_path, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=[
            {"partition": "train", "source_rel": "b_train.parquet", "row_count": 6},
            {"partition": "train", "source_rel": "a_train.parquet", "row_count": 6},
        ],
    )
    patch_read_parquet(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_training_env_readiness.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
            "--selection-policy",
            "seeded_random_episode",
            "--start-policy",
            "start_at_valid_from_row",
            "--min-remaining-steps",
            "2",
            "--seed",
            "42",
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    catalog_path = tmp_path / "runs" / run_id / "env_readiness" / "reports" / "episode_catalog.json"
    report_path = tmp_path / "runs" / run_id / "env_readiness" / "reports" / "training_env_readiness_report.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert catalog["episode_catalog_overall"] is True
    assert report["readiness_overall"] is True
    assert report["selected_episode_refs"] and len(report["selected_episode_refs"]) == 1
    assert report["selection_trace"]["eligible_domain_used"] == "training"


def test_cli_can_report_catalog_success_but_readiness_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "readiness_cli_guard_fail"
    state_root, config_path, frame_map = seed_state_run(
        tmp_path,
        run_id,
        entries=[{"partition": "train", "source_rel": "train_short.parquet", "row_count": 4, "warmup_rows": 1}],
    )
    patch_read_parquet(monkeypatch, frame_map)

    main = _load_main()
    monkeypatch.setitem(main.__globals__, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "validate_training_env_readiness.py",
            "--run-id",
            run_id,
            "--state-root",
            str(state_root),
            "--env-config",
            str(config_path),
            "--selection-policy",
            "seeded_random_episode",
            "--start-policy",
            "start_at_valid_from_row",
            "--min-remaining-steps",
            "4",
            "--seed",
            "42",
        ],
    )

    exit_code = int(main())
    assert exit_code == 2

    catalog_path = tmp_path / "runs" / run_id / "env_readiness" / "reports" / "episode_catalog.json"
    report_path = tmp_path / "runs" / run_id / "env_readiness" / "reports" / "training_env_readiness_report.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert catalog["episode_catalog_overall"] is True
    assert report["readiness_overall"] is False
    assert report["min_remaining_steps_guard_passed"] is False
