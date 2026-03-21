"""Tests for the Colab evaluation bundle convenience CLI."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import pytest

from rl.passivity_diagnostics import PASSIVITY_DIAGNOSTICS_REPORT_FILENAME
from tests.test_colab_runtime import _patch_runtime_dependencies, _seed_closure_source

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_colab_eval_bundle.py"


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload for bundle tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _command_arg(command: list[str], flag: str) -> str:
    """Extract one flag value from the explicit subprocess command."""

    index = command.index(flag)
    return str(command[index + 1])


def _seed_drive_artifact(drive_root: Path, run_id: str, artifact_attempt_id: str) -> Path:
    """Seed one canonical model artifact under the fake Drive root."""

    artifact_path = drive_root / "runs" / run_id / "ppo_artifact" / artifact_attempt_id / "canonical_ppo_model.zip"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact_path, mode="w") as archive:
        archive.writestr("metadata.txt", "bundle-test")
    return artifact_path


def _seed_eval_reports(output_dir: Path) -> None:
    """Seed the minimal evaluation report surface consumed by the summary reader."""

    _write_json(
        output_dir / "evaluation_validation_report.json",
        {
            "overall_pass": True,
            "action_masking_enabled": True,
            "passivity_diagnostics_enabled": True,
        },
    )
    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "action_masking_enabled": True,
            "passivity_diagnostics_enabled": True,
        },
    )
    _write_json(
        output_dir / "evaluation_backtest_report.json",
        {
            "evaluation_success": True,
            "action_masking_enabled": True,
            "passivity_diagnostics_enabled": True,
            "strategy_metrics": {
                "final_equity": 812.25,
                "total_return": -0.18775,
                "num_trades": 11,
                "max_drawdown": 0.08,
            },
            "startup_phase_trace": [
                {"phase": "validation", "status": "completed", "detail": {}},
                {
                    "phase": "model_load",
                    "status": "completed",
                    "detail": {
                        "model_class": "MaskablePPO",
                        "detection_source": "artifact",
                        "detected_maskable": True,
                        "action_masking_enabled": True,
                    },
                },
            ],
        },
    )
    _write_json(
        output_dir / PASSIVITY_DIAGNOSTICS_REPORT_FILENAME,
        {
            "deterministic_eval": {
                "invalid_action_ratio": 0.0,
                "action_semantic_counts": {
                    "HOLD": 12,
                    "OPEN_LONG": 0,
                    "OPEN_SHORT": 0,
                    "CLOSE_POSITION": 0,
                },
                "position_transition_counts": {
                    "-1->-1": 0,
                    "-1->0": 0,
                    "-1->1": 0,
                    "0->-1": 0,
                    "0->0": 12,
                    "0->1": 0,
                    "1->-1": 0,
                    "1->0": 0,
                    "1->1": 0,
                },
                "hold_dominance_summary": {"hold_share": 1.0},
                "action_ranking_summary": {"hold_dominance_margin_band": "wide"},
            },
            "stochastic_eval": {
                "num_trades": 11,
                "final_equity": 812.25,
            },
            "deterministic_vs_stochastic": {
                "deterministic_hold_extreme": True,
                "num_trades_delta": 11,
                "hold_share_delta": -1.0,
            },
        },
    )


def _seed_staged_eval_inputs(stage_root: Path, command: list[str]) -> None:
    """Seed the local staged evaluation inputs expected by the bundle."""

    source_map = {
        "env_config": Path(_command_arg(command, "--env-config")),
        "state_manifest": Path(_command_arg(command, "--state-manifest")),
        "env_contract_report": Path(_command_arg(command, "--env-contract-report")),
        "readiness_report": Path(_command_arg(command, "--readiness-report")),
        "episode_catalog": Path(_command_arg(command, "--episode-catalog")),
        "split_report": Path(_command_arg(command, "--split-report")),
        "eval_config": Path(_command_arg(command, "--eval-config")),
    }
    destinations = {
        "env_config": stage_root / "env_contract" / "tmp" / "bounded_training_preparation_env_config.json",
        "state_manifest": stage_root / "data_states" / "reports" / "state_manifest.json",
        "env_contract_report": stage_root / "env_contract" / "reports" / "env_contract_report.json",
        "readiness_report": stage_root / "env_readiness" / "reports" / "training_env_readiness_report.json",
        "episode_catalog": stage_root / "env_readiness" / "reports" / "episode_catalog.json",
        "split_report": stage_root / "data_features" / "reports" / "split_validation_report.json",
        "eval_config": stage_root / "configs" / "eval_config.json",
    }
    for label, source_path in source_map.items():
        dest = destinations[label]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)


def test_eval_override_config_generation(tmp_path: Path) -> None:
    main = _load_main()
    write_override = main.__globals__["_write_eval_override_config"]
    source_eval_config = tmp_path / "stage" / "configs" / "eval_config.json"
    source_eval_config.parent.mkdir(parents=True, exist_ok=True)
    source_eval_config.write_text(
        json.dumps(
            {
                "algorithm": "ppo",
                "seed": 42,
                "deterministic": True,
                "device": "cpu",
                "evaluation_mode": "episodic_eval_backtest",
                "target_mode": "explicit_partition",
                "target_partition": "validation",
                "target_fold_id": None,
                "target_episode_refs": None,
                "benchmark_mode": "buy_and_hold",
                "startup_policy": "fresh_only",
                "max_eval_episodes": 3,
                "max_eval_steps": 4096,
                "write_step_trace": False,
                "backtest_metrics": ["total_return"],
            }
        ),
        encoding="utf-8",
    )

    override_path = write_override(
        source_eval_config_path=source_eval_config,
        output_root=tmp_path / "bundle" / "configs",
        enable_action_masking=True,
        enable_passivity_diagnostics=True,
        write_step_trace=True,
    )

    source_payload = json.loads(source_eval_config.read_text(encoding="utf-8"))
    override_payload = json.loads(override_path.read_text(encoding="utf-8"))
    assert source_payload.get("action_masking") is None
    assert source_payload.get("passivity_diagnostics") is None
    assert source_payload["write_step_trace"] is False
    assert override_payload["action_masking"] is True
    assert override_payload["passivity_diagnostics"] is True
    assert override_payload["write_step_trace"] is True


def test_cli_runs_bundle_and_prints_compact_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "colab_eval_bundle_cli"
    artifact_attempt_id = "artifact_retry_20260319T224215Z"
    drive_root = tmp_path / "drive"
    stage_root = tmp_path / "stage"
    eval_out_root = tmp_path / "eval_out"
    bundle_root = tmp_path / "bundle_root"

    _patch_runtime_dependencies(monkeypatch)
    source_paths = _seed_closure_source(monkeypatch, drive_root, run_id=run_id)
    canonical_env_config_path = (
        drive_root
        / "runs"
        / run_id
        / "env_contract"
        / "tmp"
        / "bounded_training_preparation_env_config.json"
    )
    canonical_env_config_path.write_text(source_paths["env_config"].read_text(encoding="utf-8"), encoding="utf-8")
    _seed_drive_artifact(drive_root, run_id, artifact_attempt_id)

    main = _load_main()
    main.__globals__["DEFAULT_BUNDLE_BASE"] = bundle_root

    def _fake_subprocess_run(command: list[str], cwd: str, check: bool) -> SimpleNamespace:
        del cwd, check
        script_name = Path(command[1]).name
        if script_name == "stage_colab_inputs.py":
            _seed_staged_eval_inputs(Path(_command_arg(command, "--staging-root")), command)
            return SimpleNamespace(returncode=0)
        if script_name == "evaluate_policy.py":
            eval_config_path = Path(_command_arg(command, "--eval-config"))
            eval_config_payload = json.loads(eval_config_path.read_text(encoding="utf-8"))
            assert eval_config_payload["action_masking"] is True
            assert eval_config_payload["passivity_diagnostics"] is True
            assert eval_config_payload["write_step_trace"] is True
            assert eval_config_path != stage_root / "configs" / "eval_config.json"
            _seed_eval_reports(Path(_command_arg(command, "--output-dir")))
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"Unexpected subprocess command: {command}")

    monkeypatch.setattr(main.__globals__["subprocess"], "run", _fake_subprocess_run)
    monkeypatch.setattr(
        main.__globals__["sys"],
        "argv",
        [
            "run_colab_eval_bundle.py",
            "--drive-root",
            str(drive_root),
            "--run-id",
            run_id,
            "--artifact-attempt-id",
            artifact_attempt_id,
            "--repo-root",
            str(PROJECT_ROOT),
            "--stage-root",
            str(stage_root),
            "--eval-out-root",
            str(eval_out_root),
            "--enable-action-masking",
            "--enable-passivity-diagnostics",
            "--write-step-trace",
            "--print-summary",
        ],
    )

    exit_code = int(main())

    assert exit_code == 0
    summary_path = eval_out_root / "evaluation_summary.json"
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["model_class"] == "MaskablePPO"
    assert summary_payload["detected_maskable"] is True
    stdout_payload = json.loads(capsys.readouterr().out.strip())
    assert stdout_payload == {
        "evaluation_success": True,
        "model_class": "MaskablePPO",
        "detected_maskable": True,
        "action_masking_enabled": True,
        "passivity_diagnostics_enabled": True,
        "final_equity": 812.25,
        "total_return": -0.18775,
        "num_trades": 11,
        "deterministic_hold_share": 1.0,
        "deterministic_hold_dominance_margin_band": "wide",
        "eval_out": str(eval_out_root),
    }
