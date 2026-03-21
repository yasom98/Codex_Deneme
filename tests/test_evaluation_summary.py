"""Tests for compact evaluation summary extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.evaluation_summary import EVALUATION_SUMMARY_FILENAME, build_evaluation_summary, write_evaluation_summary
from rl.passivity_diagnostics import PASSIVITY_DIAGNOSTICS_REPORT_FILENAME


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload for summary tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_required_reports(output_dir: Path) -> None:
    """Seed the required evaluation report surface."""

    _write_json(
        output_dir / "evaluation_validation_report.json",
        {
            "overall_pass": True,
            "action_masking_enabled": False,
            "passivity_diagnostics_enabled": False,
        },
    )
    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "action_masking_enabled": False,
            "passivity_diagnostics_enabled": False,
        },
    )
    _write_json(
        output_dir / "evaluation_backtest_report.json",
        {
            "evaluation_success": True,
            "action_masking_enabled": True,
            "passivity_diagnostics_enabled": True,
            "strategy_metrics": {
                "final_equity": 719.421875,
                "total_return": -0.280578125,
                "num_trades": 5,
                "max_drawdown": 0.125,
            },
            "startup_phase_trace": [
                {"phase": "validation", "status": "completed", "detail": {}},
                {
                    "phase": "eval_finish",
                    "status": "completed",
                    "detail": {"trace_row_count": 32},
                },
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


def test_summary_reader_happy_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation_out"
    _seed_required_reports(output_dir)
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
                "num_trades": 9,
                "final_equity": 402.0,
            },
            "deterministic_vs_stochastic": {
                "deterministic_hold_extreme": True,
                "num_trades_delta": 9,
                "hold_share_delta": -0.75,
            },
        },
    )

    summary = build_evaluation_summary(output_dir=output_dir)
    written = write_evaluation_summary(output_dir=output_dir)

    assert summary == written
    assert summary["evaluation_success"] is True
    assert summary["validation_overall_pass"] is True
    assert summary["model_class"] == "MaskablePPO"
    assert summary["detection_source"] == "artifact"
    assert summary["detected_maskable"] is True
    assert summary["action_masking_enabled"] is True
    assert summary["passivity_diagnostics_enabled"] is True
    assert summary["final_equity"] == pytest.approx(719.421875)
    assert summary["total_return"] == pytest.approx(-0.280578125)
    assert summary["num_trades"] == 5
    assert summary["profit_factor"] is None
    assert summary["max_drawdown_pct"] == pytest.approx(12.5)
    assert summary["passivity_report_exists"] is True
    assert summary["deterministic_hold_share"] == pytest.approx(1.0)
    assert summary["deterministic_hold_extreme"] is True
    assert summary["deterministic_hold_dominance_margin_band"] == "wide"
    assert summary["stochastic_num_trades"] == 9
    assert summary["stochastic_final_equity"] == pytest.approx(402.0)
    assert summary["num_trades_delta"] == 9
    assert summary["hold_share_delta"] == pytest.approx(-0.75)
    persisted = json.loads((output_dir / EVALUATION_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert persisted == summary


def test_summary_reader_with_optional_passivity_report_absent(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation_out"
    _seed_required_reports(output_dir)

    summary = build_evaluation_summary(output_dir=output_dir)

    assert summary["passivity_report_exists"] is False
    assert summary["invalid_action_ratio"] is None
    assert summary["action_semantic_counts"] is None
    assert summary["position_transition_counts"] is None
    assert summary["deterministic_hold_share"] is None
    assert summary["deterministic_hold_extreme"] is None
    assert summary["deterministic_hold_dominance_margin_band"] is None
    assert summary["stochastic_num_trades"] is None
    assert summary["stochastic_final_equity"] is None
    assert summary["num_trades_delta"] is None
    assert summary["hold_share_delta"] is None


def test_summary_reader_fails_closed_when_required_file_is_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation_out"
    _seed_required_reports(output_dir)
    (output_dir / "evaluation_backtest_report.json").unlink()

    with pytest.raises(ValueError, match="evaluation_backtest_report.json"):
        build_evaluation_summary(output_dir=output_dir)
