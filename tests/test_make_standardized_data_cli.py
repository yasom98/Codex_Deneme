"""Integration-style tests for the standardization CLI."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "make_standardized_data.py"


def _write_config(config_path: Path, input_root: Path, runs_root: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {input_root}",
                f"runs_root: {runs_root}",
                'csv_glob: "*.csv"',
                "timestamp_aliases:",
                "  - timestamp",
                "  - ts",
                "  - datetime",
                "required_columns:",
                "  - open",
                "  - high",
                "  - low",
                "  - close",
                "  - volume",
                "float_columns:",
                "  - open",
                "  - high",
                "  - low",
                "  - close",
                "  - volume",
                "duplicate_policy: last",
                "seed: 42",
            ]
        ),
        encoding="utf-8",
    )


def _load_main() -> object:
    module = runpy.run_path(str(SCRIPT_PATH))
    return module["main"]


def test_make_standardized_data_cli_honors_input_root_override(monkeypatch: object, tmp_path: Path) -> None:
    config_input_root = tmp_path / "config_input"
    actual_input_root = tmp_path / "actual_input"
    runs_root = tmp_path / "runs"
    config_input_root.mkdir(parents=True, exist_ok=True)
    actual_input_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    csv_path = actual_input_root / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ts,open,high,low,close,volume",
                "2024-01-01 00:00:00+00:00,1,2,0.5,1.5,100",
                "2024-01-01 00:01:00+00:00,2,3,1.5,2.5,110",
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "data.yaml"
    _write_config(config_path, config_input_root, runs_root)

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        del self, index
        Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    main = _load_main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_standardized_data.py",
            "--config",
            str(config_path),
            "--run-id",
            "std_override_run",
            "--input-root",
            str(actual_input_root),
        ],
    )

    exit_code = int(main())
    assert exit_code == 0

    summary_path = runs_root / "std_override_run" / "data_standardized" / "reports" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["input_root_resolved"] == str(actual_input_root.resolve())
    assert payload["output_root_resolved"] == str((runs_root / "std_override_run" / "data_standardized" / "parquet").resolve())
