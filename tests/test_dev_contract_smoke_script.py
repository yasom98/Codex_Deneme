"""Regression tests for the developer contract smoke harness."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "dev_contract_smoke.sh"


def _write_fake_python(repo_root: Path, log_path: Path, run_id: str, fail_kind: str | None = None) -> Path:
    """Create a fake python3 shim that logs command order and simulates CLI outputs."""

    helper_path = repo_root / "fake_python_impl.py"
    helper_path.write_text(
        f"""from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REAL_PYTHON = {sys.executable!r}


def _kind(args: list[str]) -> str:
    if len(args) >= 2 and args[0] == "-m" and args[1] == "pytest":
        return "pytest"
    if args and args[0] == "-":
        return "stdin"
    if not args:
        return "python3"
    return Path(args[0]).name


def _append_log(kind: str, args: list[str]) -> None:
    log_path = Path(os.environ["FAKE_PYTHON_LOG"]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({{"kind": kind, "args": args}}, ensure_ascii=True) + "\\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    args = sys.argv[1:]
    kind = _kind(args)
    _append_log(kind, args)

    fail_kind = os.environ.get("FAKE_FAIL_KIND")
    if fail_kind and kind == fail_kind:
        return 9

    repo_root = Path(os.environ["FAKE_REPO_ROOT"]).resolve()
    run_id = os.environ["FAKE_RUN_ID"]
    run_root = repo_root / "runs" / run_id

    if kind == "stdin":
        stdin_code = sys.stdin.read()
        proc = subprocess.run(
            [REAL_PYTHON, *args],
            input=stdin_code,
            text=True,
            capture_output=False,
            check=False,
        )
        return int(proc.returncode)

    if kind == "pytest":
        return 0

    if kind == "make_features.py":
        _write_json(run_root / "data_features" / "reports" / "summary.json", {{"run_id": run_id}})
        return 0

    if kind == "validate_train_inputs.py":
        _write_json(
            run_root / "data_features" / "reports" / "train_input_validation_report.json",
            {{"run_id": run_id, "train_input_validation_overall": True}},
        )
        return 0

    if kind == "validate_splits.py":
        _write_json(
            run_root / "data_features" / "reports" / "split_validation_report.json",
            {{"run_id": run_id, "split_validation_overall": True}},
        )
        return 0

    if kind == "build_datasets.py":
        _write_json(
            run_root / "data_datasets" / "reports" / "dataset_build_report.json",
            {{"run_id": run_id, "dataset_build_overall": True}},
        )
        return 0

    if kind == "build_states.py":
        _write_json(
            run_root / "data_states" / "reports" / "state_build_report.json",
            {{"run_id": run_id, "state_build_overall": True}},
        )
        _write_json(
            run_root / "data_states" / "reports" / "state_manifest.json",
            {{
                "partition_metadata": [
                    {{
                        "scope": "partition",
                        "partition": "train",
                        "source_rel": "z_first_train.parquet",
                        "fold_id": None,
                    }},
                    {{
                        "scope": "partition",
                        "partition": "train",
                        "source_rel": "a_second_train.parquet",
                        "fold_id": None,
                    }},
                    {{
                        "scope": "partition",
                        "partition": "val",
                        "source_rel": "val_sample.parquet",
                        "fold_id": None,
                    }},
                ]
            }},
        )
        return 0

    if kind == "validate_env_contract.py":
        _write_json(
            run_root / "env_contract" / "reports" / "env_contract_report.json",
            {{"run_id": run_id, "env_contract_overall": True}},
        )
        return 0

    raise SystemExit(f"unexpected fake python invocation: {{args}}")


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )

    shim_dir = repo_root / "fake_bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "python3"
    shim_path.write_text(
        f"""#!/usr/bin/env bash
exec {shlex_quote(sys.executable)} {shlex_quote(str(helper_path))} "$@"
""",
        encoding="utf-8",
    )
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR)
    return shim_dir


def shlex_quote(value: str) -> str:
    """Return a shell-escaped single argument."""

    return "'" + value.replace("'", "'\"'\"'") + "'"


def _setup_fake_repo(tmp_path: Path, *, run_id: str, fail_kind: str | None = None) -> tuple[Path, Path]:
    """Create a temp repo copy with the harness script and fake python shim."""

    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_PATH, repo_root / "scripts" / "dev_contract_smoke.sh")
    (repo_root / "scripts" / "dev_contract_smoke.sh").chmod(
        (repo_root / "scripts" / "dev_contract_smoke.sh").stat().st_mode | stat.S_IXUSR
    )

    log_path = repo_root / "invocations.jsonl"
    fake_bin = _write_fake_python(repo_root, log_path, run_id, fail_kind=fail_kind)

    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
    env["FAKE_PYTHON_LOG"] = str(log_path)
    env["FAKE_REPO_ROOT"] = str(repo_root)
    env["FAKE_RUN_ID"] = run_id
    if fail_kind is not None:
        env["FAKE_FAIL_KIND"] = fail_kind

    return repo_root, env


def _read_invocations(log_path: Path) -> list[dict[str, Any]]:
    """Load the fake-python invocation log."""

    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _args_for_kind(invocations: list[dict[str, Any]], kind: str) -> list[str]:
    """Return the argument vector logged for a specific fake-python invocation kind."""

    for item in invocations:
        if item.get("kind") == kind:
            args = item.get("args")
            if isinstance(args, list) and all(isinstance(arg, str) for arg in args):
                return args
            raise AssertionError(f"logged args missing or invalid for kind={kind}")
    raise AssertionError(f"missing invocation for kind={kind}")


def test_harness_orders_commands_and_generates_env_config(tmp_path: Path) -> None:
    """The harness should run the validated chain in order and emit deterministic env config."""

    run_id = "smoke_success"
    repo_root, env = _setup_fake_repo(tmp_path, run_id=run_id)
    script = repo_root / "scripts" / "dev_contract_smoke.sh"

    proc = subprocess.run(
        ["bash", str(script), run_id],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "[PASS]" in proc.stdout
    assert "+ python3 -m pytest -q" in proc.stdout
    assert "+ python3 -" in proc.stdout

    invocations = _read_invocations(repo_root / "invocations.jsonl")
    assert [item["kind"] for item in invocations] == [
        "pytest",
        "make_features.py",
        "validate_train_inputs.py",
        "validate_splits.py",
        "build_datasets.py",
        "build_states.py",
        "stdin",
        "validate_env_contract.py",
    ]
    assert _args_for_kind(invocations, "build_datasets.py") == [
        str(repo_root / "scripts" / "build_datasets.py"),
        "--run-id",
        run_id,
        "--input-root",
        str(repo_root / "runs" / run_id / "data_features" / "parquet"),
        "--overwrite",
        "true",
        "--execution-price-column",
        "close",
        "--mark-to-market-column",
        "close",
    ]
    assert _args_for_kind(invocations, "build_states.py") == [
        str(repo_root / "scripts" / "build_states.py"),
        "--run-id",
        run_id,
        "--input-root",
        str(repo_root / "runs" / run_id / "data_datasets"),
        "--overwrite",
        "true",
        "--execution-price-column",
        "close",
        "--mark-to-market-column",
        "close",
    ]

    env_config_path = repo_root / "runs" / run_id / "env_contract" / "tmp" / "dev_contract_smoke_env_config.json"
    payload = json.loads(env_config_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["state_root"] == str((repo_root / "runs" / run_id / "data_states").resolve())
    assert payload["episode_ref"] == {
        "scope": "partition",
        "partition": "train",
        "source_rel": "z_first_train.parquet",
        "fold_id": None,
    }
    assert payload["execution_price_column"] == "close"
    assert payload["mark_to_market_column"] == "close"
    assert payload["allowed_safe_casts"] == ["uint8->float32"]
    assert payload["seed"] == 42


def test_harness_fails_fast_and_stops_after_first_error(tmp_path: Path) -> None:
    """The harness should stop immediately and avoid downstream commands after a failure."""

    run_id = "smoke_fail_fast"
    repo_root, env = _setup_fake_repo(tmp_path, run_id=run_id, fail_kind="build_datasets.py")
    script = repo_root / "scripts" / "dev_contract_smoke.sh"

    proc = subprocess.run(
        ["bash", str(script), run_id],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 9
    assert "[FAIL]" in proc.stdout
    assert "step=build_datasets" in proc.stdout

    invocations = _read_invocations(repo_root / "invocations.jsonl")
    assert [item["kind"] for item in invocations] == [
        "pytest",
        "make_features.py",
        "validate_train_inputs.py",
        "validate_splits.py",
        "build_datasets.py",
    ]
