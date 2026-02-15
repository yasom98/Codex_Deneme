"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline configuration values."""

    input_root: Path
    output_root: Path
    reports_root: Path
    csv_glob: str
    timestamp_aliases: tuple[str, ...]
    required_columns: tuple[str, ...]
    float_columns: tuple[str, ...]
    fail_on_critical: bool
    duplicate_policy: str
    seed: int


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _get_required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"Missing required config key: {key}")
    return data[key]


def load_config(path: Path) -> PipelineConfig:
    """Load and validate pipeline config from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_dir = path.resolve().parent

    cfg = PipelineConfig(
        input_root=_resolve_path(str(_get_required(raw, "input_root")), base_dir),
        output_root=_resolve_path(str(_get_required(raw, "output_root")), base_dir),
        reports_root=_resolve_path(str(_get_required(raw, "reports_root")), base_dir),
        csv_glob=str(raw.get("csv_glob", "**/*.csv")),
        timestamp_aliases=tuple(raw.get("timestamp_aliases", [])),
        required_columns=tuple(raw.get("required_columns", ("open", "high", "low", "close", "volume"))),
        float_columns=tuple(raw.get("float_columns", ("open", "high", "low", "close", "volume"))),
        fail_on_critical=bool(raw.get("fail_on_critical", True)),
        duplicate_policy=str(raw.get("duplicate_policy", "last")),
        seed=int(raw.get("seed", 42)),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: PipelineConfig) -> None:
    """Validate config fields and semantic constraints."""
    if not cfg.input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {cfg.input_root}")
    if not cfg.input_root.is_dir():
        raise NotADirectoryError(f"input_root is not a directory: {cfg.input_root}")
    if not cfg.csv_glob.strip():
        raise ValueError("csv_glob cannot be empty.")
    if not cfg.timestamp_aliases:
        raise ValueError("timestamp_aliases cannot be empty.")
    if cfg.duplicate_policy != "last":
        raise ValueError("Only duplicate_policy='last' is supported in this sprint.")
    if cfg.seed < 0:
        raise ValueError("seed must be >= 0.")

