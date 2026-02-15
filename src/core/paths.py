"""Path utilities for discovery and output mapping."""

from __future__ import annotations

from pathlib import Path

SYSTEM_NAMES = {"thumbs.db", "desktop.ini"}
SYSTEM_DIRS = {"__macosx"}


def ensure_within_root(path: Path, root: Path) -> None:
    """Ensure the given path resolves under the provided root path."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes root: {resolved_path} (root: {resolved_root})") from exc


def discover_csv_files(input_root: Path, pattern: str = "**/*.csv") -> list[Path]:
    """Discover CSV files recursively from the input root."""
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    files = [
        path
        for path in input_root.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".csv" and not _is_hidden_or_system(path, input_root)
    ]
    return sorted(files)


def _is_hidden_or_system(path: Path, input_root: Path) -> bool:
    """Return True when the path should be ignored as hidden/system."""
    rel_parts = path.relative_to(input_root).parts
    for part in rel_parts:
        lower_part = part.lower()
        if (
            part.startswith(".")
            or lower_part in SYSTEM_DIRS
            or lower_part in SYSTEM_NAMES
            or lower_part.startswith("thumbs.db")
            or lower_part.startswith("desktop.ini")
        ):
            return True
    return False


def build_output_parquet_path(src_csv: Path, input_root: Path, output_root: Path) -> Path:
    """Build mirrored parquet output path for a source CSV file."""
    rel_path = src_csv.resolve().relative_to(input_root.resolve())
    out_path = (output_root / rel_path).with_suffix(".parquet")
    ensure_within_root(out_path, output_root)
    return out_path


def build_report_path(src_csv: Path, input_root: Path, reports_root: Path) -> Path:
    """Build mirrored health-report path for a source CSV file."""
    rel_path = src_csv.resolve().relative_to(input_root.resolve())
    report_path = (reports_root / rel_path).with_suffix(".health.json")
    ensure_within_root(report_path, reports_root)
    return report_path
