"""Atomic file writing utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _tmp_path(dest: Path) -> Path:
    """Return deterministic temp path next to destination."""
    return dest.with_suffix(f"{dest.suffix}.tmp")


def atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Atomically write a DataFrame to parquet."""
    tmp = _tmp_path(dest)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to atomically write parquet: {dest}") from exc


def atomic_write_json(payload: dict[str, Any], dest: Path) -> None:
    """Atomically write a dictionary to JSON."""
    tmp = _tmp_path(dest)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to atomically write json: {dest}") from exc

