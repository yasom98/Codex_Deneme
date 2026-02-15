"""Tests for CSV file discovery utility."""

from __future__ import annotations

from pathlib import Path

from core.paths import discover_csv_files


def test_discover_csv_files_skips_hidden_and_system_entries(tmp_path: Path) -> None:
    visible = tmp_path / "market" / "BTCUSDT.csv"
    hidden_dir = tmp_path / ".hidden" / "ETHUSDT.csv"
    macos_dir = tmp_path / "__MACOSX" / "LTCUSDT.csv"
    thumbs = tmp_path / "market" / "Thumbs.db.csv"

    visible.parent.mkdir(parents=True, exist_ok=True)
    hidden_dir.parent.mkdir(parents=True, exist_ok=True)
    macos_dir.parent.mkdir(parents=True, exist_ok=True)
    thumbs.parent.mkdir(parents=True, exist_ok=True)

    visible.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    hidden_dir.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    macos_dir.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    thumbs.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

    found = discover_csv_files(tmp_path)
    assert found == [visible]
