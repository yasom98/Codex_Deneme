"""Tests for timestamp alias detection and normalization behavior."""

from __future__ import annotations

import pytest

from data.schema import detect_timestamp_alias


def test_datetime_alias_mapping_variations() -> None:
    columns = [" Date Time ", "open", "high", "low", "close", "volume", "CANDLE-TIME"]
    aliases = ["timestamp", "ts", "date", "datetime", "time", "candle_time", "open_time", "close_time"]
    detection = detect_timestamp_alias(columns, aliases)
    assert detection.selected_alias == "candle_time"
    assert detection.selected_column == "CANDLE-TIME"
    assert detection.confidence == 1.0


def test_datetime_alias_priority_when_multiple_candidates_exist() -> None:
    columns = ["time", "datetime", "TS", "open", "high", "low", "close", "volume"]
    aliases = ["timestamp", "ts", "date", "datetime", "time", "candle_time", "open_time", "close_time"]
    detection = detect_timestamp_alias(columns, aliases)
    assert detection.selected_alias == "ts"
    assert detection.selected_column == "TS"
    assert detection.confidence == 0.85


def test_detect_timestamp_alias_raises_when_missing() -> None:
    columns = ["open", "high", "low", "close", "volume"]
    aliases = ["timestamp", "ts"]
    with pytest.raises(ValueError, match="No datetime alias found"):
        detect_timestamp_alias(columns, aliases)
