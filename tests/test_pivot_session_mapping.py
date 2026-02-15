"""Regression tests for daily pivot session mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.features import compute_daily_pivots_with_std_bands


def test_daily_pivots_map_from_previous_session_only() -> None:
    ts = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:05:00Z",
            "2024-01-01T00:10:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-02T00:05:00Z",
            "2024-01-02T00:10:00Z",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.array([10.2, 12.2, 13.2, 13.0, 12.5, 12.2], dtype=np.float32),
            "high": np.array([12.0, 13.0, 14.0, 13.5, 13.0, 12.8], dtype=np.float32),
            "low": np.array([9.0, 10.0, 11.0, 11.8, 11.7, 11.6], dtype=np.float32),
            "close": np.array([10.0, 12.0, 13.0, 12.8, 12.2, 11.9], dtype=np.float32),
            "volume": np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0], dtype=np.float32),
        }
    )

    pivots = compute_daily_pivots_with_std_bands(df)
    day_1_mask = df["timestamp"].dt.floor("D") == pd.Timestamp("2024-01-01T00:00:00Z")
    day_2_mask = df["timestamp"].dt.floor("D") == pd.Timestamp("2024-01-02T00:00:00Z")

    assert pivots.loc[day_1_mask, "pivot_p"].isna().all()

    expected_p = (14.0 + 9.0 + 13.0) / 3.0
    expected_values = {
        "pivot_p": expected_p,
        "pivot_r1": (2.0 * expected_p) - 9.0,
        "pivot_s1": (2.0 * expected_p) - 14.0,
        "pivot_r2": expected_p + (14.0 - 9.0),
        "pivot_s2": expected_p - (14.0 - 9.0),
        "pivot_r3": 14.0 + (2.0 * (expected_p - 9.0)),
        "pivot_s3": 9.0 - (2.0 * (14.0 - expected_p)),
    }

    day_1_std = float(np.std(np.array([10.0, 12.0, 13.0], dtype=np.float64), ddof=0))
    expected_values["pivot_std_upper_1"] = expected_p + day_1_std
    expected_values["pivot_std_upper_2"] = expected_p + (2.0 * day_1_std)
    expected_values["pivot_std_upper_3"] = expected_p + (3.0 * day_1_std)
    expected_values["pivot_std_lower_1"] = expected_p - day_1_std
    expected_values["pivot_std_lower_2"] = expected_p - (2.0 * day_1_std)
    expected_values["pivot_std_lower_3"] = expected_p - (3.0 * day_1_std)

    for col, expected in expected_values.items():
        np.testing.assert_allclose(
            pivots.loc[day_2_mask, col].to_numpy(dtype=np.float64),
            np.full(int(day_2_mask.sum()), expected, dtype=np.float64),
            atol=1e-5,
            rtol=0.0,
        )
