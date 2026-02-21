"""User-provided pivot reference implementation."""

from __future__ import annotations

from typing import Literal

import pandas as pd

PivotType = Literal["Traditional", "Fibonacci"]


def compute_pivots_intraday(
    df: pd.DataFrame,
    pivot_tf: str = "1D",
    pivot_type: PivotType = "Traditional",
) -> pd.DataFrame:
    """
    Leak-free Pivot:
    - HTF OHLC is resampled
    - pivots are based on PREVIOUS HTF bar (shift(1))
    - mapped back to intraday and ffilled
    """

    req_cols = {"open", "high", "low", "close"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    ohlc = df[["open", "high", "low", "close"]].resample(pivot_tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )

    prev = ohlc.shift(1)
    pp = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    rng = prev["high"] - prev["low"]

    if pivot_type == "Traditional":
        r1 = pp * 2 - prev["low"]
        s1 = pp * 2 - prev["high"]
        r2 = pp + rng
        s2 = pp - rng
        r3 = pp * 2 + prev["high"] - 2 * prev["low"]
        s3 = pp * 2 - (2 * prev["high"] - prev["low"])
        r4 = pp * 3 + prev["high"] - 3 * prev["low"]
        s4 = pp * 3 - (3 * prev["high"] - prev["low"])
        r5 = pp * 4 + prev["high"] - 4 * prev["low"]
        s5 = pp * 4 - (4 * prev["high"] - prev["low"])
    else:
        r1 = pp + 0.382 * rng
        s1 = pp - 0.382 * rng
        r2 = pp + 0.618 * rng
        s2 = pp - 0.618 * rng
        r3 = pp + 1.000 * rng
        s3 = pp - 1.000 * rng
        r4 = pp + 1.382 * rng
        s4 = pp - 1.382 * rng
        r5 = pp + 1.618 * rng
        s5 = pp - 1.618 * rng

    piv_htf = pd.DataFrame(
        {
            "PP": pp,
            "R1": r1,
            "S1": s1,
            "R2": r2,
            "S2": s2,
            "R3": r3,
            "S3": s3,
            "R4": r4,
            "S4": s4,
            "R5": r5,
            "S5": s5,
        },
        index=ohlc.index,
    )

    piv = piv_htf.reindex(df.index, method="ffill")
    return piv
