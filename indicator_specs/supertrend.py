"""User-provided SuperTrend reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute true range."""

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def sma(x: pd.Series, n: int) -> pd.Series:
    """Compute simple moving average."""

    return x.rolling(n, min_periods=n).mean()


@dataclass
class SupertrendConfig:
    """SuperTrend config."""

    periods: int = 10
    multiplier: float = 3.0
    source: Literal["hl2", "close", "ohlc4"] = "hl2"
    change_atr_method: bool = True
    showsignals: bool = True


def _wilder_atr(tr: pd.Series, n: int) -> pd.Series:
    """Wilder-style ATR smoothing."""

    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_supertrend(
    df: pd.DataFrame,
    cfg: SupertrendConfig = SupertrendConfig(),
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute SuperTrend outputs."""

    req_cols = {"high", "low", "close", "open"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]

    if cfg.source == "hl2":
        src = (high + low) / 2.0
    elif cfg.source == "ohlc4":
        src = (open_ + high + low + close) / 4.0
    else:
        src = close

    tr = true_range(high, low, close)
    atr = _wilder_atr(tr, cfg.periods) if cfg.change_atr_method else sma(tr, cfg.periods)

    up = src - (cfg.multiplier * atr)
    dn = src + (cfg.multiplier * atr)

    up_band = np.full(len(df), np.nan, dtype=np.float64)
    dn_band = np.full(len(df), np.nan, dtype=np.float64)
    trend = np.full(len(df), np.nan, dtype=np.float64)

    it = range(len(df))
    if show_progress:
        it = tqdm(it, desc="Supertrend", leave=False)

    for i in it:
        if i == 0:
            up_band[i] = up.iat[i]
            dn_band[i] = dn.iat[i]
            trend[i] = 1.0
            continue

        up1 = up_band[i - 1]
        dn1 = dn_band[i - 1]

        if close.iat[i - 1] > up1:
            up_band[i] = max(up.iat[i], up1)
        else:
            up_band[i] = up.iat[i]

        if close.iat[i - 1] < dn1:
            dn_band[i] = min(dn.iat[i], dn1)
        else:
            dn_band[i] = dn.iat[i]

        prev_trend = trend[i - 1]

        if (prev_trend == -1.0) and (close.iat[i] > dn1):
            trend[i] = 1.0
        elif (prev_trend == 1.0) and (close.iat[i] < up1):
            trend[i] = -1.0
        else:
            trend[i] = prev_trend

    trend_s = pd.Series(trend, index=df.index)
    up_s = pd.Series(up_band, index=df.index)
    dn_s = pd.Series(dn_band, index=df.index)

    buy = ((trend_s == 1) & (trend_s.shift(1) == -1)).astype("uint8")
    sell = ((trend_s == -1) & (trend_s.shift(1) == 1)).astype("uint8")
    change = (trend_s != trend_s.shift(1)).astype("uint8")

    up_plot = up_s.where(trend_s == 1).astype("float32")
    dn_plot = dn_s.where(trend_s == -1).astype("float32")

    return pd.DataFrame(
        {
            "ST_trend": trend_s.astype("int8"),
            "ST_up": up_plot,
            "ST_dn": dn_plot,
            "ST_buy": buy,
            "ST_sell": sell,
            "ST_change": change,
        },
        index=df.index,
    )
