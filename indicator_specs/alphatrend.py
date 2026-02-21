"""User-provided AlphaTrend reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

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


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Compute RSI."""

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    """Compute MFI."""

    tp = (high + low + close) / 3.0
    mf = tp * volume
    direction = tp.diff()
    pos = mf.where(direction > 0, 0.0)
    neg = mf.where(direction < 0, 0.0)

    pos_sum = pos.rolling(n, min_periods=n).sum()
    neg_sum = neg.rolling(n, min_periods=n).sum().replace(0, np.nan)

    mr = pos_sum / neg_sum
    out = 100 - (100 / (1 + mr))
    return out.fillna(50.0)


def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    """Cross-up event."""

    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    """Cross-down event."""

    return (a < b) & (a.shift(1) >= b.shift(1))


@dataclass
class AlphaTrendConfig:
    """AlphaTrend config."""

    coeff: float = 3.0
    ap: int = 11
    use_no_volume: bool = False


def compute_alphatrend(
    df: pd.DataFrame,
    cfg: AlphaTrendConfig = AlphaTrendConfig(),
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute AlphaTrend outputs."""

    req_cols = {"high", "low", "close"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    high, low, close = df["high"], df["low"], df["close"]

    tr = true_range(high, low, close)
    atr = sma(tr, cfg.ap)

    up_t = low - atr * cfg.coeff
    down_t = high + atr * cfg.coeff

    if cfg.use_no_volume or ("volume" not in df.columns):
        regime = rsi(close, cfg.ap) >= 50
    else:
        regime = mfi(high, low, close, df["volume"], cfg.ap) >= 50

    at = np.full(len(df), np.nan, dtype=np.float64)

    it = range(len(df))
    if show_progress:
        it = tqdm(it, desc="AlphaTrend", leave=False)

    prev_at = np.nan

    for i in it:
        if i == 0:
            prev_at = np.nan
        else:
            prev_at = at[i - 1]

        if np.isnan(prev_at):
            at[i] = up_t.iat[i] if regime.iat[i] else down_t.iat[i]
            continue

        if regime.iat[i]:
            at[i] = prev_at if up_t.iat[i] < prev_at else up_t.iat[i]
        else:
            at[i] = prev_at if down_t.iat[i] > prev_at else down_t.iat[i]

    at = pd.Series(at, index=df.index, name="AlphaTrend")

    at2 = at.shift(2)

    buy = cross_up(at, at2).astype("uint8")
    sell = cross_down(at, at2).astype("uint8")

    out = pd.DataFrame(
        {
            "AlphaTrend": at.astype("float32"),
            "AlphaTrend_2": at2.astype("float32"),
            "AT_buy": buy,
            "AT_sell": sell,
        },
        index=df.index,
    )

    return out
