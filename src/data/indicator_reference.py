"""Indicator reference implementations used as formula source-of-truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from core.logging import get_logger

LOGGER = get_logger(__name__)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Return true range series."""

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def sma(x: pd.Series, n: int) -> pd.Series:
    """Return simple moving average with strict warmup."""

    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """Return exponential moving average with strict warmup."""

    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Return RSI values using Wilder-style EWMA."""

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    """Return Money Flow Index values."""

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


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return strict bar-close cross-above signal."""

    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return strict bar-close cross-below signal."""

    return (a < b) & (a.shift(1) >= b.shift(1))


def _validate_ohlcv_index(df: pd.DataFrame) -> None:
    """Validate that dataframe index is datetime and monotonic."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Indicator reference expects a DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware (UTC).")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be monotonic increasing.")


def compute_pivots_traditional(
    df: pd.DataFrame,
    pivot_tf: str = "1D",
) -> pd.DataFrame:
    """Compute leak-free traditional pivots from previous higher-timeframe bar."""

    _validate_ohlcv_index(df)
    req = {"open", "high", "low", "close"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    ohlc = df[["open", "high", "low", "close"]].resample(pivot_tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )

    prev = ohlc.shift(1)
    pp = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    rng = prev["high"] - prev["low"]

    r1 = (pp * 2) - prev["low"]
    s1 = (pp * 2) - prev["high"]
    r2 = pp + rng
    s2 = pp - rng
    r3 = (pp * 2) + prev["high"] - (2 * prev["low"])
    s3 = (pp * 2) - ((2 * prev["high"]) - prev["low"])
    r4 = (pp * 3) + prev["high"] - (3 * prev["low"])
    s4 = (pp * 3) - ((3 * prev["high"]) - prev["low"])
    r5 = (pp * 4) + prev["high"] - (4 * prev["low"])
    s5 = (pp * 4) - ((4 * prev["high"]) - prev["low"])

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
    return piv.astype("float32")


def compute_ema_set(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Compute locked EMA set (200/600/1200)."""

    if price_col not in df.columns:
        raise ValueError(f"Missing price_col: {price_col}")
    price = df[price_col]
    e200 = ema(price, 200)
    e600 = ema(price, 600)
    e1200 = ema(price, 1200)
    return pd.DataFrame(
        {
            "EMA_200": e200.astype("float32"),
            "EMA_600": e600.astype("float32"),
            "EMA_1200": e1200.astype("float32"),
        },
        index=df.index,
    )


@dataclass(frozen=True)
class AlphaTrendConfig:
    """Reference AlphaTrend configuration."""

    coeff: float = 3.0
    ap: int = 11
    use_no_volume: bool = False


def _barssince(cond: pd.Series) -> pd.Series:
    """Return number of bars since last True."""

    out = np.full(len(cond), np.nan, dtype=np.float64)
    last = -1
    cond_bool = cond.astype("boolean").fillna(False).astype(bool)
    for i, v in enumerate(cond_bool.to_numpy(dtype=np.bool_, copy=False)):
        if v:
            last = i
            out[i] = 0.0
        else:
            out[i] = (i - last) if last >= 0 else np.nan
    return pd.Series(out, index=cond.index)


def compute_alphatrend(
    df: pd.DataFrame,
    cfg: AlphaTrendConfig = AlphaTrendConfig(),
    show_progress: bool = False,
    precomputed_tr: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute reference AlphaTrend and strict buy/sell signal set."""

    del show_progress
    req = {"high", "low", "close"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    high, low, close = df["high"], df["low"], df["close"]
    tr = true_range(high, low, close) if precomputed_tr is None else precomputed_tr
    if len(tr) != len(df):
        raise ValueError("precomputed_tr length mismatch for AlphaTrend")
    if not tr.index.equals(df.index):
        raise ValueError("precomputed_tr index mismatch for AlphaTrend")
    atr = sma(tr, cfg.ap)
    up_t = low - (atr * cfg.coeff)
    down_t = high + (atr * cfg.coeff)

    if cfg.use_no_volume or ("volume" not in df.columns):
        regime = rsi(close, cfg.ap) >= 50
    else:
        regime = mfi(high, low, close, df["volume"], cfg.ap) >= 50

    rows = len(df)
    at = np.full(rows, np.nan, dtype=np.float64)
    up_arr = up_t.to_numpy(dtype=np.float64, copy=False)
    down_arr = down_t.to_numpy(dtype=np.float64, copy=False)
    regime_arr = regime.to_numpy(dtype=np.bool_, copy=False)
    it: Iterable[int] = range(rows)
    for i in it:
        if i == 0:
            at[i] = np.nan
            continue
        prev_at = at[i - 1]
        if np.isnan(prev_at):
            at[i] = up_arr[i] if regime_arr[i] else down_arr[i]
            continue
        if regime_arr[i]:
            at[i] = prev_at if up_arr[i] < prev_at else up_arr[i]
        else:
            at[i] = prev_at if down_arr[i] > prev_at else down_arr[i]

    at_s = pd.Series(at, index=df.index, name="AlphaTrend")
    at2 = at_s.shift(2)
    buy_raw = crossover(at_s, at2)
    sell_raw = crossunder(at_s, at2)

    k1 = _barssince(buy_raw)
    k2 = _barssince(sell_raw)
    o1 = _barssince(buy_raw.shift(1))
    o2 = _barssince(sell_raw.shift(1))
    buy_confirmed = (buy_raw & (o1 > k2)).fillna(False)
    sell_confirmed = (sell_raw & (o2 > k1)).fillna(False)

    return pd.DataFrame(
        {
            "AlphaTrend": at_s.astype("float32"),
            "AlphaTrend_2": at2.astype("float32"),
            "AT_buy_raw": buy_raw.fillna(False).astype("uint8"),
            "AT_sell_raw": sell_raw.fillna(False).astype("uint8"),
            "AT_buy": buy_confirmed.astype("uint8"),
            "AT_sell": sell_confirmed.astype("uint8"),
        },
        index=df.index,
    )


@dataclass(frozen=True)
class SupertrendConfig:
    """Reference Supertrend configuration."""

    periods: int = 10
    multiplier: float = 3.0
    source: str = "hl2"
    change_atr_method: bool = True


def _wilder_atr(tr: pd.Series, n: int) -> pd.Series:
    """Return Wilder ATR series."""

    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_supertrend(
    df: pd.DataFrame,
    cfg: SupertrendConfig = SupertrendConfig(),
    show_progress: bool = False,
    precomputed_tr: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute reference Supertrend and strict buy/sell signal set."""

    del show_progress
    req = {"open", "high", "low", "close"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    high, low, close = df["high"], df["low"], df["close"]
    if cfg.source == "hl2":
        src = (high + low) / 2.0
    else:
        src = close

    tr = true_range(high, low, close) if precomputed_tr is None else precomputed_tr
    if len(tr) != len(df):
        raise ValueError("precomputed_tr length mismatch for Supertrend")
    if not tr.index.equals(df.index):
        raise ValueError("precomputed_tr index mismatch for Supertrend")
    atr = _wilder_atr(tr, cfg.periods) if cfg.change_atr_method else sma(tr, cfg.periods)
    up = src - (cfg.multiplier * atr)
    dn = src + (cfg.multiplier * atr)

    rows = len(df)
    up_arr = up.to_numpy(dtype=np.float64, copy=False)
    dn_arr = dn.to_numpy(dtype=np.float64, copy=False)
    close_arr = close.to_numpy(dtype=np.float64, copy=False)

    up_band = np.full(rows, np.nan, dtype=np.float64)
    dn_band = np.full(rows, np.nan, dtype=np.float64)
    trend = np.full(rows, np.nan, dtype=np.float64)

    for i in range(rows):
        if i == 0:
            up_band[i] = up_arr[i]
            dn_band[i] = dn_arr[i]
            trend[i] = 1.0
            continue

        up1 = up_band[i - 1]
        dn1 = dn_band[i - 1]

        if close_arr[i - 1] > up1:
            up_band[i] = max(up_arr[i], up1)
        else:
            up_band[i] = up_arr[i]

        if close_arr[i - 1] < dn1:
            dn_band[i] = min(dn_arr[i], dn1)
        else:
            dn_band[i] = dn_arr[i]

        prev_trend = trend[i - 1]
        if (prev_trend == -1.0) and (close_arr[i] > dn1):
            trend[i] = 1.0
        elif (prev_trend == 1.0) and (close_arr[i] < up1):
            trend[i] = -1.0
        else:
            trend[i] = prev_trend

    trend_s = pd.Series(trend, index=df.index).astype("int8")
    up_s = pd.Series(up_band, index=df.index)
    dn_s = pd.Series(dn_band, index=df.index)

    st_buy = ((trend_s == 1) & (trend_s.shift(1) == -1)).fillna(False)
    st_sell = ((trend_s == -1) & (trend_s.shift(1) == 1)).fillna(False)

    st_up = up_s.where(trend_s == 1).astype("float32")
    st_dn = dn_s.where(trend_s == -1).astype("float32")
    return pd.DataFrame(
        {
            "ST_trend": trend_s,
            "ST_up": st_up,
            "ST_dn": st_dn,
            "ST_buy": st_buy.astype("uint8"),
            "ST_sell": st_sell.astype("uint8"),
        },
        index=df.index,
    )
