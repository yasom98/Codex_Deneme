"""Feature engineering indicator core for standardized OHLCV parquet tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.logging import get_logger

LOGGER = get_logger(__name__)

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
CONTINUOUS_FEATURE_COLUMNS: tuple[str, ...] = (
    "supertrend",
    "alphatrend",
    "rsi",
    "rsi_slope",
    "rsi_zscore",
)
EVENT_FLAG_COLUMNS: tuple[str, ...] = (
    "evt_supertrend_flip_up",
    "evt_supertrend_flip_down",
    "evt_alphatrend_flip_up",
    "evt_alphatrend_flip_down",
    "evt_rsi_cross_center_up",
    "evt_rsi_cross_center_down",
    "evt_rsi_overbought",
    "evt_rsi_oversold",
)

_ALLOWED_RULE_OPERATORS: tuple[str, ...] = (">", ">=", "<", "<=", "==", "!=")


@dataclass(frozen=True)
class ThresholdRule:
    """Declarative threshold rule for AlphaTrend regime conditions."""

    signal: str
    operator: str
    threshold: float


@dataclass(frozen=True)
class SuperTrendConfig:
    """SuperTrend hyperparameters."""

    period: int
    multiplier: float


@dataclass(frozen=True)
class AlphaTrendConfig:
    """AlphaTrend hyperparameters and rule definitions."""

    period: int
    atr_multiplier: float
    signal_period: int
    long_rule: ThresholdRule
    short_rule: ThresholdRule


@dataclass(frozen=True)
class RsiConfig:
    """RSI derivatives hyperparameters."""

    period: int
    slope_lag: int
    zscore_window: int


@dataclass(frozen=True)
class EventConfig:
    """Event threshold settings for RSI-based and trend-flip events."""

    rsi_centerline: float
    rsi_overbought: float
    rsi_oversold: float


def validate_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an OHLCV dataframe.

    Rules:
    - required columns must exist
    - timestamp is sorted increasingly (sort with warning if needed)
    - duplicate timestamps are dropped with `keep=last`
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    out = df.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()

    if not pd.api.types.is_datetime64tz_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    if out["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid or NaT values.")

    if not out["timestamp"].is_monotonic_increasing:
        LOGGER.warning("timestamp not monotonic increasing; sorting in ascending order.")
        out = out.sort_values("timestamp", kind="mergesort")

    duplicated_mask = out.duplicated(subset=["timestamp"], keep="last")
    dropped_duplicates = int(duplicated_mask.sum())
    if dropped_duplicates > 0:
        LOGGER.warning("duplicate timestamps detected; dropping duplicates with keep='last' | dropped=%d", dropped_duplicates)
        out = out.loc[~duplicated_mask].copy()

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float32")

    return out.reset_index(drop=True)


def rma(series: pd.Series, period: int) -> pd.Series:
    """Compute Wilder's RMA (running moving average)."""

    if period <= 0:
        raise ValueError("period must be > 0")
    return series.ewm(alpha=1.0 / float(period), adjust=False, min_periods=1).mean()


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range for OHLCV rows."""

    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    prev_close = df["close"].astype("float64").shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.fillna(0.0)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute Average True Range using Wilder's RMA."""

    true_range = compute_true_range(df)
    return rma(true_range, period=period)


def compute_supertrend(df: pd.DataFrame, cfg: SuperTrendConfig) -> tuple[pd.Series, pd.Series]:
    """Compute SuperTrend line and direction (+1/-1)."""

    if cfg.period <= 0:
        raise ValueError("SuperTrend period must be > 0")
    if cfg.multiplier <= 0:
        raise ValueError("SuperTrend multiplier must be > 0")

    rows = len(df)
    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)
    close = df["close"].to_numpy(dtype=np.float64, copy=False)

    atr = compute_atr(df, period=cfg.period).to_numpy(dtype=np.float64, copy=False)
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + (cfg.multiplier * atr)
    basic_lower = hl2 - (cfg.multiplier * atr)

    final_upper = np.empty(rows, dtype=np.float64)
    final_lower = np.empty(rows, dtype=np.float64)
    supertrend = np.empty(rows, dtype=np.float64)
    direction = np.empty(rows, dtype=np.int8)

    for idx in range(rows):
        if idx == 0:
            final_upper[idx] = basic_upper[idx]
            final_lower[idx] = basic_lower[idx]
            supertrend[idx] = final_lower[idx] if close[idx] >= final_lower[idx] else final_upper[idx]
            direction[idx] = 1 if close[idx] >= supertrend[idx] else -1
            continue

        prev_close = close[idx - 1]
        prev_final_upper = final_upper[idx - 1]
        prev_final_lower = final_lower[idx - 1]

        if basic_upper[idx] < prev_final_upper or prev_close > prev_final_upper:
            final_upper[idx] = basic_upper[idx]
        else:
            final_upper[idx] = prev_final_upper

        if basic_lower[idx] > prev_final_lower or prev_close < prev_final_lower:
            final_lower[idx] = basic_lower[idx]
        else:
            final_lower[idx] = prev_final_lower

        prev_supertrend = supertrend[idx - 1]
        if prev_supertrend == prev_final_upper:
            supertrend[idx] = final_upper[idx] if close[idx] <= final_upper[idx] else final_lower[idx]
        else:
            supertrend[idx] = final_lower[idx] if close[idx] >= final_lower[idx] else final_upper[idx]

        direction[idx] = 1 if close[idx] >= supertrend[idx] else -1

    return (
        pd.Series(supertrend, index=df.index, name="supertrend", dtype="float32"),
        pd.Series(direction, index=df.index, name="supertrend_direction_raw", dtype="int8"),
    )


def compute_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute RSI in range [0, 100]."""

    if period <= 0:
        raise ValueError("RSI period must be > 0")

    close = df["close"].astype("float64")
    delta = close.diff().fillna(0.0)

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = rma(gain, period=period)
    avg_loss = rma(loss, period=period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    only_loss_zero = (avg_loss == 0.0) & ~both_zero

    rsi = rsi.where(~only_loss_zero, 100.0)
    rsi = rsi.where(~both_zero, 50.0)

    return rsi.fillna(50.0).astype("float32")


def compute_mfi(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute Money Flow Index in range [0, 100]."""

    if period <= 0:
        raise ValueError("MFI period must be > 0")

    typical_price = (df["high"].astype("float64") + df["low"].astype("float64") + df["close"].astype("float64")) / 3.0
    raw_money_flow = typical_price * df["volume"].astype("float64")

    tp_delta = typical_price.diff().fillna(0.0)
    positive_flow = raw_money_flow.where(tp_delta > 0.0, 0.0)
    negative_flow = raw_money_flow.where(tp_delta < 0.0, 0.0)

    positive_sum = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_sum = negative_flow.rolling(window=period, min_periods=1).sum()

    ratio = positive_sum / negative_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))

    both_zero = (positive_sum == 0.0) & (negative_sum == 0.0)
    only_negative_zero = (negative_sum == 0.0) & ~both_zero

    mfi = mfi.where(~only_negative_zero, 100.0)
    mfi = mfi.where(~both_zero, 50.0)

    return mfi.fillna(50.0).astype("float32")


def compute_rsi_derivatives(df: pd.DataFrame, cfg: RsiConfig) -> pd.DataFrame:
    """Compute RSI, RSI slope, and RSI z-score derivatives."""

    if cfg.period <= 0:
        raise ValueError("rsi.period must be > 0")
    if cfg.slope_lag <= 0:
        raise ValueError("rsi.slope_lag must be > 0")
    if cfg.zscore_window <= 1:
        raise ValueError("rsi.zscore_window must be > 1")

    rsi = compute_rsi(df, period=cfg.period).astype("float64")
    rsi_slope = (rsi - rsi.shift(cfg.slope_lag)).fillna(0.0)

    rolling_mean = rsi.rolling(window=cfg.zscore_window, min_periods=1).mean()
    rolling_std = rsi.rolling(window=cfg.zscore_window, min_periods=1).std(ddof=0)
    rsi_zscore = ((rsi - rolling_mean) / rolling_std.replace(0.0, np.nan)).fillna(0.0)

    return pd.DataFrame(
        {
            "rsi": rsi.astype("float32"),
            "rsi_slope": rsi_slope.astype("float32"),
            "rsi_zscore": rsi_zscore.astype("float32"),
        },
        index=df.index,
    )


def _operator_callable(operator: str) -> Callable[[pd.Series, float], pd.Series]:
    if operator == ">":
        return lambda series, threshold: series > threshold
    if operator == ">=":
        return lambda series, threshold: series >= threshold
    if operator == "<":
        return lambda series, threshold: series < threshold
    if operator == "<=":
        return lambda series, threshold: series <= threshold
    if operator == "==":
        return lambda series, threshold: series == threshold
    if operator == "!=":
        return lambda series, threshold: series != threshold
    raise ValueError(f"Unsupported rule operator: {operator}")


def evaluate_rule(signal: pd.Series, rule: ThresholdRule) -> pd.Series:
    """Evaluate a threshold rule against signal values."""

    if rule.operator not in _ALLOWED_RULE_OPERATORS:
        raise ValueError(f"Unsupported rule operator: {rule.operator}")
    return _operator_callable(rule.operator)(signal, float(rule.threshold))


def _resolve_signal(df: pd.DataFrame, signal_name: str, period: int) -> pd.Series:
    """Resolve a named signal series for rule evaluation."""

    lowered = signal_name.strip().lower()
    if lowered == "rsi":
        return compute_rsi(df, period=period)
    if lowered == "mfi":
        return compute_mfi(df, period=period)
    if lowered == "close":
        return df["close"].astype("float32")

    raise ValueError(f"Unsupported signal for AlphaTrend rule: {signal_name}")


def compute_alphatrend(df: pd.DataFrame, cfg: AlphaTrendConfig) -> tuple[pd.Series, pd.Series]:
    """Compute AlphaTrend line and direction (+1/-1) from configurable threshold rules."""

    if cfg.period <= 0:
        raise ValueError("alphatrend.period must be > 0")
    if cfg.signal_period <= 0:
        raise ValueError("alphatrend.signal_period must be > 0")
    if cfg.atr_multiplier <= 0:
        raise ValueError("alphatrend.atr_multiplier must be > 0")

    if not cfg.long_rule.signal.strip() or not cfg.short_rule.signal.strip():
        raise ValueError("alphatrend long_rule.signal and short_rule.signal are required")

    atr = compute_atr(df, period=cfg.period).astype("float64")
    low = df["low"].astype("float64")
    high = df["high"].astype("float64")

    up_level = low - (cfg.atr_multiplier * atr)
    down_level = high + (cfg.atr_multiplier * atr)

    long_signal = _resolve_signal(df, cfg.long_rule.signal, period=cfg.signal_period)
    short_signal = _resolve_signal(df, cfg.short_rule.signal, period=cfg.signal_period)

    long_condition = evaluate_rule(long_signal, cfg.long_rule)
    short_condition = evaluate_rule(short_signal, cfg.short_rule)

    rows = len(df)
    alphatrend = np.empty(rows, dtype=np.float64)
    direction = np.empty(rows, dtype=np.int8)

    up_arr = up_level.to_numpy(dtype=np.float64, copy=False)
    down_arr = down_level.to_numpy(dtype=np.float64, copy=False)
    long_arr = long_condition.to_numpy(dtype=np.bool_, copy=False)
    short_arr = short_condition.to_numpy(dtype=np.bool_, copy=False)

    for idx in range(rows):
        if idx == 0:
            alphatrend[idx] = up_arr[idx]
            direction[idx] = 1
            continue

        long_active = bool(long_arr[idx] and not short_arr[idx])
        short_active = bool(short_arr[idx] and not long_arr[idx])

        if long_active:
            alphatrend[idx] = max(alphatrend[idx - 1], up_arr[idx])
            direction[idx] = 1
        elif short_active:
            alphatrend[idx] = min(alphatrend[idx - 1], down_arr[idx])
            direction[idx] = -1
        else:
            alphatrend[idx] = alphatrend[idx - 1]
            direction[idx] = direction[idx - 1]

    return (
        pd.Series(alphatrend, index=df.index, name="alphatrend", dtype="float32"),
        pd.Series(direction, index=df.index, name="alphatrend_direction_raw", dtype="int8"),
    )


def compute_indicator_core(
    df: pd.DataFrame,
    supertrend_cfg: SuperTrendConfig,
    alphatrend_cfg: AlphaTrendConfig,
    rsi_cfg: RsiConfig,
) -> pd.DataFrame:
    """Compute core continuous indicators and internal trend directions."""

    ohlcv = validate_ohlcv_frame(df)

    supertrend, supertrend_direction = compute_supertrend(ohlcv, supertrend_cfg)
    alphatrend, alphatrend_direction = compute_alphatrend(ohlcv, alphatrend_cfg)
    rsi_derivatives = compute_rsi_derivatives(ohlcv, rsi_cfg)

    features = pd.concat(
        [
            supertrend,
            alphatrend,
            rsi_derivatives,
            supertrend_direction,
            alphatrend_direction,
        ],
        axis=1,
    )

    return features


def generate_raw_event_flags(indicators: pd.DataFrame, cfg: EventConfig) -> pd.DataFrame:
    """Generate raw event flags before leak-free shift(1) enforcement."""

    required = {"supertrend_direction_raw", "alphatrend_direction_raw", "rsi"}
    missing = sorted(required.difference(indicators.columns))
    if missing:
        raise ValueError(f"Missing required indicator columns for event generation: {missing}")

    if cfg.rsi_oversold >= cfg.rsi_overbought:
        raise ValueError("events.rsi_oversold must be strictly less than events.rsi_overbought")

    super_dir = indicators["supertrend_direction_raw"].astype("int8")
    alpha_dir = indicators["alphatrend_direction_raw"].astype("int8")
    rsi = indicators["rsi"].astype("float32")

    prev_super_dir = super_dir.shift(1).fillna(super_dir.iloc[0]).astype("int8")
    prev_alpha_dir = alpha_dir.shift(1).fillna(alpha_dir.iloc[0]).astype("int8")
    prev_rsi = rsi.shift(1).fillna(rsi.iloc[0]).astype("float32")

    raw_events = pd.DataFrame(
        {
            "evt_supertrend_flip_up": (super_dir.eq(1) & prev_super_dir.eq(-1)),
            "evt_supertrend_flip_down": (super_dir.eq(-1) & prev_super_dir.eq(1)),
            "evt_alphatrend_flip_up": (alpha_dir.eq(1) & prev_alpha_dir.eq(-1)),
            "evt_alphatrend_flip_down": (alpha_dir.eq(-1) & prev_alpha_dir.eq(1)),
            "evt_rsi_cross_center_up": (rsi >= cfg.rsi_centerline) & (prev_rsi < cfg.rsi_centerline),
            "evt_rsi_cross_center_down": (rsi <= cfg.rsi_centerline) & (prev_rsi > cfg.rsi_centerline),
            "evt_rsi_overbought": (rsi >= cfg.rsi_overbought) & (prev_rsi < cfg.rsi_overbought),
            "evt_rsi_oversold": (rsi <= cfg.rsi_oversold) & (prev_rsi > cfg.rsi_oversold),
        },
        index=indicators.index,
    )
    return raw_events


def _shift_flag_one(raw_flag: pd.Series) -> pd.Series:
    """Shift a boolean event by exactly one bar and convert to uint8."""

    raw_uint8 = raw_flag.fillna(False).astype("uint8").to_numpy(dtype=np.uint8, copy=False)
    shifted = np.zeros(len(raw_uint8), dtype=np.uint8)
    if len(raw_uint8) > 1:
        shifted[1:] = raw_uint8[:-1]
    return pd.Series(shifted, index=raw_flag.index, dtype="uint8")


def enforce_shift_one(raw_events: pd.DataFrame) -> pd.DataFrame:
    """Enforce strict leak-free shift(1) for every event column."""

    out = pd.DataFrame(index=raw_events.index)
    for col in EVENT_FLAG_COLUMNS:
        if col not in raw_events.columns:
            raise ValueError(f"Missing raw event column: {col}")
        out[col] = _shift_flag_one(raw_events[col])
    return out


def validate_shift_one(raw_events: pd.DataFrame, shifted_events: pd.DataFrame) -> bool:
    """Validate strict shift(1) relation between raw and final event flags."""

    for col in EVENT_FLAG_COLUMNS:
        if col not in raw_events.columns or col not in shifted_events.columns:
            return False

        raw_uint8 = raw_events[col].fillna(False).astype("uint8").to_numpy(dtype=np.uint8, copy=False)
        shifted_uint8 = shifted_events[col].fillna(0).astype("uint8").to_numpy(dtype=np.uint8, copy=False)

        if len(raw_uint8) != len(shifted_uint8):
            return False
        if len(raw_uint8) == 0:
            continue
        if shifted_uint8[0] != 0:
            return False
        if len(raw_uint8) > 1 and not np.array_equal(shifted_uint8[1:], raw_uint8[:-1]):
            return False
    return True
