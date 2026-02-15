"""Feature engineering indicator core for standardized OHLCV parquet tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core.logging import get_logger
from core.paths import ensure_within_root

LOGGER = get_logger(__name__)

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
ALPHATREND_LOCK_PERIOD: int = 11
ALPHATREND_LOCK_ATR_MULTIPLIER: float = 3.0
CONTINUOUS_FEATURE_COLUMNS: tuple[str, ...] = (
    "supertrend",
    "alphatrend",
    "rsi",
    "rsi_slope",
    "rsi_zscore",
    "ema_200",
    "ema_600",
    "ema_1200",
    "pivot_p",
    "pivot_s1",
    "pivot_s2",
    "pivot_s3",
    "pivot_r1",
    "pivot_r2",
    "pivot_r3",
    "pivot_std_upper_1",
    "pivot_std_upper_2",
    "pivot_std_upper_3",
    "pivot_std_lower_1",
    "pivot_std_lower_2",
    "pivot_std_lower_3",
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


@dataclass(frozen=True)
class HealthPolicyConfig:
    """NaN ratio thresholds for feature health checks."""

    warn_ratio: float
    critical_warn_ratio: float
    critical_columns: tuple[str, ...]


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Feature build configuration loaded from YAML."""

    input_root: Path
    runs_root: Path
    parquet_glob: str
    seed: int
    supertrend: SuperTrendConfig
    alphatrend: AlphaTrendConfig
    rsi: RsiConfig
    events: EventConfig
    health: HealthPolicyConfig


@dataclass(frozen=True)
class FeatureBuildArtifacts:
    """Feature build artifacts for one dataframe."""

    frame: pd.DataFrame
    raw_events: pd.DataFrame
    shifted_events: pd.DataFrame


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

    if not isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype):
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


def compute_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute long-horizon EMA features from close prices."""

    close = df["close"].astype("float64")
    return pd.DataFrame(
        {
            "ema_200": close.ewm(span=200, adjust=False, min_periods=1).mean().astype("float32"),
            "ema_600": close.ewm(span=600, adjust=False, min_periods=1).mean().astype("float32"),
            "ema_1200": close.ewm(span=1200, adjust=False, min_periods=1).mean().astype("float32"),
        },
        index=df.index,
    )


def compute_daily_pivots_with_std_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Compute previous-session daily pivots and std bands mapped to intraday bars.

    All pivot/std-band levels for a session are derived from the previous UTC day.
    """

    sessions = df["timestamp"].dt.floor("D")
    grouped = (
        df.assign(_session=sessions)
        .groupby("_session", sort=True)
        .agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
    )

    close_std = (
        df.assign(_session=sessions)
        .groupby("_session", sort=True)["close"]
        .std(ddof=0)
        .rename("day_close_std")
        .fillna(0.0)
    )
    daily = grouped.join(close_std)

    pivot = (daily["day_high"] + daily["day_low"] + daily["day_close"]) / 3.0
    day_range = daily["day_high"] - daily["day_low"]

    levels = pd.DataFrame(index=daily.index)
    levels["pivot_p"] = pivot
    levels["pivot_r1"] = (2.0 * pivot) - daily["day_low"]
    levels["pivot_s1"] = (2.0 * pivot) - daily["day_high"]
    levels["pivot_r2"] = pivot + day_range
    levels["pivot_s2"] = pivot - day_range
    levels["pivot_r3"] = daily["day_high"] + (2.0 * (pivot - daily["day_low"]))
    levels["pivot_s3"] = daily["day_low"] - (2.0 * (daily["day_high"] - pivot))

    std = daily["day_close_std"]
    levels["pivot_std_upper_1"] = pivot + std
    levels["pivot_std_upper_2"] = pivot + (2.0 * std)
    levels["pivot_std_upper_3"] = pivot + (3.0 * std)
    levels["pivot_std_lower_1"] = pivot - std
    levels["pivot_std_lower_2"] = pivot - (2.0 * std)
    levels["pivot_std_lower_3"] = pivot - (3.0 * std)

    shifted_levels = levels.shift(1)

    intraday = pd.DataFrame(index=df.index)
    for col in shifted_levels.columns:
        intraday[col] = sessions.map(shifted_levels[col])

    return intraday.astype("float32")


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
    """Compute continuous indicators and internal trend-direction helpers."""

    ohlcv = validate_ohlcv_frame(df)

    supertrend, supertrend_direction = compute_supertrend(ohlcv, supertrend_cfg)
    alphatrend, alphatrend_direction = compute_alphatrend(ohlcv, alphatrend_cfg)
    rsi_derivatives = compute_rsi_derivatives(ohlcv, rsi_cfg)
    ema_features = compute_ema_features(ohlcv)
    pivot_features = compute_daily_pivots_with_std_bands(ohlcv)

    features = pd.concat(
        [
            supertrend,
            alphatrend,
            rsi_derivatives,
            ema_features,
            pivot_features,
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


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _get_required_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid dictionary config key: {key}")
    return value


def _get_required_numeric(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key, None)
    if value is None:
        raise ValueError(f"alphatrend.{key} is required")
    return float(value)


def _parse_rule(raw: dict[str, Any], key: str) -> ThresholdRule:
    node = _get_required_dict(raw, key)
    signal = str(node.get("signal", "")).strip()
    operator = str(node.get("op", "")).strip()
    if not signal:
        raise ValueError(f"alphatrend.{key}.signal is required")
    if not operator:
        raise ValueError(f"alphatrend.{key}.op is required")
    if operator not in _ALLOWED_RULE_OPERATORS:
        raise ValueError(f"Unsupported alphatrend.{key}.op: {operator}")
    if "threshold" not in node:
        raise ValueError(f"alphatrend.{key}.threshold is required")
    return ThresholdRule(signal=signal, operator=operator, threshold=float(node["threshold"]))


def load_feature_config(path: Path) -> FeatureBuildConfig:
    """Load and validate feature engineering configuration."""

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a dictionary")

    base_dir = path.resolve().parent
    supertrend_cfg = _get_required_dict(raw, "supertrend")
    alphatrend_cfg = _get_required_dict(raw, "alphatrend")
    rsi_cfg = _get_required_dict(raw, "rsi")
    events_cfg = _get_required_dict(raw, "events")
    health_cfg = raw.get("health", {})
    if not isinstance(health_cfg, dict):
        raise ValueError("health must be a dictionary if provided")

    cfg = FeatureBuildConfig(
        input_root=_resolve_path(str(raw.get("input_root", "")), base_dir),
        runs_root=_resolve_path(str(raw.get("runs_root", "runs")), base_dir),
        parquet_glob=str(raw.get("parquet_glob", "*.parquet")),
        seed=int(raw.get("seed", 42)),
        supertrend=SuperTrendConfig(
            period=int(supertrend_cfg.get("period", 10)),
            multiplier=float(supertrend_cfg.get("multiplier", 3.0)),
        ),
        alphatrend=AlphaTrendConfig(
            period=int(_get_required_numeric(alphatrend_cfg, "period")),
            atr_multiplier=float(_get_required_numeric(alphatrend_cfg, "atr_multiplier")),
            signal_period=int(alphatrend_cfg.get("signal_period", 14)),
            long_rule=_parse_rule(alphatrend_cfg, "long_rule"),
            short_rule=_parse_rule(alphatrend_cfg, "short_rule"),
        ),
        rsi=RsiConfig(
            period=int(rsi_cfg.get("period", 14)),
            slope_lag=int(rsi_cfg.get("slope_lag", 3)),
            zscore_window=int(rsi_cfg.get("zscore_window", 50)),
        ),
        events=EventConfig(
            rsi_centerline=float(events_cfg.get("rsi_centerline", 50.0)),
            rsi_overbought=float(events_cfg.get("rsi_overbought", 70.0)),
            rsi_oversold=float(events_cfg.get("rsi_oversold", 30.0)),
        ),
        health=HealthPolicyConfig(
            warn_ratio=float(health_cfg.get("warn_ratio", 0.005)),
            critical_warn_ratio=float(health_cfg.get("critical_warn_ratio", 0.001)),
            critical_columns=tuple(
                str(col) for col in health_cfg.get("critical_columns", ("supertrend", "alphatrend", "rsi"))
            ),
        ),
    )
    validate_feature_config(cfg)
    return cfg


def validate_feature_config(cfg: FeatureBuildConfig) -> None:
    """Validate semantic constraints for feature build config."""

    if not cfg.input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {cfg.input_root}")
    if not cfg.input_root.is_dir():
        raise NotADirectoryError(f"input_root is not a directory: {cfg.input_root}")
    if not cfg.parquet_glob.strip():
        raise ValueError("parquet_glob cannot be empty")
    if cfg.seed < 0:
        raise ValueError("seed must be >= 0")
    if cfg.events.rsi_oversold >= cfg.events.rsi_overbought:
        raise ValueError("events.rsi_oversold must be < events.rsi_overbought")
    if not (0.0 <= cfg.health.warn_ratio <= 1.0):
        raise ValueError("health.warn_ratio must be in [0.0, 1.0]")
    if not (0.0 <= cfg.health.critical_warn_ratio <= 1.0):
        raise ValueError("health.critical_warn_ratio must be in [0.0, 1.0]")
    if cfg.health.critical_warn_ratio > cfg.health.warn_ratio:
        raise ValueError("health.critical_warn_ratio must be <= health.warn_ratio")
    if not cfg.health.critical_columns:
        raise ValueError("health.critical_columns cannot be empty")
    if cfg.alphatrend.period != ALPHATREND_LOCK_PERIOD:
        raise ValueError(f"alphatrend.period must be fixed at {ALPHATREND_LOCK_PERIOD}")
    if cfg.alphatrend.atr_multiplier != ALPHATREND_LOCK_ATR_MULTIPLIER:
        raise ValueError(f"alphatrend.atr_multiplier must be fixed at {ALPHATREND_LOCK_ATR_MULTIPLIER}")
    if cfg.alphatrend.long_rule.signal.strip() == "":
        raise ValueError("alphatrend.long_rule.signal is required")
    if cfg.alphatrend.short_rule.signal.strip() == "":
        raise ValueError("alphatrend.short_rule.signal is required")


def discover_parquet_files(input_root: Path, parquet_glob: str = "*.parquet") -> list[Path]:
    """Discover parquet files recursively under input_root."""

    files = [path for path in input_root.glob(parquet_glob) if path.is_file() and path.suffix.lower() == ".parquet"]
    return sorted(files)


def build_feature_output_path(src_parquet: Path, input_root: Path, output_root: Path) -> Path:
    """Build output parquet path for a source standardized parquet file."""

    rel_path = src_parquet.resolve().relative_to(input_root.resolve())
    stem = "__".join(rel_path.with_suffix("").parts)
    out_path = output_root / f"{stem}.parquet"
    ensure_within_root(out_path, output_root)
    return out_path


def apply_dtype_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Cast continuous columns to float32 and event flags to uint8."""

    out = df.copy()
    continuous = ("open", "high", "low", "close", "volume", *CONTINUOUS_FEATURE_COLUMNS)

    for col in continuous:
        if col not in out.columns:
            raise ValueError(f"Missing continuous column for dtype policy: {col}")
        out[col] = out[col].astype("float32")

    for col in EVENT_FLAG_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing event flag column for dtype policy: {col}")
        out[col] = out[col].astype("uint8")

    return out


def build_feature_artifacts(df: pd.DataFrame, cfg: FeatureBuildConfig) -> FeatureBuildArtifacts:
    """Build feature table plus raw/shifted events for one input dataframe."""

    ohlcv = validate_ohlcv_frame(df)
    indicators = compute_indicator_core(
        df=ohlcv,
        supertrend_cfg=cfg.supertrend,
        alphatrend_cfg=cfg.alphatrend,
        rsi_cfg=cfg.rsi,
    )
    raw_events = generate_raw_event_flags(indicators, cfg.events)
    shifted_events = enforce_shift_one(raw_events)
    if not validate_shift_one(raw_events, shifted_events):
        raise ValueError("Strict shift(1) validation failed for event columns")

    frame = pd.concat(
        [
            ohlcv,
            indicators.loc[:, list(CONTINUOUS_FEATURE_COLUMNS)],
            shifted_events.loc[:, list(EVENT_FLAG_COLUMNS)],
        ],
        axis=1,
    )
    frame = apply_dtype_policy(frame)
    return FeatureBuildArtifacts(frame=frame, raw_events=raw_events, shifted_events=shifted_events)
