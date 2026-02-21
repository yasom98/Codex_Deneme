"""Feature engineering core locked to reference indicator formulas."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core.logging import get_logger
from core.paths import ensure_within_root
from data import indicator_reference as ref
from data.reference_pivots import (
    PIVOT_REFERENCE_SOURCE,
    PIVOT_REFERENCE_TYPE,
    compute_reference_pivots_intraday,
    pivot_reference_available,
)

LOGGER = get_logger(__name__)

INDICATOR_SPEC_VERSION: str = "indicators.v2026-02-15.1"

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
PIVOT_FEATURE_COLUMNS: tuple[str, ...] = ("PP", "R1", "S1", "R2", "S2", "R3", "S3", "R4", "S4", "R5", "S5")
CONTINUOUS_FEATURE_COLUMNS: tuple[str, ...] = (
    *PIVOT_FEATURE_COLUMNS,
    "EMA_200",
    "EMA_600",
    "EMA_1200",
    "AlphaTrend",
    "AlphaTrend_2",
    "ST_trend",
    "ST_up",
    "ST_dn",
)
PRICE_DERIVED_BASE_COLUMNS: tuple[str, ...] = (
    "log_return",
    "hl_range",
    "body_ratio",
    "zscore_return",
    "normalized_range",
)
REGIME_CONTINUOUS_COLUMNS: tuple[str, ...] = ("trend_strength", "squeeze_proxy")
REGIME_FLAG_COLUMNS: tuple[str, ...] = ("trend_regime", "volatility_regime")
PLACEHOLDER_FLAG_COLUMNS: tuple[str, ...] = ("position_placeholder", "market_state")
PLACEHOLDER_CONTINUOUS_COLUMNS: tuple[str, ...] = ("floating_pnl_placeholder", "drawdown_placeholder")
MANIFEST_VERSION: str = "features.manifest.v1"
RAW_EVENT_COLUMNS: tuple[str, ...] = ("AT_buy_raw", "AT_sell_raw", "AT_buy", "AT_sell", "ST_buy", "ST_sell")
EVENT_FLAG_COLUMNS: tuple[str, ...] = (
    "evt_at_buy_raw",
    "evt_at_sell_raw",
    "evt_at_buy",
    "evt_at_sell",
    "evt_st_buy",
    "evt_st_sell",
)
RAW_TO_EVENT_COLUMN: dict[str, str] = {
    "AT_buy_raw": "evt_at_buy_raw",
    "AT_sell_raw": "evt_at_sell_raw",
    "AT_buy": "evt_at_buy",
    "AT_sell": "evt_at_sell",
    "ST_buy": "evt_st_buy",
    "ST_sell": "evt_st_sell",
}

_LOCKED_ALPHATREND_COEFF: float = 3.0
_LOCKED_ALPHATREND_AP: int = 11
_LOCKED_SUPERTREND_PERIODS: int = 10
_LOCKED_SUPERTREND_MULTIPLIER: float = 3.0
_LOCKED_SUPERTREND_SOURCE: str = "hl2"
_LOCKED_SUPERTREND_CHANGE_ATR_METHOD: bool = True
_LOCKED_PIVOT_TF: str = "1D"
_LOCKED_SHIFT_POLICY: str = "strict_shift_1"

_ALLOWED_PIVOT_WARMUP_POLICIES: tuple[str, ...] = ("allow_first_session_nan",)
_ALLOWED_PIVOT_FIRST_SESSION_FILL: tuple[str, ...] = ("none", "ffill_from_second_session")


@dataclass(frozen=True)
class SuperTrendConfig:
    """Locked Supertrend config."""

    periods: int
    multiplier: float
    source: str
    change_atr_method: bool


@dataclass(frozen=True)
class AlphaTrendConfig:
    """Locked AlphaTrend config."""

    coeff: float
    ap: int
    use_no_volume: bool


@dataclass(frozen=True)
class PivotPolicyConfig:
    """Pivot policy configuration."""

    pivot_tf: str
    warmup_policy: str
    first_session_fill: str


@dataclass(frozen=True)
class ParityPolicyConfig:
    """Parity check configuration."""

    enabled: bool
    sample_rows: int
    float_atol: float
    float_rtol: float


@dataclass(frozen=True)
class HealthPolicyConfig:
    """NaN ratio thresholds for feature health checks."""

    warn_ratio: float
    critical_warn_ratio: float
    critical_columns: tuple[str, ...]


@dataclass(frozen=True)
class RLFeatureConfig:
    """RL-oriented additive feature configuration."""

    rolling_vol_windows: tuple[int, ...] = (20, 50)
    zscore_window: int = 50
    zscore_source: str = "return"
    volatility_regime_bins: int = 3


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Feature build configuration loaded from YAML."""

    input_root: Path
    runs_root: Path
    parquet_glob: str
    seed: int
    supertrend: SuperTrendConfig
    alphatrend: AlphaTrendConfig
    pivot: PivotPolicyConfig
    parity: ParityPolicyConfig
    health: HealthPolicyConfig
    config_hash: str
    indicator_spec_version: str
    rl_features: RLFeatureConfig = field(default_factory=RLFeatureConfig)


@dataclass(frozen=True)
class FeatureBuildArtifacts:
    """Feature build artifacts for one dataframe."""

    frame: pd.DataFrame
    raw_events: pd.DataFrame
    shifted_events: pd.DataFrame
    indicator_parity_status: str
    indicator_parity_details: dict[str, bool]
    indicator_validation_status: str
    indicator_validation_details: dict[str, bool]
    formula_fingerprints: dict[str, str]
    formula_fingerprint_bundle: str
    raw_regime_flags: pd.DataFrame
    shifted_regime_flags: pd.DataFrame
    continuous_feature_columns: tuple[str, ...]
    flag_feature_columns: tuple[str, ...]
    placeholder_columns: tuple[str, ...]
    warmup_rows_by_column: dict[str, int]
    feature_groups: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class IndicatorCoreOutput:
    """Internal indicator output blocks before event shift."""

    continuous: pd.DataFrame
    raw_events: pd.DataFrame


@dataclass(frozen=True)
class RLFeatureOutput:
    """Additive RL-oriented feature blocks."""

    continuous: pd.DataFrame
    raw_regime_flags: pd.DataFrame
    shifted_regime_flags: pd.DataFrame
    flag_placeholders: pd.DataFrame
    continuous_placeholders: pd.DataFrame
    warmup_rows_by_column: dict[str, int]


def _compact_fingerprint(payload: dict[str, Any]) -> str:
    """Return compact stable fingerprint for a formula payload."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def compute_formula_fingerprints(cfg: FeatureBuildConfig) -> dict[str, str]:
    """Build compact formula fingerprints for each indicator path."""

    payloads: dict[str, dict[str, Any]] = {
        "pivot_traditional": {
            "spec_version": INDICATOR_SPEC_VERSION,
            "pivot_tf": cfg.pivot.pivot_tf,
            "warmup_policy": cfg.pivot.warmup_policy,
            "first_session_fill": cfg.pivot.first_session_fill,
            "logic_mode": "prev_htf_shift_1_ffill",
        },
        "ema_set": {
            "spec_version": INDICATOR_SPEC_VERSION,
            "price_col": "close",
            "periods": [200, 600, 1200],
            "ewm_adjust": False,
            "min_periods": "period",
        },
        "alphatrend": {
            "spec_version": INDICATOR_SPEC_VERSION,
            "coeff": cfg.alphatrend.coeff,
            "ap": cfg.alphatrend.ap,
            "use_no_volume": cfg.alphatrend.use_no_volume,
            "atr_mode": "sma_true_range",
            "signal_mode": "crossover_at_vs_shift2_with_barssince_gating",
        },
        "supertrend": {
            "spec_version": INDICATOR_SPEC_VERSION,
            "periods": cfg.supertrend.periods,
            "multiplier": cfg.supertrend.multiplier,
            "source": cfg.supertrend.source,
            "change_atr_method": cfg.supertrend.change_atr_method,
            "signal_mode": "trend_flip_cross",
        },
        "event_shift_policy": {
            "spec_version": INDICATOR_SPEC_VERSION,
            "shift_policy": _LOCKED_SHIFT_POLICY,
            "raw_to_event": RAW_TO_EVENT_COLUMN,
        },
    }
    return {name: _compact_fingerprint(payload) for name, payload in payloads.items()}


def compute_formula_fingerprint_bundle(fingerprints: dict[str, str]) -> str:
    """Return bundle fingerprint from per-indicator fingerprints."""

    return _compact_fingerprint({"fingerprints": fingerprints, "spec_version": INDICATOR_SPEC_VERSION})


def validate_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an OHLCV dataframe."""

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
        LOGGER.warning("timestamp not monotonic increasing; sorting ascending.")
        out = out.sort_values("timestamp", kind="mergesort")

    duplicated_mask = out.duplicated(subset=["timestamp"], keep="last")
    dropped_duplicates = int(duplicated_mask.sum())
    if dropped_duplicates > 0:
        LOGGER.warning("duplicate timestamps dropped with keep='last' | dropped=%d", dropped_duplicates)
        out = out.loc[~duplicated_mask].copy()

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float32")

    return out.reset_index(drop=True)


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy with timestamp index for reference indicator functions."""

    indexed = df.copy()
    indexed = indexed.set_index("timestamp", drop=True)
    if not isinstance(indexed.index, pd.DatetimeIndex):
        raise ValueError("timestamp index conversion failed.")
    if indexed.index.tz is None:
        raise ValueError("timestamp index must be timezone-aware UTC.")
    if not indexed.index.is_monotonic_increasing:
        raise ValueError("timestamp index must be monotonic increasing.")
    return indexed


def _normalize_pivot_policies(warmup_policy: str, first_session_fill: str) -> tuple[str, str]:
    """Normalize and validate pivot policy values."""

    normalized_warmup = warmup_policy.strip().lower()
    if normalized_warmup not in _ALLOWED_PIVOT_WARMUP_POLICIES:
        raise ValueError(f"Unsupported pivot warmup policy: {warmup_policy}")

    normalized_fill = first_session_fill.strip().lower()
    if normalized_fill not in _ALLOWED_PIVOT_FIRST_SESSION_FILL:
        raise ValueError(f"Unsupported pivot first_session_fill policy: {first_session_fill}")

    return normalized_warmup, normalized_fill


def _apply_first_session_fill(
    pivots: pd.DataFrame,
    sessions: pd.Series,
    first_session_fill: str,
) -> pd.DataFrame:
    """Apply optional first-session fill to pivot columns."""

    normalized_fill = first_session_fill.strip().lower()
    if normalized_fill != "ffill_from_second_session":
        return pivots

    ordered_sessions = sessions.drop_duplicates()
    if len(ordered_sessions) < 2:
        LOGGER.warning(
            "pivot.first_session_fill requested but skipped due to insufficient sessions | sessions=%d",
            int(sessions.nunique()),
        )
        return pivots

    first_session = ordered_sessions.iloc[0]
    second_session = ordered_sessions.iloc[1]
    first_mask = sessions.eq(first_session)
    second_mask = sessions.eq(second_session)

    second_session_pivots = pivots.loc[second_mask, list(PIVOT_FEATURE_COLUMNS)]
    if second_session_pivots.empty:
        return pivots

    fill_values = second_session_pivots.iloc[0].to_numpy(dtype=np.float64, copy=False)
    pivots.loc[first_mask, list(PIVOT_FEATURE_COLUMNS)] = fill_values
    LOGGER.info("Applied pivot first-session fill from second session | rows=%d", int(first_mask.sum()))
    return pivots


def compute_daily_pivots_with_std_bands(
    df: pd.DataFrame,
    warmup_policy: str = "allow_first_session_nan",
    first_session_fill: str = "none",
    pivot_tf: str = "1D",
    assume_validated: bool = False,
    indexed_ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute leak-free traditional pivots with warmup policy handling."""

    _normalize_pivot_policies(warmup_policy, first_session_fill)
    if pivot_tf.strip() != _LOCKED_PIVOT_TF:
        raise ValueError(f"pivot.pivot_tf must be fixed at {_LOCKED_PIVOT_TF}")

    ohlcv = df if assume_validated else validate_ohlcv_frame(df)
    if indexed_ohlcv is not None and len(indexed_ohlcv) != len(ohlcv):
        raise ValueError("indexed_ohlcv length mismatch for pivot computation")
    pivots = compute_reference_pivots_intraday(ohlcv, pivot_tf=pivot_tf)

    out = pivots.copy()
    out.index = ohlcv.index
    sessions = ohlcv["timestamp"].dt.floor("D")
    out = _apply_first_session_fill(out, sessions=sessions, first_session_fill=first_session_fill)
    return out.astype("float32")


def _build_indicator_core(ohlcv: pd.DataFrame, cfg: FeatureBuildConfig) -> IndicatorCoreOutput:
    """Compute reference-locked indicator blocks."""

    indexed = _to_datetime_index(ohlcv)
    pivots = compute_daily_pivots_with_std_bands(
        ohlcv,
        warmup_policy=cfg.pivot.warmup_policy,
        first_session_fill=cfg.pivot.first_session_fill,
        pivot_tf=cfg.pivot.pivot_tf,
        assume_validated=True,
        indexed_ohlcv=indexed,
    )

    shared_tr = ref.true_range(indexed["high"], indexed["low"], indexed["close"])
    ema_block = ref.compute_ema_set(indexed, price_col="close")
    ema_block.index = ohlcv.index

    alpha_block = ref.compute_alphatrend(
        indexed,
        cfg=ref.AlphaTrendConfig(
            coeff=cfg.alphatrend.coeff,
            ap=cfg.alphatrend.ap,
            use_no_volume=cfg.alphatrend.use_no_volume,
        ),
        show_progress=False,
        precomputed_tr=shared_tr,
    )
    alpha_block.index = ohlcv.index

    supertrend_block = ref.compute_supertrend(
        indexed,
        cfg=ref.SupertrendConfig(
            periods=cfg.supertrend.periods,
            multiplier=cfg.supertrend.multiplier,
            source=cfg.supertrend.source,
            change_atr_method=cfg.supertrend.change_atr_method,
        ),
        show_progress=False,
        precomputed_tr=shared_tr,
    )
    supertrend_block.index = ohlcv.index

    continuous = pd.concat(
        [
            pivots,
            ema_block.loc[:, ["EMA_200", "EMA_600", "EMA_1200"]],
            alpha_block.loc[:, ["AlphaTrend", "AlphaTrend_2"]],
            supertrend_block.loc[:, ["ST_trend", "ST_up", "ST_dn"]],
        ],
        axis=1,
    )

    raw_events = pd.concat(
        [
            alpha_block.loc[:, ["AT_buy_raw", "AT_sell_raw", "AT_buy", "AT_sell"]],
            supertrend_block.loc[:, ["ST_buy", "ST_sell"]],
        ],
        axis=1,
    )

    return IndicatorCoreOutput(continuous=continuous, raw_events=raw_events)


def generate_raw_event_flags(raw_indicator_events: pd.DataFrame) -> pd.DataFrame:
    """Map raw indicator event signals into pipeline raw event columns."""

    missing = sorted(set(RAW_EVENT_COLUMNS).difference(raw_indicator_events.columns))
    if missing:
        raise ValueError(f"Missing raw indicator event columns: {missing}")

    out = pd.DataFrame(index=raw_indicator_events.index)
    for raw_col, event_col in RAW_TO_EVENT_COLUMN.items():
        out[event_col] = raw_indicator_events[raw_col].fillna(0).astype("uint8")
    return out


def _shift_flag_one(raw_flag: pd.Series) -> pd.Series:
    """Shift one bar and convert to uint8."""

    raw_uint8 = raw_flag.fillna(0).astype("uint8").to_numpy(dtype=np.uint8, copy=False)
    shifted = np.zeros(len(raw_uint8), dtype=np.uint8)
    if len(raw_uint8) > 1:
        shifted[1:] = raw_uint8[:-1]
    return pd.Series(shifted, index=raw_flag.index, dtype="uint8")


def enforce_shift_one_for_columns(raw_events: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Enforce strict shift(1) for provided flag columns."""

    out = pd.DataFrame(index=raw_events.index)
    for col in columns:
        if col not in raw_events.columns:
            raise ValueError(f"Missing raw event column: {col}")
        out[col] = _shift_flag_one(raw_events[col])
    return out


def validate_shift_one_for_columns(
    raw_events: pd.DataFrame,
    shifted_events: pd.DataFrame,
    columns: tuple[str, ...],
) -> bool:
    """Validate strict shift(1) relation for the selected columns."""

    for col in columns:
        if col not in raw_events.columns or col not in shifted_events.columns:
            return False

        raw_uint8 = raw_events[col].fillna(0).astype("uint8").to_numpy(dtype=np.uint8, copy=False)
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


def enforce_shift_one(raw_events: pd.DataFrame) -> pd.DataFrame:
    """Enforce strict shift(1) for every event column."""

    return enforce_shift_one_for_columns(raw_events, EVENT_FLAG_COLUMNS)


def validate_shift_one(raw_events: pd.DataFrame, shifted_events: pd.DataFrame) -> bool:
    """Validate strict shift(1) relation for event columns."""

    return validate_shift_one_for_columns(raw_events, shifted_events, EVENT_FLAG_COLUMNS)


def _float_parity_equal(actual: pd.Series, expected: pd.Series, atol: float, rtol: float) -> bool:
    """Return allclose with NaN-aware comparison."""

    actual_arr = actual.to_numpy(dtype=np.float64, copy=False)
    expected_arr = expected.to_numpy(dtype=np.float64, copy=False)
    if actual_arr.shape != expected_arr.shape:
        return False
    return bool(np.allclose(actual_arr, expected_arr, atol=atol, rtol=rtol, equal_nan=True))


def _normalize_rolling_vol_windows(windows: tuple[int, ...]) -> tuple[int, ...]:
    """Normalize rolling volatility windows into sorted unique positive ints."""

    normalized = tuple(sorted({int(window) for window in windows}))
    if not normalized:
        raise ValueError("rl_features.rolling_vol_windows cannot be empty")
    if any(window <= 1 for window in normalized):
        raise ValueError("rl_features.rolling_vol_windows values must be > 1")
    return normalized


def get_rolling_vol_columns(cfg: FeatureBuildConfig) -> tuple[str, ...]:
    """Return rolling volatility column names from config."""

    windows = _normalize_rolling_vol_windows(cfg.rl_features.rolling_vol_windows)
    return tuple(f"rolling_vol_{window}" for window in windows)


def get_feature_groups(cfg: FeatureBuildConfig) -> dict[str, tuple[str, ...]]:
    """Return stable feature groups for manifest/reporting."""

    rolling_cols = get_rolling_vol_columns(cfg)
    price_derived = (*PRICE_DERIVED_BASE_COLUMNS[:3], *rolling_cols, *PRICE_DERIVED_BASE_COLUMNS[3:])
    trend = (
        "EMA_200",
        "EMA_600",
        "EMA_1200",
        "AlphaTrend",
        "AlphaTrend_2",
        "ST_trend",
        "ST_up",
        "ST_dn",
        "trend_strength",
    )
    regime = ("trend_regime", "volatility_regime", "squeeze_proxy")
    placeholders = (
        "position_placeholder",
        "floating_pnl_placeholder",
        "drawdown_placeholder",
        "market_state",
    )
    return {
        "raw_ohlcv": REQUIRED_OHLCV_COLUMNS,
        "price_derived": price_derived,
        "trend": trend,
        "regime": regime,
        "event": EVENT_FLAG_COLUMNS,
        "placeholders": placeholders,
    }


def get_output_feature_columns(cfg: FeatureBuildConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (continuous_feature_columns, encoded_flag_columns)."""

    rolling_cols = get_rolling_vol_columns(cfg)
    continuous = (
        *CONTINUOUS_FEATURE_COLUMNS,
        "log_return",
        "hl_range",
        "body_ratio",
        *rolling_cols,
        "zscore_return",
        "normalized_range",
        *REGIME_CONTINUOUS_COLUMNS,
        *PLACEHOLDER_CONTINUOUS_COLUMNS,
    )
    flag_columns = (*EVENT_FLAG_COLUMNS, *REGIME_FLAG_COLUMNS, *PLACEHOLDER_FLAG_COLUMNS)
    return continuous, flag_columns


def get_expected_column_dtypes(cfg: FeatureBuildConfig) -> dict[str, str]:
    """Return expected output dtype map for manifest fallback."""

    continuous_features, flag_features = get_output_feature_columns(cfg)
    out: dict[str, str] = {"timestamp": "datetime64[ns, UTC]"}
    for col in ("open", "high", "low", "close", "volume", *continuous_features):
        out[col] = "float32"
    for col in flag_features:
        out[col] = "uint8"
    return out


def build_feature_manifest_payload(
    run_id: str,
    cfg: FeatureBuildConfig,
    *,
    feature_groups: dict[str, tuple[str, ...]],
    column_dtypes: dict[str, str],
    row_count: int,
    date_min_utc: str | None,
    date_max_utc: str | None,
    formula_fingerprints: dict[str, str],
    formula_fingerprint_bundle: str,
) -> dict[str, Any]:
    """Build machine-readable feature manifest payload."""

    continuous_features, flag_features = get_output_feature_columns(cfg)
    placeholder_columns = tuple(
        col for col in feature_groups.get("placeholders", ()) if col in set(column_dtypes)
    )
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "run_id": run_id,
        "feature_groups": {name: list(cols) for name, cols in feature_groups.items()},
        "column_dtypes": dict(sorted(column_dtypes.items())),
        "event_columns": list(flag_features),
        "continuous_columns": list(continuous_features),
        "placeholder_columns": list(placeholder_columns),
        "warmup_policy": {
            "pivot_warmup_policy": cfg.pivot.warmup_policy,
            "pivot_first_session_fill": cfg.pivot.first_session_fill,
            "rolling_vol_windows": list(_normalize_rolling_vol_windows(cfg.rl_features.rolling_vol_windows)),
            "zscore_window": int(cfg.rl_features.zscore_window),
            "zscore_source": str(cfg.rl_features.zscore_source),
            "volatility_regime_bins": int(cfg.rl_features.volatility_regime_bins),
        },
        "indicator_spec_version": cfg.indicator_spec_version,
        "config_hash": cfg.config_hash,
        "formula_fingerprints": dict(formula_fingerprints),
        "formula_fingerprint_bundle": str(formula_fingerprint_bundle),
        "timestamp_column": "timestamp",
        "row_count": int(row_count),
        "date_range": {"min_utc": date_min_utc, "max_utc": date_max_utc},
    }


def _safe_log_return(close: pd.Series) -> pd.Series:
    """Compute log-return with zero/non-positive guard."""

    prev_close = close.shift(1)
    ratio = close / prev_close
    ratio = ratio.where((close > 0.0) & (prev_close > 0.0))
    return np.log(ratio.astype("float64"))


def _compute_volatility_regime_raw(
    vol: pd.Series,
    bins: int,
    min_periods: int,
) -> pd.Series:
    """Compute no-lookahead volatility regime bins from expanding quantiles."""

    if bins not in (2, 3, 4):
        raise ValueError("rl_features.volatility_regime_bins must be one of {2, 3, 4}")
    if min_periods <= 1:
        raise ValueError("volatility regime min_periods must be > 1")

    quantile_levels: tuple[float, ...]
    if bins == 2:
        quantile_levels = (0.5,)
    elif bins == 3:
        quantile_levels = (1.0 / 3.0, 2.0 / 3.0)
    else:
        quantile_levels = (0.25, 0.5, 0.75)

    thresholds = [vol.expanding(min_periods=min_periods).quantile(level) for level in quantile_levels]

    raw = np.zeros(len(vol), dtype=np.uint8)
    vol_values = vol.to_numpy(dtype=np.float64, copy=False)
    threshold_arrays = [series.to_numpy(dtype=np.float64, copy=False) for series in thresholds]

    valid = np.isfinite(vol_values)
    for idx, threshold in enumerate(threshold_arrays):
        raw[valid & np.isfinite(threshold) & (vol_values > threshold)] = np.uint8(idx + 1)
    return pd.Series(raw, index=vol.index, dtype="uint8")


def _build_rl_feature_output(
    ohlcv: pd.DataFrame,
    indicator_continuous: pd.DataFrame,
    cfg: FeatureBuildConfig,
) -> RLFeatureOutput:
    """Build additive RL-ready feature blocks."""

    rolling_windows = _normalize_rolling_vol_windows(cfg.rl_features.rolling_vol_windows)
    if cfg.rl_features.zscore_window <= 1:
        raise ValueError("rl_features.zscore_window must be > 1")
    if cfg.rl_features.zscore_source.strip().lower() not in {"return", "close"}:
        raise ValueError("rl_features.zscore_source must be one of {'return', 'close'}")

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    open_ = ohlcv["open"]

    log_return = _safe_log_return(close)
    hl_range = high - low
    body_denominator = hl_range.replace(0.0, np.nan)
    body_ratio = (close - open_).abs() / body_denominator

    rolling_volatility: dict[str, pd.Series] = {}
    for window in rolling_windows:
        rolling_volatility[f"rolling_vol_{window}"] = log_return.rolling(window=window, min_periods=window).std()

    if cfg.rl_features.zscore_source.strip().lower() == "close":
        zscore_base = close
    else:
        zscore_base = log_return
    zscore_mean = zscore_base.rolling(window=cfg.rl_features.zscore_window, min_periods=cfg.rl_features.zscore_window).mean()
    zscore_std = zscore_base.rolling(window=cfg.rl_features.zscore_window, min_periods=cfg.rl_features.zscore_window).std()
    zscore_return = (zscore_base - zscore_mean) / zscore_std.replace(0.0, np.nan)

    normalized_range = hl_range / close.replace(0.0, np.nan)
    trend_strength = (close - indicator_continuous["EMA_200"]).abs() / indicator_continuous["EMA_200"].abs().replace(0.0, np.nan)

    short_window = rolling_windows[0]
    long_window = rolling_windows[-1]
    short_col = f"rolling_vol_{short_window}"
    long_col = f"rolling_vol_{long_window}"
    squeeze_proxy = rolling_volatility[short_col] / rolling_volatility[long_col].replace(0.0, np.nan)

    trend_regime_raw = (
        (indicator_continuous["EMA_200"] > indicator_continuous["EMA_600"])
        & indicator_continuous["EMA_200"].notna()
        & indicator_continuous["EMA_600"].notna()
    ).astype("uint8")
    volatility_regime_raw = _compute_volatility_regime_raw(
        vol=rolling_volatility[short_col],
        bins=cfg.rl_features.volatility_regime_bins,
        min_periods=short_window,
    )
    raw_regime = pd.DataFrame(
        {
            "trend_regime": trend_regime_raw.astype("uint8"),
            "volatility_regime": volatility_regime_raw.astype("uint8"),
        },
        index=ohlcv.index,
    )
    shifted_regime = enforce_shift_one_for_columns(raw_regime, REGIME_FLAG_COLUMNS)
    if not validate_shift_one_for_columns(raw_regime, shifted_regime, REGIME_FLAG_COLUMNS):
        raise ValueError("Strict shift(1) validation failed for regime columns")

    market_state = (
        shifted_regime["trend_regime"].astype("uint16") * np.uint16(cfg.rl_features.volatility_regime_bins)
        + shifted_regime["volatility_regime"].astype("uint16")
    ).astype("uint8")
    flag_placeholders = pd.DataFrame(
        {
            "position_placeholder": np.zeros(len(ohlcv), dtype=np.uint8),
            "market_state": market_state.to_numpy(dtype=np.uint8, copy=False),
        },
        index=ohlcv.index,
    )
    continuous_placeholders = pd.DataFrame(
        {
            "floating_pnl_placeholder": np.zeros(len(ohlcv), dtype=np.float32),
            "drawdown_placeholder": np.zeros(len(ohlcv), dtype=np.float32),
        },
        index=ohlcv.index,
    )

    continuous_rl = pd.DataFrame(
        {
            "log_return": log_return,
            "hl_range": hl_range,
            "body_ratio": body_ratio,
            **rolling_volatility,
            "zscore_return": zscore_return,
            "normalized_range": normalized_range,
            "trend_strength": trend_strength,
            "squeeze_proxy": squeeze_proxy,
        },
        index=ohlcv.index,
    )

    warmup_rows: dict[str, int] = {
        "EMA_200": 199,
        "EMA_600": 599,
        "EMA_1200": 1199,
        "log_return": 1,
        "zscore_return": int(cfg.rl_features.zscore_window),
        "trend_strength": 199,
        "squeeze_proxy": int(long_window),
    }
    for window in rolling_windows:
        warmup_rows[f"rolling_vol_{window}"] = int(window)

    return RLFeatureOutput(
        continuous=continuous_rl,
        raw_regime_flags=raw_regime,
        shifted_regime_flags=shifted_regime,
        flag_placeholders=flag_placeholders,
        continuous_placeholders=continuous_placeholders,
        warmup_rows_by_column=warmup_rows,
    )


def evaluate_indicator_parity(
    ohlcv: pd.DataFrame,
    core_output: IndicatorCoreOutput,
    cfg: FeatureBuildConfig,
) -> tuple[str, dict[str, bool]]:
    """Run deterministic parity checks against reference source on a small sample."""

    if not cfg.parity.enabled:
        return "disabled", {}

    sample_rows = min(int(cfg.parity.sample_rows), int(len(ohlcv)))
    if sample_rows <= 0:
        raise ValueError("parity.sample_rows must be > 0 when parity is enabled")

    sample = ohlcv.iloc[:sample_rows].copy()
    expected = _build_indicator_core(sample, cfg)
    actual_cont = core_output.continuous.iloc[:sample_rows]
    actual_raw = core_output.raw_events.iloc[:sample_rows]

    details: dict[str, bool] = {}

    for col in CONTINUOUS_FEATURE_COLUMNS:
        details[col] = _float_parity_equal(
            actual_cont[col],
            expected.continuous[col],
            atol=cfg.parity.float_atol,
            rtol=cfg.parity.float_rtol,
        )

    for col in RAW_EVENT_COLUMNS:
        actual_arr = actual_raw[col].fillna(0).astype("uint8").to_numpy(dtype=np.uint8, copy=False)
        expected_arr = expected.raw_events[col].fillna(0).astype("uint8").to_numpy(dtype=np.uint8, copy=False)
        details[col] = bool(np.array_equal(actual_arr, expected_arr))

    failed = sorted(key for key, passed in details.items() if not passed)
    if failed:
        LOGGER.error("Indicator parity check failed | failed_columns=%s", failed)
        return "failed", details

    return "passed", details


def _compute_pivot_reference_previous_session(ohlcv: pd.DataFrame, first_session_fill: str) -> pd.DataFrame:
    """Compute explicit traditional pivots from previous daily session OHLC."""

    sessions = ohlcv["timestamp"].dt.floor("D")
    session_ohlc = ohlcv.groupby(sessions, sort=False).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    prev = session_ohlc.shift(1)

    prev_high = sessions.map(prev["high"])
    prev_low = sessions.map(prev["low"])
    prev_close = sessions.map(prev["close"])

    pp = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low

    pivots = pd.DataFrame(
        {
            "PP": pp,
            "R1": (2.0 * pp) - prev_low,
            "S1": (2.0 * pp) - prev_high,
            "R2": pp + rng,
            "S2": pp - rng,
            "R3": (2.0 * pp) + prev_high - (2.0 * prev_low),
            "S3": (2.0 * pp) - ((2.0 * prev_high) - prev_low),
            "R4": (3.0 * pp) + prev_high - (3.0 * prev_low),
            "S4": (3.0 * pp) - ((3.0 * prev_high) - prev_low),
            "R5": (4.0 * pp) + prev_high - (4.0 * prev_low),
            "S5": (4.0 * pp) - ((4.0 * prev_high) - prev_low),
        },
        index=ohlcv.index,
    )
    return _apply_first_session_fill(
        pivots,
        sessions=sessions,
        first_session_fill=first_session_fill,
    ).astype("float32")


def _evaluate_pivot_reference_validation(
    ohlcv: pd.DataFrame,
    core_output: IndicatorCoreOutput,
    cfg: FeatureBuildConfig,
) -> tuple[bool, dict[str, bool]]:
    """Validate pivot outputs against user-provided reference implementation."""

    details: dict[str, bool] = {}
    if not pivot_reference_available():
        details["pivot_reference_available"] = False
        details["pivot_reference_execution_ok"] = True
        details["pivot_reference_parity"] = True
        return True, details

    details["pivot_reference_available"] = True
    try:
        reference = compute_reference_pivots_intraday(ohlcv, pivot_tf=cfg.pivot.pivot_tf)
        sessions = ohlcv["timestamp"].dt.floor("D")
        reference = _apply_first_session_fill(
            reference,
            sessions=sessions,
            first_session_fill=cfg.pivot.first_session_fill,
        ).astype("float32")
        details["pivot_reference_execution_ok"] = True
    except Exception as exc:  # pragma: no cover - defensive to fail closed on user spec runtime/import errors.
        LOGGER.error(
            "Pivot reference execution failed | source=%s type=%s error=%s",
            PIVOT_REFERENCE_SOURCE,
            PIVOT_REFERENCE_TYPE,
            exc,
        )
        details["pivot_reference_execution_ok"] = False
        details["pivot_reference_parity"] = False
        return False, details

    for col in PIVOT_FEATURE_COLUMNS:
        details[f"pivot_reference_{col}_parity"] = _float_parity_equal(
            core_output.continuous[col],
            reference[col],
            atol=cfg.parity.float_atol,
            rtol=cfg.parity.float_rtol,
        )
    details["pivot_reference_parity"] = all(
        details[f"pivot_reference_{col}_parity"] for col in PIVOT_FEATURE_COLUMNS
    )
    return bool(details["pivot_reference_parity"]), details


def _is_binary_signal(series: pd.Series) -> bool:
    """Return True when signal contains only 0/1 values."""

    values = series.fillna(0).to_numpy(dtype=np.int64, copy=False)
    return bool(np.isin(values, np.array([0, 1], dtype=np.int64)).all())


def _validate_alphatrend_sanity(
    core_output: IndicatorCoreOutput,
    atol: float,
    rtol: float,
) -> bool:
    """Validate AlphaTrend structural invariants."""

    continuous = core_output.continuous
    raw = core_output.raw_events

    at_shift_ok = _float_parity_equal(
        continuous["AlphaTrend_2"],
        continuous["AlphaTrend"].shift(2),
        atol=atol,
        rtol=rtol,
    )
    binary_ok = all(_is_binary_signal(raw[col]) for col in ("AT_buy_raw", "AT_sell_raw", "AT_buy", "AT_sell"))

    buy_raw = raw["AT_buy_raw"].fillna(0).astype("uint8")
    sell_raw = raw["AT_sell_raw"].fillna(0).astype("uint8")
    buy = raw["AT_buy"].fillna(0).astype("uint8")
    sell = raw["AT_sell"].fillna(0).astype("uint8")

    confirmed_subset_ok = bool(((buy <= buy_raw) & (sell <= sell_raw)).all())
    exclusivity_ok = bool((buy_raw + sell_raw <= 1).all() and (buy + sell <= 1).all())
    return bool(at_shift_ok and binary_ok and confirmed_subset_ok and exclusivity_ok)


def _validate_supertrend_sanity(core_output: IndicatorCoreOutput) -> bool:
    """Validate SuperTrend structural invariants."""

    continuous = core_output.continuous
    raw = core_output.raw_events

    trend = continuous["ST_trend"]
    trend_values = trend.fillna(0).to_numpy(dtype=np.int8, copy=False)
    trend_domain_ok = bool(np.isin(trend_values, np.array([-1, 1], dtype=np.int8)).all())

    st_up = continuous["ST_up"]
    st_dn = continuous["ST_dn"]
    st_nonnull_union = st_up.notna() | st_dn.notna()
    if bool(st_nonnull_union.any()):
        first_valid_pos = int(np.flatnonzero(st_nonnull_union.to_numpy(dtype=bool, copy=False))[0])
        post_trend = trend.iloc[first_valid_pos:]
        post_up = st_up.iloc[first_valid_pos:]
        post_dn = st_dn.iloc[first_valid_pos:]
        band_alignment_ok = bool(
            post_up.notna().eq(post_trend.eq(1)).all()
            and post_dn.notna().eq(post_trend.eq(-1)).all()
        )
        band_exclusivity_ok = not bool((post_up.notna() & post_dn.notna()).any())
    else:
        band_alignment_ok = False
        band_exclusivity_ok = False

    buy = raw["ST_buy"].fillna(0).astype("uint8").eq(1)
    sell = raw["ST_sell"].fillna(0).astype("uint8").eq(1)
    binary_ok = _is_binary_signal(raw["ST_buy"]) and _is_binary_signal(raw["ST_sell"])
    transition_ok = bool(
        ((~buy) | ((trend == 1) & (trend.shift(1) == -1))).all()
        and ((~sell) | ((trend == -1) & (trend.shift(1) == 1))).all()
    )
    signal_exclusivity_ok = not bool((buy & sell).any())

    return bool(
        trend_domain_ok
        and band_alignment_ok
        and band_exclusivity_ok
        and binary_ok
        and transition_ok
        and signal_exclusivity_ok
    )


def evaluate_indicator_validation(
    ohlcv: pd.DataFrame,
    core_output: IndicatorCoreOutput,
    raw_events: pd.DataFrame,
    shifted_events: pd.DataFrame,
    cfg: FeatureBuildConfig,
) -> tuple[str, dict[str, bool]]:
    """Run mandatory indicator correctness validation checks."""

    details: dict[str, bool] = {}
    required_checks: dict[str, bool] = {}
    close = ohlcv["close"]
    for period in (200, 600, 1200):
        expected = close.ewm(span=period, adjust=False, min_periods=period).mean()
        details[f"ema_{period}_parity"] = _float_parity_equal(
            core_output.continuous[f"EMA_{period}"],
            expected,
            atol=cfg.parity.float_atol,
            rtol=cfg.parity.float_rtol,
        )
        required_checks[f"ema_{period}_parity"] = details[f"ema_{period}_parity"]

    pivot_reference = _compute_pivot_reference_previous_session(ohlcv, first_session_fill=cfg.pivot.first_session_fill)
    details["pivot_parity"] = all(
        _float_parity_equal(
            core_output.continuous[col],
            pivot_reference[col],
            atol=cfg.parity.float_atol,
            rtol=cfg.parity.float_rtol,
        )
        for col in PIVOT_FEATURE_COLUMNS
    )
    required_checks["pivot_parity"] = details["pivot_parity"]
    details["alphatrend_sanity"] = _validate_alphatrend_sanity(
        core_output,
        atol=cfg.parity.float_atol,
        rtol=cfg.parity.float_rtol,
    )
    required_checks["alphatrend_sanity"] = details["alphatrend_sanity"]
    details["supertrend_sanity"] = _validate_supertrend_sanity(core_output)
    required_checks["supertrend_sanity"] = details["supertrend_sanity"]
    details["event_shift_one"] = validate_shift_one(raw_events, shifted_events)
    required_checks["event_shift_one"] = details["event_shift_one"]

    pivot_reference_ok, pivot_reference_details = _evaluate_pivot_reference_validation(ohlcv, core_output, cfg)
    details.update(pivot_reference_details)
    required_checks["pivot_reference_parity"] = pivot_reference_ok

    failed = sorted(key for key, passed in required_checks.items() if not passed)
    if failed:
        LOGGER.error("Indicator validation failed | checks=%s", failed)
        return "failed", details
    return "passed", details


def _resolve_path(value: str, base_dir: Path) -> Path:
    """Resolve relative path against config directory."""

    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _get_required_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return required dict node."""

    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid dictionary config key: {key}")
    return value


def _get_optional_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return optional dict node, defaulting to empty dict."""

    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Invalid dictionary config key: {key}")
    return value


def _get_required_value(raw: dict[str, Any], key: str, section: str) -> Any:
    """Get required key from a section."""

    if key not in raw:
        raise ValueError(f"{section}.{key} is required")
    return raw[key]


def _compute_config_hash(raw: dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash for config payload."""

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    pivot_cfg = _get_required_dict(raw, "pivot")
    parity_cfg = _get_required_dict(raw, "parity")
    health_cfg = _get_required_dict(raw, "health")
    rl_cfg = _get_optional_dict(raw, "rl_features")

    rolling_vol_windows_raw = rl_cfg.get("rolling_vol_windows", [20, 50])
    if not isinstance(rolling_vol_windows_raw, list):
        raise ValueError("rl_features.rolling_vol_windows must be a list")
    zscore_window_raw = rl_cfg.get("zscore_window", 50)
    zscore_source_raw = rl_cfg.get("zscore_source", "return")
    vol_bins_raw = rl_cfg.get("volatility_regime_bins", 3)

    cfg = FeatureBuildConfig(
        input_root=_resolve_path(str(_get_required_value(raw, "input_root", "root")), base_dir),
        runs_root=_resolve_path(str(_get_required_value(raw, "runs_root", "root")), base_dir),
        parquet_glob=str(_get_required_value(raw, "parquet_glob", "root")),
        seed=int(_get_required_value(raw, "seed", "root")),
        supertrend=SuperTrendConfig(
            periods=int(_get_required_value(supertrend_cfg, "periods", "supertrend")),
            multiplier=float(_get_required_value(supertrend_cfg, "multiplier", "supertrend")),
            source=str(_get_required_value(supertrend_cfg, "source", "supertrend")),
            change_atr_method=bool(_get_required_value(supertrend_cfg, "change_atr_method", "supertrend")),
        ),
        alphatrend=AlphaTrendConfig(
            coeff=float(_get_required_value(alphatrend_cfg, "coeff", "alphatrend")),
            ap=int(_get_required_value(alphatrend_cfg, "ap", "alphatrend")),
            use_no_volume=bool(_get_required_value(alphatrend_cfg, "use_no_volume", "alphatrend")),
        ),
        pivot=PivotPolicyConfig(
            pivot_tf=str(_get_required_value(pivot_cfg, "pivot_tf", "pivot")),
            warmup_policy=str(_get_required_value(pivot_cfg, "warmup_policy", "pivot")),
            first_session_fill=str(_get_required_value(pivot_cfg, "first_session_fill", "pivot")),
        ),
        parity=ParityPolicyConfig(
            enabled=bool(_get_required_value(parity_cfg, "enabled", "parity")),
            sample_rows=int(_get_required_value(parity_cfg, "sample_rows", "parity")),
            float_atol=float(_get_required_value(parity_cfg, "float_atol", "parity")),
            float_rtol=float(_get_required_value(parity_cfg, "float_rtol", "parity")),
        ),
        health=HealthPolicyConfig(
            warn_ratio=float(_get_required_value(health_cfg, "warn_ratio", "health")),
            critical_warn_ratio=float(_get_required_value(health_cfg, "critical_warn_ratio", "health")),
            critical_columns=tuple(str(col) for col in _get_required_value(health_cfg, "critical_columns", "health")),
        ),
        config_hash=_compute_config_hash(raw),
        indicator_spec_version=INDICATOR_SPEC_VERSION,
        rl_features=RLFeatureConfig(
            rolling_vol_windows=tuple(int(window) for window in rolling_vol_windows_raw),
            zscore_window=int(zscore_window_raw),
            zscore_source=str(zscore_source_raw),
            volatility_regime_bins=int(vol_bins_raw),
        ),
    )
    validate_feature_config(cfg)
    return cfg


def validate_feature_config(cfg: FeatureBuildConfig) -> None:
    """Validate semantic constraints and lock forbidden overrides."""

    if not cfg.input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {cfg.input_root}")
    if not cfg.input_root.is_dir():
        raise NotADirectoryError(f"input_root is not a directory: {cfg.input_root}")
    if not cfg.parquet_glob.strip():
        raise ValueError("parquet_glob cannot be empty")
    if cfg.seed < 0:
        raise ValueError("seed must be >= 0")

    if cfg.supertrend.periods != _LOCKED_SUPERTREND_PERIODS:
        raise ValueError(f"supertrend.periods must be fixed at {_LOCKED_SUPERTREND_PERIODS}")
    if cfg.supertrend.multiplier != _LOCKED_SUPERTREND_MULTIPLIER:
        raise ValueError(f"supertrend.multiplier must be fixed at {_LOCKED_SUPERTREND_MULTIPLIER}")
    if cfg.supertrend.source.strip().lower() != _LOCKED_SUPERTREND_SOURCE:
        raise ValueError(f"supertrend.source must be fixed at {_LOCKED_SUPERTREND_SOURCE}")
    if cfg.supertrend.change_atr_method is not _LOCKED_SUPERTREND_CHANGE_ATR_METHOD:
        raise ValueError("supertrend.change_atr_method must be fixed at true")

    if cfg.alphatrend.coeff != _LOCKED_ALPHATREND_COEFF:
        raise ValueError(f"alphatrend.coeff must be fixed at {_LOCKED_ALPHATREND_COEFF}")
    if cfg.alphatrend.ap != _LOCKED_ALPHATREND_AP:
        raise ValueError(f"alphatrend.ap must be fixed at {_LOCKED_ALPHATREND_AP}")

    if cfg.pivot.pivot_tf.strip() != _LOCKED_PIVOT_TF:
        raise ValueError(f"pivot.pivot_tf must be fixed at {_LOCKED_PIVOT_TF}")
    if cfg.pivot.warmup_policy.strip().lower() not in _ALLOWED_PIVOT_WARMUP_POLICIES:
        raise ValueError("pivot.warmup_policy has unsupported value")
    if cfg.pivot.first_session_fill.strip().lower() not in _ALLOWED_PIVOT_FIRST_SESSION_FILL:
        raise ValueError("pivot.first_session_fill has unsupported value")

    if cfg.parity.sample_rows <= 0:
        raise ValueError("parity.sample_rows must be > 0")
    if cfg.parity.float_atol < 0.0:
        raise ValueError("parity.float_atol must be >= 0")
    if cfg.parity.float_rtol < 0.0:
        raise ValueError("parity.float_rtol must be >= 0")

    if not (0.0 <= cfg.health.warn_ratio <= 1.0):
        raise ValueError("health.warn_ratio must be in [0.0, 1.0]")
    if not (0.0 <= cfg.health.critical_warn_ratio <= 1.0):
        raise ValueError("health.critical_warn_ratio must be in [0.0, 1.0]")
    if cfg.health.critical_warn_ratio > cfg.health.warn_ratio:
        raise ValueError("health.critical_warn_ratio must be <= health.warn_ratio")
    if not cfg.health.critical_columns:
        raise ValueError("health.critical_columns cannot be empty")

    normalized_windows = _normalize_rolling_vol_windows(cfg.rl_features.rolling_vol_windows)
    if len(normalized_windows) < 2:
        raise ValueError("rl_features.rolling_vol_windows must include at least two windows")
    if cfg.rl_features.zscore_window <= 1:
        raise ValueError("rl_features.zscore_window must be > 1")
    if cfg.rl_features.zscore_source.strip().lower() not in {"return", "close"}:
        raise ValueError("rl_features.zscore_source must be one of {'return', 'close'}")
    if cfg.rl_features.volatility_regime_bins not in (2, 3, 4):
        raise ValueError("rl_features.volatility_regime_bins must be one of {2, 3, 4}")


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


def apply_dtype_policy(
    df: pd.DataFrame,
    *,
    continuous_feature_columns: tuple[str, ...] | None = None,
    flag_feature_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Cast continuous columns to float32 and encoded/event flags to uint8."""

    out = df.copy()
    resolved_continuous = continuous_feature_columns or CONTINUOUS_FEATURE_COLUMNS
    resolved_flags = flag_feature_columns or EVENT_FLAG_COLUMNS
    continuous = ("open", "high", "low", "close", "volume", *resolved_continuous)

    for col in continuous:
        if col not in out.columns:
            raise ValueError(f"Missing continuous column for dtype policy: {col}")
        out[col] = out[col].astype("float32")

    for col in resolved_flags:
        if col not in out.columns:
            raise ValueError(f"Missing event flag column for dtype policy: {col}")
        out[col] = out[col].astype("uint8")

    return out


def build_feature_artifacts(df: pd.DataFrame, cfg: FeatureBuildConfig) -> FeatureBuildArtifacts:
    """Build feature table plus raw/shifted events for one input dataframe."""

    ohlcv = validate_ohlcv_frame(df)
    core_output = _build_indicator_core(ohlcv, cfg)
    rl_output = _build_rl_feature_output(ohlcv, core_output.continuous, cfg)
    continuous_feature_columns, flag_feature_columns = get_output_feature_columns(cfg)
    feature_groups = get_feature_groups(cfg)

    raw_events = generate_raw_event_flags(core_output.raw_events)
    shifted_events = enforce_shift_one(raw_events)
    if not validate_shift_one(raw_events, shifted_events):
        raise ValueError("Strict shift(1) validation failed for event columns")

    parity_status, parity_details = evaluate_indicator_parity(ohlcv, core_output, cfg)
    validation_status, validation_details = evaluate_indicator_validation(
        ohlcv=ohlcv,
        core_output=core_output,
        raw_events=raw_events,
        shifted_events=shifted_events,
        cfg=cfg,
    )
    formula_fingerprints = compute_formula_fingerprints(cfg)
    formula_fingerprint_bundle = compute_formula_fingerprint_bundle(formula_fingerprints)

    frame = pd.concat(
        [
            ohlcv,
            core_output.continuous.loc[:, list(CONTINUOUS_FEATURE_COLUMNS)],
            rl_output.continuous,
            rl_output.continuous_placeholders,
            shifted_events.loc[:, list(EVENT_FLAG_COLUMNS)],
            rl_output.shifted_regime_flags.loc[:, list(REGIME_FLAG_COLUMNS)],
            rl_output.flag_placeholders.loc[:, list(PLACEHOLDER_FLAG_COLUMNS)],
        ],
        axis=1,
    )
    frame = apply_dtype_policy(
        frame,
        continuous_feature_columns=continuous_feature_columns,
        flag_feature_columns=flag_feature_columns,
    )

    return FeatureBuildArtifacts(
        frame=frame,
        raw_events=raw_events,
        shifted_events=shifted_events,
        indicator_parity_status=parity_status,
        indicator_parity_details=parity_details,
        indicator_validation_status=validation_status,
        indicator_validation_details=validation_details,
        formula_fingerprints=formula_fingerprints,
        formula_fingerprint_bundle=formula_fingerprint_bundle,
        raw_regime_flags=rl_output.raw_regime_flags,
        shifted_regime_flags=rl_output.shifted_regime_flags,
        continuous_feature_columns=continuous_feature_columns,
        flag_feature_columns=flag_feature_columns,
        placeholder_columns=feature_groups["placeholders"],
        warmup_rows_by_column=rl_output.warmup_rows_by_column,
        feature_groups=feature_groups,
    )
