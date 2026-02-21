"""Feature engineering core locked to reference indicator formulas."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class IndicatorCoreOutput:
    """Internal indicator output blocks before event shift."""

    continuous: pd.DataFrame
    raw_events: pd.DataFrame


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
    indexed = indexed_ohlcv if indexed_ohlcv is not None else _to_datetime_index(ohlcv)
    if len(indexed) != len(ohlcv):
        raise ValueError("indexed_ohlcv length mismatch for pivot computation")
    pivots = ref.compute_pivots_traditional(indexed, pivot_tf=pivot_tf)

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


def enforce_shift_one(raw_events: pd.DataFrame) -> pd.DataFrame:
    """Enforce strict shift(1) for every event column."""

    out = pd.DataFrame(index=raw_events.index)
    for col in EVENT_FLAG_COLUMNS:
        if col not in raw_events.columns:
            raise ValueError(f"Missing raw event column: {col}")
        out[col] = _shift_flag_one(raw_events[col])
    return out


def validate_shift_one(raw_events: pd.DataFrame, shifted_events: pd.DataFrame) -> bool:
    """Validate strict shift(1) relation for event columns."""

    for col in EVENT_FLAG_COLUMNS:
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


def _float_parity_equal(actual: pd.Series, expected: pd.Series, atol: float, rtol: float) -> bool:
    """Return allclose with NaN-aware comparison."""

    actual_arr = actual.to_numpy(dtype=np.float64, copy=False)
    expected_arr = expected.to_numpy(dtype=np.float64, copy=False)
    if actual_arr.shape != expected_arr.shape:
        return False
    return bool(np.allclose(actual_arr, expected_arr, atol=atol, rtol=rtol, equal_nan=True))


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
    close = ohlcv["close"]
    for period in (200, 600, 1200):
        expected = close.ewm(span=period, adjust=False, min_periods=period).mean()
        details[f"ema_{period}_parity"] = _float_parity_equal(
            core_output.continuous[f"EMA_{period}"],
            expected,
            atol=cfg.parity.float_atol,
            rtol=cfg.parity.float_rtol,
        )

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
    details["alphatrend_sanity"] = _validate_alphatrend_sanity(
        core_output,
        atol=cfg.parity.float_atol,
        rtol=cfg.parity.float_rtol,
    )
    details["supertrend_sanity"] = _validate_supertrend_sanity(core_output)
    details["event_shift_one"] = validate_shift_one(raw_events, shifted_events)

    failed = sorted(key for key, passed in details.items() if not passed)
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
    core_output = _build_indicator_core(ohlcv, cfg)

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
            shifted_events.loc[:, list(EVENT_FLAG_COLUMNS)],
        ],
        axis=1,
    )
    frame = apply_dtype_policy(frame)

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
    )
