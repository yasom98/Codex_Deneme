# Indicator Specification Lock

Spec Version: `indicators.v2026-02-15.1`

## Scope
This document is the formula contract for production feature generation.
Reference source-of-truth implementation lives in `src/data/indicator_reference.py`.

## Changelog
- `indicators.v2026-02-15.1`
  - Locked formulas to externally provided reference code.
  - Locked default parameters for Pivot/EMA/AlphaTrend/Supertrend.
  - Locked strict bar-close signal logic and shift(1) event policy integration.

## Global Input Contract
- Required OHLCV columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- `timestamp` must be UTC, monotonic increasing.
- Feature engine validates and sorts timestamps, then computes indicators on sorted data only.

## Helper Indicators
### True Range
- Formula:
  - `prev_close = close.shift(1)`
  - `TR = max(|high-low|, |high-prev_close|, |low-prev_close|)`
- Warmup:
  - First bar uses NaN in `prev_close`; max across components is applied as in pandas behavior.

### SMA
- Formula: `x.rolling(n, min_periods=n).mean()`
- Warmup: NaN for first `n-1` rows.

### EMA
- Formula: `x.ewm(span=n, adjust=False, min_periods=n).mean()`
- Warmup: NaN for first `n-1` rows.

### RSI
- Formula:
  - `delta = close.diff()`
  - `gain = clip(delta, lower=0)`
  - `loss = clip(-delta, lower=0)`
  - `avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()`
  - `avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()`
  - `rs = avg_gain / avg_loss.replace(0, NaN)`
  - `RSI = 100 - (100 / (1 + rs))`
  - `fillna(50.0)`
- Warmup: first `n-1` rows -> 50.0 after fill.

### MFI
- Formula:
  - `tp = (high + low + close)/3`
  - `mf = tp * volume`
  - `direction = tp.diff()`
  - `pos = mf where direction > 0 else 0`
  - `neg = mf where direction < 0 else 0`
  - `pos_sum = pos.rolling(n, min_periods=n).sum()`
  - `neg_sum = neg.rolling(n, min_periods=n).sum().replace(0, NaN)`
  - `mr = pos_sum / neg_sum`
  - `MFI = 100 - (100/(1+mr))`
  - `fillna(50.0)`
- Warmup: first `n-1` rows -> 50.0 after fill.

### Cross Logic
- `crossover(a,b)`: `a[t] > b[t]` and `a[t-1] <= b[t-1]`
- `crossunder(a,b)`: `a[t] < b[t]` and `a[t-1] >= b[t-1]`
- Strict bar-close based, no intrabar repaint assumption.

## Pivot (Traditional)
- Base timeframe: `pivot_tf="1D"` (locked).
- Leak-free logic:
  - Resample OHLC on `pivot_tf`.
  - Use previous resampled bar (`shift(1)`).
  - Compute levels from previous bar only.
  - Forward-fill levels to intraday rows.
- Formulas:
  - `PP = (H_prev + L_prev + C_prev)/3`
  - `R1 = 2*PP - L_prev`
  - `S1 = 2*PP - H_prev`
  - `R2 = PP + (H_prev - L_prev)`
  - `S2 = PP - (H_prev - L_prev)`
  - `R3 = 2*PP + H_prev - 2*L_prev`
  - `S3 = 2*PP - (2*H_prev - L_prev)`
  - `R4 = 3*PP + H_prev - 3*L_prev`
  - `S4 = 3*PP - (3*H_prev - L_prev)`
  - `R5 = 4*PP + H_prev - 4*L_prev`
  - `S5 = 4*PP - (4*H_prev - L_prev)`
- Warmup:
  - First session has no previous session => NaN allowed by policy.
  - Optional policy fill: `ffill_from_second_session` copies second-session pivot levels to first session only.

## EMA Set
- Columns: `EMA_200`, `EMA_600`, `EMA_1200`
- Price source: `close`
- Formula: EMA helper with strict `min_periods=n`.
- Warmup:
  - NaN before each lookback window is complete.

## AlphaTrend
- Locked parameters:
  - `coeff=3.0`
  - `ap=11`
  - `use_no_volume=False` by default
- Formula:
  - `atr = sma(true_range, ap)` (not Wilder ATR)
  - `upT = low - atr*coeff`
  - `downT = high + atr*coeff`
  - regime:
    - if `use_no_volume=True` or no volume: `rsi(close, ap) >= 50`
    - else: `mfi(high, low, close, volume, ap) >= 50`
  - recursive `AlphaTrend`:
    - first row: NaN
    - first valid recursive row initializes from `upT`/`downT` by regime
    - up regime: `max(prev_at, upT)`
    - down regime: `min(prev_at, downT)`
- Signals:
  - `AT_buy_raw = crossover(AlphaTrend, AlphaTrend.shift(2))`
  - `AT_sell_raw = crossunder(AlphaTrend, AlphaTrend.shift(2))`
  - `barssince` gating:
    - `K1 = barssince(AT_buy_raw)`
    - `K2 = barssince(AT_sell_raw)`
    - `O1 = barssince(AT_buy_raw.shift(1))`
    - `O2 = barssince(AT_sell_raw.shift(1))`
  - `AT_buy = AT_buy_raw and (O1 > K2)`
  - `AT_sell = AT_sell_raw and (O2 > K1)`
- Warmup:
  - Early rows can be NaN in `AlphaTrend` due to ATR/SMA warmup.

## Supertrend
- Locked parameters:
  - `periods=10`
  - `multiplier=3.0`
  - `source="hl2"`
  - `change_atr_method=True` (Wilder ATR)
- Formula:
  - `src = (high+low)/2` for `hl2`
  - `atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()` (when locked flag true)
  - `up = src - multiplier*atr`
  - `dn = src + multiplier*atr`
  - recursive carry and trend transition exactly as reference code.
- Signals:
  - `ST_buy = (ST_trend==1) and (ST_trend.shift(1)==-1)`
  - `ST_sell = (ST_trend==-1) and (ST_trend.shift(1)==1)`
- Warmup:
  - ATR warmup causes early NaN in bands, trend initializes at `1`.

## Event Shift Policy (Pipeline Contract)
- Raw indicator signals are computed on bar close.
- Output event flags are strict `shift(1)` versions of raw signals.
- Final event columns are `uint8` and satisfy:
  - first row `0`
  - `out[t] == raw[t-1]` for `t>=1`

## Dtype Contract
- Continuous indicator columns: `float32`
- Event/signal output columns: `uint8`

## Ambiguity Handling
- Any unsupported/unknown formula override must fail fast with explicit error.
- No silent fallback is allowed for locked parameters.
