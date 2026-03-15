# Binance Perpetual Gap-Fill on Windows Host

## Purpose

This operation fills the missing Binance USD-M perpetual BTCUSDT history gap for:

- `1m`
- `5m`
- `15m`

It writes only into the separate perpetual lineage under `runs/<refresh_session_id>/...` and then runs the existing repo pipeline:

- raw download
- OHLCV standardization
- feature build
- feature contract compatibility check

This is an operational runbook for the already-proven repo path. It does not describe a new architecture.

## When To Use This

Use this runbook when all of the following are true:

- you need to extend the separate Binance perpetual lineage forward from the existing immutable legacy anchors
- you want a controlled historical gap-fill with an explicit UTC cutoff
- you need operator-visible progress bars and checkpoint-safe resume behavior
- you must preserve the immutable legacy root CSV lineage

## Why The Windows Host Route Was Used

The Windows host route is the proven operational path for this repo state.

Reason:

- Binance USD-M futures bootstrap was fixed and validated
- WSL DNS/connectivity was previously the blocker for Binance futures metadata access
- Windows host execution completed the full BTCUSDT historical gap-fill successfully
- the existing repo processing pipeline also completed successfully on that route

This is why the wrapper in this repo runs the proven Windows host path instead of the earlier WSL route.

## Why WSL Was Not The Chosen Operational Route

WSL was not chosen for the production gap-fill route because Binance USD-M futures metadata access previously failed there due to environment-specific DNS/runtime behavior.

That problem was diagnosed and the repo was moved to the Windows host route for Binance fetch execution. The Windows route is therefore the stored operational path for this runbook.

## Immutable Legacy CSV Rules

These files are reference-only and must remain untouched:

- `BTC_USDT_1m_price_data.csv`
- `BTC_USDT_5m_price_data.csv`
- `BTC_USDT_15m_price_data.csv`

Never:

- append to them
- overwrite them
- merge new perpetual data into them
- rename them as part of the gap-fill

The historical Binance perpetual run must write only into the separate lineage under `runs/<refresh_session_id>/...`.

## Verified BTCUSDT Legacy Anchors

These anchors were already verified from the immutable legacy root CSV lineage:

- `1m` legacy end: `2025-04-25T20:58:00+00:00` -> Binance gap starts at `2025-04-25T20:59:00+00:00`
- `5m` legacy end: `2025-04-25T21:10:00+00:00` -> Binance gap starts at `2025-04-25T21:15:00+00:00`
- `15m` legacy end: `2025-04-25T21:00:00+00:00` -> Binance gap starts at `2025-04-25T21:15:00+00:00`

These anchors are derived by the repo code. Do not hand-edit them in the wrapper.

## Exact Wrapper Command Example

This is an exact repo-specific Windows host command pattern:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_binance_perpetual_gapfill_windows.ps1 `
  -RepoPath "$env:USERPROFILE\OneDrive\Desktop\Codex_Deneme" `
  -Symbol BTCUSDT `
  -CutoffUtc 2026-03-14T10:00:00+00:00 `
  -AcceptedRunId 20260215T230000Z_clean15m `
  -RefreshSessionId 20260314Tbinance_perp_hist_full_003_rerun `
  -RequestLimit 1000
```

The wrapper prints the exact underlying Python command before execution.

## Underlying Command Pattern

The wrapper preserves this underlying repo command shape:

```text
python scripts/refresh_market_data_tail.py ^
  --mode separate_binance_perpetual_backfill ^
  --accepted-run-id <ACCEPTED_RUN_ID> ^
  --refresh-session-id <REFRESH_SESSION_ID> ^
  --legacy-input-root <REPO_PATH> ^
  --data-config <TEMP_WINDOWS_DATA_CONFIG> ^
  --features-config <FEATURES_CONFIG> ^
  --request-limit 1000 ^
  --historical-max-candles-per-timeframe 0 ^
  --target-end-utc <EXPLICIT_UTC_CUTOFF> ^
  --max-retries 2 ^
  --retry-backoff-seconds 1.0 ^
  --log-level INFO
```

The wrapper is intentionally transparent. It is not a hidden orchestration layer.

## Explicit Cutoff Usage

Always use an explicit UTC cutoff for a controlled historical run.

Why:

- it keeps the run deterministic
- it avoids accidental "latest available" drift
- it lets the repo fail closed if the requested cutoff is not yet reachable for one of the timeframes

The cutoff must be:

- timezone-aware UTC
- aligned to the requested timeframe boundaries
- reachable for all target timeframes

Example safe common cutoff for `1m / 5m / 15m`:

- `2026-03-14T10:00:00+00:00`

## Progress Bars and ETA

The current repo path exposes operator-visible `tqdm` progress bars during the Binance historical loop.

Visible behavior:

- one progress bar per timeframe
- separate runs for `1m`, then `5m`, then `15m`
- percentage complete
- speed in rows per second
- page/request progress via `pages=current/total`
- ETA via standard `tqdm` remaining-time display

If a fail-closed guard triggers before the loop starts, progress bars will not appear. That is expected.

## Produced Reports And Artifacts

The run produces repo-local artifacts under `runs/<refresh_session_id>/...`.

Key reports:

- `data_tail_refresh/reports/provider_strategy_report.json`
- `data_tail_refresh/reports/separate_parallel_lineage_download_report.json`
- `data_tail_refresh/reports/historical_backfill_checkpoint.json`
- `data_tail_refresh/reports/separate_parallel_lineage_processing_report.json`

Key raw outputs:

- `data_tail_refresh/separate_parallel_lineage/raw/historical_backfill/binance_perpetual/BTC_USDT_1m_price_data.csv`
- `data_tail_refresh/separate_parallel_lineage/raw/historical_backfill/binance_perpetual/BTC_USDT_5m_price_data.csv`
- `data_tail_refresh/separate_parallel_lineage/raw/historical_backfill/binance_perpetual/BTC_USDT_15m_price_data.csv`

Key processing outputs:

- `data_standardized/reports/summary.json`
- `data_features/reports/summary.json`
- `data_features/reports/feature_manifest.json`

## Success Checklist

Before signing off the run, verify all of the following:

- per-timeframe checkpoint status is `completed` for `1m`, `5m`, and `15m`
- download report status is `success`
- processing report status is `success`
- `standardize` stage exit code is `0`
- `feature_build` stage exit code is `0`
- feature contract compatibility status is `success`
- expected raw output files exist under the separate perpetual lineage
- expected report files exist
- `feature_manifest.json` reports `timestamp = datetime64[ns, UTC]`
- legacy immutable CSV files remained untouched

## Fail-Closed Guards To Expect

This path intentionally stops early if any of the following are invalid:

- missing legacy reference files
- invalid or unreadable legacy timestamps
- wrong Binance bootstrap surface
- wrong market resolution
- unreachable explicit cutoff
- empty or partial page result
- timeframe alignment mismatch
- processing failure
- feature contract mismatch

Fail-closed behavior is expected and should not be bypassed with manual file edits.

## Do Not Do This

Do not:

- run this as a WSL-native Binance fetch path for this repo state
- append new data into the immutable legacy root CSV files
- remove the explicit cutoff from a controlled historical run
- hand-edit checkpoint files to force completion
- bypass the processing report or feature contract compatibility checks
- change request pagination behavior in the wrapper
- treat ETHUSDT or XRPUSDT as already-proven with the current BTC-only pipeline state

## Adapting Later For ETHUSDT / XRPUSDT

This wrapper stores the proven BTCUSDT route.

For future ETHUSDT or XRPUSDT work, the operator-facing parameter that would change first is:

- `-Symbol`

But that alone is not sufficient today. The current repo path is still BTC-specific in the historical lineage anchors and symbol normalization.

Before ETHUSDT / XRPUSDT can use the same wrapper operationally, these repo fields/surfaces must be parameterized explicitly:

- immutable legacy reference file names, currently `BTC_USDT_<timeframe>_price_data.csv`
- Binance symbol normalization currently bound to BTCUSDT / BTC/USDT:USDT
- output file naming that currently emits `BTC_USDT_<timeframe>_price_data.csv`
- any accepted-run lineage/report assumptions that remain BTC-specific

Until those repo surfaces are intentionally extended, this Windows wrapper should be treated as BTCUSDT-only and should fail closed for other symbols.
