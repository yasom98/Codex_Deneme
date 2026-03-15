# Post Gap-Fill Audit: BTCUSDT Perpetual State Capture

## 1. Purpose

This document captures the currently proven BTCUSDT Binance USD-M perpetual gap-fill state before the project moves to training-related audit work.

It is a repo-local, evidence-driven state capture of an already successful operation. It is not a new run, not a redesign, and not a claim of operational readiness for non-BTC symbols.

## 2. Canonical operation identity

- Repo: `Codex_Deneme`
- Symbol: `BTCUSDT`
- Route: `Windows host execution`
- Session id: `20260314Tbinance_perp_hist_full_003`
- Explicit cutoff: `2026-03-14T10:00:00+00:00`

## 3. Legacy immutable baseline

Immutable root CSV surface:

- `BTC_USDT_1m_price_data.csv`
- `BTC_USDT_5m_price_data.csv`
- `BTC_USDT_15m_price_data.csv`

Verified legacy last timestamps:

- `1m`: `2025-04-25T20:58:00+00:00`
- `5m`: `2025-04-25T21:10:00+00:00`
- `15m`: `2025-04-25T21:00:00+00:00`

Verified gap start anchors:

- `1m`: `2025-04-25T20:59:00+00:00`
- `5m`: `2025-04-25T21:15:00+00:00`
- `15m`: `2025-04-25T21:15:00+00:00`

Legacy mutation status:

- Verified during this audit: `git diff --name-only -- BTC_USDT_1m_price_data.csv BTC_USDT_5m_price_data.csv BTC_USDT_15m_price_data.csv`
- Result: empty output
- Conclusion: legacy immutable root CSV files were not mutated

## 4. Successful completion summary

| Timeframe | Completion | Pages / Requests | Rows Downloaded |
| --- | --- | ---: | ---: |
| `1m` | `completed` | `465` | `464462` |
| `5m` | `completed` | `93` | `92890` |
| `15m` | `completed` | `31` | `30964` |

## 5. Processing and contract outcomes

- Raw download result: `success`
- Standardization result: `success`
- Feature build result: `success`
- Feature contract compatibility result: `success`

Verified processing details:

- `standardize` stage exit code: `0`
- `feature_build` stage exit code: `0`
- `feature_contract_compatibility.checks.column_dtypes_match`: `true`
- Refreshed feature manifest timestamp dtype: `datetime64[ns, UTC]`

## 6. Evidence artifacts

| Artifact | What It Proves | Verified During This Audit |
| --- | --- | --- |
| `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/historical_backfill_checkpoint.json` | Session identity, cutoff, bootstrap proof, per-timeframe `completed` status, pages completed, rows downloaded, start/end timestamps, `legacy_lineage_touched=false` | `yes` |
| `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/separate_parallel_lineage_download_report.json` | Download status `success`, separate lineage identity, legacy anchors, download windows, rows downloaded, raw output file paths | `yes` |
| `runs/20260314Tbinance_perp_hist_full_003/data_tail_refresh/reports/separate_parallel_lineage_processing_report.json` | Processing status `success`, `standardize` and `feature_build` stage success, feature contract compatibility success | `yes` |
| `runs/20260314Tbinance_perp_hist_full_003/data_features/reports/feature_manifest.json` | Supporting evidence that the refreshed feature artifacts exist and record canonical `timestamp = datetime64[ns, UTC]` | `yes` |

## 7. Operator verification checklist

- [ ] Session id is `20260314Tbinance_perp_hist_full_003`
- [ ] Cutoff is exactly `2026-03-14T10:00:00+00:00`
- [ ] `1m`, `5m`, and `15m` all show `completed`
- [ ] Requests/pages are `465`, `93`, and `31`
- [ ] Rows downloaded are `464462`, `92890`, and `30964`
- [ ] `separate_parallel_lineage_download_report.json` shows `status = success`
- [ ] `historical_backfill_checkpoint.json` shows `status = completed`
- [ ] `separate_parallel_lineage_processing_report.json` shows `status = success`
- [ ] Feature contract compatibility is green / `success`
- [ ] `feature_manifest.json` records `timestamp = datetime64[ns, UTC]`
- [ ] Legacy immutable CSV files remained untouched

## 8. Known boundaries

- This audit captures the currently proven BTCUSDT route.
- The proven operational path for this state is Windows-host execution.
- Non-BTC symbols are not yet claimed as operationally proven by this audit.
- Future extension should be designed parametrically for other eligible Binance-listed symbols, rather than hard-coded to named examples.
- This document is a state capture of a completed run, not a new execution record.

## 9. Next project transition

The next correct project focus is:

1. training-ready audit of the refreshed separate perpetual lineage
2. then a bounded training smoke

This document does not design or start the training phase. It only records that the BTCUSDT perpetual gap-fill and processing path reached a contract-valid, repo-ready state.
