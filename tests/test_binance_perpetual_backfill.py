"""Tests for the separate Binance perpetual historical backfill flow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from data.binance_perpetual_backfill import BinancePerpetualBackfillOptions, run_separate_binance_perpetual_backfill
from data.tail_refresh import TIMEFRAME_SECONDS, _validate_timeframe_alignment
from tests.test_gateio_perpetual_lineage import _seed_project


def _options(
    project_root: Path,
    data_config_path: Path,
    features_config_path: Path,
    *,
    refresh_session_id: str,
    max_candles_per_timeframe: int = 4,
    target_end_utc: str | None = None,
) -> BinancePerpetualBackfillOptions:
    return BinancePerpetualBackfillOptions(
        project_root=project_root,
        accepted_run_id="accepted_ref",
        refresh_session_id=refresh_session_id,
        legacy_input_root=project_root,
        data_config_path=data_config_path,
        features_config_path=features_config_path,
        request_limit=2,
        max_retries=1,
        retry_backoff_seconds=0.0,
        python_executable=project_root / ".venv" / "bin" / "python",
        max_candles_per_timeframe=max_candles_per_timeframe,
        target_end_utc=target_end_utc,
        log_level="INFO",
    )


def _target_end_by_timeframe() -> dict[str, pd.Timestamp]:
    return {
        "1m": pd.Timestamp("2024-01-01T00:05:00+00:00"),
        "5m": pd.Timestamp("2024-01-01T00:25:00+00:00"),
        "15m": pd.Timestamp("2024-01-01T00:30:00+00:00"),
    }


def _fake_row(timestamp: pd.Timestamp, seed: int) -> list[object]:
    return [
        int(timestamp.timestamp() * 1000),
        float(seed),
        float(seed) + 0.1,
        float(seed) - 0.1,
        float(seed) + 0.05,
        float(seed) * 10.0,
    ]


def _fake_binance_usdm_market() -> dict[str, object]:
    return {
        "id": "BTCUSDT",
        "symbol": "BTC/USDT:USDT",
        "type": "swap",
        "contract": True,
        "linear": True,
        "swap": True,
        "future": False,
        "spot": False,
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
    }


def test_validate_timeframe_alignment_accepts_datetime64_ms_utc() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.Series(
                pd.to_datetime(
                    [
                        "2025-04-25T20:59:00+00:00",
                        "2025-04-25T21:00:00+00:00",
                        "2025-04-25T21:01:00+00:00",
                    ],
                    utc=True,
                )
            ).astype("datetime64[ms, UTC]")
        }
    )

    assert str(frame["timestamp"].dtype) == "datetime64[ms, UTC]"
    assert _validate_timeframe_alignment(frame, "1m") == (True, 0, None)


def test_run_binance_backfill_writes_historical_outputs_and_checkpoint(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config_path, features_config_path, _ = _seed_project(tmp_path)
    options = _options(project_root, data_config_path, features_config_path, refresh_session_id="refresh_hist")
    targets = _target_end_by_timeframe()

    class _FakeClient:
        id = "binanceusdm"
        urls = {"api": {"fapiPublic": "https://fapi.binance.com/fapi/v1"}}
        options = {"defaultType": "swap", "defaultSubType": "linear", "fetchMarkets": {"types": ["linear"]}}

        def load_markets(self) -> dict[str, dict[str, str]]:
            return {"BTC/USDT:USDT": _fake_binance_usdm_market()}

        def close(self) -> None:
            return None

    def fake_retry(
        client: object,
        *,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> list[list[object]]:
        del client, symbol, max_retries, retry_backoff_seconds
        start = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        rows: list[list[object]] = []
        current = start
        seed = 1
        while len(rows) < limit and current <= targets[timeframe]:
            rows.append(_fake_row(current, seed))
            current = current + pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            seed += 1
        return rows

    monkeypatch.setattr("data.binance_perpetual_backfill._utc_now", lambda: datetime(2024, 1, 1, 0, 46, tzinfo=timezone.utc))
    monkeypatch.setattr("data.binance_perpetual_backfill._build_binance_usdm_client", lambda: _FakeClient())
    monkeypatch.setattr("data.binance_perpetual_backfill._retry_fetch_ohlcv", fake_retry)
    monkeypatch.setattr(
        "data.binance_perpetual_backfill._run_processing_chain",
        lambda options, raw_input_root, **kwargs: {
            "status": "success",
            "report": {
                "status": "success",
                "refresh_session_id": options.refresh_session_id,
                "exchange_used": "binance",
                "market_type": "perpetual",
                "source_lineage": "separate_parallel_lineage",
                "provider_track": "historical_backfill",
                "provider_choice_reason": "historical",
                "legacy_lineage_touched": False,
                "raw_input_root": str(raw_input_root.resolve()),
                "feature_contract_compatibility": {"status": "success", "checks": {}},
                "stages": [],
                "errors": [],
            },
        },
    )

    result = run_separate_binance_perpetual_backfill(options)

    assert result["status"] == "success"
    for timeframe, expected_rows in {"1m": 4, "5m": 4, "15m": 1}.items():
        output_path = (
            project_root
            / "runs"
            / "refresh_hist"
            / "data_tail_refresh"
            / "separate_parallel_lineage"
            / "raw"
            / "historical_backfill"
            / "binance_perpetual"
            / f"BTC_USDT_{timeframe}_price_data.csv"
        )
        assert output_path.exists()
        frame = pd.read_csv(output_path)
        assert len(frame) == expected_rows

    report_path = project_root / "runs" / "refresh_hist" / "data_tail_refresh" / "reports" / "historical_backfill_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["provider_track"] == "historical_backfill"
    assert payload["provider_choice_reason"]
    assert payload["legacy_lineage_touched"] is False
    assert payload["completion_scope"] == "bounded_smoke"
    assert payload["checkpoint_report_path"]
    assert payload["bootstrap_proof"]["status"] == "proven"
    assert payload["bootstrap_proof"]["resolved"]["ccxt_exchange_id"] == "binanceusdm"
    assert payload["bootstrap_proof"]["resolved"]["market_type"] == "swap"
    assert payload["bootstrap_proof"]["resolved"]["spot"] is False

    checkpoint_path = project_root / "runs" / "refresh_hist" / "data_tail_refresh" / "reports" / "historical_backfill_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "completed"
    assert checkpoint["bootstrap_status"] == "proven"
    assert checkpoint["timeframes"]["1m"]["pages_completed"] == 2
    assert checkpoint["timeframes"]["5m"]["pages_completed"] == 2
    assert checkpoint["timeframes"]["15m"]["pages_completed"] == 1


def test_run_binance_backfill_resumes_from_checkpoint(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config_path, features_config_path, _ = _seed_project(tmp_path)
    options = _options(project_root, data_config_path, features_config_path, refresh_session_id="refresh_resume")
    targets = _target_end_by_timeframe()

    class _FakeClient:
        id = "binanceusdm"
        urls = {"api": {"fapiPublic": "https://fapi.binance.com/fapi/v1"}}
        options = {"defaultType": "swap", "defaultSubType": "linear", "fetchMarkets": {"types": ["linear"]}}

        def load_markets(self) -> dict[str, dict[str, str]]:
            return {"BTC/USDT:USDT": _fake_binance_usdm_market()}

        def close(self) -> None:
            return None

    call_counts: dict[tuple[str, int], int] = {}

    def flaky_retry(
        client: object,
        *,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> list[list[object]]:
        del client, symbol, max_retries, retry_backoff_seconds
        key = (timeframe, since_ms)
        call_counts[key] = call_counts.get(key, 0) + 1
        if timeframe == "1m" and since_ms > int(pd.Timestamp("2024-01-01T00:02:00+00:00").timestamp() * 1000):
            raise RuntimeError("forced second-page failure")
        start = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        rows: list[list[object]] = []
        current = start
        seed = 1
        while len(rows) < limit and current <= targets[timeframe]:
            rows.append(_fake_row(current, seed))
            current = current + pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            seed += 1
        return rows

    def stable_retry(
        client: object,
        *,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> list[list[object]]:
        del client, symbol, max_retries, retry_backoff_seconds
        start = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        rows: list[list[object]] = []
        current = start
        seed = 1
        while len(rows) < limit and current <= targets[timeframe]:
            rows.append(_fake_row(current, seed))
            current = current + pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            seed += 1
        return rows

    monkeypatch.setattr("data.binance_perpetual_backfill._utc_now", lambda: datetime(2024, 1, 1, 0, 46, tzinfo=timezone.utc))
    monkeypatch.setattr("data.binance_perpetual_backfill._build_binance_usdm_client", lambda: _FakeClient())
    monkeypatch.setattr("data.binance_perpetual_backfill._retry_fetch_ohlcv", flaky_retry)
    monkeypatch.setattr(
        "data.binance_perpetual_backfill._run_processing_chain",
        lambda options, raw_input_root, **kwargs: {
            "status": "success",
            "report": {
                "status": "success",
                "refresh_session_id": options.refresh_session_id,
                "exchange_used": "binance",
                "market_type": "perpetual",
                "source_lineage": "separate_parallel_lineage",
                "provider_track": "historical_backfill",
                "provider_choice_reason": "historical",
                "legacy_lineage_touched": False,
                "raw_input_root": str(raw_input_root.resolve()),
                "feature_contract_compatibility": {"status": "success", "checks": {}},
                "stages": [],
                "errors": [],
            },
        },
    )

    first_result = run_separate_binance_perpetual_backfill(options)
    assert first_result["status"] == "failed"

    checkpoint_path = project_root / "runs" / "refresh_resume" / "data_tail_refresh" / "reports" / "historical_backfill_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "failed"
    assert checkpoint["bootstrap_status"] == "proven"
    assert checkpoint["timeframes"]["1m"]["pages_completed"] == 1
    assert checkpoint["timeframes"]["1m"]["status"] == "failed"

    monkeypatch.setattr("data.binance_perpetual_backfill._retry_fetch_ohlcv", stable_retry)
    second_result = run_separate_binance_perpetual_backfill(options)
    assert second_result["status"] == "success"

    resumed_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert resumed_checkpoint["status"] == "completed"
    assert resumed_checkpoint["timeframes"]["1m"]["pages_completed"] == 2
    assert len(resumed_checkpoint["timeframes"]["1m"]["pages"]) == 2


def test_run_binance_backfill_uses_explicit_cutoff_and_initializes_progress(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config_path, features_config_path, _ = _seed_project(tmp_path)
    options = _options(
        project_root,
        data_config_path,
        features_config_path,
        refresh_session_id="refresh_cutoff",
        max_candles_per_timeframe=0,
        target_end_utc="2024-01-01T00:30:00+00:00",
    )

    class _FakeClient:
        id = "binanceusdm"
        urls = {"api": {"fapiPublic": "https://fapi.binance.com/fapi/v1"}}
        options = {"defaultType": "swap", "defaultSubType": "linear", "fetchMarkets": {"types": ["linear"]}}

        def load_markets(self) -> dict[str, dict[str, str]]:
            return {"BTC/USDT:USDT": _fake_binance_usdm_market()}

        def close(self) -> None:
            return None

    class _FakeProgress:
        def __init__(self, *, total: int, initial: int, desc: str, unit: str, dynamic_ncols: bool) -> None:
            self.total = total
            self.initial = initial
            self.desc = desc
            self.unit = unit
            self.dynamic_ncols = dynamic_ncols
            self.updates: list[int] = []
            self.postfixes: list[dict[str, str]] = []

        def __enter__(self) -> _FakeProgress:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

        def update(self, value: int) -> None:
            self.updates.append(value)

        def set_postfix(self, payload: dict[str, str]) -> None:
            self.postfixes.append(dict(payload))

    progress_instances: list[_FakeProgress] = []

    def fake_tqdm(*, total: int, initial: int, desc: str, unit: str, dynamic_ncols: bool) -> _FakeProgress:
        progress = _FakeProgress(total=total, initial=initial, desc=desc, unit=unit, dynamic_ncols=dynamic_ncols)
        progress_instances.append(progress)
        return progress

    target_end = pd.Timestamp(options.target_end_utc)

    def fake_retry(
        client: object,
        *,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> list[list[object]]:
        del client, symbol, max_retries, retry_backoff_seconds
        start = pd.Timestamp(since_ms, unit="ms", tz="UTC")
        rows: list[list[object]] = []
        current = start
        seed = 1
        while len(rows) < limit and current <= target_end:
            rows.append(_fake_row(current, seed))
            current = current + pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            seed += 1
        return rows

    monkeypatch.setattr("data.binance_perpetual_backfill._utc_now", lambda: datetime(2024, 1, 1, 0, 46, tzinfo=timezone.utc))
    monkeypatch.setattr("data.binance_perpetual_backfill._build_binance_usdm_client", lambda: _FakeClient())
    monkeypatch.setattr("data.binance_perpetual_backfill._retry_fetch_ohlcv", fake_retry)
    monkeypatch.setattr("data.binance_perpetual_backfill.tqdm", fake_tqdm)
    monkeypatch.setattr(
        "data.binance_perpetual_backfill._run_processing_chain",
        lambda options, raw_input_root, **kwargs: {
            "status": "success",
            "report": {
                "status": "success",
                "refresh_session_id": options.refresh_session_id,
                "exchange_used": "binance",
                "market_type": "perpetual",
                "source_lineage": "separate_parallel_lineage",
                "provider_track": "historical_backfill",
                "provider_choice_reason": "historical",
                "legacy_lineage_touched": False,
                "raw_input_root": str(raw_input_root.resolve()),
                "feature_contract_compatibility": {"status": "success", "checks": {}},
                "stages": [],
                "errors": [],
            },
        },
    )

    result = run_separate_binance_perpetual_backfill(options)

    assert result["status"] == "success"
    assert [item.desc for item in progress_instances] == ["Binance 1m", "Binance 5m", "Binance 15m"]
    assert all(item.initial == 0 for item in progress_instances)

    checkpoint_path = project_root / "runs" / "refresh_cutoff" / "data_tail_refresh" / "reports" / "historical_backfill_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["target_end_utc"] == "2024-01-01T00:30:00+00:00"
    assert checkpoint["timeframes"]["1m"]["target_end_utc"] == "2024-01-01T00:30:00+00:00"
    assert checkpoint["timeframes"]["5m"]["target_end_utc"] == "2024-01-01T00:30:00+00:00"
    assert checkpoint["timeframes"]["15m"]["target_end_utc"] == "2024-01-01T00:30:00+00:00"
    assert [item.total for item in progress_instances] == [
        checkpoint["timeframes"]["1m"]["expected_rows"],
        checkpoint["timeframes"]["5m"]["expected_rows"],
        checkpoint["timeframes"]["15m"]["expected_rows"],
    ]


def test_run_binance_backfill_fails_closed_on_spot_bootstrap(monkeypatch: object, tmp_path: Path) -> None:
    project_root, data_config_path, features_config_path, _ = _seed_project(tmp_path)
    options = _options(project_root, data_config_path, features_config_path, refresh_session_id="refresh_spot_fail")

    class _SpotClient:
        id = "binance"
        urls = {"api": {"fapiPublic": "https://api.binance.com/api/v3"}}
        options = {"defaultType": "spot", "fetchMarkets": {"types": ["spot"]}}

        def load_markets(self) -> dict[str, dict[str, object]]:
            return {
                "BTC/USDT": {
                    "id": "BTCUSDT",
                    "symbol": "BTC/USDT",
                    "type": "spot",
                    "contract": False,
                    "linear": False,
                    "swap": False,
                    "future": False,
                    "spot": True,
                    "base": "BTC",
                    "quote": "USDT",
                    "settle": None,
                }
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr("data.binance_perpetual_backfill._build_binance_usdm_client", lambda: _SpotClient())
    monkeypatch.setattr(
        "data.binance_perpetual_backfill._run_binance_network_probe_suite",
        lambda: {
            "prod": {
                "url": "https://fapi.binance.com/fapi/v1/exchangeInfo",
                "result": "failed",
                "http_status": 403,
                "error_class": "waf_or_forbidden",
                "error_message": "HTTP 403",
            },
            "testnet": {
                "url": "https://demo-fapi.binance.com/fapi/v1/exchangeInfo",
                "result": "success",
                "http_status": 200,
                "error_class": None,
                "error_message": None,
            },
        },
    )

    result = run_separate_binance_perpetual_backfill(options)

    assert result["status"] == "failed"
    raw_root = (
        project_root
        / "runs"
        / "refresh_spot_fail"
        / "data_tail_refresh"
        / "separate_parallel_lineage"
        / "raw"
        / "historical_backfill"
    )
    assert not raw_root.exists()

    checkpoint_path = project_root / "runs" / "refresh_spot_fail" / "data_tail_refresh" / "reports" / "historical_backfill_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "failed"
    assert checkpoint["bootstrap_status"] == "failed"
    assert checkpoint["bootstrap_proof"]["status"] == "failed"
    assert "usd-m futures" in checkpoint["bootstrap_proof"]["error"].lower()
    assert checkpoint["bootstrap_failure_class"] == "waf_or_forbidden"
    assert "waf" in checkpoint["recommended_next_action"].lower() or "restricted" in checkpoint["recommended_next_action"].lower()
    assert checkpoint["network_probe"]["prod"]["error_class"] == "waf_or_forbidden"
    assert checkpoint["network_probe"]["testnet"]["result"] == "success"
