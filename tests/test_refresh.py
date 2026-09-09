import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from ashare_gauntlet.data.fetch import MARKET_ENDPOINTS, TokenExpiredError
from scripts import refresh
from scripts.backfill import TradeCalendarUnavailableError
from scripts.c2_review import _is_month_end


def _january_calendar() -> pd.DataFrame:
    days = pd.date_range("2026-01-01", "2026-01-31")
    return pd.DataFrame({
        "cal_date": [day.strftime("%Y%m%d") for day in days],
        "is_open": [int(day.weekday() < 5) for day in days],
    })


def _cross_month_calendar() -> pd.DataFrame:
    days = pd.date_range("2025-12-01", "2026-01-31")
    return pd.DataFrame({
        "cal_date": [day.strftime("%Y%m%d") for day in days],
        "is_open": [int(day.weekday() < 5) for day in days],
    })


class _FakePro:
    def __init__(self, calendar: pd.DataFrame) -> None:
        self.calendar = calendar
        self.calendar_calls: list[dict[str, str]] = []

    def trade_cal(self, **kwargs) -> pd.DataFrame:
        self.calendar_calls.append(kwargs)
        dates = self.calendar["cal_date"].astype(str)
        return self.calendar.loc[
            (dates >= kwargs["start_date"]) & (dates <= kwargs["end_date"])
        ]


@pytest.mark.parametrize(("today", "expected_month_end"), [
    (dt.date(2026, 1, 29), False),
    (dt.date(2026, 1, 30), True),
])
def test_refresh_persists_full_month_calendar_without_fetching_future_market_days(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    today: dt.date,
    expected_month_end: bool,
) -> None:
    pro = _FakePro(_january_calendar())
    market_calls: list[tuple[str, str]] = []
    cache = tmp_path / "data/cache"
    monkeypatch.setattr(refresh, "tushare_pro", lambda: pro)
    monkeypatch.setattr(
        refresh,
        "fetch_market_day",
        lambda pro, endpoint, day, cache_dir: market_calls.append((endpoint, day)),
    )

    refresh.main(10, str(cache), today=today)

    shard = cache / "trade_cal/20260101_20260131.parquet"
    assert shard.is_file()
    assert pro.calendar_calls == [{
        "exchange": "SSE",
        "start_date": "20260101",
        "end_date": "20260131",
    }]
    assert _is_month_end(tmp_path, today.strftime("%Y%m%d")) is expected_month_end
    lookback_start = (today - dt.timedelta(days=10)).strftime("%Y%m%d")
    expected_days = [
        day.strftime("%Y%m%d")
        for day in pd.date_range(lookback_start, today.strftime("%Y%m%d"))
        if day.weekday() < 5
    ]
    assert market_calls == [
        (endpoint, day) for endpoint in refresh.ENDPOINTS for day in expected_days
    ]


def test_early_month_refresh_preserves_cross_month_market_lookback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = dt.date(2026, 1, 3)
    pro = _FakePro(_cross_month_calendar())
    market_calls: list[tuple[str, str]] = []
    cache = tmp_path / "data/cache"
    monkeypatch.setattr(refresh, "tushare_pro", lambda: pro)
    monkeypatch.setattr(
        refresh,
        "fetch_market_day",
        lambda pro, endpoint, day, cache_dir: market_calls.append((endpoint, day)),
    )

    refresh.main(10, str(cache), today=today)

    assert (cache / "trade_cal/20251201_20251231.parquet").is_file()
    assert (cache / "trade_cal/20260101_20260131.parquet").is_file()
    assert pro.calendar_calls == [
        {"exchange": "SSE", "start_date": "20251201", "end_date": "20251231"},
        {"exchange": "SSE", "start_date": "20260101", "end_date": "20260131"},
    ]
    assert _is_month_end(tmp_path, "20260102") is False
    expected_days = [
        day.strftime("%Y%m%d")
        for day in pd.date_range("2025-12-24", "2026-01-03")
        if day.weekday() < 5
    ]
    assert market_calls == [
        (endpoint, day) for endpoint in refresh.ENDPOINTS for day in expected_days
    ]

    market_calls.clear()
    refresh.main(10, str(cache), today=dt.date(2026, 1, 4))

    assert len(pro.calendar_calls) == 2
    expected_days = [
        day.strftime("%Y%m%d")
        for day in pd.date_range("2025-12-25", "2026-01-04")
        if day.weekday() < 5
    ]
    assert market_calls == [
        (endpoint, day) for endpoint in refresh.ENDPOINTS for day in expected_days
    ]


def test_refresh_fails_before_market_fetch_when_full_month_calendar_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = _january_calendar().iloc[:-1]
    pro = _FakePro(incomplete)
    market_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(refresh, "tushare_pro", lambda: pro)
    monkeypatch.setattr(
        refresh,
        "fetch_market_day",
        lambda pro, endpoint, day, cache_dir: market_calls.append((endpoint, day)),
    )

    with pytest.raises(TradeCalendarUnavailableError, match="trade_cal"):
        refresh.main(10, str(tmp_path / "data/cache"), today=dt.date(2026, 1, 29))

    assert market_calls == []


def test_refresh_token_expiry_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pro = _FakePro(_january_calendar())
    monkeypatch.setattr(refresh, "tushare_pro", lambda: pro)
    monkeypatch.setattr(
        refresh,
        "fetch_market_day",
        lambda *args, **kwargs: (_ for _ in ()).throw(TokenExpiredError("expired")),
    )

    with pytest.raises(SystemExit) as exc:
        refresh.main(
            1,
            str(tmp_path / "data/cache"),
            today=dt.date(2026, 1, 5),
        )

    assert exc.value.code == 1
    assert "token 耗尽" in capsys.readouterr().out


def _patch_refresh_fetch(
    monkeypatch: pytest.MonkeyPatch,
    calendar: pd.DataFrame,
    fetch_impl,
) -> list[tuple[str, str]]:
    pro = _FakePro(calendar)
    market_calls: list[tuple[str, str]] = []

    def _fetch(_pro, endpoint, day, cache_dir):
        market_calls.append((endpoint, day))
        return fetch_impl(endpoint, day)

    monkeypatch.setattr(refresh, "tushare_pro", lambda: pro)
    monkeypatch.setattr(refresh, "fetch_market_day", _fetch)
    return market_calls


def test_refresh_core_endpoint_failure_is_fail_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fetch_impl(endpoint: str, day: str) -> None:
        if endpoint == "daily_basic":
            raise RuntimeError("simulated core failure")

    market_calls = _patch_refresh_fetch(monkeypatch, _january_calendar(), fetch_impl)

    with pytest.raises(SystemExit) as exc:
        refresh.main(1, str(tmp_path / "data/cache"), today=dt.date(2026, 1, 5))

    assert exc.value.code == 1
    called_endpoints = [endpoint for endpoint, _day in market_calls]
    assert "daily_basic" in called_endpoints
    assert "stk_limit" in called_endpoints
    assert "hk_hold" in called_endpoints
    out = capsys.readouterr().out
    assert "核心失败" in out
    assert "daily_basic" in out
    assert "refresh done" not in out


def test_refresh_non_core_failure_does_not_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fetch_impl(endpoint: str, day: str) -> None:
        if endpoint == "hk_hold":
            raise RuntimeError("simulated hk_hold zero-column")

    market_calls = _patch_refresh_fetch(monkeypatch, _january_calendar(), fetch_impl)

    refresh.main(1, str(tmp_path / "data/cache"), today=dt.date(2026, 1, 5))

    called_endpoints = [endpoint for endpoint, _day in market_calls]
    for endpoint in MARKET_ENDPOINTS:
        assert endpoint in called_endpoints
    assert "hk_hold" in called_endpoints
    assert "moneyflow_hsgt" in called_endpoints
    out = capsys.readouterr().out
    assert "非核心失败" in out
    assert "hk_hold" in out
    assert "refresh done" not in out
    assert "核心完成" in out


def test_refresh_core_endpoints_run_before_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_calls = _patch_refresh_fetch(
        monkeypatch, _january_calendar(), lambda endpoint, day: None,
    )

    refresh.main(10, str(tmp_path / "data/cache"), today=dt.date(2026, 1, 16))

    core = set(MARKET_ENDPOINTS)
    optional = [endpoint for endpoint in refresh.ENDPOINTS if endpoint not in core]
    last_core = max(i for i, (endpoint, _day) in enumerate(market_calls) if endpoint in core)
    first_optional = min(
        i for i, (endpoint, _day) in enumerate(market_calls) if endpoint in set(optional)
    )
    assert last_core < first_optional
    unique_order = list(dict.fromkeys(endpoint for endpoint, _day in market_calls))
    assert unique_order[:len(MARKET_ENDPOINTS)] == list(MARKET_ENDPOINTS)
    assert refresh.ENDPOINTS[:len(MARKET_ENDPOINTS)] == MARKET_ENDPOINTS


def test_refresh_all_endpoints_fail_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fetch_impl(endpoint: str, day: str) -> None:
        raise RuntimeError(f"simulated {endpoint} failure")

    market_calls = _patch_refresh_fetch(monkeypatch, _january_calendar(), fetch_impl)

    with pytest.raises(SystemExit) as exc:
        refresh.main(1, str(tmp_path / "data/cache"), today=dt.date(2026, 1, 5))

    assert exc.value.code == 1
    assert {endpoint for endpoint, _day in market_calls} == set(refresh.ENDPOINTS)
    out = capsys.readouterr().out
    assert "全部端点失败" in out
    assert "refresh done" not in out
