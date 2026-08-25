import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from scripts import refresh
from scripts.backfill import TradeCalendarUnavailableError
from scripts.c2_review import _is_month_end


def _january_calendar() -> pd.DataFrame:
    days = pd.date_range("2026-01-01", "2026-01-31")
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
        return self.calendar


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
        (endpoint, day) for day in expected_days for endpoint in refresh.ENDPOINTS
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
