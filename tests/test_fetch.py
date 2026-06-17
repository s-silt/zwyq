"""Tests for the fetch layer's pure parsing (network calls are smoke-tested live)."""

import pandas as pd
import pytest
import requests

from ashare_gauntlet.data.fetch import (
    TokenExpiredError,
    call_with_retry,
    fetch_symbol_history,
    trading_days_from_cal,
)


class _FakePro:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append((ts_code, start_date, end_date))
        return pd.DataFrame({"ts_code": [ts_code], "trade_date": [start_date]})


def test_fetch_symbol_history_cache_key_encodes_the_date_window(tmp_path):
    pro = _FakePro()

    fetch_symbol_history(pro, "daily", "A", "20230101", "20231231", tmp_path)
    fetch_symbol_history(pro, "daily", "A", "20230101", "20231231", tmp_path)  # same window -> cached
    fetch_symbol_history(pro, "daily", "A", "20240101", "20241231", tmp_path)  # new window -> fresh

    # A different window must NOT collide with the cached one (else changing the
    # backfill window silently serves stale data).
    assert pro.calls == [
        ("A", "20230101", "20231231"),
        ("A", "20240101", "20241231"),
    ]


def test_trading_days_from_cal_keeps_open_days_sorted():
    cal = pd.DataFrame(
        {
            "exchange": ["SSE"] * 4,
            "cal_date": ["20240105", "20240101", "20240104", "20240103"],
            "is_open": [1, 0, 1, 1],
        }
    )

    days = trading_days_from_cal(cal)

    # Closed days dropped (20240101), result sorted ascending.
    assert days == ["20240103", "20240104", "20240105"]


def test_call_with_retry_succeeds_after_transient_network_failures():
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ChunkedEncodingError("connection broken mid-stream")
        return "ok"

    out = call_with_retry(flaky, attempts=4, base_delay=0.0, sleep=lambda _s: None)

    assert out == "ok"
    assert len(calls) == 3


def test_call_with_retry_reraises_after_exhausting_attempts():
    def always_fail() -> str:
        raise requests.exceptions.ReadTimeout("never responds")

    with pytest.raises(requests.exceptions.ReadTimeout):
        call_with_retry(always_fail, attempts=2, base_delay=0.0, sleep=lambda _: None)


def test_call_with_retry_aborts_immediately_on_expired_token():
    # An expired/exhausted token is fatal — retrying just burns more credits, so
    # it must raise TokenExpiredError on the FIRST failure, no retries.
    calls: list[int] = []

    def expired() -> str:
        calls.append(1)
        raise Exception("token已过期")

    with pytest.raises(TokenExpiredError):
        call_with_retry(expired, attempts=5, base_delay=0.0, sleep=lambda _: None)
    assert calls == [1]


def test_call_with_retry_retries_on_rate_limit_message():
    # A rate-limit (frequency) API error IS transient — back off and retry.
    calls: list[int] = []

    def throttled() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise Exception("抱歉，您每分钟最多访问该接口500次")
        return "ok"

    out = call_with_retry(throttled, attempts=5, base_delay=0.0, sleep=lambda _: None)

    assert out == "ok"
    assert len(calls) == 3


def test_call_with_retry_does_not_retry_unknown_api_errors():
    # A non-transient, non-fatal API error should propagate, not be retried.
    calls: list[int] = []

    def bad_param() -> str:
        calls.append(1)
        raise Exception("ts_code 参数错误")

    with pytest.raises(Exception, match="参数错误"):
        call_with_retry(bad_param, attempts=5, base_delay=0.0, sleep=lambda _: None)
    assert calls == [1]
