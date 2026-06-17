"""Tests for the fetch layer's pure parsing (network calls are smoke-tested live)."""

import pandas as pd
import pytest
import requests

from ashare_gauntlet.data.fetch import call_with_retry, trading_days_from_cal


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
        call_with_retry(always_fail, attempts=2, base_delay=0.0, sleep=lambda _s: None)
