"""Tests for the fetch layer's pure parsing (network calls are smoke-tested live)."""

import pandas as pd
import pytest
import requests

from ashare_gauntlet.data.fetch import (
    EmptyCoreTableError,
    TokenExpiredError,
    call_with_retry,
    fetch_symbol_history,
    fetch_symbol_table,
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


class _TablePro:
    """Returns whatever DataFrame is registered per endpoint; records calls."""

    def __init__(self, returns: dict[str, pd.DataFrame]) -> None:
        self._returns = returns
        self.calls: list[tuple[str, str]] = []

    def __getattr__(self, endpoint: str):
        def _method(ts_code: str) -> pd.DataFrame:
            self.calls.append((endpoint, ts_code))
            return self._returns[endpoint]

        return _method


# --- 契约C1 (fetch/cache 侧): 核心财报表空 -> 响亮抛错, 事件表空 -> 允许 ---


@pytest.mark.parametrize("endpoint", ["income", "fina_indicator", "balancesheet", "cashflow"])
def test_fetch_symbol_table_raises_when_core_table_is_empty(tmp_path, endpoint):
    # 一张核心财报表取到 0 行不是 "公司没利润", 而是没取到 -> 必须响亮抛错,
    # 且异常里带上 ts_code + endpoint 方便定位, 不能把空当真值返回。
    pro = _TablePro({endpoint: pd.DataFrame()})

    with pytest.raises(EmptyCoreTableError) as excinfo:
        fetch_symbol_table(pro, endpoint, "600519.SH", tmp_path)

    message = str(excinfo.value)
    assert "600519.SH" in message
    assert endpoint in message


def test_fetch_symbol_table_does_not_cache_empty_core_table(tmp_path):
    # 空的核心表绝不能落盘 — 否则下次 read_or_fetch 命中缓存把空当真值返回,
    # 错误被永久固化。落盘前就该抛。
    pro = _TablePro({"income": pd.DataFrame()})

    with pytest.raises(EmptyCoreTableError):
        fetch_symbol_table(pro, "income", "600519.SH", tmp_path)

    assert not (tmp_path / "income" / "600519.SH.parquet").exists()


def test_fetch_symbol_table_raises_when_reading_back_empty_core_cache(tmp_path):
    # 即便磁盘上已存在一份历史遗留的空核心表缓存, 读回来也要抛 —
    # 断言加在知道 endpoint 类别的这层, 读写两条路径都护住。
    path = tmp_path / "cashflow" / "600519.SH.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_parquet(path, index=False)
    pro = _TablePro({"cashflow": pd.DataFrame({"x": [1]})})  # would-be re-pull, must not be reached

    with pytest.raises(EmptyCoreTableError):
        fetch_symbol_table(pro, "cashflow", "600519.SH", tmp_path)
    assert pro.calls == []  # cache hit; no re-pull


def test_fetch_symbol_table_returns_non_empty_core_table(tmp_path):
    df = pd.DataFrame({"ts_code": ["600519.SH"], "end_date": ["20251231"], "n_income": [1.0]})
    pro = _TablePro({"income": df})

    out = fetch_symbol_table(pro, "income", "600519.SH", tmp_path)

    pd.testing.assert_frame_equal(out, df)
    assert (tmp_path / "income" / "600519.SH.parquet").exists()


@pytest.mark.parametrize(
    "endpoint", ["share_float", "pledge_stat", "stk_holdertrade", "forecast", "express"]
)
def test_fetch_symbol_table_allows_empty_event_table(tmp_path, endpoint):
    # 事件表空是合法的 "确认无事件" 信号 (CORE agent 在 record 层区分未知/确认无),
    # fetch 侧照常缓存返回, 不抛。
    pro = _TablePro({endpoint: pd.DataFrame()})

    out = fetch_symbol_table(pro, endpoint, "600519.SH", tmp_path)

    assert out.empty
    assert (tmp_path / endpoint / "600519.SH.parquet").exists()


# --- #11: 收紧 _RATE_MARKERS, 含 "频繁/频率" 字样的真实参数错误别误判为限频重试 ---


def test_call_with_retry_does_not_retry_param_error_mentioning_frequency():
    # 真实 API 参数错误的措辞里偶尔带 "频繁"/"频率" 字样 (非限频), 子串匹配会
    # 误判成瞬时限频, 白白重试 attempts 次再抛。收紧后应一次就抛、不重试。
    calls: list[int] = []

    def bad_param() -> str:
        calls.append(1)
        raise Exception("参数错误:start_date 不能比 end_date 更频繁")

    with pytest.raises(Exception, match="参数错误"):
        call_with_retry(bad_param, attempts=5, base_delay=0.0, sleep=lambda _: None)
    assert calls == [1]
