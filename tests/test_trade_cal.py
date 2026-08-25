"""Tests for backfill 的交易日历展开 —— trade_cal + 区间 → 应拉日期列表(纯函数)+ 缓存/回退。

节假日不再靠 EmptyMarketDayError 试错:先拉一次 trade_cal,只对 is_open==1 的日期拉数。
日历故障时 fail-loud(打 warning)回退全自然日试错,不让日历卡住行情拉取。
"""
import pandas as pd
import pytest

from ashare_gauntlet.data.fetch import TokenExpiredError
from scripts.backfill import TradeCalendarUnavailableError, days_to_pull, fetch_trade_cal


def _cal(rows: list[tuple[str, int]]) -> pd.DataFrame:
    """(cal_date, is_open) 行 → trade_cal 形状的 df。"""
    return pd.DataFrame(rows, columns=["cal_date", "is_open"])


# ---- days_to_pull:纯函数,trade_cal df + 区间 → 应拉日期列表 ----

def test_days_to_pull_keeps_only_open_days():
    # 20260628 是周日(is_open=0):不应进拉取队列
    cal = _cal([
        ("20260628", 0),
        ("20260629", 1),
        ("20260630", 1),
        ("20260701", 1),
        ("20260702", 1),
    ])
    assert days_to_pull(cal, "20260628", "20260702") == [
        "20260629", "20260630", "20260701", "20260702",
    ]


def test_days_to_pull_sorted_ascending_even_if_cal_unsorted():
    cal = _cal([("20260630", 1), ("20260629", 1)])
    assert days_to_pull(cal, "20260629", "20260630") == ["20260629", "20260630"]


def test_days_to_pull_clips_to_range():
    # 防御:日历若带了区间外的日期(缓存复用/接口宽松),不应拉区间外的
    cal = _cal([("20260626", 1), ("20260629", 1), ("20260703", 1)])
    assert days_to_pull(cal, "20260628", "20260702") == ["20260629"]


def test_days_to_pull_all_closed_returns_empty():
    cal = _cal([("20261001", 0), ("20261002", 0)])
    assert days_to_pull(cal, "20261001", "20261002") == []


def test_days_to_pull_none_falls_back_to_all_calendar_days():
    # 日历拉不到(None)→ 回退旧行为:区间内全部自然日逐日试(含周末)
    assert days_to_pull(None, "20260628", "20260702") == [
        "20260628", "20260629", "20260630", "20260701", "20260702",
    ]


# ---- fetch_trade_cal:按 (start,end) 缓存 + 故障回退 ----

class _FakePro:
    """只实现 trade_cal 的假 pro:可注入返回值或异常,并记调用次数。"""

    def __init__(self, df: pd.DataFrame | None = None, error: Exception | None = None):
        self.calls = 0
        self._df = df
        self._error = error

    def trade_cal(self, exchange: str = "SSE", start_date: str = "", end_date: str = "") -> pd.DataFrame:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._df is not None
        return self._df


def test_fetch_trade_cal_caches_by_range(tmp_path):
    pro = _FakePro(df=_cal([("20260629", 1), ("20260628", 0)]))
    first = fetch_trade_cal(pro, "20260628", "20260629", str(tmp_path))
    second = fetch_trade_cal(pro, "20260628", "20260629", str(tmp_path))
    assert pro.calls == 1  # 第二次命中 data/cache/trade_cal/<start>_<end>.parquet
    assert (tmp_path / "trade_cal" / "20260628_20260629.parquet").exists()
    assert first is not None and second is not None
    assert list(second["cal_date"]) == list(first["cal_date"])


def test_fetch_trade_cal_failure_warns_and_returns_none(tmp_path, capsys):
    # ValueError 非瞬态(call_with_retry 不重试直接抛)→ warning + None,不阻塞行情拉取
    pro = _FakePro(error=ValueError("calendar service down"))
    assert fetch_trade_cal(pro, "20260628", "20260629", str(tmp_path)) is None
    assert "warning" in capsys.readouterr().out.lower()


def test_fetch_trade_cal_empty_result_not_cached_returns_none(tmp_path, capsys):
    # 任何非空区间的日历必有行(开市/休市日都在表里),0 行只可能是拉挂了:不落盘、warning、回退
    pro = _FakePro(df=pd.DataFrame(columns=["cal_date", "is_open"]))
    assert fetch_trade_cal(pro, "20260628", "20260629", str(tmp_path)) is None
    assert not (tmp_path / "trade_cal" / "20260628_20260629.parquet").exists()
    assert "warning" in capsys.readouterr().out.lower()


def test_fetch_trade_cal_token_expired_propagates(tmp_path):
    # 额度耗尽是全局致命错:回退试错只会烧掉更多失败调用,必须向上抛而非静默回退
    pro = _FakePro(error=Exception("您的token已过期"))
    with pytest.raises(TokenExpiredError):
        fetch_trade_cal(pro, "20260628", "20260629", str(tmp_path))


def test_strict_invalid_fresh_calendar_is_not_cached_and_can_recover(tmp_path):
    days = pd.date_range("2026-01-01", "2026-01-31")
    complete = _cal([
        (day.strftime("%Y%m%d"), int(day.weekday() < 5)) for day in days
    ])

    class RecoveringPro:
        def __init__(self):
            self.calls = 0

        def trade_cal(self, **kwargs):
            self.calls += 1
            return complete.iloc[:-1] if self.calls == 1 else complete

    pro = RecoveringPro()
    target = tmp_path / "trade_cal/20260101_20260131.parquet"

    with pytest.raises(TradeCalendarUnavailableError, match="未覆盖区间内全部自然日"):
        fetch_trade_cal(pro, "20260101", "20260131", tmp_path, strict=True)

    assert not target.exists()
    recovered = fetch_trade_cal(pro, "20260101", "20260131", tmp_path, strict=True)
    assert recovered is not None
    assert pro.calls == 2
    assert target.is_file()


def test_strict_invalid_cached_calendar_recovers_without_losing_failed_evidence(tmp_path):
    days = pd.date_range("2026-01-01", "2026-01-31")
    complete = _cal([
        (day.strftime("%Y%m%d"), int(day.weekday() < 5)) for day in days
    ])
    target = tmp_path / "trade_cal/20260101_20260131.parquet"
    target.parent.mkdir(parents=True)
    complete.iloc[:-1].to_parquet(target, index=False)
    invalid_bytes = target.read_bytes()

    class RecoveringPro:
        def __init__(self):
            self.calls = 0

        def trade_cal(self, **kwargs):
            self.calls += 1
            return complete.iloc[:-1] if self.calls == 1 else complete

    pro = RecoveringPro()

    with pytest.raises(TradeCalendarUnavailableError, match="未覆盖区间内全部自然日"):
        fetch_trade_cal(pro, "20260101", "20260131", tmp_path, strict=True)

    assert pro.calls == 1
    assert target.read_bytes() == invalid_bytes
    recovered = fetch_trade_cal(pro, "20260101", "20260131", tmp_path, strict=True)
    assert recovered is not None
    assert pro.calls == 2
    assert len(pd.read_parquet(target)) == 31
