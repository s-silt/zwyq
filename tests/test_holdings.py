"""Tests for holdings —— 持仓盯盘估值(只算不判:数字给盯盘 Claude,信号判断留给它)。

全部纯函数;口径钉死(close/盈亏/距stop=不复权,ma20/距20低=前复权);
单只坏数据降级标注、缓存整体陈旧 fail-loud(不拿旧价当今日)。
"""
import pytest

from ashare_gauntlet.holdings import (
    StaleCacheError,
    build_position_record,
    downside_to_stop,
    held_trading_days,
    is_date_partition,
    min_low,
    moving_average,
    pct_change,
    qfq_series,
    verify_as_of,
)


# ---------- is_date_partition:只认日期分区文件,滤掉 per-symbol 缓存污染 ----------

def test_is_date_partition_accepts_date_file():
    assert is_date_partition("20260703.parquet") is True


def test_is_date_partition_rejects_symbol_range_file():
    # data/cache/daily/ 里混有 <ts_code>_<start>_<end>.parquet,字典序排在日期文件后,
    # 不滤掉会让 sorted(glob)[-1] 取到它、把 as_of 污染成代码串(实测踩过)
    assert is_date_partition("603979.SH_20260501_20260703.parquet") is False


def test_is_date_partition_rejects_non_eight_digit():
    assert is_date_partition("2026070.parquet") is False   # 7 位
    assert is_date_partition("foo.parquet") is False
    assert is_date_partition("20260703.csv") is False       # 非 parquet


# ---------- pct_change:通用相对距离(盈亏/距MA20/距20低共用) ----------

def test_pct_change_basic():
    assert abs(pct_change(150.0, 100.0) - 50.0) < 1e-9
    assert abs(pct_change(94.0, 100.0) - (-6.0)) < 1e-9


def test_pct_change_ref_nonpositive_fails_loud():
    # 参照(cost/ma/low)不可能 ≤0;≤0 只能是坏数据,fail-loud 不返回诡异值
    with pytest.raises(ValueError):
        pct_change(10.0, 0.0)
    with pytest.raises(ValueError):
        pct_change(10.0, -5.0)


# ---------- downside_to_stop:从现价再跌多少%到止损(分母=现价) ----------

def test_downside_to_stop_above_stop():
    # 收57.59 stop55.0 → (57.59-55)/57.59 = +4.50%(现价上方,还有4.5%缓冲)
    assert abs(downside_to_stop(57.59, 55.0) - 4.497308) < 1e-4


def test_downside_to_stop_below_stop_is_negative():
    assert downside_to_stop(54.0, 55.0) < 0


def test_downside_to_stop_close_nonpositive_fails_loud():
    with pytest.raises(ValueError):
        downside_to_stop(0.0, 55.0)


# ---------- qfq_series:前复权归一(逐点 raw*adj/latest_adj) ----------

def test_qfq_series_normalizes_to_latest_factor():
    # latest_adj=4:某日 adj=2 的价 ×(2/4)=半价;当日 adj=4 的价不变
    out = qfq_series([10.0, 20.0], [2.0, 4.0], latest_adj=4.0)
    assert abs(out[0] - 5.0) < 1e-9
    assert abs(out[1] - 20.0) < 1e-9


def test_qfq_series_length_mismatch_fails_loud():
    with pytest.raises(ValueError):
        qfq_series([10.0, 20.0], [2.0], latest_adj=4.0)


def test_qfq_series_latest_adj_nonpositive_fails_loud():
    with pytest.raises(ValueError):
        qfq_series([10.0], [2.0], latest_adj=0.0)


# ---------- moving_average / min_low:不足 window → None(单字段降级,不崩) ----------

def test_moving_average_last_window():
    ma = moving_average([float(i) for i in range(1, 21)], 20)
    assert ma is not None and abs(ma - 10.5) < 1e-9


def test_moving_average_uses_only_last_window():
    # 21 个值,只取最后 20 个(1e9 是更久远历史,不进均线)
    vals = [1e9] + [float(i) for i in range(1, 21)]
    ma = moving_average(vals, 20)
    assert ma is not None and abs(ma - 10.5) < 1e-9


def test_moving_average_insufficient_returns_none():
    # 次新股历史不足 20 日 → None(降级),而非缩窗伪造均线
    assert moving_average([1.0, 2.0, 3.0], 20) is None


def test_min_low_last_window():
    vals = [float(i) for i in range(5, 25)]  # 5..24
    low = min_low(vals, 20)
    assert low is not None and abs(low - 5.0) < 1e-9


def test_min_low_insufficient_returns_none():
    assert min_low([10.0, 11.0], 20) is None


# ---------- held_trading_days:entry→as_of 的交易日数(短线10日时间止损用) ----------

def test_held_trading_days_counts_inclusive():
    # entry 当天算第1日;[0701,0702,0703] 三个交易日 → 3
    days = ["20260630", "20260701", "20260702", "20260703"]
    assert held_trading_days("20260701", "20260703", days) == 3


def test_held_trading_days_same_day_is_one():
    days = ["20260702", "20260703"]
    assert held_trading_days("20260703", "20260703", days) == 1


# ---------- verify_as_of:缓存最新日≠今日 → 系统性 fail-loud(拒绝拿旧价) ----------

def test_verify_as_of_match_ok():
    verify_as_of("20260703", "20260703")  # 不抛


def test_verify_as_of_stale_fails_loud():
    # backfill 没跑成功 → 缓存停在昨天,所有数字都是旧的,硬失败
    with pytest.raises(StaleCacheError, match="20260703"):
        verify_as_of("20260702", "20260703")


# ---------- build_position_record:组装一只(降级标注,round(2) 展示友好) ----------

def _pos(**kw):
    base = {"ts_code": "603979.SH", "name": "金诚信", "theme": "铜", "bucket": "长线",
            "bucket_note": "宽止损", "shares": 300, "cost": 61.237, "stop": 55.0}
    base.update(kw)
    return base


def test_build_record_full_fields():
    qfq_c = [float(i) for i in range(40, 60)]  # 20 日:40..59,均值 49.5
    qfq_l = [float(i) for i in range(38, 58)]  # 低点 38..57,20低=38
    rec = build_position_record(
        _pos(), close=57.59, pct_chg=-0.29, qfq_closes=qfq_c, qfq_lows=qfq_l,
        as_of="20260703", trade_days=["20260703"], window=20)
    assert rec["ts_code"] == "603979.SH"
    assert rec["error"] is None
    assert rec["close"] == 57.59
    assert abs(rec["pnl_pct"] - round((57.59 - 61.237) / 61.237 * 100, 2)) < 1e-9
    assert abs(rec["dist_stop_pct"] - 4.5) < 0.05
    assert rec["ma20"] == 49.5
    assert rec["held_days"] is None          # 无 entry_date
    assert rec["bucket"] == "长线" and rec["bucket_note"] == "宽止损"


def test_build_record_no_quote_degrades():
    # 停牌/无当日行情 → 该只 error 标注,不崩(其余持仓照算)
    rec = build_position_record(
        _pos(), close=None, pct_chg=None, qfq_closes=[], qfq_lows=[],
        as_of="20260703", trade_days=["20260703"], window=20)
    assert rec["error"] is not None
    assert rec["close"] is None
    assert rec["ts_code"] == "603979.SH"     # 身份字段仍在,方便报告列出


def test_build_record_short_history_nulls_ma():
    # 次新股不足 20 日 → ma20/dist_ma20/dist_low20 为 null,其余字段照算
    rec = build_position_record(
        _pos(), close=57.59, pct_chg=-0.29, qfq_closes=[57.0, 58.0], qfq_lows=[56.0, 57.0],
        as_of="20260703", trade_days=["20260703"], window=20)
    assert rec["error"] is None
    assert rec["ma20"] is None
    assert rec["dist_ma20_pct"] is None
    assert rec["dist_low20_pct"] is None
    assert rec["pnl_pct"] is not None        # 盈亏不依赖历史,仍要算


def test_build_record_with_entry_date_counts_held_days():
    rec = build_position_record(
        _pos(entry_date="20260701"), close=57.59, pct_chg=-0.29,
        qfq_closes=[float(i) for i in range(40, 60)], qfq_lows=[float(i) for i in range(38, 58)],
        as_of="20260703", trade_days=["20260630", "20260701", "20260702", "20260703"], window=20)
    assert rec["held_days"] == 3
