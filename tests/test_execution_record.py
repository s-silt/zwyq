"""execution_record:影子建议→次日现实的闭环纯函数(spec §13.2 影子核对)。"""
from __future__ import annotations

import pytest


def test_verify_buy_executability_codes():
    from scripts.execution_record import verify_buy

    assert verify_buy(open_px=10.0, locked=False, suspended=False) == "BUYABLE"
    assert verify_buy(open_px=11.0, locked=True, suspended=False) == "LIMIT_UP_LOCKED"
    assert verify_buy(open_px=None, locked=False, suspended=True) == "SUSPENDED"


def test_verify_buy_contradictory_state_fails_loud():
    from scripts.execution_record import verify_buy

    # 停牌却有开盘价 / 既锁又停:上游数据矛盾,不得静默选一个
    with pytest.raises(ValueError):
        verify_buy(open_px=10.0, locked=False, suspended=True)
    with pytest.raises(ValueError):
        verify_buy(open_px=None, locked=True, suspended=True)


def test_verify_buy_missing_price_fails_loud():
    from scripts.execution_record import verify_buy

    # 非停牌却无有效价=上游缺口(Codex P1:锁板无价不得绕过、NaN 不得判 BUYABLE)
    with pytest.raises(ValueError):
        verify_buy(open_px=None, locked=True, suspended=False)
    with pytest.raises(ValueError):
        verify_buy(open_px=None, locked=False, suspended=False)
    with pytest.raises(ValueError):
        verify_buy(open_px=float("nan"), locked=False, suspended=False)


def test_next_trade_date_is_first_after_as_of():
    from scripts.execution_record import next_trade_date

    dates = ["20260716", "20260717", "20260720"]
    assert next_trade_date(dates, "20260717") == "20260720"   # 跨周末取次一交易日
    assert next_trade_date(dates, "20260716") == "20260717"
    with pytest.raises(ValueError):
        next_trade_date(dates, "20260720")                    # 快照后尚无行情=不可核对


def test_next_trade_date_sorts_and_validates_input():
    from scripts.execution_record import next_trade_date

    # 不依赖调用方清洗:乱序输入不得误选更晚日期(Codex P1)
    assert next_trade_date(["20260720", "20260717"], "20260716") == "20260717"
    # 非 YYYYMMDD 混入 fail-loud,不得静默跳过或误比较
    with pytest.raises(ValueError):
        next_trade_date(["20260717", "600519_20200101_20260101"], "20260716")


def test_upsert_record_freezes_history():
    from scripts.execution_record import upsert_record

    r1 = {"decision_as_of": "20260717", "verify_date": "20260720", "buy_checks": []}
    records, action = upsert_record([], r1)
    assert action == "inserted" and records == [r1]

    # 同日重跑:幂等覆盖
    r1b = {"decision_as_of": "20260717", "verify_date": "20260720", "buy_checks": [{"x": 1}]}
    records, action = upsert_record(records, r1b)
    assert action == "replaced" and records == [r1b]

    # 隔日再跑同一快照:历史纪律事实冻结,不得用当日 holdings 改写(Codex P1)
    r1c = {"decision_as_of": "20260717", "verify_date": "20260721", "buy_checks": []}
    records2, action = upsert_record(records, r1c)
    assert action == "frozen" and records2 == [r1b]


def test_divergences_four_cases():
    from scripts.execution_record import divergences

    decisions = [
        {"ts_code": "600001.SH", "state": "BUY"},
        {"ts_code": "600002.SH", "state": "BUY"},
        {"ts_code": "600003.SH", "state": "EXIT"},
        {"ts_code": "600004.SH", "state": "EXIT"},
        {"ts_code": "600005.SH", "state": "HOLD"},
    ]
    held_now = {"600001.SH", "600003.SH", "600005.SH", "600099.SH"}
    prev_held = {"600003.SH", "600004.SH", "600005.SH"}
    out = {d["ts_code"]: d for d in divergences(decisions, held_now, prev_held)}
    assert out["600001.SH"]["outcome"] == "FOLLOWED"          # BUY→已持有
    assert out["600002.SH"]["outcome"] == "NOT_FOLLOWED"      # BUY→未买
    assert out["600003.SH"]["outcome"] == "NOT_FOLLOWED"      # EXIT→仍持有
    assert out["600004.SH"]["outcome"] == "FOLLOWED"          # EXIT→已清
    assert out["600099.SH"]["outcome"] == "OFF_LIST_TRADE"    # 未建议却新增持仓(纪律偏差)
    assert "600005.SH" not in out                             # HOLD 持有中=一致,不记偏差
