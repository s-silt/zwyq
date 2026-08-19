"""止损政策一致性检查(只读):裸窗口/写反/带外必须 surface,未知不当 OK。"""
from __future__ import annotations

import pytest

from ashare_gauntlet import stop_policy as sp


def _pos(**kw):
    base = {"ts_code": "600001.SH", "bucket": "long", "cost": 10.0, "stop": 8.7}
    base.update(kw)
    return base


def test_long_band_uses_institution_range_endpoints():
    """长线带=制度区间 -12~-15% 的端点,不是"中值±容差"(后者会放行区间外的 -11%)。"""
    assert sp.policy_band("long", 10.0) == (8.5, 8.8)      # -15% ~ -12%
    # -11% 的长线止损比制度更紧,必须被判出带外(跨层审计整改)
    assert sp.check_position(_pos(stop=8.9))["status"] == "OUT_OF_BAND"
    # 短线 -7% 硬止损 ± 2pt 人工填价容差 → [9.1, 9.5]
    assert sp.policy_band("short", 10.0) == (9.1, 9.5)
    # 中英两种写法同义(经 account_state 权威归一)
    assert sp.policy_band("长线", 10.0) == sp.policy_band("long", 10.0)
    assert sp.policy_band("短线", 10.0) == sp.policy_band("short", 10.0)
    # 制度前老仓不套新规
    assert sp.policy_band("制度前", 10.0) is None


def test_in_band_is_ok():
    assert sp.check_position(_pos(stop=8.7))["status"] == "OK"
    assert sp.check_position(_pos(bucket="short", stop=9.3))["status"] == "OK"


def test_missing_stop_is_bare_window():
    # trade_record --buy 落账后的裸窗口:哨兵跳过 BREACH/NEAR
    r = sp.check_position(_pos(stop=None))
    assert r["status"] == "MISSING_STOP"
    assert "哨兵" in r["detail"]
    # 非正/非法值同样无保护,不当作已填
    assert sp.check_position(_pos(stop=0))["status"] == "MISSING_STOP"
    assert sp.check_position(_pos(stop="8.7"))["status"] == "MISSING_STOP"


def test_stop_above_cost_flags_direction_error():
    # 止损写在成本上方=方向写反/抄错
    r = sp.check_position(_pos(stop=11.0))
    assert r["status"] == "ABOVE_COST"
    r2 = sp.check_position(_pos(stop=10.0))
    assert r2["status"] == "ABOVE_COST"


def test_out_of_band_too_tight_or_loose():
    tight = sp.check_position(_pos(stop=9.6))     # 距成本 -4%,长线偏紧
    assert tight["status"] == "OUT_OF_BAND" and "偏紧" in tight["detail"]
    loose = sp.check_position(_pos(stop=7.0))     # 距成本 -30%,偏松
    assert loose["status"] == "OUT_OF_BAND" and "偏松" in loose["detail"]
    # 少写一位(8.7 → 0.87)必须被抓到
    assert sp.check_position(_pos(stop=0.87))["status"] == "OUT_OF_BAND"


def test_unknown_never_reported_as_ok():
    assert sp.check_position(_pos(cost=None))["status"] == "UNKNOWN"
    assert sp.check_position(_pos(bucket="制度前"))["status"] == "UNKNOWN"
    # UNKNOWN 计入需人工处理(未知不解释为安全)
    rows = sp.check_positions([_pos(cost=None), _pos(stop=8.7)])
    assert [r["status"] for r in sp.needs_attention(rows)] == ["UNKNOWN"]


def test_bucket_normalization_is_single_source():
    """跨层审计 P1 回归:bucket 中英双轨制曾让短线席位/+25%锁利对英文仓静默失效。"""
    from ashare_gauntlet.account_state import normalize_account_state, normalize_bucket
    assert normalize_bucket("short") == normalize_bucket("短线") == "short"
    assert normalize_bucket("long") == normalize_bucket("长线") == "long"
    assert normalize_bucket("制度前") == "legacy"
    assert normalize_bucket("乱写") is None and normalize_bucket(None) is None
    # 英文 "short" 仓必须占用短线席位(此前 occupied 恒 False → 可开第二只短线仓)
    acct = normalize_account_state({
        "as_of": "20260818", "cash": 1000, "conditional_orders": "无",
        "positions": [{"ts_code": "600001.SH", "mv": 9000, "cost": 10.0,
                       "shares": 900, "industry": "电气设备", "bucket": "short"}]})
    assert acct["short_slot"]["occupied"] is True
    assert acct["short_slot"]["count"] == 1


def test_time_stop_short_bucket_only():
    # 短线仓达 10 交易日窗口 → 提醒(制度常数,非新阈值)
    hit = sp.check_time_stop({"ts_code": "1.SH", "bucket": "short", "held_days": 10})
    assert hit and hit["status"] == "TIME_STOP" and hit["held_days"] == 10
    assert sp.check_time_stop({"ts_code": "1.SH", "bucket": "短线", "held_days": 12})
    # 未到窗口 / 非短线 / held_days 缺失 → 不报(不伪造判定)
    assert sp.check_time_stop({"ts_code": "1.SH", "bucket": "short", "held_days": 9}) is None
    assert sp.check_time_stop({"ts_code": "1.SH", "bucket": "long", "held_days": 99}) is None
    assert sp.check_time_stop({"ts_code": "1.SH", "bucket": "short", "held_days": None}) is None
    assert sp.check_time_stop({"ts_code": "1.SH", "bucket": "short"}) is None


def test_check_time_stops_returns_only_hits():
    rows = [
        {"ts_code": "1.SH", "bucket": "short", "held_days": 11},
        {"ts_code": "2.SZ", "bucket": "short", "held_days": 3},
        {"ts_code": "3.SH", "bucket": "long", "held_days": 300},
    ]
    assert [r["ts_code"] for r in sp.check_time_stops(rows)] == ["1.SH"]


def test_summarize_and_bad_input():
    rows = sp.check_positions([_pos(), _pos(stop=None), _pos(stop=11.0)])
    assert sp.summarize(rows) == {"OK": 1, "MISSING_STOP": 1, "ABOVE_COST": 1}
    with pytest.raises(sp.StopPolicyError):
        sp.check_position("not-a-dict")  # type: ignore[arg-type]


def test_conditional_order_coverage_never_assumes_safe():
    """哨兵停用后条件单是唯一防线;未验证/无数据一律记 uncovered(未知不解释为安全)。"""
    from ashare_gauntlet.stop_policy import conditional_order_coverage as cov

    acct = {"positions": [{"ts_code": "600001.SH"}, {"ts_code": "000002.SZ"}],
            "conditional_orders": {"status": "unverified", "format": "free_text"}}
    r = cov(acct)
    assert r["status"] == "UNVERIFIED" and r["verified"] is False
    assert set(r["uncovered"]) == {"600001.SH", "000002.SZ"} and r["covered"] == []

    # 结构化 v2 且 verified:逐仓判定,只认 active 的 SELL 单
    acct2 = {"positions": [{"ts_code": "600001.SH"}, {"ts_code": "000002.SZ"}],
             "conditional_orders": {"status": "verified", "format": "structured_v2",
                                    "orders": [
                                        {"ts_code": "600001.SH", "side": "SELL", "status": "active"},
                                        {"ts_code": "000002.SZ", "side": "SELL", "status": "cancelled"},
                                        {"ts_code": "000002.SZ", "side": "BUY", "status": "active"}]}}
    r2 = cov(acct2)
    assert r2["status"] == "VERIFIED"
    assert r2["covered"] == ["600001.SH"] and r2["uncovered"] == ["000002.SZ"]

    assert cov({"positions": []})["status"] == "NO_POSITIONS"
