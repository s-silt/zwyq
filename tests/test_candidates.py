"""candidates:生产候选资格与硬否决真值表(spec §5;reason code 顺序确定)。"""
from __future__ import annotations


def _row(**kw) -> dict:
    base = {"ts_code": "600001.SH", "name": "好公司", "industry": "化工原料",
            "decile": 10, "tier": "🟢", "spec_crowd": False, "spike_limit": False,
            "score": 0.9, "last": 10.0}
    base.update(kw)
    return base


def _ov(verdict="clear", expires="20270101") -> dict:
    return {"ts_code": "600001.SH", "as_of": "20260701", "verdict": verdict,
            "reason": "t", "expires_on": expires}


AS_OF = "20260717"


def test_perfect_candidate_is_buy_eligible():
    from ashare_gauntlet.candidates import candidate_assessment

    a = candidate_assessment(_row(), _ov(), AS_OF)
    assert a["eligible_buy"] is True
    assert a["reason_codes"] == ["D10", "TIER_GREEN", "FACTCHECK_CLEAR"]


def test_veto_truth_table():
    from ashare_gauntlet.candidates import candidate_assessment

    cases = [
        (_row(decile=9), "NOT_D10"),
        (_row(tier="🟡"), "TIER_NOT_GREEN"),
        (_row(spec_crowd=True), "SPEC_CROWD"),
        (_row(spike_limit=True), "SPIKE_LIMIT"),
        (_row(poll_mark=True), "POLLUTION_PENDING_FACTCHECK"),
        (_row(name="*ST烂"), "ST_NAME"),
        (_row(ts_code="300001.SZ"), "NOT_MAIN_BOARD"),
    ]
    for row, code in cases:
        a = candidate_assessment(row, _ov(), AS_OF)
        assert a["eligible_buy"] is False, code
        assert code in a["reason_codes"], code


def test_factcheck_gate():
    from ashare_gauntlet.candidates import candidate_assessment

    # 第四关强制:无覆盖 → 不可 BUY(FACTCHECK_REQUIRED);过期 → FACTCHECK_EXPIRED
    a = candidate_assessment(_row(), None, AS_OF)
    assert a["eligible_buy"] is False and "FACTCHECK_REQUIRED" in a["reason_codes"]
    b = candidate_assessment(_row(), _ov(expires="20260710"), AS_OF)
    assert b["eligible_buy"] is False and "FACTCHECK_EXPIRED" in b["reason_codes"]
    c = candidate_assessment(_row(), _ov(verdict="red"), AS_OF)
    assert c["eligible_buy"] is False and "GOVERNANCE_RED" in c["reason_codes"]


def test_reason_codes_deterministic_order():
    from ashare_gauntlet.candidates import candidate_assessment

    # 多重否决:输出顺序=固定检查序,不随 dict 顺序/运行漂移
    row = _row(decile=8, tier="🟡", spec_crowd=True, spike_limit=True, name="ST差")
    a = candidate_assessment(row, None, AS_OF)
    assert a["reason_codes"] == ["ST_NAME", "NOT_D10", "TIER_NOT_GREEN",
                                 "SPEC_CROWD", "SPIKE_LIMIT", "FACTCHECK_REQUIRED"]


def test_missing_fields_fail_loud():
    from ashare_gauntlet.candidates import candidate_assessment
    import pytest

    row = _row()
    del row["decile"]
    with pytest.raises(KeyError):
        candidate_assessment(row, _ov(), AS_OF)
