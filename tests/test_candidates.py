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


def test_b8_band_member_is_buy_eligible_with_band_code():
    """X-14:D9 且 b8_band=True(上期成员仍在带内)→ 资格成立,来源码 B8_BAND。"""
    from ashare_gauntlet.candidates import candidate_assessment

    a = candidate_assessment(_row(decile=9, b8_band=True), _ov(), AS_OF)
    assert a["eligible_buy"] is True
    assert a["reason_codes"] == ["B8_BAND", "TIER_GREEN", "FACTCHECK_CLEAR"]
    # D8 也在带内
    a8 = candidate_assessment(_row(decile=8, b8_band=True), _ov(), AS_OF)
    assert a8["eligible_buy"] is True


def test_b8_band_requires_decile_in_band():
    """防御:跌穿带(decile≤7)即使误标 b8_band 也被 NOT_D10 否决;不注入字段=旧口径。"""
    from ashare_gauntlet.candidates import candidate_assessment

    for row in (_row(decile=7, b8_band=True), _row(decile=9), _row(decile=9, b8_band=False)):
        a = candidate_assessment(row, _ov(), AS_OF)
        assert a["eligible_buy"] is False, row
        assert "NOT_D10" in a["reason_codes"], row


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


def test_future_factcheck_does_not_rewrite_old_decision():
    from ashare_gauntlet.candidates import candidate_assessment

    for verdict in ("clear", "red"):
        future = _ov(verdict=verdict)
        future["as_of"] = "20260809"
        result = candidate_assessment(_row(), future, "20260807")
        assert result["eligible_buy"] is False
        assert "FACTCHECK_AFTER_AS_OF" in result["reason_codes"]
        assert "GOVERNANCE_RED" not in result["reason_codes"]


def test_factcheck_dates_fail_loud_when_invalid_or_reversed():
    import pytest
    from ashare_gauntlet.candidates import candidate_assessment

    bad_decision_date = _ov()
    with pytest.raises(ValueError, match="decision as_of"):
        candidate_assessment(_row(), bad_decision_date, "20260230")

    bad_factcheck_date = _ov()
    bad_factcheck_date["as_of"] = "20260230"
    with pytest.raises(ValueError, match="factcheck as_of"):
        candidate_assessment(_row(), bad_factcheck_date, AS_OF)

    reversed_window = _ov(expires="20260630")
    with pytest.raises(ValueError, match="不能早于"):
        candidate_assessment(_row(), reversed_window, AS_OF)


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


def test_hard_veto_codes_covers_all_non_factcheck_checks():
    """HARD_VETO_CODES 必须=全部"写 clear 也解不开"的码,漏一个就会把🎰/⚡票
    呈现成"只差一步就能买"(跨层审计;突变验证发现此修复原先无测试守住)。"""
    from ashare_gauntlet.candidates import HARD_VETO_CODES, _CHECKS

    # 与检查序逐条对齐:非 FACTCHECK_ 前缀的码全部是硬否决
    assert HARD_VETO_CODES == {c for c in _CHECKS if not c.startswith("FACTCHECK_")}
    # 这两个是最容易被漏掉的(它们与 FACTCHECK_REQUIRED 共现,且 clear 解不开)
    assert "SPEC_CROWD" in HARD_VETO_CODES
    assert "SPIKE_LIMIT" in HARD_VETO_CODES
    assert "NOT_D10" in HARD_VETO_CODES and "TIER_NOT_GREEN" in HARD_VETO_CODES
    # fact-check 相关码不是硬否决(它们正是"写 verdict 能解开"的那些)
    assert not any(c.startswith("FACTCHECK_") for c in HARD_VETO_CODES)


def test_spec_crowd_stock_not_listed_as_only_pending_factcheck():
    """🎰投机拥挤 + ⚡涨停的票即便只差 fact-check,也不得进"唯一未决项"候选集。"""
    from scripts.eod_ops import pending_factcheck_set

    snap = {"decisions": [
        {"ts_code": "1.SZ", "state": "WAIT",
         "reason_codes": ["SPEC_CROWD", "SPIKE_LIMIT", "FACTCHECK_REQUIRED"]},
        {"ts_code": "2.SZ", "state": "WAIT", "reason_codes": ["FACTCHECK_REQUIRED"]},
    ]}
    assert pending_factcheck_set(snap) == {"2.SZ"}
