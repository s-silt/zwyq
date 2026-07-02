"""⛔ 档 needs_human 拆分:监管/算术事实(ST/资不抵债)否决安全;量级判断(亏损/双降/商誉
常数未标定)一票否决必须过人工复核——别无人复核误杀(审计:误杀概率不可知)。"""
from ashare_gauntlet.record import tier_of


def _base(**over):
    rec = {
        "fundamental": {"profitable": True, "np_yoy": 10.0, "dedt_yoy": 10.0, "rev_yoy": 10.0,
                        "np_yi": 5.0, "dedt_yi": 4.8},
        "quality": {"op_cashflow_yi": 5.0},
        "balance": {}, "status": {}, "flags": [],
    }
    for k, v in over.items():
        rec[k] = {**rec.get(k, {}), **v}
    return rec


def test_st_veto_is_factual_no_human_needed():
    t = tier_of(_base(status={"is_st": True}))
    assert t["grade"] == "⛔" and t["needs_human"] is False


def test_insolvency_veto_is_factual_no_human_needed():
    t = tier_of(_base(balance={"net_assets_yi": -1.0}))
    assert t["grade"] == "⛔" and t["needs_human"] is False


def test_loss_veto_needs_human():
    t = tier_of(_base(fundamental={"profitable": False}))
    assert t["grade"] == "⛔" and t["needs_human"] is True


def test_double_decline_veto_needs_human():
    t = tier_of(_base(fundamental={"np_yoy": -50.0, "dedt_yoy": -55.0}))
    assert t["grade"] == "⛔" and t["needs_human"] is True


def test_goodwill_veto_needs_human():
    t = tier_of(_base(balance={"goodwill_yi": 60.0, "net_assets_yi": 100.0}))
    assert t["grade"] == "⛔" and t["needs_human"] is True
