"""portfolio_decision:四态生成与组合约束(spec §7/§8;确定性分配)。"""
from __future__ import annotations

from ashare_gauntlet.portfolio_decision import decide_states

POLICY = {"policy_version": "1", "target_positions": 10, "target_weight": 0.10,
          "industry_cap": 0.20, "lot_size": 100, "min_cash": 0}


def _a(ts, score=0.9, industry="化工原料", last=10.0, eligible=True, codes=None) -> dict:
    return {"ts_code": ts, "name": ts, "industry": industry, "score": score, "last": last,
            "eligible_buy": eligible,
            "reason_codes": codes or (["D10", "TIER_GREEN", "FACTCHECK_CLEAR"] if eligible else ["NOT_D10"]),
            "governance_red": False}


def _held(ts, mv=10000.0, industry="煤化工") -> dict:
    return {"ts_code": ts, "name": ts, "industry": industry, "mv": mv}


def test_buy_allocation_lot_and_cash():
    from ashare_gauntlet.portfolio_decision import decide_states

    ds = decide_states([_a("600001.SH", last=33.3)], {}, POLICY,
                       account_value=100_000.0, cash=50_000.0)
    d = {x["ts_code"]: x for x in ds}["600001.SH"]
    # 10% × 100k = 10k → 33.3 元 → 300 股(整手向下)
    assert d["state"] == "BUY" and d["execution"]["shares"] == 300


def test_industry_cap_blocks_second_buy():
    from ashare_gauntlet.portfolio_decision import decide_states

    ds = decide_states([_a("600001.SH", score=0.99), _a("600002.SH", score=0.98)],
                       {"600003.SH": _held("600003.SH", mv=10_000, industry="化工原料")},
                       POLICY, account_value=100_000.0, cash=90_000.0)
    d = {x["ts_code"]: x for x in ds}
    # 已持同行业 10% + 新买 10% = 20% 顶格;第二只再买破 20% 上限 → WAIT
    assert d["600001.SH"]["state"] == "BUY"
    assert d["600002.SH"]["state"] == "WAIT"
    assert "INDUSTRY_CAP" in d["600002.SH"]["reason_codes"]


def test_cash_shortage_downsizes_then_waits():
    from ashare_gauntlet.portfolio_decision import decide_states

    # 现金只够 1 手:按现金缩量成交,不透支
    ds = decide_states([_a("600001.SH", last=30.0)], {}, POLICY,
                       account_value=100_000.0, cash=3_500.0)
    d = ds[0]
    assert d["state"] == "BUY" and d["execution"]["shares"] == 100
    # 连 1 手都不够 → WAIT INSUFFICIENT_CASH
    ds2 = decide_states([_a("600002.SH", last=30.0)], {}, POLICY,
                        account_value=100_000.0, cash=2_000.0)
    assert ds2[0]["state"] == "WAIT" and "INSUFFICIENT_CASH" in ds2[0]["reason_codes"]


def test_account_state_missing_gives_zero_shares():
    from ashare_gauntlet.portfolio_decision import decide_states

    ds = decide_states([_a("600001.SH")], {}, POLICY, account_value=None, cash=None)
    d = ds[0]
    assert d["state"] == "BUY" and d["execution"]["shares"] == 0
    assert "ACCOUNT_STATE_MISSING" in d["reason_codes"]


def test_held_stays_hold_without_new_buy_and_after_decile_drop():
    from ashare_gauntlet.portfolio_decision import decide_states

    # 跌出 D10 的持仓:生产退出规则=C2 月度审视(methodology §10),日频快照挂语义码不退
    ds = decide_states([_a("600003.SH", eligible=False, codes=["NOT_D10"])],
                       {"600003.SH": _held("600003.SH")}, POLICY,
                       account_value=100_000.0, cash=10_000.0)
    d = ds[0]
    assert d["state"] == "HOLD"
    assert "EXIT_RULE_C2_MONTHLY" in d["reason_codes"]


def test_confirmed_c2_becomes_advisory_exit() -> None:
    code = "600003.SH"
    decisions = decide_states([_a(code, eligible=False, codes=["NOT_D10"])],
                              {code: _held(code)}, POLICY,
                              account_value=100_000, cash=0,
                              c2_exit_eligible={code})
    row = next(d for d in decisions if d["ts_code"] == code)
    assert row["state"] == "EXIT"
    assert row["reason_codes"] == ["EXIT_RULE_C2_CONFIRMED"]


def test_governance_exit_precedes_confirmed_c2() -> None:
    code = "600003.SH"
    assessment = _a(code, eligible=False, codes=["GOVERNANCE_RED"])
    assessment["governance_red"] = True
    decisions = decide_states([assessment], {code: _held(code)}, POLICY,
                              account_value=100_000, cash=0,
                              c2_exit_eligible={code})
    row = next(d for d in decisions if d["ts_code"] == code)
    assert row["reason_codes"] == ["GOVERNANCE_RED"]


def test_risk_and_manual_exit_precede_confirmed_c2() -> None:
    code = "600003.SH"
    assessment = _a(code, eligible=False, codes=["NOT_D10"])
    decisions = decide_states([assessment], {code: _held(code)}, POLICY,
                              account_value=100_000, cash=0,
                              risk_breach={code}, manual_exit={code},
                              c2_exit_eligible={code})
    row = next(d for d in decisions if d["ts_code"] == code)
    assert row["reason_codes"] == ["RISK_LINE_BREACH", "MANUAL_LOGIC_FAIL"]


def test_confirmed_c2_applies_to_eligible_daily_assessment() -> None:
    code = "600003.SH"
    decisions = decide_states([_a(code, eligible=True)], {code: _held(code)}, POLICY,
                              account_value=100_000, cash=0,
                              c2_exit_eligible={code})
    row = next(d for d in decisions if d["ts_code"] == code)
    assert row["state"] == "EXIT"
    assert row["reason_codes"] == ["EXIT_RULE_C2_CONFIRMED"]


def test_exit_only_from_predefined_reasons():
    from ashare_gauntlet.portfolio_decision import decide_states

    a = _a("600003.SH", eligible=False, codes=["GOVERNANCE_RED"])
    a["governance_red"] = True
    ds = decide_states([a], {"600003.SH": _held("600003.SH")}, POLICY,
                       account_value=100_000.0, cash=0.0,
                       risk_breach={"600004.SH"}, manual_exit=set())
    assert ds[0]["state"] == "EXIT" and "GOVERNANCE_RED" in ds[0]["reason_codes"]
    ds2 = decide_states([], {"600004.SH": _held("600004.SH")}, POLICY,
                        account_value=100_000.0, cash=0.0,
                        risk_breach={"600004.SH"}, manual_exit=set())
    assert ds2[0]["state"] == "EXIT" and "RISK_LINE_BREACH" in ds2[0]["reason_codes"]


def test_portfolio_full_and_deterministic_tiebreak():
    from ashare_gauntlet.portfolio_decision import decide_states

    held = {f"60000{i}.SH": _held(f"60000{i}.SH", industry=f"行业{i}") for i in range(1, 10)}
    cands = [_a("600011.SH", score=0.5, industry="行业A"),
             _a("600010.SH", score=0.5, industry="行业B")]
    ds = decide_states(cands, held, POLICY, account_value=100_000.0, cash=90_000.0)
    d = {x["ts_code"]: x for x in ds}
    # 9 持仓 + 1 空位;同分 → ts_code 升序优先 → 600010 BUY,600011 WAIT PORTFOLIO_FULL
    assert d["600010.SH"]["state"] == "BUY"
    assert d["600011.SH"]["state"] == "WAIT"
    assert "PORTFOLIO_FULL" in d["600011.SH"]["reason_codes"]


def test_account_missing_still_enforces_industry_cap_by_planned_weight():
    from ashare_gauntlet.portfolio_decision import decide_states

    # Codex P1-1:账户未知时行业上限仍须按"计划权重"执行(两只10%顶格,第三只拦)
    cands = [_a("600001.SH", score=0.9, industry="同业"),
             _a("600002.SH", score=0.8, industry="同业"),
             _a("600003.SH", score=0.7, industry="同业")]
    ds = decide_states(cands, {}, POLICY, account_value=None, cash=None)
    d = {x["ts_code"]: x for x in ds}
    assert d["600001.SH"]["state"] == "BUY" and d["600002.SH"]["state"] == "BUY"
    assert d["600003.SH"]["state"] == "WAIT" and "INDUSTRY_CAP" in d["600003.SH"]["reason_codes"]


def test_governance_red_unheld_is_wait_not_exit():
    from ashare_gauntlet.portfolio_decision import decide_states

    a = _a("600005.SH", eligible=False, codes=["GOVERNANCE_RED"])
    a["governance_red"] = True
    ds = decide_states([a], {}, POLICY, account_value=100_000.0, cash=10_000.0)
    assert ds[0]["state"] == "WAIT" and "GOVERNANCE_RED" in ds[0]["reason_codes"]


def test_decision_contract_has_invalidations_and_rich_evidence():
    from ashare_gauntlet.portfolio_decision import decide_states

    a = _a("600001.SH")
    a.update(decile=10, spec_crowd=False, spike_limit=False)
    ds = decide_states([a], {}, POLICY, account_value=100_000.0, cash=50_000.0)
    d = ds[0]
    assert d["state"] == "BUY"
    assert "LEAVE_PRODUCTION_UNIVERSE" in d["invalidations"]     # spec §9
    assert d["evidence"]["decile"] == 10 and d["evidence"]["spec_crowd"] is False


def test_policy_sanity_fail_loud():
    from ashare_gauntlet.portfolio_decision import validate_policy
    import pytest

    bad = dict(POLICY, target_positions=15, target_weight=0.10)   # 150% 目标敞口
    with pytest.raises(ValueError):
        validate_policy(bad)
    validate_policy(POLICY)   # 合法不抛
    # 生产口径 3 × 0.24 / 行业上限 0.25 必须合法
    validate_policy({"policy_version": "1", "target_positions": 3,
                     "target_weight": 0.24, "industry_cap": 0.25, "lot_size": 100})


def test_validate_policy_rejects_missing_bool_string_and_nonfinite():
    from ashare_gauntlet.portfolio_decision import validate_policy
    import pytest

    with pytest.raises(ValueError, match="缺字段"):
        validate_policy({"target_positions": 3, "target_weight": 0.24})
    with pytest.raises(ValueError, match="整数"):
        validate_policy(dict(POLICY, target_positions=True))
    with pytest.raises(ValueError, match="整数"):
        validate_policy(dict(POLICY, target_positions="3"))
    with pytest.raises(ValueError, match="整数"):
        validate_policy(dict(POLICY, lot_size=100.0))
    with pytest.raises(ValueError, match="有限数"):
        validate_policy(dict(POLICY, target_weight="0.24"))
    with pytest.raises(ValueError, match="有限数"):
        validate_policy(dict(POLICY, industry_cap=float("nan")))
    with pytest.raises(ValueError, match="有限数"):
        validate_policy(dict(POLICY, target_weight=float("inf")))
    with pytest.raises(ValueError, match="非法"):
        validate_policy(dict(POLICY, target_positions=0))
    with pytest.raises(ValueError, match="非法"):
        validate_policy(dict(POLICY, industry_cap=0))
    with pytest.raises(ValueError, match=r"industry_cap=.*>1"):
        validate_policy(dict(POLICY, industry_cap=1.5))
    validate_policy(dict(POLICY, industry_cap=1.0))
    with pytest.raises(ValueError, match="有限数") as overflow_cap:
        validate_policy(dict(POLICY, industry_cap=10 ** 400))
    assert isinstance(overflow_cap.value.__cause__, OverflowError)
    with pytest.raises(ValueError, match="有限数") as overflow_w:
        validate_policy(dict(POLICY, target_weight=10 ** 400))
    assert isinstance(overflow_w.value.__cause__, OverflowError)
    with pytest.raises(ValueError) as overflow_n:
        validate_policy(dict(POLICY, target_positions=10 ** 400, target_weight=0.24,
                             industry_cap=1.0))
    assert isinstance(overflow_n.value.__cause__, OverflowError)


def test_held_overweight_stays_hold_not_exit():
    """已有持仓超 target_weight / industry_cap 不得擅自转 EXIT。"""
    policy = {"policy_version": "1", "target_positions": 3, "target_weight": 0.24,
              "industry_cap": 0.25, "lot_size": 100, "min_cash": 0}
    ds = decide_states([_a("600001.SH", eligible=True, industry="化工原料")],
                       {"600001.SH": _held("600001.SH", mv=50_000, industry="化工原料")},
                       policy, account_value=100_000.0, cash=50_000.0)
    row = next(d for d in ds if d["ts_code"] == "600001.SH")
    assert row["state"] == "HOLD"
    assert "EXIT" != row["state"]
