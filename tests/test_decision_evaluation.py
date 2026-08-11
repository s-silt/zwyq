"""历史决策链冻结输出审计与反事实可执行性测试。"""
from __future__ import annotations

import pandas as pd
import pytest

from ashare_gauntlet.decision_evaluation import (
    DecisionEvaluationError,
    aggregate_metrics,
    audit_snapshot,
    build_market_tables,
    evaluate_code_horizon,
    evaluate_episodes,
    extract_buy_episodes,
    factcheck_status,
)


def _factor(code="600001.SH", decile=10):
    return [{"ts_code": code, "name": "甲", "decile": decile, "score": 0.9}]


def _decision(code="600001.SH", state="BUY", reasons=None, decile=10, shares=100):
    return {
        "ts_code": code,
        "name": "甲",
        "state": state,
        "reason_codes": reasons if reasons is not None else ["D10", "TIER_GREEN", "FACTCHECK_CLEAR"],
        "evidence": {"decile": decile, "score": 0.9},
        "execution": {"eligible_from": "NEXT_TRADING_DAY", "max_entry_price": None,
                      "target_weight": 0.1 if state == "BUY" else None, "shares": shares},
        "invalidations": [],
    }


def _snapshot(date="20260101", decisions=None, *, account=False):
    out = {
        "as_of": date,
        "generated_at": f"{date[:4]}-{date[4:6]}-{date[6:]}T18:00:00+08:00",
        "factor_snapshot": f"data/holdscore/{date}_factor.json",
        "policy_version": "1",
        "entry_model_version": "research-only",
        "data_status": "complete",
        "decisions": decisions if decisions is not None else [_decision()],
    }
    if account:
        out["account_as_of"] = date
        out["account_source_schema"] = "account_state.v1"
    return out


def test_audit_legacy_snapshot_is_frozen_replayable_but_partial():
    audit = audit_snapshot(_snapshot(), "20260101", _factor())
    assert audit["valid"] is True
    assert audit["frozen_output_replayability"] == "frozen_output_readable"
    assert audit["content_integrity_status"] == "unverified_legacy"
    assert audit["pit_evidence_status"] == "partial"
    assert audit["full_input_recomputability"] == "not_recomputable"
    assert audit["factcheck_encoded_coverage"] == 1.0


def test_audit_rejects_filename_factor_and_buy_contract_mismatches():
    snap = _snapshot(decisions=[_decision(decile=9, reasons=["D10", "FACTCHECK_REQUIRED"])])
    snap["factor_snapshot"] = "data/holdscore/20260102_factor.json"
    audit = audit_snapshot(snap, "20260102", _factor(decile=9))
    assert audit["valid"] is False
    text = " ".join(audit["errors"])
    assert "文件日期" in text
    assert "factor_snapshot" in text
    assert "decile" in text
    assert "FACTCHECK_CLEAR" in text


def test_audit_allows_held_position_missing_from_factor_with_warning():
    snap = _snapshot(decisions=[_decision(code="600009.SH", state="HOLD", reasons=["HELD"],
                                          decile=None, shares=0)])
    audit = audit_snapshot(snap, "20260101", _factor())
    assert audit["valid"] is True
    assert "掉出 factor snapshot" in " ".join(audit["warnings"])


@pytest.mark.parametrize(("reasons", "expected"), [
    (["FACTCHECK_CLEAR"], "clear_as_recorded"),
    (["GOVERNANCE_RED"], "red_as_recorded"),
    (["FACTCHECK_EXPIRED"], "expired_as_recorded"),
    (["FACTCHECK_REQUIRED"], "missing_as_recorded"),
    (["FACTCHECK_AFTER_AS_OF"], "future_evidence_rejected"),
    (["D10"], "unknown"),
])
def test_factcheck_status_uses_frozen_reason_codes_only(reasons, expected):
    assert factcheck_status(reasons) == expected


def test_factcheck_status_rejects_contradiction():
    with pytest.raises(DecisionEvaluationError, match="矛盾"):
        factcheck_status(["FACTCHECK_CLEAR", "GOVERNANCE_RED"])


def test_buy_episode_deduplicates_consecutive_buy_and_reopens_after_wait():
    states = ["WAIT", "BUY", "BUY", "WAIT", "BUY"]
    snaps = []
    for i, state in enumerate(states, start=1):
        reasons = ["D10", "TIER_GREEN", "FACTCHECK_CLEAR"] if state == "BUY" else ["FACTCHECK_REQUIRED"]
        snaps.append(_snapshot(f"2026010{i}", [_decision(state=state, reasons=reasons,
                                                          shares=100 if state == "BUY" else 0)]))
    episodes = extract_buy_episodes(snaps)
    assert [e["as_of"] for e in episodes] == ["20260102", "20260105"]
    assert all(e["actual_execution"] is False for e in episodes)


def _market(prices, *, missing=None):
    """prices={date: open};同一代码，复权因子恒1。"""
    daily = []
    adj = []
    missing = set(missing or [])
    for date, price in prices.items():
        if date not in missing:
            daily.append({"ts_code": "600001.SH", "trade_date": date, "open": price,
                          "high": price + 0.2, "low": price - 0.2, "close": price + 0.1})
            adj.append({"ts_code": "600001.SH", "trade_date": date, "adj_factor": 1.0})
        # 每日放一个陪衬代码，确保停牌日仍有市场交易日
        daily.append({"ts_code": "600999.SH", "trade_date": date, "open": 5.0,
                      "high": 5.1, "low": 4.9, "close": 5.0})
        adj.append({"ts_code": "600999.SH", "trade_date": date, "adj_factor": 1.0})
    return build_market_tables(pd.DataFrame(daily), pd.DataFrame(adj))


def _limits(dates, *, up_locked=None, down_locked=None):
    up_locked = set(up_locked or [])
    down_locked = set(down_locked or [])
    out = {}
    for date in dates:
        # 默认涨跌停价远离真实价；锁定测试会在下方配合改写行情 OHLC
        out[date] = pd.DataFrame([
            {"ts_code": "600001.SH", "up_limit": 99.0, "down_limit": 0.1},
            {"ts_code": "600999.SH", "up_limit": 99.0, "down_limit": 0.1},
        ])
    return out


def test_t_plus_one_entry_and_horizon_exit_use_next_opens():
    dates = ["20260102", "20260105", "20260106", "20260107"]
    trade_days, market = _market(dict(zip(dates, [10.0, 11.0, 12.0, 13.0])))
    outcome = evaluate_code_horizon("600001.SH", "20260102", 2, trade_days, market,
                                    _limits(dates), commission_rate=0.0, slippage_rate=0.0)
    assert outcome["entry_date"] == "20260105"  # 周五决策，周一开盘
    assert outcome["exit_date"] == "20260107"
    assert outcome["gross_return"] == pytest.approx(13.0 / 11.0 - 1.0)


def test_generated_after_next_open_defers_entry_to_following_trading_day():
    dates = ["20260101", "20260102", "20260105", "20260106"]
    trade_days, market = _market(dict(zip(dates, [9.0, 10.0, 11.0, 12.0])))
    outcome = evaluate_code_horizon(
        "600001.SH", "20260101", 1, trade_days, market, _limits(dates),
        generated_at="2026-01-02T10:00:00+08:00", commission_rate=0.0, slippage_rate=0.0)
    assert outcome["entry_date"] == "20260105"
    assert outcome["exit_date"] == "20260106"


def test_audit_requires_timezone_aware_generated_at():
    snap = _snapshot()
    snap["generated_at"] = "2026-01-01T18:00:00"
    audit = audit_snapshot(snap, "20260101", _factor())
    assert audit["valid"] is False
    assert "generated_at" in " ".join(audit["errors"])


def test_audit_rejects_impossible_dates_and_generation_before_eod_availability():
    impossible = _snapshot("20260230")
    audit = audit_snapshot(impossible, "20260230", _factor())
    assert audit["valid"] is False
    assert "真实 YYYYMMDD" in " ".join(audit["errors"])

    prior_day = _snapshot("20260102")
    prior_day["generated_at"] = "2026-01-01T23:00:00+08:00"
    audit = audit_snapshot(prior_day, "20260102", _factor())
    assert audit["valid"] is False
    assert "不能早于 as_of" in " ".join(audit["errors"])

    before_close = _snapshot("20260102")
    before_close["generated_at"] = "2026-01-02T14:59:59+08:00"
    audit = audit_snapshot(before_close, "20260102", _factor())
    assert audit["valid"] is False
    assert "早于 as_of 收盘" in " ".join(audit["errors"])


def test_entry_suspension_is_not_deferred():
    dates = ["20260101", "20260102", "20260105"]
    trade_days, market = _market(dict.fromkeys(dates, 10.0), missing={"20260102"})
    outcome = evaluate_code_horizon("600001.SH", "20260101", 1, trade_days, market, _limits(dates))
    assert outcome["outcome_status"] == "suspended_next_day"
    assert "entry_price" not in outcome


def test_one_word_limit_up_is_unfilled_and_not_deferred():
    dates = ["20260101", "20260102", "20260105"]
    trade_days, market = _market(dict.fromkeys(dates, 10.0))
    row = market["20260102"].loc[market["20260102"]["ts_code"] == "600001.SH",
                                 ["open", "high", "low", "close", "adj_open"]]
    row.loc[:, ["open", "high", "low", "close", "adj_open"]] = 11.0
    market["20260102"].loc[row.index, row.columns] = row
    limits = _limits(dates)
    limits["20260102"].loc[limits["20260102"]["ts_code"] == "600001.SH", "up_limit"] = 11.0
    outcome = evaluate_code_horizon("600001.SH", "20260101", 1, trade_days, market, limits)
    assert outcome["outcome_status"] == "one_word_limit_up"
    assert "entry_price" not in outcome


def test_exit_one_word_limit_down_defers_to_first_sellable_open():
    dates = ["20260101", "20260102", "20260105", "20260106"]
    trade_days, market = _market(dict(zip(dates, [9.0, 10.0, 8.0, 9.0])))
    locked = market["20260105"].loc[market["20260105"]["ts_code"] == "600001.SH",
                                    ["open", "high", "low", "close", "adj_open"]]
    locked.loc[:, ["open", "high", "low", "close", "adj_open"]] = 8.0
    market["20260105"].loc[locked.index, locked.columns] = locked
    limits = _limits(dates)
    limits["20260105"].loc[limits["20260105"]["ts_code"] == "600001.SH", "down_limit"] = 8.0
    outcome = evaluate_code_horizon("600001.SH", "20260101", 1, trade_days, market, limits,
                                    commission_rate=0.0, slippage_rate=0.0)
    assert outcome["target_exit_date"] == "20260105"
    assert outcome["exit_date"] == "20260106"
    assert outcome["exit_deferred_days"] == 1
    assert outcome["gross_return"] == pytest.approx(-0.1)


def test_immature_horizon_is_missing_not_zero():
    dates = ["20260101", "20260102"]
    trade_days, market = _market(dict.fromkeys(dates, 10.0))
    outcome = evaluate_code_horizon("600001.SH", "20260101", 5, trade_days, market, _limits(dates))
    assert outcome["outcome_status"] == "insufficient_maturity"
    assert "net_return" not in outcome


def test_max_entry_price_is_an_actual_fill_constraint():
    dates = ["20260101", "20260102", "20260105"]
    trade_days, market = _market({"20260101": 9.0, "20260102": 12.0, "20260105": 13.0})
    outcome = evaluate_code_horizon("600001.SH", "20260101", 1, trade_days, market,
                                    _limits(dates), max_entry_price=10.0)
    assert outcome["outcome_status"] == "above_max_entry_price"
    assert "net_return" not in outcome


def test_partial_ohlc_is_data_error_not_suspension():
    daily = pd.DataFrame([{
        "ts_code": "600001.SH", "trade_date": "20260102", "open": None,
        "high": 10.0, "low": 9.0, "close": 9.5,
    }])
    adj = pd.DataFrame([{
        "ts_code": "600001.SH", "trade_date": "20260102", "adj_factor": 1.0,
    }])
    with pytest.raises(DecisionEvaluationError, match="四价全空"):
        build_market_tables(daily, adj)


def test_missing_target_limit_row_fails_loud():
    dates = ["20260101", "20260102", "20260105"]
    trade_days, market = _market(dict.fromkeys(dates, 10.0))
    limits = _limits(dates)
    limits["20260102"] = limits["20260102"][limits["20260102"]["ts_code"] != "600001.SH"]
    with pytest.raises(DecisionEvaluationError, match="up_limit"):
        evaluate_code_horizon("600001.SH", "20260101", 1, trade_days, market, limits)


def test_unknown_boundary_marks_new_buy_left_censored():
    sequence = [
        _snapshot("20260101", [_decision(state="BUY")]),
        {"_unknown_boundary": True, "as_of": "20260102"},
        _snapshot("20260103", [_decision(state="BUY")]),
    ]
    episodes = extract_buy_episodes(sequence)
    assert [e["left_censored"] for e in episodes] == [True, True]


def test_known_trade_day_gap_between_snapshots_is_unknown_boundary():
    """codex P1-3:相邻快照之间隔着已知交易日却无快照文件 → 不得跨缺口合并。"""
    sequence = [
        _snapshot("20260101", [_decision(state="BUY")]),
        _snapshot("20260105", [_decision(state="BUY")]),
    ]
    # 无缺口信息(退回旧语义):20260105 视为同一 episode 的延续,不开新事件
    merged = extract_buy_episodes(sequence)
    assert [e["as_of"] for e in merged] == ["20260101"]
    # 已知 20260102 是交易日却无快照:缺口=unknown boundary,20260105 新开且左删失
    split = extract_buy_episodes(sequence, known_trade_days=["20260102"])
    assert [(e["as_of"], e["left_censored"]) for e in split] == [
        ("20260101", True), ("20260105", True)]
    # 缺口日不在快照区间内(如周末)时不触发
    outside = extract_buy_episodes(sequence, known_trade_days=["20251230", "20260106"])
    assert [e["as_of"] for e in outside] == ["20260101"]


def test_nw_requires_both_four_dates_and_requested_lag_support():
    events = []
    for i in range(3):
        events.append({
            "as_of": f"2026010{i + 1}",
            "outcomes": {"2": {
                "outcome_status": "resolved", "status": "fillable_next_open",
                "net_return": 0.01 + i * 0.001, "excess_vs_hs300": 0.0,
                "increment_vs_d10": 0.002 + i * 0.001,
            }},
        })
    metrics = aggregate_metrics(events, (2,))["2"]
    assert metrics["increment_nw_status"] == "insufficient_signal_dates_for_required_lag"
    assert metrics["increment_nw_lag"] == 1
    assert metrics["increment_nw_t"] is None


def test_evaluate_episodes_reports_d10_common_support():
    dates = ["20260101", "20260102", "20260105"]
    daily = []
    adj = []
    for code, prices in {"600001.SH": [9.0, 10.0, 12.0], "600002.SH": [9.0, 10.0, 11.0]}.items():
        for date, price in zip(dates, prices):
            daily.append({"ts_code": code, "trade_date": date, "open": price,
                          "high": price + 0.2, "low": price - 0.2, "close": price})
            adj.append({"ts_code": code, "trade_date": date, "adj_factor": 1.0})
    trade_days, market = build_market_tables(pd.DataFrame(daily), pd.DataFrame(adj))
    limits = {d: pd.DataFrame([
        {"ts_code": "600001.SH", "up_limit": 99.0, "down_limit": 0.1},
        {"ts_code": "600002.SH", "up_limit": 99.0, "down_limit": 0.1},
    ]) for d in dates}
    episodes = [{"episode_id": "x", "as_of": "20260101", "ts_code": "600001.SH",
                 "name": "甲", "actual_execution": False}]
    factors = {"20260101": [{"ts_code": "600001.SH", "decile": 10},
                            {"ts_code": "600002.SH", "decile": 10}]}
    events, metrics = evaluate_episodes(episodes, factors, trade_days, market, limits, (1,),
                                       commission_rate=0.0, slippage_rate=0.0)
    outcome = events[0]["outcomes"]["1"]
    assert outcome["d10_attempted_count"] == 2
    assert outcome["d10_resolved_count"] == 2
    # commission/slippage 设零后仍有卖出印花税 5bp（PIT 监管成本不能关闭）。
    assert outcome["d10_net_return"] == pytest.approx(0.1495)
    assert outcome["increment_vs_d10"] == pytest.approx(0.05)
    assert metrics["1"]["resolved_count"] == 1
    assert metrics["1"]["increment_nw_lag"] == 0
    assert metrics["1"]["increment_nw_status"] == "insufficient_signal_dates_for_required_lag"
