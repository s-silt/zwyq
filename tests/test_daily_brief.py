"""daily_brief 每日一屏测试:只读聚合、机器状态逐字读、股息叠加、退出码 0/1/2。"""
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import daily_brief as db

AS_OF = "20260818"


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def _decision(ts_code, state, *, decile=None, reason_codes=None,
              max_entry=None, shares=0, name=None) -> dict:
    evidence = {}
    if decile is not None:
        evidence["decile"] = decile
    execution = {"shares": shares, "max_entry_price": max_entry}
    if state == "BUY":
        execution["eligible_from"] = "NEXT_TRADING_DAY"
    return {
        "ts_code": ts_code, "name": name or ts_code, "state": state,
        "reason_codes": list(reason_codes or []), "evidence": evidence,
        "invalidations": [], "execution": execution,
    }


def _setup_root(tmp_path: Path, *, decisions, as_of=AS_OF, holdings=None,
                dv_rows=None, core_endpoints=("daily", "adj_factor", "stk_limit"),
                account_state=None) -> Path:
    _dump(tmp_path / "data/trading_policy.json", {
        "policy_version": "1", "target_positions": 10, "target_weight": 0.1,
        "industry_cap": 0.2, "lot_size": 100, "min_cash": 1000})
    _dump(tmp_path / "data/profile.json", {"as_of": as_of, "excluded_industries": []})
    _dump(tmp_path / "data/factcheck_overrides.json", {"overrides": []})
    _dump(tmp_path / "data/holdings.json",
          holdings if holdings is not None else {"as_of": as_of, "cash": 10000, "positions": []})
    for ep in core_endpoints:
        _parquet(tmp_path / f"data/cache/{ep}/{as_of}.parquet", [{"ts_code": "600000.SH"}])
    _parquet(tmp_path / f"data/cache/daily_basic/{as_of}.parquet",
             dv_rows if dv_rows is not None
             else [{"ts_code": "600000.SH", "dv_ttm": 4.2, "dv_ratio": 3.9}])
    _dump(tmp_path / f"data/holdscore/{as_of}_factor.json", [])
    _dump(tmp_path / f"data/decisions/{as_of}_buy_decisions.json", {
        "as_of": as_of, "data_status": "complete",
        "generated_at": "2026-08-18T17:40:00+08:00", "decisions": decisions})
    if account_state is not None:
        _dump(tmp_path / f"data/account_state/{as_of}_account_state.json", account_state)
    return tmp_path


def test_machine_states_read_verbatim_and_dividend_overlay(tmp_path: Path) -> None:
    decisions = [
        _decision("600000.SH", "BUY", decile=10, max_entry=12.3, shares=1000,
                  reason_codes=["D10", "TIER_GREEN", "FACTCHECK_CLEAR"]),
        _decision("000001.SZ", "WAIT", decile=10, reason_codes=["FACTCHECK_REQUIRED"]),
        _decision("600519.SH", "EXIT", reason_codes=["EXIT_RULE_RISK_LINE"]),
    ]
    root = _setup_root(tmp_path, decisions=decisions, dv_rows=[
        {"ts_code": "600000.SH", "dv_ttm": 4.2, "dv_ratio": 3.9},
        {"ts_code": "000001.SZ", "dv_ttm": 1.8, "dv_ratio": 1.5},
    ])
    brief = db.build_brief(root)

    # 状态计数逐字来自快照
    assert brief["decision_snapshot"]["state_counts"] == {"BUY": 1, "WAIT": 1, "HOLD": 0, "EXIT": 1}
    buy = brief["machine"]["buys"][0]
    assert buy["ts_code"] == "600000.SH" and buy["actionable"] is True
    assert buy["max_entry_price"] == 12.3 and buy["planned_shares"] == 1000
    assert buy["dv_ttm"] == 4.2  # 股息展示叠加挂到 BUY 行
    # 仅差 fact-check 候选仍是 WAIT(绝不提升为 BUY)
    pend = brief["machine"]["pending_factcheck"]
    assert [p["ts_code"] for p in pend] == ["000001.SZ"]
    assert pend[0]["dv_ttm"] == 1.8
    assert [e["ts_code"] for e in brief["machine"]["exits"]] == ["600519.SH"]
    assert brief["dividends"]["status"] == "OK"
    # 有 BUY/EXIT/待核查 → 需人工 → 退出码 2
    assert brief["exit_code"] == 2
    # 渲染不崩
    assert "每日简报" in db.render_text(brief)


def test_buy_missing_entry_price_forced_wait(tmp_path: Path) -> None:
    decisions = [
        _decision("600000.SH", "BUY", decile=10, max_entry=None, shares=0,
                  reason_codes=["D10", "TIER_GREEN", "FACTCHECK_CLEAR"]),
    ]
    root = _setup_root(tmp_path, decisions=decisions)
    brief = db.build_brief(root)
    buy = brief["machine"]["buys"][0]
    # 缺核验入场价 → _actionable_view 强制 WAIT,不可执行(不用现价补造)
    assert buy["actionable"] is False
    assert buy["user_action"] == "WAIT"
    assert buy["reason"] == "VERIFIED_ENTRY_PRICE_MISSING"
    # 没有可执行 BUY → 无 ⑤ 动作
    assert not any(a.startswith("⑤") for a in brief["next_actions"])


def test_core_eod_missing_forces_exit_1(tmp_path: Path) -> None:
    # 缺 stk_limit 核心端点 → readiness CORE_EOD_MISSING_OR_MISALIGNED → 退出码 1
    root = _setup_root(tmp_path, decisions=[_decision("600000.SH", "WAIT", decile=5)],
                       core_endpoints=("daily", "adj_factor"))
    brief = db.build_brief(root)
    assert "CORE_EOD_MISSING_OR_MISALIGNED" in brief["readiness"]["blockers"]
    assert brief["exit_code"] == 1


def test_dividend_unavailable_when_partition_degraded(tmp_path: Path) -> None:
    # daily_basic 分区存在(readiness 只看文件名)但无股息列 → 股息叠加 UNAVAILABLE,不崩
    decisions = [_decision("600000.SH", "BUY", decile=10, max_entry=9.9, shares=100,
                           reason_codes=["D10", "TIER_GREEN", "FACTCHECK_CLEAR"])]
    root = _setup_root(tmp_path, decisions=decisions,
                       dv_rows=[{"ts_code": "600000.SH", "pe": 10.0}])
    brief = db.build_brief(root)
    assert brief["dividends"]["status"] == "UNAVAILABLE"
    assert brief["machine"]["buys"][0]["dv_ttm"] is None


def test_invalid_snapshot_exit_1(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, decisions=[_decision("600000.SH", "WAIT", decile=5)])
    # 破坏快照契约:BUY 缺 D10/FACTCHECK_CLEAR
    _dump(root / f"data/decisions/{AS_OF}_buy_decisions.json", {
        "as_of": AS_OF, "data_status": "complete", "generated_at": "x",
        "decisions": [_decision("600000.SH", "BUY", decile=3, max_entry=1.0, shares=100)]})
    brief = db.build_brief(root)
    assert brief["decision_snapshot"]["status"] == "invalid"
    assert brief["exit_code"] == 1


def test_calm_day_exit_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 用 stub readiness 隔离退出码逻辑:全对齐、无 blocker、无机器待办 → 退出码 0
    root = _setup_root(tmp_path, decisions=[_decision("600000.SH", "WAIT", decile=5)])

    def _ready(_root=None):
        return {
            "ok": True,
            "recommendation_readiness": {
                "ready": True, "status": "ready", "blockers": [], "warnings": [],
                "as_of": AS_OF,
                "components": {"eod": {"status": "ready", "as_of": AS_OF},
                               "holdings": {"freshness": "aligned", "as_of": AS_OF}},
            },
        }

    monkeypatch.setattr(db.svc, "healthcheck", _ready)
    # 门禁基线存在且新鲜,否则 ⑫ 会正确地报"证据未复核"(缺基线不当健康)
    _dump(root / "data/holdscore/gate_baseline.json",
          {"frozen_at": "2026-08-18T17:00:00+08:00", "factors": [], "composite": {}})
    brief = db.build_brief(root)
    assert brief["gate_evidence"]["status"] == "FRESH"
    assert brief["next_actions"] == []
    assert brief["exit_code"] == 0


def test_missing_gate_baseline_is_surfaced(tmp_path: Path) -> None:
    """门禁证据基线缺失/过期必须进待办——准入证据静默变旧是深读 R1 的头号缺口。"""
    root = _setup_root(tmp_path, decisions=[_decision("600000.SH", "WAIT", decile=5)])
    brief = db.build_brief(root)
    assert brief["gate_evidence"]["status"] == "MISSING"
    assert any(a.startswith("⑫") for a in brief["next_actions"])

    _dump(root / "data/holdscore/gate_baseline.json",
          {"frozen_at": "2025-01-01T17:00:00+08:00", "factors": [], "composite": {}})
    stale = db.build_brief(root)
    assert stale["gate_evidence"]["status"] == "STALE"
    assert stale["gate_evidence"]["days"] > 100


def test_stop_none_does_not_crash_eod_valuation():
    """对抗复核 P1:新建仓 stop=None 曾让 build_position_record 直接 TypeError,
    连累整轮 holdings_watch 产不出任何持仓的 EOD 估值。必须单字段降级并标 warn。"""
    from ashare_gauntlet.holdings import build_position_record

    rec = build_position_record(
        {"ts_code": "600001.SH", "name": "甲", "cost": 10.0, "stop": None,
         "shares": 100, "bucket": "long", "entry_date": "20260817"},
        close=10.5, pct_chg=1.0, qfq_closes=[10.0, 10.5], qfq_lows=[9.8, 10.1],
        as_of="20260818", trade_days=["20260817", "20260818"], window=20)
    assert rec["error"] is None                 # 不是整行失败
    assert rec["dist_stop_pct"] is None         # 只有这一个字段降级
    assert rec["stop_warn"] and "无止损警报" in rec["stop_warn"]
    assert rec["pnl_pct"] == 5.0 and rec["held_days"] == 2   # 其余照常算


def test_time_stop_checked_branch_with_fresh_snapshot(tmp_path: Path) -> None:
    """对抗复核 P2:acct_state_fresh=True 分支(CHECKED + 时间止损命中)此前零覆盖。"""
    decisions = [_decision("600000.SH", "WAIT", decile=5)]
    root = _setup_root(tmp_path, decisions=decisions, holdings={
        "as_of": AS_OF, "cash": 1000,
        "positions": [{"ts_code": "600000.SH", "name": "甲", "mv": 9000, "cost": 10.0,
                       "shares": 900, "industry": "电气设备", "bucket": "短线", "stop": 9.3}]})
    _dump(root / f"data/account_state/{AS_OF}_account_state.json", {
        "as_of": AS_OF, "data_status": "complete", "valuation": {"status": "complete"},
        "positions": [{"ts_code": "600000.SH", "name": "甲", "bucket": "短线", "shares": 900,
                       "cost": 10.0, "stop": 9.3, "close": 10.2, "pnl_pct": 2.0,
                       "dist_stop_pct": 9.7, "ma20": 10.0, "held_days": 12,
                       "stop_warn": None, "error": None}]})
    brief = db.build_brief(root)
    assert brief["time_stop_check"]["status"] == "CHECKED"
    assert [h["ts_code"] for h in brief["time_stops"]] == ["600000.SH"]
    assert brief["holdings_risk_source"] == "account_state_snapshot"
    # 短线仓不得显示 +25% 减半线(那是长线纪律)
    assert brief["holdings_risk"][0]["profit_take_line"] is None
    assert any(a.startswith("⑩") for a in brief["next_actions"])


def test_stale_snapshot_marks_not_checked(tmp_path: Path) -> None:
    """data_status 缺失=未知,不得当新鲜(对抗复核 P2);时间止损须显式 NOT_CHECKED。"""
    root = _setup_root(tmp_path, decisions=[_decision("600000.SH", "WAIT", decile=5)])
    _dump(root / f"data/account_state/{AS_OF}_account_state.json",
          {"as_of": AS_OF, "positions": []})      # 无 data_status
    brief = db.build_brief(root)
    assert brief["time_stop_check"]["status"] == "NOT_CHECKED"
