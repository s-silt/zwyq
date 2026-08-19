"""账本对账:捕捉 trade_journal --add 与 trade_record 双写造成的漂移(只读只报)。"""
from __future__ import annotations

import pytest

from ashare_gauntlet import ledger_audit as la


def _h(positions=(), closed=()):
    return {"as_of": "20260818", "cash": 100.0,
            "positions": list(positions), "closed": list(closed)}


def _t(code="600001.SH", **kw):
    base = {"code": code, "bucket": "长线", "entry_date": "20260810", "entry_px": 24.0,
            "shares": 600, "exit_date": "20260817", "exit_px": 26.5, "pnl_pct": 10.4,
            "hold_days": 3, "reason": "人工判断", "approx": False}
    base.update(kw)
    return base


def test_clean_ledger_has_no_findings():
    holdings = _h(positions=[{"ts_code": "000002.SZ", "shares": 500}],
                  closed=[{"ts_code": "600001.SH", "name": "甲"}])
    assert la.reconcile(holdings, {"trades": [_t()]}) == []


def test_held_and_closed_is_review_not_drift():
    """卖出后重新建仓是合法常态(closed=历史清仓记录),不得判成矛盾诱导删除真实历史。

    实测用户账本正是此情形:002204/000589 于 7 月清仓、其后以不同成本重新买入。
    """
    holdings = _h(positions=[{"ts_code": "600001.SH", "shares": 600}],
                  closed=[{"ts_code": "600001.SH", "name": "甲"}])
    out = la.reconcile(holdings, {"trades": []})
    hit = [f for f in out if f["issue"] == "HELD_AND_CLOSED"]
    assert hit and hit[0]["level"] == "REVIEW"
    assert "重新建仓" in hit[0]["detail"]


def test_exited_but_still_held_is_review():
    # trade_journal --add 写了平仓笔但 holdings 没减仓(或是部分减仓)→ 需人确认
    holdings = _h(positions=[{"ts_code": "600001.SH", "shares": 600}])
    out = la.reconcile(holdings, {"trades": [_t()]})
    hit = [f for f in out if f["issue"] == "EXITED_BUT_STILL_HELD"]
    assert hit and hit[0]["level"] == "REVIEW"


def test_closed_without_journal_is_review():
    holdings = _h(closed=[{"ts_code": "600001.SH", "name": "甲"}])
    out = la.reconcile(holdings, {"trades": []})
    assert any(f["issue"] == "CLOSED_WITHOUT_JOURNAL" for f in out)


def test_duplicate_trade_is_drift():
    out = la.reconcile(_h(), {"trades": [_t(), _t()]})
    assert any(f["issue"] == "DUPLICATE_TRADE" and f["level"] == "DRIFT" for f in out)
    # 同日不同腿(股数不同)不算重复——减半锁利+清仓是合法序列
    out2 = la.reconcile(_h(), {"trades": [_t(shares=600), _t(shares=400, exit_px=20.9)]})
    assert not any(f["issue"] == "DUPLICATE_TRADE" for f in out2)


def test_contract_gaps_flagged():
    out = la.reconcile(_h(), {"trades": [_t(pnl_pct=None), _t(code="2.SZ", shares=None)]})
    issues = {f["issue"] for f in out}
    assert "EXIT_WITHOUT_PNL" in issues and "EXIT_WITHOUT_SHARES" in issues
    assert all(f["level"] == "CONTRACT" for f in out if f["issue"].startswith("EXIT_WITHOUT"))


def test_open_position_without_journal_is_not_a_finding():
    # 还没卖过的持仓不该被报(常态)
    assert la.reconcile(_h(positions=[{"ts_code": "600001.SH", "shares": 600}]),
                        {"trades": []}) == []


def test_bad_structure_fails_loud():
    with pytest.raises(la.LedgerAuditError):
        la.reconcile({"positions": "x"}, {"trades": []})
    with pytest.raises(la.LedgerAuditError):
        la.reconcile(_h(), {"trades": None})


def test_summarize_counts_levels():
    holdings = _h(positions=[{"ts_code": "600001.SH", "shares": 600}],
                  closed=[{"ts_code": "600001.SH", "name": "甲"}])
    counts = la.summarize(la.reconcile(holdings, {"trades": [_t()]}))
    assert counts["REVIEW"] >= 1
