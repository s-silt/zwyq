"""eod_ops:状态对比判据 + 编排关键路径测试(run_step 打桩,无真子进程无网络)。"""
from __future__ import annotations

import json

import pytest

from scripts import eod_ops
from scripts.eod_ops import diff_alerts, pending_factcheck_set


def _snap(decisions):
    return {"decisions": decisions}


def test_pending_set_excludes_non_green():
    snap = _snap([
        {"ts_code": "1.SZ", "state": "WAIT", "reason_codes": ["FACTCHECK_REQUIRED"]},
        {"ts_code": "2.SZ", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED", "TIER_NOT_GREEN"]},
        {"ts_code": "3.SZ", "state": "HOLD", "reason_codes": ["HELD"]},
    ])
    assert pending_factcheck_set(snap) == {"1.SZ"}


def test_diff_alerts_buy_exit_c2_and_new_pending():
    prev = _snap([
        {"ts_code": "1.SZ", "name": "甲", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED"]},
    ])
    cur = _snap([
        {"ts_code": "1.SZ", "name": "甲", "state": "BUY",
         "reason_codes": ["D10", "TIER_GREEN", "FACTCHECK_CLEAR"],
         "execution": {"shares": 700, "max_entry_price": 14.2}},
        {"ts_code": "4.SH", "name": "乙", "state": "HOLD",
         "reason_codes": ["HELD", "EXIT_RULE_C2_MONTHLY"]},
        {"ts_code": "5.SH", "name": "丙", "state": "EXIT",
         "reason_codes": ["RISK_STOP"]},
        {"ts_code": "6.SZ", "name": "丁", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED"]},
    ])
    alerts = diff_alerts(prev, cur)
    joined = "\n".join(alerts)
    assert "BUY 出现: 甲 1.SZ 700股" in joined
    assert "C2 月度审视: 乙 4.SH" in joined
    assert "EXIT 信号: 丙 5.SH" in joined
    assert "新增待 fact-check 绿灯候选 1 只: 6.SZ" in joined
    assert "1.SZ" not in joined.split("新增待 fact-check")[1]   # 已 BUY 的不算新增待核


def test_main_detects_same_day_snapshot_overwrite(tmp_path, monkeypatch, capsys):
    """codex P1:同一交易日重跑覆盖同名快照,必须按内容 diff,不得因文件名相同漏报。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "decisions").mkdir(parents=True)
    path = tmp_path / "data" / "decisions" / "20260817_buy_decisions.json"
    old = {"as_of": "20260817", "decisions": [
        {"ts_code": "1.SZ", "name": "甲", "state": "WAIT",
         "reason_codes": ["FACTCHECK_AFTER_AS_OF"]}]}
    new = {"as_of": "20260817", "decisions": [
        {"ts_code": "1.SZ", "name": "甲", "state": "BUY",
         "reason_codes": ["D10", "TIER_GREEN", "FACTCHECK_CLEAR"],
         "execution": {"shares": 700, "max_entry_price": 14.2}}]}
    path.write_text(json.dumps(old), encoding="utf-8")

    def fake_run_step(args):
        # buy_list 步骤重写同名快照(模拟当日重跑覆盖)
        if "scripts.buy_list" in args:
            path.write_text(json.dumps(new), encoding="utf-8")
        return True, ""

    monkeypatch.setattr(eod_ops, "run_step", fake_run_step)
    with pytest.raises(SystemExit) as exc:
        eod_ops.main(["--skip-probe"])
    assert exc.value.code == 2
    assert "BUY 出现: 甲 1.SZ 700股" in capsys.readouterr().out


def test_diff_alerts_quiet_when_unchanged():
    snap = _snap([
        {"ts_code": "1.SZ", "name": "甲", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED"]},
        {"ts_code": "2.SZ", "name": "乙", "state": "HOLD", "reason_codes": ["HELD"]},
    ])
    assert diff_alerts(snap, snap) == []
