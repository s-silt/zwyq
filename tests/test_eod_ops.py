"""eod_ops:状态对比判据 + 编排关键路径测试(run_step 打桩,无真子进程无网络)。"""
from __future__ import annotations

import json

import pytest

from scripts import eod_ops
from scripts.eod_ops import diff_alerts, pending_factcheck_set, run_step


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
    # C2 是"连续 2 个有效月度审视仍在档外才退出"的跨期规则,日频快照既无 streak 也
    # 不知今天算不算审视日 → 只报**集合新增**,不逐日催"待退出"(跨层审计整改)
    assert "C2 观察新增 1 只: 乙 4.SH" in joined
    assert "非今日待办" in joined
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


def test_write_alerts_persists_action_state(tmp_path):
    alert_dir = str(tmp_path / "data" / "alerts")
    path = eod_ops.write_alerts("20260817", ["BUY 出现: 甲 1.SZ"], alert_dir=alert_dir)
    assert path is not None
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["as_of"] == "20260817"
    assert payload["status"] == "action_required"
    assert payload["alert_count"] == 1
    assert payload["alerts"] == ["BUY 出现: 甲 1.SZ"]


def test_write_alerts_calm_and_skips_without_as_of(tmp_path):
    alert_dir = str(tmp_path / "data" / "alerts")
    path = eod_ops.write_alerts("20260817", [], alert_dir=alert_dir)
    assert json.loads(open(path, encoding="utf-8").read())["status"] == "calm"
    # 无 as_of 锚点 → 不落盘(返回 None,不伪造文件名)
    assert eod_ops.write_alerts(None, ["x"], alert_dir=alert_dir) is None


def test_run_step_echoes_live_and_keeps_tail(capsys):
    """★进度必须实时回显:此前 capture_output 全吞,factor_rank 算几分钟屏幕无动静,
    被当成死机而中断(中断会留半成品产物)。同时尾部仍要留给 ALERT 判定。"""
    ok, tail = run_step(["-c", "print('A'); print('B')"])
    assert ok is True
    assert "A" in tail and "B" in tail
    out = capsys.readouterr().out
    assert "| A" in out and "| B" in out          # 逐行直通到终端


def test_run_step_can_silence_echo(capsys):
    ok, tail = run_step(["-c", "print('QUIET')"], echo=False)
    assert ok is True and "QUIET" in tail
    assert "QUIET" not in capsys.readouterr().out


def test_run_step_failure_keeps_stderr_in_tail():
    """stderr 合并进 stdout:失败原因必须进 ALERT,不能只剩退出码。"""
    ok, tail = run_step(
        ["-c", "import sys; sys.stderr.write('BOOM\n'); sys.exit(3)"])
    assert ok is False and "BOOM" in tail


def test_run_step_timeout_kills_and_labels():
    """子进程长时间不输出时,阻塞读不得耽误计时(独立读取线程 + wait 守超时)。"""
    ok, tail = run_step(
        ["-c", "import time; print('start', flush=True); time.sleep(30)"], timeout=1)
    assert ok is False and tail.startswith("超时(>1s)")
    assert "start" in tail


def test_run_step_tail_is_bounded():
    """尾部有上限,单步刷屏不会灌爆 ALERT。"""
    ok, tail = run_step(["-c", "\n".join(f"print({i})" for i in range(50))])
    assert ok is True
    assert len(tail.splitlines()) == eod_ops.TAIL_LINES
    assert tail.splitlines()[-1] == "49"          # 留的是最后几行
