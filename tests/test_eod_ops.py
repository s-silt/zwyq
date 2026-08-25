"""eod_ops:状态对比判据 + 编排关键路径测试(run_step 打桩,无真子进程无网络)。"""
from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd
import pytest

from ashare_gauntlet.data.fetch import TokenExpiredError
from scripts import eod_ops, refresh
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
    monkeypatch.setattr(
        eod_ops, "run_step_code", lambda args, **kwargs: (0, '{"status":"NOT_DUE"}'),
    )
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


def _run_main_with_new_snapshot(
    tmp_path, monkeypatch, capsys, *, c2_code: int, c2_tail: str,
):
    monkeypatch.chdir(tmp_path)
    decision_dir = tmp_path / "data" / "decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "20260826_buy_decisions.json"
    calls: list[list[str]] = []

    def fake_run_step_code(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["-m", "scripts.buy_list"]:
            decision_path.write_text(
                json.dumps({"as_of": "20260826", "decisions": []}), encoding="utf-8",
            )
        if args[:2] == ["-m", "scripts.c2_review"]:
            return c2_code, c2_tail
        return 0, ""

    monkeypatch.setattr(eod_ops, "run_step_code", fake_run_step_code)
    with pytest.raises(SystemExit) as exc:
        eod_ops.main(["--skip-probe"])
    return exc.value.code, calls, capsys.readouterr().out, decision_path


def test_c2_action_is_alert_not_pipeline_failure(tmp_path, monkeypatch, capsys):
    code, calls, out, _ = _run_main_with_new_snapshot(
        tmp_path,
        monkeypatch,
        capsys,
        c2_code=2,
        c2_tail='{"status":"VALID","newly_exit_eligible":["A"]}',
    )

    assert code == 2
    assert "C2 退出资格成立" in out
    assert "newly_exit_eligible" in out
    assert "C2 月度审视数据失败" not in out
    assert calls[-1][:2] == ["-m", "scripts.c2_review"]


def test_c2_data_failure_returns_one(tmp_path, monkeypatch, capsys):
    code, _, out, _ = _run_main_with_new_snapshot(
        tmp_path,
        monkeypatch,
        capsys,
        c2_code=1,
        c2_tail='{"status":"REVIEW_BLOCKED_DATA"}',
    )

    assert code == 1
    assert "C2 月度审视数据失败" in out
    assert "REVIEW_BLOCKED_DATA" in out


def test_c2_not_due_keeps_calm_eod_success(tmp_path, monkeypatch, capsys):
    code, _, out, _ = _run_main_with_new_snapshot(
        tmp_path,
        monkeypatch,
        capsys,
        c2_code=0,
        c2_tail='{"status":"NOT_DUE"}',
    )

    assert code == 0
    assert "C2 退出资格成立" not in out
    assert "C2 月度审视数据失败" not in out
    assert "=== 无状态变化 ===" in out


def test_core_failure_does_not_invoke_c2_and_returns_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "decisions").mkdir(parents=True)
    calls: list[list[str]] = []

    def fail_refresh(args, **kwargs):
        calls.append(list(args))
        return 1, "refresh failed"

    monkeypatch.setattr(eod_ops, "run_step_code", fail_refresh)
    with pytest.raises(SystemExit) as exc:
        eod_ops.main(["--skip-probe"])

    assert exc.value.code == 1
    assert calls == [["-m", "scripts.refresh"]]
    assert all("scripts.c2_review" not in args for args in calls)


def test_refresh_token_expiry_stops_eod_before_downstream(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "decisions").mkdir(parents=True)
    monkeypatch.setattr(refresh, "tushare_pro", lambda: object())
    monkeypatch.setattr(
        refresh,
        "fetch_trade_cal",
        lambda *args, **kwargs: pd.DataFrame(
            {"cal_date": ["20260826"], "is_open": [1]},
        ),
    )
    monkeypatch.setattr(refresh, "trading_days_from_cal", lambda _cal: ["20260826"])
    monkeypatch.setattr(
        refresh,
        "fetch_market_day",
        lambda *args, **kwargs: (_ for _ in ()).throw(TokenExpiredError("expired")),
    )
    calls: list[list[str]] = []

    def run_with_refresh_main(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["-m", "scripts.refresh"]:
            try:
                refresh.main(
                    0,
                    str(tmp_path / "data/cache"),
                    today=date(2026, 8, 26),
                )
            except SystemExit as exc:
                return int(exc.code), "token 耗尽"
        return 0, ""

    monkeypatch.setattr(eod_ops, "run_step_code", run_with_refresh_main)

    with pytest.raises(SystemExit) as exc:
        eod_ops.main(["--skip-probe"])

    assert exc.value.code == 1
    assert calls == [["-m", "scripts.refresh"]]


def test_no_new_snapshot_does_not_invoke_c2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    decision_dir = tmp_path / "data" / "decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "20260826_buy_decisions.json"
    decision_path.write_text(
        json.dumps({"as_of": "20260826", "decisions": []}), encoding="utf-8",
    )
    calls: list[list[str]] = []

    def succeed_without_change(args, **kwargs):
        calls.append(list(args))
        return 0, ""

    monkeypatch.setattr(eod_ops, "run_step_code", succeed_without_change)
    with pytest.raises(SystemExit) as exc:
        eod_ops.main(["--skip-probe"])

    assert exc.value.code == 0
    assert calls == [
        ["-m", "scripts.refresh"],
        ["-m", "scripts.factor_rank"],
        ["-m", "scripts.buy_list"],
    ]


def test_same_content_atomic_snapshot_replacement_retries_c2_after_failure(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    decision_dir = tmp_path / "data" / "decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "20260826_buy_decisions.json"
    payload = json.dumps({"as_of": "20260826", "decisions": []})
    calls: list[list[str]] = []
    replacements = 0

    def replace_snapshot() -> None:
        nonlocal replacements
        replacements += 1
        temporary = decision_dir / f".tmp_buy_decisions_{replacements}.json"
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, decision_path)

    def fail_c2(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["-m", "scripts.buy_list"]:
            replace_snapshot()
        if args[:2] == ["-m", "scripts.c2_review"]:
            return 1, '{"status":"REVIEW_BLOCKED_DATA"}'
        return 0, ""

    monkeypatch.setattr(eod_ops, "run_step_code", fail_c2)
    exit_codes = []
    for _ in range(2):
        with pytest.raises(SystemExit) as exc:
            eod_ops.main(["--skip-probe"])
        exit_codes.append(exc.value.code)

    c2_calls = [args for args in calls if args[:2] == ["-m", "scripts.c2_review"]]
    assert exit_codes == [1, 1]
    assert len(c2_calls) == 2
    assert all(args[-1] == eod_ops.snapshot_paths()[-1] for args in c2_calls)


def test_run_step_code_timeout_reaps_process_before_joining_reader(monkeypatch):
    fake_threads = []

    class FakeProcess:
        def __init__(self):
            self.stdout = [f"line {i}\n" for i in range(10)]
            self.killed = False
            self.reaped = False
            self.returncode = None
            self.wait_calls = []

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                raise eod_ops.subprocess.TimeoutExpired("fake", timeout)
            self.reaped = True
            self.returncode = -9
            return self.returncode

        def kill(self):
            self.killed = True

    process = FakeProcess()

    class FakeThread:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon
            self.completed = False
            self.joined_after_reap = False
            fake_threads.append(self)

        def start(self):
            pass

        def join(self, timeout=None):
            self.joined_after_reap = process.reaped
            if process.reaped:
                self.target()
                self.completed = True

    monkeypatch.setattr(eod_ops.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(eod_ops.threading, "Thread", FakeThread)

    code, tail = eod_ops.run_step_code(["fake"], timeout=3, echo=False)

    assert code == 1
    assert process.killed is True
    assert process.reaped is True
    assert process.returncode == -9
    assert process.wait_calls == [3, None]
    assert len(fake_threads) == 1
    assert fake_threads[0].joined_after_reap is True
    assert fake_threads[0].completed is True
    assert tail.splitlines() == ["超时(>3s): line 7", "line 8", "line 9"]


def test_run_step_code_timeout_does_not_wait_forever_for_inherited_pipe(monkeypatch):
    join_calls = []

    class ReapedProcess:
        def __init__(self):
            self.stdout = []
            self.wait_calls = []

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                raise eod_ops.subprocess.TimeoutExpired("fake", timeout)
            return -9

        def kill(self):
            pass

    process = ReapedProcess()

    class PipeHeldOpenReader:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

        def join(self, timeout=None):
            join_calls.append(timeout)
            if timeout is None:
                raise AssertionError("unbounded reader join would wait forever for pipe EOF")

    monkeypatch.setattr(eod_ops.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(eod_ops.threading, "Thread", PipeHeldOpenReader)

    code, tail = eod_ops.run_step_code(["fake"], timeout=3, echo=False)

    assert code == 1
    assert tail.startswith("超时(>3s):")
    assert process.wait_calls == [3, None]
    assert join_calls == [5]


def test_c2_runs_before_factcheck_probe_with_each_step_once(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    decision_dir = tmp_path / "data" / "decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "20260826_buy_decisions.json"
    calls: list[list[str]] = []

    def succeed(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["-m", "scripts.buy_list"]:
            decision_path.write_text(json.dumps({
                "as_of": "20260826",
                "decisions": [{
                    "ts_code": "1.SZ",
                    "state": "WAIT",
                    "reason_codes": ["FACTCHECK_REQUIRED"],
                }],
            }), encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(eod_ops, "run_step_code", succeed)
    with pytest.raises(SystemExit) as exc:
        eod_ops.main([])

    assert exc.value.code == 2
    assert calls == [
        ["-m", "scripts.refresh"],
        ["-m", "scripts.factor_rank"],
        ["-m", "scripts.buy_list"],
        ["-m", "scripts.c2_review", "--decision", eod_ops.snapshot_paths()[-1]],
        ["-m", "scripts.factcheck_probe", "--codes", "1.SZ"],
    ]


def test_c2_uses_new_snapshot_after_core_once_without_rerunning_buy_list(
    tmp_path, monkeypatch, capsys,
):
    code, calls, _, _ = _run_main_with_new_snapshot(
        tmp_path,
        monkeypatch,
        capsys,
        c2_code=0,
        c2_tail='{"status":"NOT_DUE"}',
    )

    assert code == 0
    assert calls == [
        ["-m", "scripts.refresh"],
        ["-m", "scripts.factor_rank"],
        ["-m", "scripts.buy_list"],
        ["-m", "scripts.c2_review", "--decision", eod_ops.snapshot_paths()[-1]],
    ]
    assert sum("scripts.buy_list" in args for args in calls) == 1


def test_run_step_code_preserves_real_exit_code():
    code, tail = eod_ops.run_step_code(
        ["-c", "import sys; print('ACTION'); sys.exit(2)"], echo=False,
    )
    assert code == 2
    assert "ACTION" in tail
