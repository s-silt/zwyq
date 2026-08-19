"""收盘编排:人工签字不得被自动化掉,失败即停。"""
from __future__ import annotations

import io
import json

import pytest

from scripts import eod_close as ec


def test_date_must_be_real_yyyymmdd():
    for bad in ("2026819", "20261332", "", "abcdefgh"):
        with pytest.raises(SystemExit):
            ec.main([bad, "--skip-eod"])


def test_non_tty_refuses_to_sign_for_user(capsys):
    """定时任务/管道/智能体不得代人推进 as_of——这是账本的人工签字环节。"""
    assert ec.confirm_prompt("20260819", "20260818", is_tty=False) is False
    err = capsys.readouterr().err
    assert "不得代人签字" in err
    assert "holdings_confirm 20260819" in err


def test_tty_requires_exact_date_as_signature():
    # 逐字输入目标日期才算签字
    assert ec.confirm_prompt("20260819", "20260818",
                             stream=io.StringIO("20260819\n"), is_tty=True) is True
    # 其他输入一律取消(含 yes / 空 / 错日期)
    for answer in ("yes\n", "\n", "20260818\n", "y\n"):
        assert ec.confirm_prompt("20260819", "20260818",
                                 stream=io.StringIO(answer), is_tty=True) is False


def test_current_as_of_reads_without_guessing(tmp_path):
    p = tmp_path / "holdings.json"
    p.write_text(json.dumps({"as_of": "20260818", "positions": []}), encoding="utf-8")
    assert ec.current_as_of(str(p)) == "20260818"
    # 文件缺失/坏 JSON/无 as_of → None(不猜)
    assert ec.current_as_of(str(tmp_path / "nope.json")) is None
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert ec.current_as_of(str(tmp_path / "bad.json")) is None
    (tmp_path / "noas.json").write_text(json.dumps({"positions": []}), encoding="utf-8")
    assert ec.current_as_of(str(tmp_path / "noas.json")) is None


def test_aligned_as_of_skips_signature(monkeypatch, capsys):
    """as_of 已对齐 → 无需签字(也就不该在非 TTY 下被拒),其余步骤照跑。"""
    monkeypatch.setattr(ec, "current_as_of", lambda *a, **k: "20260819")
    calls = []

    def fake_run(module, args=None):
        calls.append(module)
        return 0

    monkeypatch.setattr(ec, "_run", fake_run)
    monkeypatch.setattr(ec, "confirm_prompt",
                        lambda *a, **k: pytest.fail("as_of 已对齐不该要签字"))
    with pytest.raises(SystemExit) as exc:
        ec.main(["20260819", "--skip-eod"])
    assert exc.value.code == 0
    assert calls == ["scripts.holdings_watch", "scripts.daily_brief"]


def test_refresh_failure_stops_before_touching_ledger(monkeypatch):
    """refresh 失败必须在碰账本前停——不在残缺行情上继续。"""
    def fake_run(module, args=None):
        return 1 if module == "scripts.refresh" else 0

    monkeypatch.setattr(ec, "_run", fake_run)
    monkeypatch.setattr(ec, "confirm_prompt",
                        lambda *a, **k: pytest.fail("失败后不该走到签字"))
    with pytest.raises(SystemExit) as exc:
        ec.main(["20260819"])
    assert "refresh 失败" in str(exc.value.code)


def test_eod_exit_code_2_is_not_a_failure(monkeypatch):
    """eod_ops 退出码 2 = 有状态变化(正常),不得当成管线故障中断收盘。"""
    monkeypatch.setattr(ec, "current_as_of", lambda *a, **k: "20260819")
    seen = []

    def fake_run(module, args=None):
        seen.append(module)
        return 2 if module == "scripts.eod_ops" else 0

    monkeypatch.setattr(ec, "_run", fake_run)
    with pytest.raises(SystemExit) as exc:
        ec.main(["20260819"])
    assert exc.value.code == 0
    assert "scripts.daily_brief" in seen


def test_cancelled_signature_leaves_ledger_untouched(monkeypatch):
    monkeypatch.setattr(ec, "current_as_of", lambda *a, **k: "20260818")
    monkeypatch.setattr(ec, "confirm_prompt", lambda *a, **k: False)
    calls = []
    monkeypatch.setattr(ec, "_run", lambda m, args=None: calls.append(m) or 0)
    with pytest.raises(SystemExit) as exc:
        ec.main(["20260819", "--skip-eod"])
    assert exc.value.code == 2
    assert calls == []      # 取消后一步都不跑,账本零改动


def test_daily_brief_exit_code_passes_through(monkeypatch):
    """一屏的 0/2/1 语义必须直通,否则调度器读不出"今天要不要人管"。"""
    monkeypatch.setattr(ec, "current_as_of", lambda *a, **k: "20260819")
    for brief_code in (0, 1, 2):
        monkeypatch.setattr(
            ec, "_run",
            lambda m, args=None, c=brief_code: c if m == "scripts.daily_brief" else 0)
        with pytest.raises(SystemExit) as exc:
            ec.main(["20260819", "--skip-eod"])
        assert exc.value.code == brief_code


def test_empty_input_gives_manual_hint(capsys):
    """isatty 在本平台不可靠(Git Bash `< /dev/null` 报 True):读到 EOF/空行时
    必须也走"未签字"并给出手动指引,不能只依赖 isatty 一道防线。"""
    assert ec.confirm_prompt("20260819", "20260818",
                             stream=io.StringIO(""), is_tty=True) is False
    err = capsys.readouterr().err
    assert "未收到签字输入" in err and "holdings_confirm 20260819" in err


def test_piped_date_cannot_sign_without_tty(capsys):
    """管道喂正确日期(自动化最可能的形态)也不得签字——两道防线的第一道。"""
    assert ec.confirm_prompt("20260819", "20260818",
                             stream=io.StringIO("20260819\n"), is_tty=False) is False
    assert "不得代人签字" in capsys.readouterr().err


def test_step_order_refresh_then_sign_then_decide(monkeypatch):
    """★顺序回归锁(实测死锁):holdings_confirm 要求目标日是已知交易日 → 刷新必须
    在签字前;buy_list 要求账户 as_of == 行情日 → 决策必须在签字后。把整条 eod_ops
    放在签字前会让 buy_list 卡在账户门禁、管线 break、连签字都走不到。
    """
    monkeypatch.setattr(ec, "current_as_of", lambda *a, **k: "20260818")
    seq = []
    monkeypatch.setattr(ec, "_run", lambda m, args=None: seq.append(m) or 0)
    monkeypatch.setattr(ec, "confirm_prompt",
                        lambda *a, **k: seq.append("<签字>") or True)
    with pytest.raises(SystemExit):
        ec.main(["20260819"])
    assert seq == ["scripts.refresh", "<签字>", "scripts.holdings_confirm",
                   "scripts.eod_ops", "scripts.holdings_watch", "scripts.daily_brief"]
    # 决策管线必须在签字之后(否则撞 buy_list 的账户日期门禁)
    assert seq.index("scripts.eod_ops") > seq.index("<签字>")
    # 刷新必须在签字之前(否则 holdings_confirm 认不出目标交易日)
    assert seq.index("scripts.refresh") < seq.index("<签字>")
