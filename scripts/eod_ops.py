"""每日 EOD 管线:refresh → factor_rank → buy_list → c2_review → factcheck_probe。

设计边界:
- 本脚本只做编排与状态对比,不含任何研究口径;各步失败 fail-loud 并入 ALERT,
  绝不把失败解释为"无事发生"(CLAUDE.md:缺失/退化必须显式失败)。
- buy_list 的账户日期门禁(require_account_as_of)是有意的人工环节:账户 as_of
  未经 holdings_confirm 推进时,快照生成失败属于**正常提醒**,不是管线故障。
- 状态对比只读 data/decisions 冻结快照;alert 判据(定义性,零参数):
  ①出现 BUY;②出现 EXIT;③C2 观察集合(EXIT_RULE_C2_MONTHLY)有新增;
  ④"仅差 fact-check"绿灯候选集合有新增;⑤任一管线步骤失败。
- C2 只报集合新增、不逐日重报:它是"连续 2 个有效月度审视仍在档外才退出"的规则
  (methodology §10 M3),日频快照既不累计 streak 也不标审视日;逐日催"待退出"
  等于把 C2 退化成立即退出变体,而 C2 的收益优势正来自换手 41%→22%。

Usage: E:\\zwyq\\.venv\\Scripts\\python.exe -m scripts.eod_ops [--skip-probe]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import deque
from datetime import datetime, timedelta, timezone

from ashare_gauntlet.candidates import HARD_VETO_CODES

DECISION_DIR = "data/decisions"
ALERT_DIR = "data/alerts"

STEPS: tuple[tuple[str, list[str]], ...] = (
    ("refresh", ["-m", "scripts.refresh"]),
    ("factor_rank", ["-m", "scripts.factor_rank"]),
    ("buy_list", ["-m", "scripts.buy_list"]),
)


STEP_TIMEOUT = 1800
TAIL_LINES = 6      # 进 ALERT 的输出尾行数(失败定位够用,不灌爆提醒)


def run_step_code(args: list[str], *, timeout: int = STEP_TIMEOUT,
                  echo: bool = True) -> tuple[int, str]:
    """跑一步管线;返回 (退出码, 输出尾部)。失败/超时原文进 ALERT(codex P1)。

    **边跑边回显**(echo=True):此前用 capture_output 把子进程输出全吞了,factor_rank
    这种要算几分钟的步骤在屏幕上毫无动静,实测被当成死机而中断——中断会留下半成品
    产物。这里改 Popen 逐行读:既实时打印,又把尾部留给 ALERT 判定。
    stderr 合并进 stdout,保持两者的真实先后顺序(分开取再拼接会错乱因果)。
    超时用独立读取线程 + wait(timeout) 守:子进程长时间不输出时,阻塞读不会耽误计时。
    """
    proc = subprocess.Popen(
        [sys.executable, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    tail: deque[str] = deque(maxlen=TAIL_LINES)

    def _pump() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.rstrip("\n")
            tail.append(text)
            if echo:
                print(f"  | {text}", flush=True)

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader.join(timeout=5)
        return 1, f"超时(>{timeout}s): " + "\n".join(list(tail)[-3:])
    reader.join(timeout=5)
    return code, "\n".join(tail)


def run_step(args: list[str], *, timeout: int = STEP_TIMEOUT,
             echo: bool = True) -> tuple[bool, str]:
    """兼容旧调用方,返回 (成功, 输出尾部)。"""
    code, tail = run_step_code(args, timeout=timeout, echo=echo)
    return code == 0, tail


def snapshot_paths(decision_dir: str = DECISION_DIR) -> list[str]:
    files = sorted(f for f in os.listdir(decision_dir)
                   if re.fullmatch(r"\d{8}_buy_decisions\.json", f))
    return [os.path.join(decision_dir, f) for f in files]


def pending_factcheck_set(snapshot: dict) -> set[str]:
    """"唯一未决项=fact-check"的 WAIT 集合(判据单一来源=candidates.HARD_VETO_CODES)。

    此前手写三码排除表,漏了 SPEC_CROWD/SPIKE_LIMIT——那两码写 clear 也解不开,
    把🎰/⚡票当候选会白跑 probe 并误导用户(跨层审计)。
    """
    out: set[str] = set()
    for d in snapshot.get("decisions", []):
        codes = set(d.get("reason_codes", []))
        if (d.get("state") == "WAIT" and "FACTCHECK_REQUIRED" in codes
                and not codes & HARD_VETO_CODES):
            out.add(str(d["ts_code"]))
    return out


def c2_watch_set(snapshot: "dict | None") -> set[str]:
    """C2 观察集合:跌出 D10 但按规则不当日退出的持仓(EXIT_RULE_C2_MONTHLY)。

    只做集合,不做"该不该退"的判断——退出要连续 2 个**有效月度审视**仍在档外,
    而日频快照既不含 streak 也不知道今天算不算审视日(methodology §10 M3)。
    """
    return {str(d["ts_code"]) for d in (snapshot or {}).get("decisions", [])
            if "EXIT_RULE_C2_MONTHLY" in d.get("reason_codes", [])}


def diff_alerts(prev: "dict | None", cur: dict) -> list[str]:
    """两份快照间的 alert 行(见模块 docstring 判据;prev=None 时只报当前状态)。"""
    alerts: list[str] = []
    for d in cur.get("decisions", []):
        code, name = str(d["ts_code"]), str(d.get("name", d["ts_code"]))
        if d.get("state") == "BUY":
            ex = d.get("execution", {})
            alerts.append(f"BUY 出现: {name} {code} {ex.get('shares', 0)}股 "
                          f"max_entry={ex.get('max_entry_price')}")
        if d.get("state") == "EXIT":
            alerts.append(f"EXIT 信号: {name} {code} "
                          f"{'/'.join(d.get('reason_codes', []))}")
    new_c2 = c2_watch_set(cur) - c2_watch_set(prev)
    if new_c2:
        names = {str(d["ts_code"]): str(d.get("name", d["ts_code"]))
                 for d in cur.get("decisions", [])}
        alerts.append(f"C2 观察新增 {len(new_c2)} 只: "
                      + ",".join(f"{names.get(c, c)} {c}" for c in sorted(new_c2))
                      + "(跌出 D10;退出需连续 2 个有效月度审视仍在档外,非今日待办)")
    new_pending = pending_factcheck_set(cur) - (pending_factcheck_set(prev) if prev else set())
    if new_pending:
        alerts.append(f"新增待 fact-check 绿灯候选 {len(new_pending)} 只: "
                      + ",".join(sorted(new_pending)))
    return alerts


def write_alerts(as_of: "str | None", alerts: list[str], *,
                 alert_dir: str = ALERT_DIR, now: "datetime | None" = None) -> "str | None":
    """把本次运行状态落 data/alerts/<as_of>_alerts.json(持久化:防 17:30 会话丢失 alert)。

    原子替换(tmp+fsync+os.replace),失败向上抛由调用方决定;as_of 缺失则不写(无锚点)。
    这是新增可观测性产物,不参与任何门禁,也不改变退出码语义。
    """
    if not as_of:
        return None
    now = now or datetime.now(timezone(timedelta(hours=8)))
    payload = {
        "as_of": as_of,
        "generated_at": now.isoformat(),
        "status": "action_required" if alerts else "calm",
        "alert_count": len(alerts),
        "alerts": list(alerts),
    }
    os.makedirs(alert_dir, exist_ok=True)
    path = os.path.join(alert_dir, f"{as_of}_alerts.json")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix=".tmp_alerts_", dir=alert_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def main(argv: "list[str] | None" = None) -> None:
    ap = argparse.ArgumentParser(description="每日 EOD 编排:管线+状态变化提醒")
    ap.add_argument("--skip-probe", action="store_true",
                    help="跳过对新增候选的 factcheck_probe(证据抓取)")
    a = ap.parse_args(argv)

    before = snapshot_paths()
    prev_snapshot = None
    if before:
        prev_snapshot = json.load(open(before[-1], encoding="utf-8"))

    alerts: list[str] = []
    core_succeeded = True
    for name, args in STEPS:
        ok, tail = run_step(args)
        print(f"[{name}] {'ok' if ok else 'FAILED'}")
        if not ok:
            if name == "buy_list" and ("account" in tail or "as_of" in tail):
                alerts.append(f"{name} 失败(多半是账户 as_of 未确认,先跑 holdings_confirm): {tail}")
            else:
                alerts.append(f"{name} 失败: {tail}")
            core_succeeded = False
            break   # 后续步骤依赖前置产物,断链即停(不在残缺数据上继续)

    # 按内容而非文件名判断快照更新:同一交易日重跑 buy_list 会覆盖同名文件,
    # 比较路径会漏报当次新产生的 BUY/EXIT/候选变化(codex P1)
    after = snapshot_paths()
    new_snapshot = json.load(open(after[-1], encoding="utf-8")) if after else None
    snapshot_changed = new_snapshot is not None and new_snapshot != prev_snapshot
    c2_failed = False
    if core_succeeded and snapshot_changed:
        code, tail = run_step_code([
            "-m", "scripts.c2_review", "--decision", after[-1],
        ])
        if code == 0:
            print("[c2_review] ok")
        elif code == 2:
            print("[c2_review] ACTION")
            alerts.append(f"C2 退出资格成立: {tail}")
        else:
            print("[c2_review] FAILED")
            alerts.append(f"C2 月度审视数据失败(退出码 {code}): {tail}")
            c2_failed = True

    if snapshot_changed:
        alerts.extend(diff_alerts(prev_snapshot, new_snapshot))
        pending_new = (pending_factcheck_set(new_snapshot)
                       - (pending_factcheck_set(prev_snapshot) if prev_snapshot else set()))
        if pending_new and not a.skip_probe:
            ok, tail = run_step(["-m", "scripts.factcheck_probe",
                                 "--codes", ",".join(sorted(pending_new))])
            print(f"[factcheck_probe] {'ok' if ok else 'FAILED'}")
            if not ok:
                alerts.append(f"factcheck_probe 失败: {tail}")
    elif not alerts:
        print("无新快照(数据未更新或今日非交易日)")

    # C2 观察名单按状态行打印(不进 alerts、不改退出码):集合没变就不该催人行动,
    # 但也不能让它彻底隐身——把它推进到"退出"的资格只属于月度审视例行 + 人工终判
    watching = c2_watch_set(new_snapshot)
    if watching:
        print(f"[C2 观察] {len(watching)} 只跌出 D10 待月度审视: {','.join(sorted(watching))}")

    print("\n=== ALERT ===" if alerts else "\n=== 无状态变化 ===")
    for line in alerts:
        print(f"! {line}")

    # 持久化本次状态到 data/alerts/<as_of>_alerts.json(新增可观测性;不改退出码语义)
    as_of = None
    for snap in (new_snapshot, prev_snapshot):
        if isinstance(snap, dict) and snap.get("as_of"):
            as_of = str(snap["as_of"])
            break
    try:
        alert_path = write_alerts(as_of, alerts)
        if alert_path:
            print(f"[alerts] 已落盘 {alert_path}")
    except OSError as exc:
        print(f"[alerts] 落盘失败(不影响本次判定): {exc}", file=sys.stderr)

    # 退出码语义:1=数据失败,2=有需人工处理的状态变化,0=平静
    raise SystemExit(1 if c2_failed else (2 if alerts else 0))


if __name__ == "__main__":
    main()
