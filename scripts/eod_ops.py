"""每日 EOD 运维管线:refresh → factor_rank → buy_list → factcheck_probe,只报状态变化。

设计边界:
- 本脚本只做编排与状态对比,不含任何研究口径;各步失败 fail-loud 并入 ALERT,
  绝不把失败解释为"无事发生"(CLAUDE.md:缺失/退化必须显式失败)。
- buy_list 的账户日期门禁(require_account_as_of)是有意的人工环节:账户 as_of
  未经 holdings_confirm 推进时,快照生成失败属于**正常提醒**,不是管线故障。
- 状态对比只读 data/decisions 冻结快照;alert 判据(定义性,零参数):
  ①出现 BUY;②出现 EXIT 或 C2 审视(EXIT_RULE_C2_MONTHLY);③"仅差 fact-check"
  绿灯候选集合有新增;④任一管线步骤失败。

Usage: E:\\zwyq\\.venv\\Scripts\\python.exe -m scripts.eod_ops [--skip-probe]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

DECISION_DIR = "data/decisions"

STEPS: tuple[tuple[str, list[str]], ...] = (
    ("refresh", ["-m", "scripts.refresh"]),
    ("factor_rank", ["-m", "scripts.factor_rank"]),
    ("buy_list", ["-m", "scripts.buy_list"]),
)


def run_step(args: list[str]) -> tuple[bool, str]:
    """跑一步管线;返回 (成功, 输出尾部)。不吞错——失败原文进 ALERT。"""
    proc = subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    tail = "\n".join(((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
                     .splitlines()[-6:])
    return proc.returncode == 0, tail


def snapshot_paths(decision_dir: str = DECISION_DIR) -> list[str]:
    files = sorted(f for f in os.listdir(decision_dir)
                   if re.fullmatch(r"\d{8}_buy_decisions\.json", f))
    return [os.path.join(decision_dir, f) for f in files]


def pending_factcheck_set(snapshot: dict) -> set[str]:
    """"仅差 fact-check"的绿灯 WAIT 集合(与 factcheck_probe.select_candidates 同判据)。"""
    out: set[str] = set()
    for d in snapshot.get("decisions", []):
        codes = set(d.get("reason_codes", []))
        if (d.get("state") == "WAIT" and "FACTCHECK_REQUIRED" in codes
                and not codes & {"TIER_NOT_GREEN", "GOVERNANCE_RED",
                                 "POLLUTION_PENDING_FACTCHECK"}):
            out.add(str(d["ts_code"]))
    return out


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
        if "EXIT_RULE_C2_MONTHLY" in d.get("reason_codes", []):
            alerts.append(f"C2 月度审视: {name} {code}(掉出 D10,按规则待退出)")
    new_pending = pending_factcheck_set(cur) - (pending_factcheck_set(prev) if prev else set())
    if new_pending:
        alerts.append(f"新增待 fact-check 绿灯候选 {len(new_pending)} 只: "
                      + ",".join(sorted(new_pending)))
    return alerts


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
    for name, args in STEPS:
        ok, tail = run_step(args)
        print(f"[{name}] {'ok' if ok else 'FAILED'}")
        if not ok:
            if name == "buy_list" and ("account" in tail or "as_of" in tail):
                alerts.append(f"{name} 失败(多半是账户 as_of 未确认,先跑 holdings_confirm): {tail}")
            else:
                alerts.append(f"{name} 失败: {tail}")
            break   # 后续步骤依赖前置产物,断链即停(不在残缺数据上继续)

    after = snapshot_paths()
    new_snapshot = None
    if after and (not before or after[-1] != before[-1]):
        new_snapshot = json.load(open(after[-1], encoding="utf-8"))
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

    print("\n=== ALERT ===" if alerts else "\n=== 无状态变化 ===")
    for line in alerts:
        print(f"! {line}")
    # 退出码语义:2=有需人工处理的状态变化(供调度器/通知层判断),0=平静
    raise SystemExit(2 if alerts else 0)


if __name__ == "__main__":
    main()
