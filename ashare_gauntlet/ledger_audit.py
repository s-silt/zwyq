"""账本对账(纯函数,只读):交叉核对 holdings 与 trade_journal 是否自洽。

补的缺口(深读 R5):journal 存在**双写路径**——`trade_record` 写 journal+holdings 且
带护栏,而 `trade_journal --add` 只写 journal、不动 holdings、无一致性校验。混用会造成
"journal 有平仓笔而 holdings 未减仓"的漂移,**目前无人对账**。

边界:只读、只 surface,绝不自动修账。判定刻意保守——A股允许"卖出后重新建仓"和
"部分减仓后继续持有",故这类组合标 REVIEW(需人看)而非 DRIFT(确定不一致),
未知不解释为安全,也不伪造确定性。

import 阶段无 I/O 副作用。
"""
from __future__ import annotations

from typing import Any, Iterable

# 判定等级:DRIFT=账本自相矛盾(必须处理);REVIEW=可由合法操作解释但需人确认;
# CONTRACT=journal 行本身不合契约(统计会受污染)
LEVELS = ("DRIFT", "REVIEW", "CONTRACT")


class LedgerAuditError(ValueError):
    """输入结构非法——不猜不静默。"""


def _codes(rows: Iterable[Any], key: str) -> list[str]:
    out = []
    for r in rows:
        if isinstance(r, dict) and r.get(key):
            out.append(str(r[key]))
    return out


def reconcile(holdings: dict[str, Any], journal: dict[str, Any]) -> list[dict[str, Any]]:
    """返回发现的问题行(空列表=账本自洽)。每行 {level, code, issue, detail}。"""
    if not isinstance(holdings, dict) or not isinstance(holdings.get("positions"), list):
        raise LedgerAuditError("holdings 需含 positions 列表")
    if not isinstance(journal, dict) or not isinstance(journal.get("trades"), list):
        raise LedgerAuditError("journal 需含 trades 列表")

    positions = [p for p in holdings["positions"] if isinstance(p, dict)]
    closed = [c for c in holdings.get("closed", []) if isinstance(c, dict)]
    trades = [t for t in journal["trades"] if isinstance(t, dict)]

    held = set(_codes(positions, "ts_code"))
    closed_codes = set(_codes(closed, "ts_code"))
    findings: list[dict[str, Any]] = []

    # ① 同一 code 既在持仓又在 closed:**卖出后重新建仓是合法常态**(closed 是历史
    #    清仓记录,不是"当前不持有"的断言),故标 REVIEW 而非 DRIFT——实测用户账本
    #    正是此情形(002204/000589 7 月清仓后重新买入),误判为矛盾会诱导删除真实历史
    for code in sorted(held & closed_codes):
        findings.append({
            "level": "REVIEW", "code": code, "issue": "HELD_AND_CLOSED",
            "detail": "同时在 positions 与 closed 段——若是卖出后重新建仓属正常(closed 为"
                      "历史清仓记录);若是清仓后 positions 未删则需修正"})

    # ② journal 已平仓的 code 仍在持仓 —— 可能是 trade_journal --add 未减仓(漂移),
    #    也可能是部分减仓/卖出后重建仓(合法)。保守标 REVIEW。
    exited: dict[str, list[str]] = {}
    for t in trades:
        if t.get("exit_date") and t.get("code"):
            exited.setdefault(str(t["code"]), []).append(str(t["exit_date"]))
    for code in sorted(set(exited) & held):
        findings.append({
            "level": "REVIEW", "code": code, "issue": "EXITED_BUT_STILL_HELD",
            "detail": f"journal 有平仓笔(exit_date={','.join(sorted(exited[code]))})但仍在持仓"
                      "——若非部分减仓/卖后重建仓,则是 journal 与 holdings 漂移"})

    # ③ closed 段有但 journal 无任何平仓笔 —— 卖出未记流水,复盘统计缺样本
    for code in sorted(closed_codes - set(exited)):
        findings.append({
            "level": "REVIEW", "code": code, "issue": "CLOSED_WITHOUT_JOURNAL",
            "detail": "已在 closed 段但 journal 无平仓笔——该笔不进胜率/期望统计(复盘缺样本)"})

    # ④ journal 完全重复行(同 code+exit_date+shares+exit_px)——重复计入统计
    seen: set[tuple] = set()
    for t in trades:
        if not t.get("exit_date"):
            continue
        key = (str(t.get("code")), str(t.get("exit_date")),
               t.get("shares"), t.get("exit_px"))
        if key in seen:
            findings.append({
                "level": "DRIFT", "code": str(t.get("code")), "issue": "DUPLICATE_TRADE",
                "detail": f"journal 存在完全相同的平仓行(exit_date={t.get('exit_date')},"
                          f"{t.get('shares')}股@{t.get('exit_px')})——胜率/期望被重复计入"})
        seen.add(key)

    # ⑤ 已平仓行缺 pnl_pct —— stats 会静默跳过该笔(样本缺失而非报错)
    for t in trades:
        if t.get("exit_date") and t.get("pnl_pct") is None:
            findings.append({
                "level": "CONTRACT", "code": str(t.get("code")), "issue": "EXIT_WITHOUT_PNL",
                "detail": f"exit_date={t.get('exit_date')} 但无 pnl_pct"
                          "——trade_journal.stats 会跳过该笔,胜率/期望少一个样本"})

    # ⑥ 平仓行缺 shares —— shares 加权统计(expectancy_w)无法计入
    for t in trades:
        if t.get("exit_date") and not isinstance(t.get("shares"), int):
            findings.append({
                "level": "CONTRACT", "code": str(t.get("code")), "issue": "EXIT_WITHOUT_SHARES",
                "detail": f"exit_date={t.get('exit_date')} 但 shares 非整数"
                          "——不计入 shares 加权统计(expectancy_w)"})
    return findings


def summarize(findings: list[dict[str, Any]]) -> dict[str, int]:
    """按 level 计数。"""
    counts = {lv: 0 for lv in LEVELS}
    for f in findings:
        counts[f["level"]] = counts.get(f["level"], 0) + 1
    return counts
