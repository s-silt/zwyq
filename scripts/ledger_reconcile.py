"""账本对账(只读 CLI):交叉核对 holdings 与 trade_journal 是否自洽。

用途:捕捉 `trade_journal --add`(只写流水)与 `trade_record`(写流水+持仓)双写路径
造成的账本漂移。**只读只报,绝不改账**——修法永远由用户人工决定。

退出码:0=自洽;2=有需人工确认/处理的发现;1=账本文件读取或结构失败。

Usage: E:\\zwyq\\.venv\\Scripts\\python.exe -m scripts.ledger_reconcile [--json]
"""
from __future__ import annotations

import argparse
import json

from ashare_gauntlet.config import HOLDINGS_PATH, TRADE_JOURNAL_PATH
from ashare_gauntlet.ledger_audit import LedgerAuditError, reconcile, summarize


def _load(path: str, what: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"{path} 不存在——{what} 缺失")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} 不是合法 JSON: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"{path} 顶层必须是对象")
    return data


def main(argv: "list[str] | None" = None) -> None:
    ap = argparse.ArgumentParser(description="账本对账(只读):holdings ↔ trade_journal")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON")
    ap.add_argument("--holdings", default=HOLDINGS_PATH)
    ap.add_argument("--journal", default=TRADE_JOURNAL_PATH)
    a = ap.parse_args(argv)

    holdings = _load(a.holdings, "账户状态")
    journal = _load(a.journal, "交易流水")
    try:
        findings = reconcile(holdings, journal)
    except LedgerAuditError as exc:
        raise SystemExit(f"账本结构非法: {exc}")
    counts = summarize(findings)

    if a.json:
        print(json.dumps({"findings": findings, "counts": counts},
                         ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"=== 账本对账 === 持仓 {len(holdings.get('positions', []))} 只 | "
              f"流水 {len(journal.get('trades', []))} 笔")
        if not findings:
            print("✅ 自洽:未发现 holdings 与 journal 的矛盾")
        else:
            print(f"DRIFT={counts['DRIFT']}(矛盾,须处理) "
                  f"REVIEW={counts['REVIEW']}(需确认) "
                  f"CONTRACT={counts['CONTRACT']}(统计口径受影响)")
            for f in findings:
                print(f"  [{f['level']}] {f['code']} {f['issue']}: {f['detail']}")
            print("\n(本工具只读只报;如何修由你决定——改账请走 trade_record/手工编辑)")
    raise SystemExit(2 if findings else 0)


if __name__ == "__main__":
    main()
