"""治理雷本地核查:最大股东质押与最新审计意见。

本脚本只读取 ``data/cache/pledge_detail`` 与 ``data/cache/fina_audit``。
联网刷新必须先运行 ``refresh_stock_financials``；未覆盖不等于无风险。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.governance_check
           [--codes 600875.SH,601138.SH]
默认读 data/holdings.json 的 positions。``--force`` 仅为旧调用兼容并会明确拒绝。
"""

import argparse
import json

from ashare_gauntlet import mcp_service
from ashare_gauntlet.config import HOLDINGS_PATH as HOLDINGS


def holdings_codes(path: str) -> list[tuple[str, str]]:
    """holdings.json 的 positions → [(ts_code, name)];文件/键缺失直接抛(fail-loud)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [(p["ts_code"], p.get("name", "")) for p in data["positions"]]


def _pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "?(字段缺,≠0)"


def print_one(ts_code: str, name: str, result: dict) -> None:
    print(f"--- {ts_code} {name}")
    pledge = result["pledge"]
    if pledge["status"] == "uncovered":
        reason = pledge["coverage"]["reason"]
        print(f"  质押: 未覆盖(≠无风险)—— {reason}")
    elif pledge["status"] == "none_current":
        print("  质押: 本地历史已覆盖，当前无未解押质押记录")
    else:
        top = pledge["controller_pledge"]
        print(
            f"  质押: 最大股东 {top['holder_name']} | "
            f"占其持股 {_pct(top['pledged_ratio_of_holding'])} | "
            f"占总股本 {_pct(top['ratio_of_total'])} | 快照 {top['asof']}"
        )

    audit = result["audit"]
    if audit["status"] == "uncovered":
        reason = audit["coverage"]["reason"]
        print(f"  审计: 未覆盖(≠无风险)—— {reason}")
        return
    opinion = audit["opinion"]
    flag = "⚠️ 非标(治理红旗)" if opinion["is_nonstandard"] else "标准"
    audit_result = opinion["audit_result"] if opinion["audit_result"] is not None else "(缺失)"
    print(f"  审计: {opinion['end_date']} {audit_result} → {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="治理雷本地核查(pledge_detail + fina_audit)")
    parser.add_argument("--codes", help="逗号分隔 ts_code 列表;缺省读 holdings.json positions")
    parser.add_argument(
        "--force", action="store_true",
        help="已禁用;请先通过 refresh_stock_financials 显式刷新",
    )
    args = parser.parse_args()
    if args.force:
        raise ValueError(
            "governance_check never refreshes data; call refresh_stock_financials first"
        )

    if args.codes:
        targets = [(c.strip(), "") for c in args.codes.split(",") if c.strip()]
    else:
        targets = holdings_codes(HOLDINGS)
    checked = mcp_service.governance_check(
        [code for code, _ in targets], force=args.force,
    )
    by_code = checked["codes"]
    for ts_code, name in targets:
        print_one(ts_code.upper(), name, by_code[ts_code.upper()])


if __name__ == "__main__":
    main()
