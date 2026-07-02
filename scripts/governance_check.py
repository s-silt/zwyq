"""治理雷核查:逐只打印最大股东质押(占其持股%/占总股本%)+ 最新审计意见是否非标。

预警表 ⛔ 抓不到控股股东个人质押/非标审计 —— 本脚本用 pledge_detail + fina_audit
补这两个盲点。只报数不设阈值,判断归人工/factcheck。fail-loud:表拉不到打印
"未覆盖(≠无风险)",绝不当无风险吞掉;fina_audit 空表同样按未覆盖上报
(上市公司必有年审,空 = 数据源没盖到,≠ 意见干净)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.governance_check
           [--codes 600875.SH,601138.SH] [--force]
默认读 data/holdings.json 的 positions;--force 整只重拉(有新质押/审计公告时用,
否则缓存优先会把治理状态冻结在旧公告)。
"""

import argparse
import json
import os

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import fetch_symbol_table
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.governance import audit_opinion, controller_pledge

CACHE = "data/cache"
HOLDINGS = "data/holdings.json"


def holdings_codes(path: str) -> list[tuple[str, str]]:
    """holdings.json 的 positions → [(ts_code, name)];文件/键缺失直接抛(fail-loud)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [(p["ts_code"], p.get("name", "")) for p in data["positions"]]


def _pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "?(字段缺,≠0)"


def check_one(pro: object, ts_code: str, name: str, force: bool) -> None:
    print(f"--- {ts_code} {name}")
    # 股东质押明细:空表 = 无质押公告(合法值);拉取失败 = 未覆盖。
    try:
        pledge = fetch_symbol_table(pro, "pledge_detail", ts_code, CACHE, force=force)
    except Exception as error:
        print(f"  质押: 未覆盖(≠无风险)—— pledge_detail 拉取失败: {str(error)[:80]}")
    else:
        top = controller_pledge(pledge)
        if not top:
            print("  质押: 无未解押质押记录")
        else:
            print(
                f"  质押: 最大股东 {top['holder_name']} | "
                f"占其持股 {_pct(top['pledged_ratio_of_holding'])} | "
                f"占总股本 {_pct(top['ratio_of_total'])} | 快照 {top['asof']}"
            )
    # 审计意见:上市公司必有年审 → 空表不是合法值,按未覆盖上报。
    try:
        audit = fetch_symbol_table(pro, "fina_audit", ts_code, CACHE, force=force)
    except Exception as error:
        print(f"  审计: 未覆盖(≠无风险)—— fina_audit 拉取失败: {str(error)[:80]}")
        return
    opinion = audit_opinion(audit)
    if not opinion:
        print("  审计: 未覆盖(≠无风险)—— fina_audit 返回空表")
        return
    flag = "⚠️ 非标(治理红旗)" if opinion["is_nonstandard"] else "标准"
    result = opinion["audit_result"] if opinion["audit_result"] is not None else "(缺失)"
    print(f"  审计: {opinion['end_date']} {result} → {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="治理雷核查(pledge_detail + fina_audit)")
    parser.add_argument("--codes", help="逗号分隔 ts_code 列表;缺省读 holdings.json positions")
    parser.add_argument("--force", action="store_true", help="跳过缓存整只重拉(有新公告时用)")
    args = parser.parse_args()

    load_env_local()
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])

    if args.codes:
        targets = [(c.strip(), "") for c in args.codes.split(",") if c.strip()]
    else:
        targets = holdings_codes(HOLDINGS)

    for ts_code, name in targets:
        check_one(pro, ts_code, name, args.force)


if __name__ == "__main__":
    main()
