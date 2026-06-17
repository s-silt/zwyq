"""Interface-first fundamentals card — pull+cache the structured tushare tables
and print the deterministic facts (Q1 业绩 + 质押 + 即将解禁), replacing the slow
web-scrape. Also refreshes the macro index snapshot (index_daily) into the cache.

This is the fixed analysis mode's step 3, interface-first half: numbers come from
here; the web layer (named workflow `factcheck`) only adds narrative/消息面.

Usage: PYTHONIOENCODING=utf-8 python scripts/fundamentals.py <ts_code> [ts_code ...]
"""

import glob
import os
import sys

import pandas as pd

from ashare_gauntlet.data.fetch import fetch_symbol_history, fetch_symbol_table
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.fundamentals import (
    index_changes,
    latest_quarter,
    pledge_ratio,
    upcoming_unlocks,
)

SYMBOL_TABLES = ("income", "fina_indicator", "share_float", "pledge_stat")
INDEXES = (("000001.SH", "上证"), ("399001.SZ", "深成"), ("399006.SZ", "创业板"), ("000688.SH", "科创50"))


def _as_of(cache_dir: str) -> str:
    files = sorted(glob.glob(f"{cache_dir}/daily/*.parquet"))
    return os.path.basename(files[-1])[:8] if files else ""


def _macro(pro: object, as_of: str, cache_dir: str) -> None:
    frames = [fetch_symbol_history(pro, "index_daily", code, "20251001", as_of, cache_dir)
              for code, _ in INDEXES]
    idx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ch = index_changes(idx, as_of)
    print(f"=== 宏观指数 (接口, 截至 {as_of}) ===")
    for code, name in INDEXES:
        c = ch.get(code)
        if c and c["close"] is not None:
            print(f"  {name}({code}) {c['close']:.2f}  {c['pct_chg']:+.2f}%")
    print()


def main(codes: list[str], cache_dir: str = "data/cache") -> None:
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    as_of = _as_of(cache_dir)
    _macro(pro, as_of, cache_dir)

    for code in codes:
        tabs = {t: fetch_symbol_table(pro, t, code, cache_dir) for t in SYMBOL_TABLES}
        q = latest_quarter(tabs["income"], tabs["fina_indicator"])
        print(f"=== {code} 基本面 (接口口径,确定) ===")
        if q and q.get("net_profit_yi") is not None:
            gm = f"{q['gross_margin_pct']:.1f}%" if q.get("gross_margin_pct") is not None else "-"
            ry = f"{q['revenue_yoy_pct']:+.1f}%" if q.get("revenue_yoy_pct") is not None else "-"
            ny = f"{q['net_profit_yoy_pct']:+.1f}%" if q.get("net_profit_yoy_pct") is not None else "-"
            rev = f"{q['revenue_yi']:.2f}亿" if q.get("revenue_yi") is not None else "-"
            print(f"  {q['end_date']}: 营收 {rev}(同比{ry}) | 归母净利 {q['net_profit_yi']:.2f}亿(同比{ny}) "
                  f"| 毛利率{gm} | {'盈利' if q['profitable'] else '亏损'}")
        else:
            print("  无利润表数据(接口)")
        pr = pledge_ratio(tabs["pledge_stat"], as_of)
        print(f"  控股体系质押比例: {pr:.2f}%" if pr is not None else "  控股体系质押比例: 无数据")
        unlocks = upcoming_unlocks(tabs["share_float"], as_of, 180) if as_of else []
        if unlocks:
            for u in unlocks[:3]:
                fr = f"占{u['float_ratio']:.2f}%" if u["float_ratio"] is not None else ""
                sh = f"{u['float_share'] / 1e4:.0f}万股" if u["float_share"] is not None else "?"
                print(f"  ⚠️ 即将解禁 {u['float_date']}: {sh}{fr} {u['holder_name'][:18]}")
        else:
            print("  未来180日无限售解禁(接口)")
        print("  —— 数字=接口确定口径;叙事/消息面(重组/澄清/业绩说明会)用 named workflow `factcheck` 补 ——")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    stocks = [a for a in args if a.endswith((".SH", ".SZ", ".BJ"))]
    cdir = next((a for a in args if "/" in a), "data/cache")
    if not stocks:
        print("usage: python scripts/fundamentals.py <ts_code> [ts_code ...]")
    else:
        main(stocks, cdir)
