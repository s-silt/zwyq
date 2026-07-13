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

from ashare_gauntlet.config import CACHE_DIR
from ashare_gauntlet.data.fetch import call_with_retry, fetch_symbol_history, fetch_symbol_table
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.fundamentals import (
    balance_facts,
    cashflow_facts,
    index_changes,
    latest_express,
    latest_forecast,
    latest_quarter,
    peg,
    pledge_ratio,
    receivables_ratio,
    recent_holder_trades,
    st_status,
    upcoming_unlocks,
)

SYMBOL_TABLES = ("income", "fina_indicator", "balancesheet", "cashflow", "share_float",
                 "pledge_stat", "stk_holdertrade", "namechange", "forecast", "express")
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


def main(codes: list[str], cache_dir: str = CACHE_DIR) -> None:
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    as_of = _as_of(cache_dir)
    _macro(pro, as_of, cache_dir)

    for code in codes:
        tabs = {t: fetch_symbol_table(pro, t, code, cache_dir) for t in SYMBOL_TABLES}
        q = latest_quarter(tabs["income"], tabs["fina_indicator"])
        db = call_with_retry(lambda: pro.daily_basic(ts_code=code, trade_date=as_of, fields="ts_code,pe_ttm"))
        pe_ttm = None
        if db is not None and not db.empty:
            v = db.iloc[0].get("pe_ttm")
            pe_ttm = float(v) if v is not None and str(v) != "nan" else None
        print(f"=== {code} 基本面 (接口口径,确定) ===")
        if q and q.get("net_profit_yi") is not None:
            gm = f"{q['gross_margin_pct']:.1f}%" if q.get("gross_margin_pct") is not None else "-"
            ry = f"{q['revenue_yoy_pct']:+.1f}%" if q.get("revenue_yoy_pct") is not None else "-"
            ny = f"{q['net_profit_yoy_pct']:+.1f}%" if q.get("net_profit_yoy_pct") is not None else "-"
            rev = f"{q['revenue_yi']:.2f}亿" if q.get("revenue_yi") is not None else "-"
            pg = peg(pe_ttm, q.get("net_profit_yoy_pct"))
            pgs = f" | PE_TTM{pe_ttm:.0f}/PEG{pg:.2f}" if pg is not None else (
                f" | PE_TTM{pe_ttm:.0f}" if pe_ttm is not None else "")
            print(f"  {q['end_date']}: 营收 {rev}(同比{ry}) | 归母净利 {q['net_profit_yi']:.2f}亿(同比{ny}) "
                  f"| 毛利率{gm}{pgs} | {'盈利' if q['profitable'] else '亏损'}")
        else:
            print("  无利润表数据(接口)")
        bf = balance_facts(tabs["balancesheet"])
        cf = cashflow_facts(tabs["cashflow"])
        bc = []
        if bf.get("accounts_receiv_yi") is not None:
            bc.append(f"应收{bf['accounts_receiv_yi']:.1f}亿")
        if bf.get("goodwill_yi") is not None:
            bc.append(f"商誉{bf['goodwill_yi']:.2f}亿")
        if bf.get("money_cap_yi") is not None:
            bc.append(f"货币资金{bf['money_cap_yi']:.0f}亿")
        if cf.get("op_cashflow_yi") is not None:
            bc.append(f"经营现金流{cf['op_cashflow_yi']:+.0f}亿")
        rr = receivables_ratio(tabs["balancesheet"], tabs["income"])
        if rr is not None:
            bc.append(f"应收/年净利{rr:.0f}%")
        if bc:
            print("  " + " | ".join(bc))
        fc = latest_forecast(tabs["forecast"])
        ex = latest_express(tabs["express"])
        q_end = q.get("end_date", "") if q else ""
        if fc and fc["end_date"] > q_end and fc.get("p_change_min") is not None:
            print(f"  📅 业绩预告 {fc['end_date']}: {fc['type']} 净利同比 {fc['p_change_min']:.0f}~{fc['p_change_max']:.0f}%")
        elif ex and ex["end_date"] > q_end and ex.get("net_profit_yi") is not None:
            yo = f"{ex['yoy_net_profit_pct']:+.0f}%" if ex.get("yoy_net_profit_pct") is not None else "-"
            print(f"  📅 业绩快报 {ex['end_date']}: 营收{ex['revenue_yi']:.1f}亿 净利{ex['net_profit_yi']:.2f}亿(同比{yo})")
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
        trades = recent_holder_trades(tabs["stk_holdertrade"], as_of, 365) if as_of else []
        reductions = [t for t in trades if t["direction"] == "减持"]
        if reductions:
            for t in reductions[:2]:
                vol = f"{t['change_vol'] / 1e4:.0f}万股" if t["change_vol"] is not None else "?"
                rt = f"占{t['change_ratio']:.2f}%" if t["change_ratio"] is not None else ""
                print(f"  ⚠️ 近一年减持 {t['ann_date']}: {t['holder_name'][:14]} {vol}{rt}")
        else:
            print("  近一年无股东减持(接口)")
        st = st_status(tabs["namechange"])
        if st.get("is_st"):
            print(f"  ⛔ 当前 ST: {st['current_name']}")
        elif st.get("ever_st"):
            print(f"  ⚠️ 曾 ST/摘帽: 现名 {st['current_name']}, 最近改名 {st['last_change_date']}({st['last_change_reason']})")
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
