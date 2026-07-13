"""一次性 factcard:把技术面+估值(screen)与接口基本面(fundamentals)合并成单卡,
按基本面质地分四层分组输出。数字全接口确定口径,扣非走 fina_indicator 直出。

固定模式第4步「诚实面板」的卡片视图。四层映射由人工核实(接口+factcheck)给定。

Usage: PYTHONIOENCODING=utf-8 python scripts/factcard.py
"""

import glob
import os
import sys

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR as CACHE
from ashare_gauntlet.data.fetch import call_with_retry, fetch_symbol_history, fetch_symbol_table
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factsheet import daily_tech_facts, entry_rank, market_returns
from ashare_gauntlet.lit_factors import (
    accrual_ratio,
    earnings_yield,
    latest_annual_end,
    net_cash_ratio,
    reversal,
    volatility,
)
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

INDEXES = (("000001.SH", "上证"), ("399001.SZ", "深成"), ("399006.SZ", "创业板"), ("000688.SH", "科创50"))
SYMBOL_TABLES = ("income", "fina_indicator", "balancesheet", "cashflow", "share_float",
                 "pledge_stat", "stk_holdertrade", "namechange", "forecast", "express")

TIERS = [
    ("🟢 强且干净", ["601138.SH", "002463.SZ", "002245.SZ", "000021.SZ", "603890.SH", "600885.SH"]),
    ("🟡 盈利但瑕疵", ["600522.SH", "000100.SZ", "002156.SZ", "002185.SZ", "001287.SZ", "001314.SZ",
                  "600563.SH", "000922.SZ", "601231.SH", "002925.SZ", "002803.SZ", "603296.SH"]),
    ("🔴 题材背离·警示", ["603267.SH", "000681.SZ", "603859.SH", "002409.SZ", "002138.SZ", "000725.SZ",
                    "002938.SZ", "603989.SH", "000823.SZ", "002056.SZ", "600237.SH", "605058.SH",
                    "002600.SZ", "603228.SH", "000063.SZ"]),
    ("⛔ 地雷/重度恶化", ["001229.SZ", "603341.SH", "000733.SZ", "603328.SH", "002993.SZ", "605258.SH",
                    "002106.SZ", "002897.SZ"]),
]


def annual_lit_factors(income: pd.DataFrame | None, cashflow: pd.DataFrame | None,
                       balancesheet: pd.DataFrame | None) -> tuple[str | None, float | None, float | None]:
    """动态取最近年报期并算净现比/应计,返回 (年报期, 净现比, 应计强度)。

    P2-9:年报期不再硬编码,取自 latest_annual_end(跨表 max …1231);
    取不到任何年报期 → (None, None, None),由渲染层标「年报缺失」,不套固定日期伪造。
    """
    end = latest_annual_end(income, cashflow, balancesheet)
    if end is None:
        return None, None, None
    return end, net_cash_ratio(income, cashflow, end=end), accrual_ratio(income, cashflow, balancesheet, end=end)


def lit_factor_label(annual_end: str | None) -> str:
    """文献因子行的口径标注:按真实年报期渲染(如 '净现比/应计=2024年报'),缺失则明示。"""
    if annual_end is None:
        return "净现比/应计=年报缺失"
    return f"净现比/应计={annual_end[:4]}年报"  # end_date 为 yyyymmdd,前4位即年份


def _load(ep: str) -> pd.DataFrame:
    fs = glob.glob(f"{CACHE}/{ep}/*.parquet")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _dedt_yoy(code: str, end_date: str) -> float | None:
    fs = glob.glob(f"{CACHE}/fina_indicator/{code}.parquet")
    if not fs:
        return None
    df = pd.read_parquet(fs[0])
    df = df[df["end_date"] == end_date]
    if df.empty:
        return None
    v = df.iloc[0].get("dt_netprofit_yoy")
    return float(v) if v is not None and str(v) != "nan" else None


def main() -> None:
    daily, adj = _load("daily"), _load("adj_factor")
    as_of = str(daily["trade_date"].max())
    mr = market_returns(daily, adj, (5, 20))
    rank20 = (mr[20].rank(pct=True) * 100).to_dict()

    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])

    frames = [fetch_symbol_history(pro, "index_daily", c, "20251001", as_of, CACHE) for c, _ in INDEXES]
    ch = index_changes(pd.concat(frames, ignore_index=True), as_of)
    macro = " / ".join(f"{n}{ch[c]['close']:.0f}({ch[c]['pct_chg']:+.2f}%)"
                       for c, n in INDEXES if ch.get(c) and ch[c]["close"] is not None)
    print(f"宏观指数(接口,截至{as_of}): {macro}")
    print("分 名称(代码) 业务 | 现价 距60高 近20%(分位) PE PB 市值 ‖ 营收/净利(扣非)/毛利/PEG ‖ 应收/商誉/货币/现金流/应收净利 ‖ 质押/解禁/减持/ST/前瞻\n")

    tiers = [t for t in TIERS if t[0][0] in "🟢🟡"] if "--gy" in sys.argv else TIERS
    for label, codes in tiers:
        print(f"════════ {label} ({len(codes)}) ════════")
        for code in codes:
            f = daily_tech_facts(code, daily[daily["ts_code"] == code], adj[adj["ts_code"] == code], mr)
            score, _ = entry_rank(f)
            dbr = call_with_retry(lambda c=code: pro.daily_basic(ts_code=c, trade_date=as_of, fields="ts_code,pe_ttm,pb,total_mv"))
            sbr = call_with_retry(lambda c=code: pro.stock_basic(ts_code=c, fields="ts_code,name,industry"))
            has_sb = sbr is not None and not sbr.empty
            has_db = dbr is not None and not dbr.empty
            name = str(sbr.iloc[0]["name"]) if has_sb else ""
            ind = str(sbr.iloc[0]["industry"]) if has_sb and pd.notna(sbr.iloc[0]["industry"]) else "-"
            pe = dbr.iloc[0]["pe_ttm"] if has_db else None
            pb = dbr.iloc[0]["pb"] if has_db else None
            mv = dbr.iloc[0]["total_mv"] if has_db else None
            pes = f"{float(pe):.0f}" if pd.notna(pe) else "-"
            pbs = f"{float(pb):.1f}" if pd.notna(pb) else "-"
            mvs = f"{float(mv) / 1e4:.0f}" if pd.notna(mv) else "-"
            sc = f"{score:>3.0f}" if score is not None else "  —"  # 入场分缺失(契约C2)
            print(f"[{sc}] {name}({code}) {ind} | 现{f['close']:.2f} 距60高{f['dist_60d_high_pct']:+.0f}% "
                  f"近20{f['ret20_pct']:+.1f}%(分{rank20.get(code, 0):.0f}) PE{pes} PB{pbs} 市值{mvs}亿")

            tabs = {t: fetch_symbol_table(pro, t, code, CACHE) for t in SYMBOL_TABLES}
            q = latest_quarter(tabs["income"], tabs["fina_indicator"])
            if q and q.get("net_profit_yi") is not None:
                ry = f"{q['revenue_yoy_pct']:+.1f}%" if q.get("revenue_yoy_pct") is not None else "-"
                ny = f"{q['net_profit_yoy_pct']:+.1f}%" if q.get("net_profit_yoy_pct") is not None else "-"
                rev = f"{q['revenue_yi']:.2f}亿" if q.get("revenue_yi") is not None else "-"
                gm = f"{q['gross_margin_pct']:.1f}%" if q.get("gross_margin_pct") is not None else "-"
                dty = _dedt_yoy(code, q.get("end_date", ""))
                dts = f",扣非{dty:+.1f}%" if dty is not None else ""
                pg = peg(float(pe) if pd.notna(pe) else None, q.get("net_profit_yoy_pct"))
                pgs = f" PE{float(pe):.0f}/PEG{pg:.2f}" if pg is not None else (f" PE{float(pe):.0f}" if pd.notna(pe) else "")
                print(f"      {q['end_date']}: 营收{rev}({ry}) | 归母净利{q['net_profit_yi']:.2f}亿({ny}{dts}) | "
                      f"毛利{gm} |{pgs} | {'盈利' if q['profitable'] else '亏损'}")
            else:
                print("      无利润表数据(接口)")

            bf, cf = balance_facts(tabs["balancesheet"]), cashflow_facts(tabs["cashflow"])
            bc = []
            if bf.get("accounts_receiv_yi") is not None:
                bc.append(f"应收{bf['accounts_receiv_yi']:.1f}亿")
            bc.append(f"商誉{bf['goodwill_yi']:.2f}亿" if bf.get("goodwill_yi") is not None else "商誉-")
            if bf.get("money_cap_yi") is not None:
                bc.append(f"货币{bf['money_cap_yi']:.0f}亿")
            if cf.get("op_cashflow_yi") is not None:
                bc.append(f"经营现金流{cf['op_cashflow_yi']:+.0f}亿")
            rr = receivables_ratio(tabs["balancesheet"], tabs["income"])
            if rr is not None:
                bc.append(f"应收/年净利{rr:.0f}%")
            print("      " + " | ".join(bc))

            flags = []
            pr = pledge_ratio(tabs["pledge_stat"], as_of)
            if pr is not None and pr > 0:
                flags.append(f"质押{pr:.1f}%")
            unlocks = upcoming_unlocks(tabs["share_float"], as_of, 180) if as_of else []
            if unlocks:
                u = unlocks[0]
                fr = f"占{u['float_ratio']:.1f}%" if u["float_ratio"] is not None else ""
                flags.append(f"⚠️解禁{u['float_date']}{fr}")
            trades = recent_holder_trades(tabs["stk_holdertrade"], as_of, 365) if as_of else []
            reds = [t for t in trades if t["direction"] == "减持"]
            if reds:
                flags.append(f"⚠️近一年减持{len(reds)}笔")
            st = st_status(tabs["namechange"])
            if st.get("is_st"):
                flags.append(f"⛔当前ST({st['current_name']})")
            elif st.get("ever_st"):
                flags.append("曾ST/摘帽")
            q_end = q.get("end_date", "") if q else ""
            fc, ex = latest_forecast(tabs["forecast"]), latest_express(tabs["express"])
            if fc and fc["end_date"] > q_end and fc.get("p_change_min") is not None:
                flags.append(f"📅预告{fc['end_date']} {fc['type']}{fc['p_change_min']:.0f}~{fc['p_change_max']:.0f}%")
            elif ex and ex["end_date"] > q_end and ex.get("net_profit_yi") is not None:
                yo = f"同比{ex['yoy_net_profit_pct']:+.0f}%" if ex.get("yoy_net_profit_pct") is not None else ""
                flags.append(f"📅快报{ex['end_date']} 净利{ex['net_profit_yi']:.2f}亿{yo}")
            print("      " + (" / ".join(flags) if flags else "无质押/解禁/减持/ST旗标"))

            dsub, asub = daily[daily["ts_code"] == code], adj[adj["ts_code"] == code]
            # P2-9:年报期动态取自数据(latest_annual_end),不再依赖硬编码 20251231
            a_end, ncr, acc = annual_lit_factors(tabs["income"], tabs["cashflow"], tabs["balancesheet"])
            ep = earnings_yield(float(pe) if pd.notna(pe) else None)
            rev20, vol60 = reversal(dsub, asub, 20), volatility(dsub, asub, 60)
            lf = [
                f"净现比{ncr:.2f}" if ncr is not None else "净现比-",
                f"应计{acc * 100:+.0f}%" if acc is not None else "应计-",
                f"EP{ep * 100:.1f}%" if ep is not None else "EP-",
                f"近20反转{rev20 * 100:+.0f}%" if rev20 is not None else "反转-",
                f"波动{vol60 * 100:.1f}%" if vol60 is not None else "波动-",
            ]
            print(f"      📚文献因子({lit_factor_label(a_end)}): " + " | ".join(lf) + "\n")
        print()


if __name__ == "__main__":
    main()
