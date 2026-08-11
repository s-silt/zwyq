"""精简基本面:只拉 🟢 判定必需的 4 张表(income/fina_indicator/balancesheet/cashflow)
+ 扣非(dt_netprofit_yoy)+ daily_basic(PE),每只间隔降速规避镜像限频。

用于跨板块快速筛 🟢(净利&扣非双增 + 营收同步 + 现金流正 + 应收/商誉无雷)。
Usage: PYTHONIOENCODING=utf-8 python scripts/quickfund.py <ts_code...> [--sleep 8]
"""

import glob
import sys
import time

import pandas as pd

from ashare_gauntlet.data.fetch import call_with_retry, fetch_symbol_table
from ashare_gauntlet.fundamentals import balance_facts, cashflow_facts, latest_quarter, peg, receivables_ratio
import os

from ashare_gauntlet.config import CACHE_DIR as CACHE, tushare_pro

TABLES = ("income", "fina_indicator", "balancesheet", "cashflow")


def _as_of() -> str:
    fs = sorted(glob.glob(f"{CACHE}/daily/*.parquet"))
    return os.path.basename(fs[-1])[:8] if fs else ""


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


def main(codes: list[str], sleep_s: float) -> None:
    pro = tushare_pro()
    as_of = _as_of()
    for i, code in enumerate(codes):
        try:
            tabs = {t: fetch_symbol_table(pro, t, code, CACHE) for t in TABLES}
            q = latest_quarter(tabs["income"], tabs["fina_indicator"])
            dbr = call_with_retry(lambda c=code: pro.daily_basic(ts_code=c, trade_date=as_of, fields="ts_code,pe_ttm"))
            pe = float(dbr.iloc[0]["pe_ttm"]) if dbr is not None and not dbr.empty and pd.notna(dbr.iloc[0]["pe_ttm"]) else None
            if not q or q.get("net_profit_yi") is None:
                print(f"{code}: 无利润表数据"); continue
            ny = q.get("net_profit_yoy_pct")
            ry = q.get("revenue_yoy_pct")
            dty = _dedt_yoy(code, q.get("end_date", ""))
            cf = cashflow_facts(tabs["cashflow"])
            bf = balance_facts(tabs["balancesheet"])
            rr = receivables_ratio(tabs["balancesheet"], tabs["income"])
            ocf = cf.get("op_cashflow_yi")
            gw = bf.get("goodwill_yi")
            pg = peg(pe, ny)
            s_ny = f"{ny:+.1f}%" if ny is not None else "-"
            s_ry = f"{ry:+.1f}%" if ry is not None else "-"
            s_dty = f"{dty:+.1f}%" if dty is not None else "-"
            s_pe = f"{pe:.0f}" if pe is not None else "-"
            s_pg = f"{pg:.2f}" if pg is not None else "-"
            s_ocf = f"{ocf:+.0f}亿" if ocf is not None else "-"
            s_gw = f"{gw:.1f}亿" if gw is not None else "-"
            s_rr = f"{rr:.0f}%" if rr is not None else "-"
            gm = q.get("gross_margin_pct")
            s_gm = f"{gm:.1f}%" if gm is not None else "-"
            # 🟢 粗判:净利>15 & 扣非>10 & 营收>10 & 现金流>0 & 应收/净利<400 & 商誉无大雷
            green = (ny is not None and ny > 15 and dty is not None and dty > 10
                     and ry is not None and ry > 10 and ocf is not None and ocf > 0
                     and rr is not None and rr < 400)
            tag = "🟢候选" if green else ""
            print(f"{code} {q['end_date']}: 营收{q['revenue_yi']:.1f}亿({s_ry}) | 净利{q['net_profit_yi']:.2f}亿({s_ny},扣非{s_dty}) | "
                  f"毛利{s_gm} | 现金流{s_ocf} | 应收/净利{s_rr} | 商誉{s_gw} | PE{s_pe}/PEG{s_pg} {tag}")
        except Exception as e:
            print(f"{code}: 失败 {str(e)[:40]}")
        if i < len(codes) - 1:
            time.sleep(sleep_s)


if __name__ == "__main__":
    args = sys.argv[1:]
    sleep_s = 8.0
    if "--sleep" in args:
        j = args.index("--sleep")
        sleep_s = float(args[j + 1])
        args = args[:j] + args[j + 2:]
    stocks = [a for a in args if a.endswith((".SH", ".SZ", ".BJ"))]
    main(stocks, sleep_s)
