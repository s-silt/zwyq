"""持有分排序(质地优先) —— 对"动量优先漏斗"的纠正(见 memory stock-analysis-mode 2026-06-25 修正)。

流程:① daily_basic 全市场做"价值+盈利"预筛(非动量:主板+盈利+PE/PB/市值合理)
→ ② 对候选(已缓存的免费 + 价值预筛新名)逐只 build_record → ③ compute_holdscore 算持有分
→ ④ 按持有分由高到低排,**位置(MA20)只作买入时机标注(回调位/正常/涨高),不参与排序**。

先找"最干净的"(持有分),再看"什么时候买"(位置)——而非先抓今天涨得猛的。

数据源约束:这源不能按报告期一次拉全市场财务(fina_indicator/income 全市场返回空),
故用 daily_basic(全市场可拉)价值预筛 + 候选逐只建档,而非真全市场扫描。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.holdscore
       [--board main] [--max-pe 40] [--max-pb 8] [--min-mv 50] [--build-cap 100] [--top 30]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, cast

import pandas as pd

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry, fetch_symbol_table
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factsheet import market_returns
from ashare_gauntlet.record import build_record, compute_holdscore
from ashare_gauntlet.screen import board_of

CACHE = "data/cache"
OUT_DIR = "data/holdscore"
SYM = ("income", "fina_indicator", "balancesheet", "cashflow", "share_float",
       "pledge_stat", "stk_holdertrade", "namechange", "forecast", "express")
BOARDS: dict[str, list[str] | None] = {
    "sh_main": ["沪主板"], "sz_main": ["深主板"], "main": ["沪主板", "深主板"], "all": None,
}


def _load(ep: str) -> pd.DataFrame:
    fs = glob.glob(f"{CACHE}/{ep}/*.parquet")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def pos_flag(px: float, ma20: float) -> str:
    """位置标注(买入时机,不参与持有分):回调位 / 正常 / 涨高。纯函数。"""
    d = (px / ma20 - 1) * 100
    if d < -3:
        return f"回调位{d:+.0f}%"
    if d <= 5:
        return f"正常{d:+.0f}%"
    return f"涨高{d:+.0f}%"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main", choices=list(BOARDS))
    ap.add_argument("--max-pe", type=float, default=40.0)
    ap.add_argument("--max-pb", type=float, default=8.0)
    ap.add_argument("--min-mv", type=float, default=50.0, help="最小总市值(亿)")
    ap.add_argument("--build-cap", type=int, default=100, help="新建档上限(已缓存的不计)")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args(argv)

    load_env_local()
    daily, adj = _load("daily"), _load("adj_factor")
    if daily.empty:
        raise SystemExit("data/cache/daily 为空 —— 先 backfill")
    as_of = str(daily["trade_date"].max())
    mr = market_returns(daily, adj, (5, 20))

    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    db = call_with_retry(lambda: pro.daily_basic(trade_date=as_of, fields="ts_code,pe_ttm,pb,total_mv,dv_ttm"))
    sb = call_with_retry(lambda: pro.stock_basic(list_status="L", fields="ts_code,name,industry"))
    name_d = dict(zip(sb["ts_code"], sb["name"]))
    ind_d = dict(zip(sb["ts_code"], sb["industry"]))
    boards = BOARDS[a.board]

    # 价值+盈利预筛(非动量):盈利 + PE/PB 合理 + 市值够 + 主板
    db = db.copy()
    db["mv_yi"] = db["total_mv"].astype(float) / 1e4  # 万元 → 亿
    cand = db[(db["pe_ttm"] > 0) & (db["pe_ttm"] <= a.max_pe)
              & (db["pb"] > 0) & (db["pb"] <= a.max_pb) & (db["mv_yi"] >= a.min_mv)]
    if boards is not None:
        cand = cand[cand["ts_code"].apply(lambda c: board_of(str(c)) in boards)]
    # 新名按市值降序(建大盘蓝筹,避开微盘),质地排序交给 holdscore
    val_codes = list(cand.sort_values("mv_yi", ascending=False)["ts_code"])

    cached = {os.path.basename(p).replace(".parquet", "") for p in glob.glob(f"{CACHE}/fina_indicator/*.parquet")}
    cached = {c for c in cached if boards is None or board_of(c) in boards}
    new_codes = [c for c in val_codes if c not in cached][:a.build_cap]
    build = list(cached) + new_codes
    print(f"as_of={as_of} | 价值预筛通过 {len(cand)} 只 | 已缓存 {len(cached)} + 新建 {len(new_codes)} "
          f"= 建档 {len(build)} 只 → 算持有分…", flush=True)

    db_d = {str(r["ts_code"]): r for _, r in db.iterrows()}
    recs: list[dict[str, Any]] = []
    for i, code in enumerate(build):
        try:
            ft = {t: fetch_symbol_table(pro, t, code, CACHE) for t in SYM}
            ind = ind_d.get(code)
            rec = build_record(code, name=str(name_d.get(code, code)),
                               industry=ind if isinstance(ind, str) and ind else "-", as_of=as_of,
                               daily_sub=cast(pd.DataFrame, daily[daily["ts_code"] == code]),
                               adj_sub=cast(pd.DataFrame, adj[adj["ts_code"] == code]),
                               mr=mr, fund_tables=ft, db_row=db_d.get(code))
            rec["holdscore"] = compute_holdscore(rec)
            recs.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  …{i + 1}/{len(build)}", flush=True)
        except Exception as e:  # 单只失败响亮上报、跳过
            print(f"  {code} 建档失败: {type(e).__name__}: {str(e)[:40]}", file=sys.stderr, flush=True)

    recs.sort(key=lambda r: r.get("holdscore", 0.0), reverse=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{as_of}.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=2, allow_nan=False)

    last = daily[daily["trade_date"] == as_of].set_index("ts_code")["close"].astype(float)
    print(f"\n=== 持有分 Top {a.top}(质地优先;位置只作买入时机标注)→ {OUT_DIR}/{as_of}.json ===")
    print(f"{'#':>2} {'分':>4} {'档':>2} {'票':<9}{'行业':<7} {'净/扣/营%':>15} {'PE':>5} {'市值亿':>7}  位置")
    for i, r in enumerate(recs[:a.top], 1):
        f, v = r["fundamental"], r["valuation"]
        code = r["ts_code"]
        px = last.get(code)
        sc = daily[daily["ts_code"] == code].sort_values("trade_date")["close"].astype(float)
        ma20 = sc.tail(20).mean() if len(sc) >= 20 else float("nan")
        pos = pos_flag(float(px), float(ma20)) if (px is not None and ma20 == ma20) else "—"
        pp = lambda x: f"{x:+.0f}" if isinstance(x, (int, float)) else "—"
        nn = lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else "—"
        g3 = f"{pp(f.get('np_yoy'))}/{pp(f.get('dedt_yoy'))}/{pp(f.get('rev_yoy'))}"
        print(f"{i:>2} {r['holdscore']:>4.0f} {r['tier']['grade']:>2} {r['name'][:8]:<9}{str(r['industry'])[:6]:<7} "
              f"{g3:>15} {nn(v.get('pe_ttm')):>5} {nn(v.get('mv_yi')):>7}  {pos}")


if __name__ == "__main__":
    main()
