"""全市场质地排序·核心版 —— 用本地 4 核心财报表对全主板 build_record 算完整财务 tier,按持有分排序。

接 `backfill_fina --mode core` 之后跑。比精简版(holdscore_lean)强在:
- 净现比用真·现金流表(cashflow)算,不再是 ocfps 每股现金流代理;
- 扣非用 income 表真值;tier 是完整 tier_of(三增/盈利质量/现金/警示),只缺预警表那档 ⛔。

每只只读本地核心表 + 预警表留空(**不触发任何 per-symbol API 拉取**),全程仅 daily_basic/stock_basic
各 1 次全市场调用。⛔ 地雷(质押/减持/预亏)留给 Top 名单 factcheck 或之后 full 预警表。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.holdscore_core
       [--board main] [--industry 关键词] [--max-pe N] [--min-mv 亿] [--grades 🟢🟡] [--top 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import cast

import pandas as pd

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factsheet import market_returns
from ashare_gauntlet.record import build_record, compute_holdscore
from ashare_gauntlet.screen import board_of

CACHE = "data/cache"
OUT_DIR = "data/holdscore"
MAIN = ("沪主板", "深主板")
CORE = ("income", "fina_indicator", "balancesheet", "cashflow")
WARN = ("forecast", "express", "share_float", "pledge_stat", "stk_holdertrade", "namechange")


def core_universe(board: str) -> list[str]:
    """有全部 4 张核心表的主板代码(预警表无所谓,build 时留空)。"""
    have = None
    for t in CORE:
        codes = {os.path.basename(p).replace(".parquet", "") for p in glob.glob(f"{CACHE}/{t}/*.parquet")}
        have = codes if have is None else (have & codes)
    universe = sorted(have or set())
    if board == "main":
        universe = [c for c in universe if board_of(c) in MAIN]
    return universe


def load_core_tables(code: str) -> dict[str, pd.DataFrame]:
    """核心表 + 预警表都从本地读(预警表已 backfill 全主板,tier_of 才能做 ⛔ 治理雷检测:
    ST/质押/解禁/预亏);本地缺则留空 DataFrame(graceful,不拉 API)。"""
    ft: dict[str, pd.DataFrame] = {}
    for t in (*CORE, *WARN):
        p = f"{CACHE}/{t}/{code}.parquet"
        ft[t] = pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()
    return ft


def pos_flag(px: float, ma20: float) -> str:
    d = (px / ma20 - 1) * 100
    if d < -3:
        return f"回调位{d:+.0f}%"
    if d <= 5:
        return f"正常{d:+.0f}%"
    return f"涨高{d:+.0f}%"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main", choices=("main", "all"))
    ap.add_argument("--industry", default=None)
    ap.add_argument("--max-pe", type=float, default=None)
    ap.add_argument("--min-mv", type=float, default=0.0, help="最小总市值(亿)")
    ap.add_argument("--grades", default="🟢🟡")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)

    load_env_local()
    universe = core_universe(a.board)
    daily = pd.concat([pd.read_parquet(f)  # 全列:build_record 技术指标要 amount/vol/high/low
                       for f in sorted(glob.glob(f"{CACHE}/daily/*.parquet"))], ignore_index=True)
    adj = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{CACHE}/adj_factor/*.parquet"))],
                    ignore_index=True)
    as_of = str(daily["trade_date"].max())
    mr = market_returns(daily, adj, (5, 20))
    # 预先按 ts_code 分组成字典(O(1) 查),避免循环里对全表反复过滤(O(n²))
    daily_g = {str(c): g for c, g in daily.groupby("ts_code")}
    adj_g = {str(c): g for c, g in adj.groupby("ts_code")}
    _empty = daily.iloc[:0]

    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    db = call_with_retry(lambda: pro.daily_basic(trade_date=as_of, fields="ts_code,pe_ttm,pb,total_mv"))
    sb = call_with_retry(lambda: pro.stock_basic(list_status="L", fields="ts_code,name,industry"))
    db_d = {str(r["ts_code"]): r for _, r in db.iterrows()}
    name_d = dict(zip(sb["ts_code"], sb["name"]))
    ind_d = dict(zip(sb["ts_code"], sb["industry"]))

    print(f"core 排序:全主板有完整核心表 {len(universe)} 只 → build_record…", flush=True)
    recs = []
    for i, code in enumerate(universe):
        ind = ind_d.get(code)
        if a.industry and a.industry not in str(ind or ""):
            continue
        try:
            rec = build_record(code, name=str(name_d.get(code, code)),
                               industry=str(ind) if isinstance(ind, str) and ind else "-", as_of=as_of,
                               daily_sub=cast(pd.DataFrame, daily_g.get(code, _empty)),
                               adj_sub=cast(pd.DataFrame, adj_g.get(code, _empty)),
                               mr=mr, fund_tables=load_core_tables(code), db_row=db_d.get(code))
        except Exception as e:
            print(f"  {code} 失败: {type(e).__name__}: {str(e)[:40]}", flush=True)
            continue
        if (rec["tier"]["grade"] not in a.grades):
            continue
        mv = (rec.get("valuation") or {}).get("mv_yi")
        if a.min_mv and (not isinstance(mv, (int, float)) or mv < a.min_mv):
            continue
        pe = (rec.get("valuation") or {}).get("pe_ttm")
        if a.max_pe is not None and (not isinstance(pe, (int, float)) or pe <= 0 or pe > a.max_pe):
            continue
        rec["holdscore"] = compute_holdscore(rec)
        recs.append(rec)
        if (i + 1) % 500 == 0:
            print(f"  …{i + 1}/{len(universe)}", flush=True)

    recs.sort(key=lambda x: x["holdscore"], reverse=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{as_of}_core.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=2, allow_nan=False)

    top = recs[: a.top]
    last = daily[daily["trade_date"] == as_of].set_index("ts_code")["close"].astype(float)
    scope = f"{a.industry}板块" if a.industry else "全主板"
    print(f"\n=== 全市场质地排序·核心版({scope}, as_of={as_of}, {len(recs)}只过滤后)→ {OUT_DIR}/{as_of}_core.json ===")
    print(f"{'#':>2} {'分':>4} {'档':>2} {'票':<9}{'行业':<7}{'净/扣/营%':>15}{'净现比':>6}{'PE':>5}{'市值亿':>7}  位置")
    for i, r in enumerate(top, 1):
        f, v, q = r["fundamental"], r["valuation"], r["quality"]
        code = r["ts_code"]
        px = last.get(code)
        s = daily[daily["ts_code"] == code].sort_values("trade_date")["close"].astype(float)
        ma20 = s.tail(20).mean() if len(s) >= 20 else float("nan")
        pos = pos_flag(float(px), float(ma20)) if (px is not None and ma20 == ma20) else "—"
        pp = lambda x: f"{x:+.0f}" if isinstance(x, (int, float)) else "—"
        nn = lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else "—"
        ncr = q.get("net_cash_ratio")
        g3 = f"{pp(f.get('np_yoy'))}/{pp(f.get('dedt_yoy'))}/{pp(f.get('rev_yoy'))}"
        print(f"{i:>2} {r['holdscore']:>4.0f} {r['tier']['grade']:>2} {r['name'][:8]:<9}{str(r['industry'])[:6]:<7}"
              f"{g3:>15}{(f'{ncr:.2f}' if isinstance(ncr,(int,float)) else '—'):>6}{nn(v.get('pe_ttm')):>5}"
              f"{nn(v.get('mv_yi')):>7}  {pos}")


if __name__ == "__main__":
    main()
