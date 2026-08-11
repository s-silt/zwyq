"""幸存者清单(核心池):跨沪深主板"四关漏斗"的前三关,自动跑出 🟢 强干净候选。

关① 技术(有强度·不破位)+ 关② 估值(PE/价格合理·盈利)—— 复用 screen_candidates;
关③ 基本面(净利&扣非&营收三增 + 经营现金流>0 + 无警示 = 🟢)—— build_record/tier_of;
关④ web 核实(named workflow `factcheck`)对幸存者按需深挖(web 重,非本脚本)。

落库 data/survivors/<as_of>.json + 打印;周期重跑即更新核心池。基本面缺数据者保守不入池。
非荐股:只做"四关前三关"的事实层漏斗;买卖与 conviction 是人的判断。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.survivors
       [--board sh_main|sz_main|main|all] [--min-pct20 60] [--max-pe 50]
       [--max-price 300] [--grades 🟢|🟢🟡] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, cast

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR as CACHE, SURVIVORS_DIR as OUT_DIR, tushare_pro
from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry, fetch_symbol_table
from ashare_gauntlet.factsheet import daily_tech_facts, entry_rank, market_returns
from ashare_gauntlet.record import build_record
from ashare_gauntlet.screen import board_of, screen_candidates

SYM = ("income", "fina_indicator", "balancesheet", "cashflow", "share_float",
       "pledge_stat", "stk_holdertrade", "namechange", "forecast", "express")
_TIER_ORDER = {"🟢": 0, "🟡": 1, "🔴": 2, "⛔": 3}
_BOARDS: dict[str, list[str] | None] = {
    "sh_main": ["沪主板"], "sz_main": ["深主板"], "main": ["沪主板", "深主板"], "all": None,
}


def pick_survivors(records: list[dict[str, Any]],
                   grades: tuple[str, ...] = ("🟢",)) -> list[dict[str, Any]]:
    """选出达标质地档的幸存者,按 tier 优先级(🟢→🟡)+ entry 分降序(缺分排末)。纯函数。"""
    def _key(r: dict[str, Any]) -> tuple[int, float]:
        grade = str((r.get("tier") or {}).get("grade") or "")
        score = (r.get("entry") or {}).get("score")
        score_f = float(score) if isinstance(score, (int, float)) else -1.0
        return (_TIER_ORDER.get(grade, 9), -score_f)

    return sorted((r for r in records if (r.get("tier") or {}).get("grade") in grades), key=_key)


def _load(ep: str) -> pd.DataFrame:
    fs = glob.glob(f"{CACHE}/{ep}/*.parquet")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _load_env(path: str = ".env.local") -> None:
    load_env_local(path)  # .env.local 权威覆盖(见 ashare_gauntlet.data.env)


def _tech_table(daily: pd.DataFrame, adj: pd.DataFrame, mr: dict[int, pd.Series], a: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """关①:主板 + 历史够 + 强度(全市场20日分位≥min)+ 价格 → 技术事实表。"""
    as_of = str(daily["trade_date"].max())
    rank20 = mr[20].rank(pct=True) * 100
    last = daily[daily["trade_date"] == as_of].set_index("ts_code")["close"].astype(float)
    counts = daily["ts_code"].value_counts()
    boards = _BOARDS[a.board]
    c_d, r_d, l_d = counts.to_dict(), rank20.to_dict(), last.to_dict()
    pre = [str(c) for c in mr[20].index
           if (boards is None or board_of(str(c)) in boards)
           and c_d.get(c, 0) >= 60 and r_d.get(c, 0.0) >= a.min_pct20
           and c in l_d and 0.0 < l_d[c] <= a.max_price]
    dsub = cast(pd.DataFrame, daily[daily["ts_code"].isin(pre)])
    asub = cast(pd.DataFrame, adj[adj["ts_code"].isin(pre)])
    recs = []
    for code in pre:
        f = daily_tech_facts(code, dsub, asub, mr)
        score, _ = entry_rank(f)
        recs.append({"ts_code": code, "close": f["close"], "trend": f["trend"], "rsi": f["rsi"],
                     "dist60": f["dist_60d_high_pct"], "pct20": f.get("pct20"),
                     "ret20": f["ret20_pct"], "entry": score})
    return pd.DataFrame(recs), as_of


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main", choices=list(_BOARDS))
    ap.add_argument("--min-pct20", type=float, default=60.0)
    ap.add_argument("--max-pe", type=float, default=50.0)
    ap.add_argument("--max-price", type=float, default=300.0)
    ap.add_argument("--grades", default="🟢")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--industries", default=None,
                    help="逗号分隔的行业子串过滤(substring),如 通信设备,元器件,半导体,IT设备,软件服务,互联网")
    a = ap.parse_args(argv)

    _load_env()
    daily, adj = _load("daily"), _load("adj_factor")
    if daily.empty:
        raise SystemExit("data/cache/daily 为空 —— 先 backfill")
    mr = market_returns(daily, adj, (5, 20))
    df, as_of = _tech_table(daily, adj, mr, a)

    pro = tushare_pro()
    # 估值表(daily_basic)常比行情(daily)晚发布几小时 —— 当天 as_of 取不到时
    # 优雅回退到最近一个有估值的交易日(PE/PB 慢变,隔一两个交易日口径可接受),
    # 免得 "估值还没发" 把整个筛选卡死(周五早跑的周更任务也吃这个保护)。
    db, val_as_of = None, as_of
    for d in sorted({str(x) for x in daily["trade_date"].unique()}, reverse=True)[:6]:
        try:
            cand = call_with_retry(lambda dd=d: pro.daily_basic(trade_date=dd, fields="ts_code,pe_ttm,pb,total_mv"))
        except Exception as e:  # 源暂不可用/未发布 —— 回退上一交易日,不炸整池
            print(f"  daily_basic({d}) 不可用({str(e)[:24]}),回退上一交易日…", file=sys.stderr, flush=True)
            continue
        if cand is not None and not cand.empty:
            db, val_as_of = cand, d
            break
    if db is None or db.empty:
        raise SystemExit("daily_basic 连续多日取不到 —— 估值数据源异常,稍后重试")
    if val_as_of != as_of:
        print(f"⚠ 估值口径={val_as_of}(as_of={as_of} 估值尚未发布);价格/动量仍为 {as_of},PE/PB 滞后 1+ 交易日", flush=True)
    sb = call_with_retry(lambda: pro.stock_basic(list_status="L", fields="ts_code,name,industry"))
    df = df.merge(db, on="ts_code", how="left").merge(sb, on="ts_code", how="left")

    # 关②:估值/盈利/趋势过滤(全行业,不限板块内部)
    inds = tuple(s for s in a.industries.split(",") if s) if a.industries else None
    cand = screen_candidates(df, boards=_BOARDS[a.board], max_price=a.max_price,
                             min_pct20=a.min_pct20, max_dist60=None, trends=("多头", "纠缠"),
                             require_profitable=True, max_pe=a.max_pe, max_pb=None,
                             industries=inds, sort_by="pct20", ascending=False, top=999)
    codes = list(cand["ts_code"])
    if a.limit:
        codes = codes[:a.limit]
    print(f"关①②(技术+估值)通过 {len(codes)} 只 → 关③逐只建档定基本面…", flush=True)

    name_d = dict(zip(sb["ts_code"], sb["name"]))
    ind_d = dict(zip(sb["ts_code"], sb["industry"]))
    db_d = {str(r["ts_code"]): r for _, r in db.iterrows()}
    records: list[dict[str, Any]] = []
    for i, code in enumerate(codes):
        try:
            ft = {t: fetch_symbol_table(pro, t, code, CACHE) for t in SYM}
            ind = ind_d.get(code)
            rec = build_record(
                code, name=str(name_d.get(code, code)),
                industry=ind if isinstance(ind, str) and ind else "-",
                as_of=as_of,
                daily_sub=cast(pd.DataFrame, daily[daily["ts_code"] == code]),
                adj_sub=cast(pd.DataFrame, adj[adj["ts_code"] == code]),
                mr=mr, fund_tables=ft, db_row=db_d.get(code))
            records.append(rec)
            print(f"  [{i + 1}/{len(codes)}] {rec['tier']['grade']} {rec['name']}({code})", flush=True)
        except Exception as e:  # 单只建档失败响亮上报、跳过,不炸整池(IO 层容错)
            print(f"  [{i + 1}/{len(codes)}] {code} 建档失败: {type(e).__name__}: {e}", file=sys.stderr)

    grades = tuple(g for g in a.grades if g in _TIER_ORDER) or ("🟢",)
    surv = pick_survivors(records, grades)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{as_of}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(surv, fh, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"\n=== 幸存者核心池(关①②③全过 {''.join(grades)}){len(surv)}/{len(records)} 只 → {out_path} ===")
    print("—— 关④ web 核实(named workflow `factcheck`)对这些幸存者按需深挖;事实层非荐股 ——")
    for r in surv:
        f, v, q = r["fundamental"], r["valuation"], r["quality"]
        unk = [k for k, s in (r.get("data_coverage") or {}).items() if s == "unknown"]
        unk_s = f" ⚐未取到×{len(unk)}" if unk else ""
        pp = lambda x: f"{x:+.0f}%" if isinstance(x, (int, float)) else "—"
        nn = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "—"
        print(f"  {r['tier']['grade']} {r['name']}({r['ts_code']}) {r['industry']} | "
              f"净利{pp(f.get('np_yoy'))} 扣非{pp(f.get('dedt_yoy'))} 营收{pp(f.get('rev_yoy'))} | "
              f"净现比{nn(q.get('net_cash_ratio'))} PE{nn(v.get('pe_ttm'))} PEG{nn(v.get('peg'))}{unk_s}")


if __name__ == "__main__":
    main()
