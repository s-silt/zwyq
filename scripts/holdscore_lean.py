"""全市场质地排序·精简版 —— 只用本地 fina_indicator(全主板)算质地,真·全市场首过。

接 `scripts/backfill_fina --mode lean`(把全主板 fina_indicator 拉到本地)之后跑。对每只主板:
lean_tier 定档 + compute_holdscore 算持有分(三增 + 扣非真 + 经营现金流方向 + ROE + 估值),按持有分排序;
位置(MA20)只对 Top 标注买入时机,不参与排序。

这就把关① 质地排序从"daily_basic 价值预筛近似"升级成"真·全主板基本面排序"。
注:精简版用 fina_indicator 单表(无预警表),抓不到 ⛔ 地雷/质押减持——是"全市场首过";
Top 名单再走完整 build_record/factcheck 深核(完整版数据见 backfill_fina --mode full)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.holdscore_lean
       [--board main] [--industry 关键词] [--max-pe N] [--min-mv 亿] [--grades 🟢🟡] [--top 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.record import compute_holdscore, lean_tier
from ashare_gauntlet.screen import board_of

CACHE = "data/cache"
OUT_DIR = "data/holdscore"
MAIN = ("沪主板", "深主板")
FCOLS = ["ts_code", "end_date", "netprofit_yoy", "dt_netprofit_yoy", "tr_yoy", "ocfps", "roe"]


def latest_fina() -> pd.DataFrame:
    """每只 fina_indicator 取最新报告期一行(全主板首过的基本面来源)。"""
    rows = []
    for f in glob.glob(f"{CACHE}/fina_indicator/*.parquet"):
        try:
            df = pd.read_parquet(f, columns=FCOLS)
        except Exception:
            df = pd.read_parquet(f)
        if df.empty:
            continue
        rows.append(df.sort_values("end_date").iloc[-1])
    return pd.DataFrame(rows).reset_index(drop=True)


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
    ap.add_argument("--industry", default=None, help="行业子串过滤(如 半导体/元器件/通信)")
    ap.add_argument("--max-pe", type=float, default=None)
    ap.add_argument("--min-mv", type=float, default=0.0, help="最小总市值(亿)")
    ap.add_argument("--grades", default="🟢🟡", help="保留的档(默认 🟢🟡,剔 🔴)")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)

    load_env_local()
    fina = latest_fina()
    if a.board == "main":
        fina = fina[fina["ts_code"].apply(lambda c: board_of(str(c)) in MAIN)].reset_index(drop=True)

    daily = pd.read_parquet(sorted(glob.glob(f"{CACHE}/daily/*.parquet"))[-1])
    as_of = str(daily["trade_date"].max())

    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    db = call_with_retry(lambda: pro.daily_basic(trade_date=as_of, fields="ts_code,pe_ttm,pb,total_mv"))
    sb = call_with_retry(lambda: pro.stock_basic(list_status="L", fields="ts_code,name,industry"))
    pe_d = dict(zip(db["ts_code"], db["pe_ttm"]))
    pb_d = dict(zip(db["ts_code"], db["pb"]))
    mv_d = {c: float(m) / 1e4 for c, m in zip(db["ts_code"], db["total_mv"])}  # 万元→亿
    name_d = dict(zip(sb["ts_code"], sb["name"]))
    ind_d = dict(zip(sb["ts_code"], sb["industry"]))

    recs = []
    for _, r in fina.iterrows():
        code = str(r["ts_code"])
        ind = str(ind_d.get(code) or "")
        if a.industry and a.industry not in ind:
            continue
        mv = _f(mv_d.get(code))
        if a.min_mv and (mv is None or mv < a.min_mv):
            continue
        grade = lean_tier(r.get("netprofit_yoy"), r.get("dt_netprofit_yoy"), r.get("tr_yoy"),
                          ocfps=r.get("ocfps"), roe=r.get("roe"))
        if grade not in a.grades:
            continue
        pe = pe_d.get(code)
        if a.max_pe is not None and (pe is None or pe <= 0 or pe > a.max_pe):
            continue
        rec = {
            "ts_code": code, "name": str(name_d.get(code, code)), "industry": ind or "-",
            "tier": {"grade": grade},
            "fundamental": {"np_yoy": _f(r.get("netprofit_yoy")), "dedt_yoy": _f(r.get("dt_netprofit_yoy")),
                            "rev_yoy": _f(r.get("tr_yoy"))},
            "quality": {"net_cash_ratio": _f(r.get("ocfps"))},  # ocfps>0 作现金流方向代理
            "valuation": {"pe_ttm": _f(pe), "pb": _f(pb_d.get(code)), "peg": None, "mv_yi": _f(mv)},
        }
        rec["holdscore"] = compute_holdscore(rec)
        recs.append(rec)

    recs.sort(key=lambda x: x["holdscore"], reverse=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{as_of}_lean.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=2, allow_nan=False)

    # 位置只对 Top N 算(快)
    top = recs[: a.top]
    topcodes = {r["ts_code"] for r in top}
    dall = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
                      for f in sorted(glob.glob(f"{CACHE}/daily/*.parquet"))], ignore_index=True)
    dall = dall[dall["ts_code"].isin(list(topcodes))]
    ma = {}
    for code, g in dall.groupby("ts_code"):
        s = g.sort_values("trade_date")["close"].astype(float)
        ma[code] = (s.iloc[-1], s.tail(20).mean() if len(s) >= 20 else float("nan"))

    scope = f"{a.industry}板块" if a.industry else "全主板"
    print(f"=== 全市场质地排序·精简版({scope}, as_of={as_of}, 共{len(recs)}只过滤后)→ {OUT_DIR}/{as_of}_lean.json ===")
    print(f"{'#':>2} {'分':>4} {'档':>2} {'票':<9}{'行业':<7}{'净/扣/营%':>15}{'PE':>5}{'市值亿':>7}  位置")
    for i, r in enumerate(top, 1):
        f, v = r["fundamental"], r["valuation"]
        px, ma20 = ma.get(r["ts_code"], (None, float("nan")))
        pos = pos_flag(float(px), float(ma20)) if (px is not None and ma20 == ma20) else "—"
        pp = lambda x: f"{x:+.0f}" if isinstance(x, (int, float)) else "—"
        nn = lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else "—"
        g3 = f"{pp(f.get('np_yoy'))}/{pp(f.get('dedt_yoy'))}/{pp(f.get('rev_yoy'))}"
        print(f"{i:>2} {r['holdscore']:>4.0f} {r['tier']['grade']:>2} {r['name'][:8]:<9}{str(r['industry'])[:6]:<7}"
              f"{g3:>15}{nn(v.get('pe_ttm')):>5}{nn(v.get('mv_yi')):>7}  {pos}")


def _f(x: object) -> float | None:
    if isinstance(x, (int, float)) and x == x:
        return float(x)
    return None


if __name__ == "__main__":
    main()
