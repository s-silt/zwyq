"""Reusable stock screen — the fixed '维度优' lens as one command.

Defaults encode the user's mode (see the stock-analysis-mode memory):
沪市主板 + 通信电子 + 盈利 / 不破位(非空头) / 仍有强度 / 估值不离谱.
Technical facts are computed OFFLINE from the cache; PE/PB/行业 come from the
interface (needs a token) — without one it falls back to a technical-only screen.

Usage: PYTHONIOENCODING=utf-8 python scripts/screen.py
       [--board sh_main|all] [--sector comm_elec|tech|all]
       [--max-price 300] [--min-pct20 50] [--max-dist60 -12]
       [--max-pe 80] [--max-pb N] [--no-profitable]
       [--sort pct20|pe_ttm|entry|dist60] [--top 30] [--cache data/cache]
"""

import argparse
import glob
import os
from typing import cast

import pandas as pd

from ashare_gauntlet.factsheet import daily_tech_facts, entry_rank, market_returns
from ashare_gauntlet.screen import board_of, screen_candidates

SECTORS: dict[str, tuple[str, ...] | None] = {
    "comm_elec": ("通信设备", "通信运营", "元器件", "半导体", "光学光电子", "消费电子",
                  "IT设备", "电子元件", "集成电路", "光电子"),
    "tech": ("通信设备", "通信运营", "元器件", "半导体", "光学光电子", "消费电子", "IT设备",
             "电子元件", "集成电路", "光电子", "软件服务", "计算机", "互联网", "电气设备"),
    "all": None,
}
_SORT = {"pct20": ("pct20", False), "pe_ttm": ("pe_ttm", True),
         "entry": ("entry", False), "dist60": ("dist60", True)}


def _load(cache_dir: str, endpoint: str) -> pd.DataFrame:
    files = glob.glob(f"{cache_dir}/{endpoint}/*.parquet")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _facts_table(daily: pd.DataFrame, adj: pd.DataFrame, codes: list[str],
                 mr: dict[int, pd.Series]) -> pd.DataFrame:
    dsub = cast(pd.DataFrame, daily[daily["ts_code"].isin(codes)])
    asub = cast(pd.DataFrame, adj[adj["ts_code"].isin(codes)])
    recs = []
    for code in codes:
        f = daily_tech_facts(code, dsub, asub, mr)
        score, _ = entry_rank(f)
        recs.append({
            "ts_code": code, "close": f["close"], "trend": f["trend"], "rsi": f["rsi"],
            "dist60": f["dist_60d_high_pct"], "pct20": float(cast(float, f.get("pct20", 0.0))),
            "ret20": f["ret20_pct"], "entry": score,
        })
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="sh_main", choices=["sh_main", "all"])
    ap.add_argument("--sector", default="comm_elec", choices=list(SECTORS))
    ap.add_argument("--max-price", type=float, default=300.0)
    ap.add_argument("--min-pct20", type=float, default=50.0)
    ap.add_argument("--max-dist60", type=float, default=None)
    ap.add_argument("--max-pe", type=float, default=80.0)
    ap.add_argument("--max-pb", type=float, default=None)
    ap.add_argument("--no-profitable", action="store_true")
    ap.add_argument("--sort", default="pct20", choices=list(_SORT))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()

    daily, adj = _load(a.cache, "daily"), _load(a.cache, "adj_factor")
    if daily.empty:
        print(f"no cached data in {a.cache}")
        return
    as_of = str(daily["trade_date"].max())
    mr = market_returns(daily, adj, (5, 20))
    rank20 = mr[20].rank(pct=True) * 100
    last = daily[daily["trade_date"] == as_of].set_index("ts_code")["close"].astype(float)
    counts = daily["ts_code"].value_counts()
    boards = ["沪主板"] if a.board == "sh_main" else None

    counts_d, rank_d, last_d = counts.to_dict(), rank20.to_dict(), last.to_dict()
    codes = [str(c) for c in mr[20].index
             if (boards is None or board_of(str(c)) in boards)
             and counts_d.get(c, 0) >= 60 and rank_d.get(c, 0.0) >= a.min_pct20
             and c in last_d and 0.0 < last_d[c] <= a.max_price]
    df = _facts_table(daily, adj, codes, mr)

    have_token = "TUSHARE_TOKEN" in os.environ and "TUSHARE_HTTP_URL" in os.environ
    if have_token:
        from ashare_gauntlet.data.fetch import call_with_retry
        from ashare_gauntlet.data.tushare_source import make_pro_api
        pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
        db = call_with_retry(lambda: pro.daily_basic(trade_date=as_of, fields="ts_code,pe_ttm,pb,total_mv"))
        sb = call_with_retry(lambda: pro.stock_basic(
            exchange="SSE" if a.board == "sh_main" else "", list_status="L",
            fields="ts_code,name,industry"))
        df = df.merge(db, on="ts_code", how="left").merge(sb, on="ts_code", how="left")
        industries = SECTORS[a.sector]
    else:
        print("!! 无 TUSHARE_TOKEN/URL —— 跳过估值+行业过滤(纯技术面)", flush=True)
        for col in ("pe_ttm", "pb", "total_mv", "name", "industry"):
            df[col] = pd.NA
        industries = None

    sort_by, ascending = _SORT[a.sort]
    out = screen_candidates(
        df, boards=boards, max_price=a.max_price, min_pct20=a.min_pct20,
        max_dist60=a.max_dist60, trends=("多头", "纠缠"),
        require_profitable=have_token and not a.no_profitable,
        max_pe=a.max_pe if have_token else None,
        max_pb=a.max_pb if have_token else None,
        industries=industries, sort_by=sort_by, ascending=ascending, top=a.top,
    )

    print(f"=== 筛选 (截至 {as_of}) board={a.board} sector={a.sector} 价≤{a.max_price:.0f} "
          f"pct20≥{a.min_pct20:.0f} 距60高≤{a.max_dist60} PE≤{a.max_pe:.0f} "
          f"盈利={have_token and not a.no_profitable} → {len(out)} 只 ===")
    print("—— 维度优初筛(技术+估值);基本面/消息面需 named workflow `factcheck` 核实,非荐股 ——")
    for _, r in out.iterrows():
        pe = r.get("pe_ttm"); pe = f"{float(pe):.0f}" if pd.notna(pe) else "-"
        pb = r.get("pb"); pb = f"{float(pb):.1f}" if pd.notna(pb) else "-"
        mv = r.get("total_mv"); mv = f"{float(mv) / 1e4:.0f}" if pd.notna(mv) else "-"
        name = "" if pd.isna(r.get("name")) else str(r.get("name"))
        ind = "" if pd.isna(r.get("industry")) else str(r.get("industry"))
        print(f"[{r['entry']:>3.0f}] {name:<6}{r['ts_code']} {ind:<7} {r['close']:7.2f} "
              f"{r['trend']:<4} RSI{r['rsi']:.0f} 距60高{r['dist60']:+.0f}% "
              f"近20{r['ret20']:+.1f}%(分位{r['pct20']:.0f}) PE{pe} PB{pb} 市值{mv}亿")


if __name__ == "__main__":
    main()
