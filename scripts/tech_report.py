"""Daily tech-stock factual report (data layer).

Per-stock technical facts + whole-market cross-sectional percentile for a tech
watchlist (头部/中部), computed OFFLINE from the cached full-market data — zero
token, reproducible. DESCRIPTIVE ONLY: trend / momentum / position / relative
strength, never a buy/sell call.

The macro narrative and northbound aggregate flow are a LIVE web layer (the
script can't web-search); they are appended at delivery time, with sources.
Per-stock northbound holdings are no longer daily (changed to quarterly
2024-08-19), so they are intentionally not here.

Usage: python scripts/tech_report.py [cache_dir] [ts_code ...]
"""

import glob
import sys

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR
from ashare_gauntlet.factsheet import (
    NORTH_FLOW_SEMANTICS_CUTOFF,
    daily_tech_facts,
    entry_rank,
    market_returns,
    north_flow_disclosure,
    north_turnover,
)
from ashare_gauntlet.fundamentals import index_changes

# Macro indices shown in the report header (from cached index_daily; pull via
# scripts/fundamentals.py). 宏观叙事仍走 web,但指数点位现在接口直出。
INDEX_HEADER = (("000001.SH", "上证"), ("399001.SZ", "深成"), ("399006.SZ", "创业板"), ("000688.SH", "科创50"))

# Default tech watchlist (头部 large-cap leaders + 中部 mid-cap), labelled tier·theme.
DEFAULT_WATCH: dict[str, str] = {
    "601138.SH": "工业富联·头部·AI服务器",
    "002475.SZ": "立讯精密·头部·消费电子",
    "688981.SH": "中芯国际·头部·晶圆代工",
    "300308.SZ": "中际旭创·头部·光模块CPO",
    "002371.SZ": "北方华创·头部·半导体设备",
    "000063.SZ": "中兴通讯·头部·通信",
    "603986.SH": "兆易创新·中部·存储MCU",
    "688012.SH": "中微公司·中部·刻蚀设备",
    "300782.SZ": "卓胜微·中部·射频芯片",
    "688008.SH": "澜起科技·中部·内存接口",
}


def _load(cache_dir: str, endpoint: str) -> pd.DataFrame:
    files = glob.glob(f"{cache_dir}/{endpoint}/*.parquet")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(cache_dir: str = CACHE_DIR, watch: dict[str, str] | None = None) -> None:
    watch = watch or DEFAULT_WATCH
    daily = _load(cache_dir, "daily")
    adj = _load(cache_dir, "adj_factor")
    mf = _load(cache_dir, "moneyflow_hsgt")
    idx = _load(cache_dir, "index_daily")
    if daily.empty:
        print(f"no cached data in {cache_dir}")
        return

    as_of = str(daily["trade_date"].max())
    mret = market_returns(daily, adj, (5, 20))
    # Market breadth on the latest session (a real market-state fact).
    last = daily[daily["trade_date"] == as_of]
    up = int((last["pct_chg"] > 0).sum()) if "pct_chg" in last.columns else -1
    dn = int((last["pct_chg"] < 0).sum()) if "pct_chg" in last.columns else -1

    print(f"=== 科技股事实报告 (截至 {as_of};前复权,分位=全市场横截面) ===")
    ich = index_changes(idx, as_of) if not idx.empty else {}
    parts = [f"{name}{c['close']:.0f}({c['pct_chg']:+.2f}%)"
             for code, name in INDEX_HEADER
             if (c := ich.get(code)) and c["close"] is not None]
    if parts:
        print("宏观指数(接口): " + " | ".join(parts))
    print(f"市场广度(当日): 涨 {up} / 跌 {dn}   | 宏观叙事为 live web 层,交付时附(带 source)")
    print(north_flow_disclosure())
    # The one official daily aggregate that survived: 北向总成交额 (NOT net flow).
    if not mf.empty and (mf["trade_date"] == as_of).any() and as_of >= NORTH_FLOW_SEMANTICS_CUTOFF:
        nt = north_turnover(mf, as_of)
        two_mkt_yi = float(last["amount"].sum()) / 1e5  # cache amount 千元 -> 亿元
        pct = nt["total_yi"] / two_mkt_yi * 100 if two_mkt_yi else float("nan")
        print(
            f"北向总成交额(官方,成交额非净额): {nt['total_yi']:.0f} 亿"
            f"(沪{nt['hgt_yi']:.0f}+深{nt['sgt_yi']:.0f}) | 占两市A股成交 {pct:.1f}%"
        )
    print("排序=入场纪律分(强势+趋势确认,惩罚追高/接刀);这是组织信息的镜头,非预测会涨。")
    print("—— 描述现状,非预测/非买卖建议 ——")

    rows = []
    for code, label in watch.items():
        if (daily["ts_code"] == code).sum() < 60:
            rows.append((label, code, None, (-1e9, "数据不足")))
            continue
        f = daily_tech_facts(code, daily, adj, mret)
        rows.append((label, code, f, entry_rank(f)))

    rows.sort(key=lambda r: (r[3][0] if r[3][0] is not None else -1e9), reverse=True)
    for label, code, f, (score, tag) in rows:
        if f is None:
            print(f"{label} [{code}] 数据不足")
            continue
        p20 = f.get("pct20", float("nan"))
        sc = f"{score:>3.0f}" if score is not None else "  —"  # 入场分缺失(契约C2)
        print(
            f"[{sc}〕{tag}〕{label} {f['close']:.2f} | {f['trend']} | RSI {f['rsi']:.0f}{f['rsi_dir']} "
            f"| 距60高 {f['dist_60d_high_pct']:+.0f}% | 近5 {f['ret5_pct']:+.1f}% 近20 {f['ret20_pct']:+.1f}%(分位{p20:.0f}) "
            f"| 量比 {f['vol_ratio']:.2f}"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    codes = [a for a in args if a.endswith((".SH", ".SZ", ".BJ"))]
    cdir = next((a for a in args if "/" in a), "data/cache")
    main(cdir, {c: c for c in codes} if codes else None)
