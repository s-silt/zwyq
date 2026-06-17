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

from ashare_gauntlet.factsheet import daily_tech_facts, market_returns

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


def main(cache_dir: str = "data/cache", watch: dict[str, str] | None = None) -> None:
    watch = watch or DEFAULT_WATCH
    daily = _load(cache_dir, "daily")
    adj = _load(cache_dir, "adj_factor")
    if daily.empty:
        print(f"no cached data in {cache_dir}")
        return

    as_of = daily["trade_date"].max()
    mret = market_returns(daily, adj, (5, 20))
    # Market breadth on the latest session (a real market-state fact).
    last = daily[daily["trade_date"] == as_of]
    up = int((last["pct_chg"] > 0).sum()) if "pct_chg" in last.columns else -1
    dn = int((last["pct_chg"] < 0).sum()) if "pct_chg" in last.columns else -1

    print(f"=== 科技股事实报告 (截至 {as_of};前复权,分位=全市场横截面) ===")
    print(f"市场广度(当日): 涨 {up} / 跌 {dn}   | 宏观叙事 + 北向净流入为 live web 层,交付时附")
    print("—— 描述现状,非预测/非买卖建议 ——")

    rows = []
    for code, label in watch.items():
        if (daily["ts_code"] == code).sum() < 60:
            rows.append((label, code, None))
            continue
        rows.append((label, code, daily_tech_facts(code, daily, adj, mret)))

    rows.sort(key=lambda r: (r[2]["pct20"] if r[2] and "pct20" in r[2] else -1.0), reverse=True)
    for label, code, f in rows:
        if f is None:
            print(f"{label} [{code}] 数据不足")
            continue
        p5 = f.get("pct5", float("nan"))
        p20 = f.get("pct20", float("nan"))
        print(
            f"{label} {f['close']:.2f} | {f['trend']}(价{'>' if f['close'] > f['ema_long'] else '<'}EMA20) "
            f"| RSI {f['rsi']:.0f}{f['rsi_dir']} | 距60高 {f['dist_60d_high_pct']:+.0f}% "
            f"| 近5 {f['ret5_pct']:+.1f}%(分位{p5:.0f}) 近20 {f['ret20_pct']:+.1f}%(分位{p20:.0f}) "
            f"| 量比 {f['vol_ratio']:.2f}"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    codes = [a for a in args if a.endswith((".SH", ".SZ", ".BJ"))]
    cdir = next((a for a in args if "/" in a), "data/cache")
    main(cdir, {c: c for c in codes} if codes else None)
