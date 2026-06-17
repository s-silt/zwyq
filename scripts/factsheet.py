"""Print the factual-layer factsheet for one A-share, from cached full-market data.

Descriptive only — real price, self-computed daily indicators (后复权), real
ranges, and the REAL 北向持股 level/change. No predictions, no trade setups, no
invented positioning. News is verified separately via web search at report time.

Usage: python scripts/factsheet.py <ts_code> [cache_dir]   e.g. 600519.SH
"""

import glob
import sys

import pandas as pd

from ashare_gauntlet.factsheet import build_factsheet


def load(cache_dir: str, endpoint: str) -> pd.DataFrame:
    files = glob.glob(f"{cache_dir}/{endpoint}/*.parquet")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(ts_code: str, cache_dir: str = "data/cache") -> None:
    daily = load(cache_dir, "daily")
    adj = load(cache_dir, "adj_factor")
    hk = load(cache_dir, "hk_hold")
    if daily.empty or (daily["ts_code"] == ts_code).sum() == 0:
        print(f"no cached daily data for {ts_code} in {cache_dir}")
        return

    fs = build_factsheet(ts_code, daily, adj, hk if not hk.empty else None)
    lo, mid, up = fs["boll"]  # type: ignore[misc]
    print(f"=== {ts_code} 事实层 (截至 {fs['as_of']};价格/指标前复权,与现价同标度) ===")
    print(f"  现价 {fs['close_raw']:.2f}  日涨跌 {fs['pct_chg_1d_pct']:+.2f}%  成交额 {fs['amount']:.0f} 千元")
    print(f"  区间: 20日 {fs['low_20d']:.2f}~{fs['high_20d']:.2f} | 60日 {fs['low_60d']:.2f}~{fs['high_60d']:.2f}")
    print(f"  EMA5 {fs['ema_short']:.2f}  EMA20 {fs['ema_long']:.2f}  RSI14 {fs['rsi']:.1f}  布林[下 {lo:.2f}/中 {mid:.2f}/上 {up:.2f}]")
    nr = fs.get("north_ratio")
    chg = fs.get("north_ratio_chg_5")
    if nr is not None and chg == chg:  # chg==chg is False only when NaN
        print(f"  北向持股占比 {nr:.2f}%  近5日变化 {chg:+.3f}pp  (真实数据)")
    elif nr is not None:
        print(f"  北向持股占比 {nr:.2f}%(快照)  日频变化不可得 —— 个股北向自 2024-08-19 改季度披露")
    else:
        print("  北向持股: 无日频数据(2024-08-19 起个股北向改季度披露)")
    print("  —— 描述现状,非预测、非买卖建议;相关新闻需 web 搜索核实后再附。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/factsheet.py <ts_code> [cache_dir]")
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/cache")
