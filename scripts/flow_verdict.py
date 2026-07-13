"""Northbound-flow (北向资金) signal verdict.

⚠️ LEGACY(2026-07 标注):早期一次性验证脚本,消费未标定阈值的旧 gauntlet
(见 ashare_gauntlet/gauntlet.py 头注)。不在现役链路;且 2024-08-19 后
moneyflow_hsgt 语义变更(净流→成交额,见 ashare_gauntlet/factsheet.py 的
NORTH_FLOW_SEMANTICS_CUTOFF),北向净流信号已不可延续。留档备查。

Pulls hk_hold (北向持股明细) for the dates already cached as daily prices, builds
the flow panel (signal = -Δ北向占比 over k days; fwd_ret from cached prices), and
runs the same gauntlet. Needs a live token for the hk_hold pull (~25 credits per
full-market day); the price/forward side is fully cached.

Usage: python scripts/flow_verdict.py [cache_dir]
"""

import glob
import os
import sys

import pandas as pd

from ashare_gauntlet.data.fetch import TokenExpiredError, fetch_market_day
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.gauntlet import run_gauntlet
from ashare_gauntlet.panel import assemble_flow_panel, universe_from_daily


def load_cached(cache_dir: str, endpoint: str) -> pd.DataFrame:
    files = glob.glob(f"{cache_dir}/{endpoint}/*.parquet")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(cache_dir: str = "data/cache") -> None:
    daily = load_cached(cache_dir, "daily")
    adj = load_cached(cache_dir, "adj_factor")
    dates = sorted(daily["trade_date"].unique())
    print(f"price dates={len(dates)} ({dates[0]}..{dates[-1]}); pulling hk_hold...", flush=True)

    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    try:
        for i, day in enumerate(dates, 1):
            fetch_market_day(pro, "hk_hold", day, cache_dir)
            if i % 20 == 0 or i == len(dates):
                print(f"  hk_hold [{i}/{len(dates)}]", flush=True)
    except TokenExpiredError as exc:
        print(f"!!! TOKEN EXPIRED / OUT OF CREDITS: {exc} — using hk_hold cached so far", flush=True)

    hk = load_cached(cache_dir, "hk_hold")
    print(
        f"hk_hold rows={len(hk)} codes={hk['ts_code'].nunique()} dates={hk['trade_date'].nunique()}",
        flush=True,
    )

    universe = universe_from_daily(daily)
    panel = assemble_flow_panel(
        daily, adj, hk, universe, k=5, h=5, rebalance=5, min_amount=50000.0, min_list_days=0
    )
    print(
        f"flow panel: {len(panel)} rows over {panel['trade_date'].nunique()} dates, "
        f"{panel['ts_code'].nunique()} 北向-eligible names",
        flush=True,
    )

    rep = run_gauntlet(panel, n_buckets=10, periods_per_year=50.0)
    print("=" * 64)
    print(f"NORTHBOUND-FLOW VERDICT: {rep.verdict}")
    print(f"  decision dates:          {rep.n_decisions}")
    print(f"  long-short Sharpe:       {rep.long_short_sharpe:.3f}")
    print(f"  long-only excess Sharpe: {rep.long_only_excess_sharpe:.3f}  (demeaned selection)")
    print(f"  top-symbol concentration:{rep.top_symbol} = {rep.top_symbol_share:.1%} of gross")
    print("  OOS splits (cut / in_sharpe / oos_sharpe):")
    print(rep.oos.to_string(index=False))
    if rep.reasons:
        print("  NO_GO reasons:")
        for reason in rep.reasons:
            print(f"    - {reason}")
    print("=" * 64)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cache")
