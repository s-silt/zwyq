"""Load the cached per-symbol data, assemble the panel, run the gauntlet, print
the GO/NO_GO verdict.

v1 universe-wide reversal: past-5-day back-adjusted return signal, weekly
(non-overlapping 5-day) holds, deciles, entry at next open. Liquidity floor on
daily turnover (amount, 千元); 次新 excluded; 一字板/turnover refinements pending.

Usage: python scripts/first_verdict.py [cache_dir]
"""

import glob
import sys

import pandas as pd

from ashare_gauntlet.gauntlet import run_gauntlet
from ashare_gauntlet.panel import assemble_panel, universe_from_daily


def load_cached(cache_dir: str, endpoint: str) -> pd.DataFrame:
    files = glob.glob(f"{cache_dir}/{endpoint}/*.parquet")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(cache_dir: str = "data/cache") -> None:
    daily = load_cached(cache_dir, "daily")
    adj = load_cached(cache_dir, "adj_factor")
    print(
        f"daily rows={len(daily)} codes={daily['ts_code'].nunique()} "
        f"dates={daily['trade_date'].nunique()} ({daily['trade_date'].min()}..{daily['trade_date'].max()}) "
        f"| adj rows={len(adj)}",
        flush=True,
    )

    # Universe derived from cached prices (token exhausted). 次新 filter off
    # (min_list_days=0) since list_date is unavailable — preliminary.
    universe = universe_from_daily(daily)
    panel = assemble_panel(
        daily, adj, universe, k=5, h=5, rebalance=5, min_amount=50000.0, min_list_days=0
    )
    print(
        f"panel: {len(panel)} decision-rows over {panel['trade_date'].nunique()} dates, "
        f"{panel['ts_code'].nunique()} names",
        flush=True,
    )

    rep = run_gauntlet(panel, n_buckets=10, periods_per_year=50.0)
    print("=" * 64)
    print(f"VERDICT: {rep.verdict}")
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
