"""Backfill the per-day full-market parquet cache for a date range.

Usage:
    python scripts/backfill.py <start YYYYMMDD> <end YYYYMMDD> [cache_dir]

Idempotent: already-cached (endpoint, trade_date) pulls are skipped, so it is
safe to re-run / resume after an interruption.
"""

import os
import sys

from ashare_gauntlet.data.fetch import MARKET_ENDPOINTS, fetch_market_day, trading_days_from_cal
from ashare_gauntlet.data.tushare_source import make_pro_api


def main(start: str, end: str, cache_dir: str = "data/cache") -> None:
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    days = trading_days_from_cal(cal)
    print(f"{len(days)} trading days {start}..{end}; endpoints={MARKET_ENDPOINTS}", flush=True)
    for i, day in enumerate(days, 1):
        last_shape: tuple[int, ...] | None = None
        for endpoint in MARKET_ENDPOINTS:
            last_shape = fetch_market_day(pro, endpoint, day, cache_dir).shape
        if i % 10 == 0 or i == len(days):
            print(f"  [{i}/{len(days)}] {day} (last pull {last_shape})", flush=True)
    print("backfill done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "data/cache")
