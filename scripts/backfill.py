"""Backfill the per-day full-market parquet cache for a date range.

Usage:
    python scripts/backfill.py <start YYYYMMDD> <end YYYYMMDD> [cache_dir]

Idempotent: already-cached (endpoint, trade_date) pulls are skipped, so it is
safe to re-run / resume after an interruption.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from ashare_gauntlet.data.fetch import (
    TokenExpiredError,
    fetch_market_day,
    trading_days_from_cal,
)
from ashare_gauntlet.data.tushare_source import make_pro_api

# v1 pulls only what the reversal gauntlet needs. Per-day full-market pulls are
# credit-efficient: one call = the whole market for a day (~480 calls/year for
# daily+adj), vs ~10k for per-stock. Modest concurrency speeds it up without a
# credit burst — the mirror drops the large daily response sometimes and retry
# recovers it.
V1_ENDPOINTS: tuple[str, ...] = ("daily", "adj_factor")
MAX_WORKERS = 8


def _pull_day(pro: object, day: str, cache_dir: str) -> list[str]:
    errs: list[str] = []
    for endpoint in V1_ENDPOINTS:
        try:
            fetch_market_day(pro, endpoint, day, cache_dir)
        except TokenExpiredError:
            raise  # fatal — let the pool abort
        except Exception as exc:  # noqa: BLE001 — record + continue
            errs.append(f"{day}/{endpoint}:{type(exc).__name__}")
    return errs


def main(start: str, end: str, cache_dir: str = "data/cache") -> None:
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    days = trading_days_from_cal(cal)
    print(
        f"{len(days)} trading days {start}..{end}; endpoints={V1_ENDPOINTS}; "
        f"~{len(days) * len(V1_ENDPOINTS)} credit-calls (before retries); workers={MAX_WORKERS}",
        flush=True,
    )
    failed: list[str] = []
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            # Newest-first: the recent (usable, contiguous) window fills first so a
            # first verdict is possible while the backfill keeps marching back.
            futures = [pool.submit(_pull_day, pro, day, cache_dir) for day in reversed(days)]
            for future in as_completed(futures):
                failed.extend(future.result())
                done += 1
                if done % 20 == 0 or done == len(days):
                    print(f"  [{done}/{len(days)}] failed_calls={len(failed)}", flush=True)
    except TokenExpiredError as exc:
        print(f"!!! TOKEN EXPIRED / OUT OF CREDITS: {exc} — aborting; cached so far is kept", flush=True)
        return
    print(f"backfill done; failed_calls={len(failed)}", flush=True)
    if failed:
        print("failed sample:", failed[:10], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "data/cache")
