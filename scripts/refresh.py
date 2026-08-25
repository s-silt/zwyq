"""Daily incremental refresh of the cache and current-month trade calendar.

Caches the authoritative full natural-month trade calendar, then pulls the last
few trading days (daily+adj+hk_hold) only up to today. Idempotent: cached data is
reused, so only genuinely new partitions hit the network. Needs a live token.

Usage: python scripts/refresh.py [lookback_days] [cache_dir]
"""

import calendar
import datetime as dt
import sys
from pathlib import Path

from ashare_gauntlet.config import CACHE_DIR, tushare_pro
from ashare_gauntlet.data.fetch import TokenExpiredError, fetch_market_day, trading_days_from_cal
from scripts.backfill import fetch_trade_cal

# moneyflow_hsgt is market-level (1 row/day): 北向总成交额(沪/深股通). Post
# 2024-08-19 its north_money column is TURNOVER, not net flow — see
# factsheet.NORTH_FLOW_SEMANTICS_CUTOFF. Cheap (1 row) so cached every refresh.
ENDPOINTS = ("daily", "adj_factor", "hk_hold", "moneyflow_hsgt")


def main(
    lookback_days: int = 10,
    cache_dir: str = CACHE_DIR,
    *,
    today: dt.date | None = None,
) -> None:
    current_day = today or dt.date.today()
    start = current_day - dt.timedelta(days=lookback_days)
    month_start = current_day.replace(day=1)
    month_end = current_day.replace(
        day=calendar.monthrange(current_day.year, current_day.month)[1]
    )
    pro = tushare_pro()
    cal = fetch_trade_cal(
        pro,
        month_start.strftime("%Y%m%d"),
        month_end.strftime("%Y%m%d"),
        cache_dir,
        strict=True,
    )
    assert cal is not None
    end = current_day.strftime("%Y%m%d")
    first = start.strftime("%Y%m%d")
    days = [day for day in trading_days_from_cal(cal) if first <= day <= end]
    print(f"refresh: 检查 {len(days)} 个交易日 {days[0] if days else '-'}..{days[-1] if days else '-'}", flush=True)

    new_files = 0
    try:
        for day in days:
            for endpoint in ENDPOINTS:
                existed = Path(cache_dir, endpoint, f"{day}.parquet").exists()
                fetch_market_day(pro, endpoint, day, cache_dir)
                if not existed:
                    new_files += 1
        print(f"refresh done: 新增 {new_files} 个缓存文件(最新交易日 {days[-1] if days else '-'})", flush=True)
    except TokenExpiredError as exc:
        print(f"!!! token 耗尽: {exc} — 已增量到此,新增 {new_files} 个文件", flush=True)


if __name__ == "__main__":
    lb = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    cdir = next((a for a in sys.argv[1:] if "/" in a), "data/cache")
    main(lb, cdir)
