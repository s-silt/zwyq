"""Daily incremental refresh of the cache.

Pulls the last few trading days (daily+adj+hk_hold) up to today into the cache.
Idempotent: already-cached days are skipped, so only genuinely new days hit the
network — cheap (~1-3 new days x 3 endpoints per run). Needs a live token.

Usage: python scripts/refresh.py [lookback_days] [cache_dir]
"""

import datetime as dt
import sys
from pathlib import Path

from ashare_gauntlet.config import CACHE_DIR, tushare_pro
from ashare_gauntlet.data.fetch import TokenExpiredError, fetch_market_day, trading_days_from_cal

# moneyflow_hsgt is market-level (1 row/day): 北向总成交额(沪/深股通). Post
# 2024-08-19 its north_money column is TURNOVER, not net flow — see
# factsheet.NORTH_FLOW_SEMANTICS_CUTOFF. Cheap (1 row) so cached every refresh.
ENDPOINTS = ("daily", "adj_factor", "hk_hold", "moneyflow_hsgt")


def main(lookback_days: int = 10, cache_dir: str = CACHE_DIR) -> None:
    today = dt.date.today()
    start = today - dt.timedelta(days=lookback_days)
    pro = tushare_pro()
    cal = pro.trade_cal(
        exchange="SSE", start_date=start.strftime("%Y%m%d"), end_date=today.strftime("%Y%m%d")
    )
    days = trading_days_from_cal(cal)
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
