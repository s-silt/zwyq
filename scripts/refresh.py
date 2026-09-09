"""Daily incremental refresh of the cache and current-month trade calendar.

Caches the authoritative full natural-month trade calendar, then pulls the last
few trading days. Four core EOD endpoints (daily/adj_factor/daily_basic/stk_limit)
run first and any miss/failure is fail-loud. Optional extras (hk_hold,
moneyflow_hsgt) are attempted after the cores: a failure is listed in the
summary and does not abort the remaining endpoints. Idempotent: cached data is
reused, so only genuinely new partitions hit the network. Needs a live token.

Usage: python scripts/refresh.py [lookback_days] [cache_dir]
"""

import calendar
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR, tushare_pro
from ashare_gauntlet.data.fetch import (
    MARKET_ENDPOINTS,
    TokenExpiredError,
    fetch_market_day,
    refresh_market_endpoints,
    trading_days_from_cal,
)
from scripts.backfill import fetch_trade_cal

# moneyflow_hsgt is market-level (1 row/day): 北向总成交额(沪/深股通). Post
# 2024-08-19 its north_money column is TURNOVER, not net flow — see
# factsheet.NORTH_FLOW_SEMANTICS_CUTOFF. Cheap (1 row) so cached every refresh.
# Order is core-first via refresh_market_endpoints().
ENDPOINTS = refresh_market_endpoints()
CORE_ENDPOINTS = MARKET_ENDPOINTS


def _format_failures(failures: list[tuple[str, str, str, str]]) -> str:
    return "; ".join(
        f"{endpoint} {day} {error_type}: {error}"
        for endpoint, day, error_type, error in failures
    )


def main(
    lookback_days: int = 10,
    cache_dir: str = CACHE_DIR,
    *,
    today: dt.date | None = None,
) -> None:
    current_day = today or dt.date.today()
    start = current_day - dt.timedelta(days=lookback_days)
    month_start = current_day.replace(day=1)
    pro = tushare_pro()
    calendars: list[pd.DataFrame] = []
    calendar_month = start.replace(day=1)
    while calendar_month <= month_start:
        calendar_end = calendar_month.replace(
            day=calendar.monthrange(calendar_month.year, calendar_month.month)[1]
        )
        cal = fetch_trade_cal(
            pro,
            calendar_month.strftime("%Y%m%d"),
            calendar_end.strftime("%Y%m%d"),
            cache_dir,
            strict=True,
        )
        assert cal is not None
        calendars.append(cal)
        calendar_month = calendar_end + dt.timedelta(days=1)
    combined_calendar = pd.concat(calendars, ignore_index=True)
    end = current_day.strftime("%Y%m%d")
    first = start.strftime("%Y%m%d")
    days = [
        day for day in trading_days_from_cal(combined_calendar) if first <= day <= end
    ]
    print(f"refresh: 检查 {len(days)} 个交易日 {days[0] if days else '-'}..{days[-1] if days else '-'}", flush=True)

    new_files = 0
    attempted = 0
    successes = 0
    failures: list[tuple[str, str, str, str]] = []
    core_set = frozenset(CORE_ENDPOINTS)
    try:
        # Endpoint-outer so every core partition lands before any optional extra.
        for endpoint in ENDPOINTS:
            for day in days:
                attempted += 1
                existed = Path(cache_dir, endpoint, f"{day}.parquet").exists()
                try:
                    fetch_market_day(pro, endpoint, day, cache_dir)
                except TokenExpiredError:
                    raise
                except Exception as exc:  # noqa: BLE001 - isolate per endpoint, classify later.
                    failures.append((endpoint, day, type(exc).__name__, str(exc)[:300]))
                    print(
                        f"refresh 失败: {endpoint} {day} {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                successes += 1
                if not existed:
                    new_files += 1
    except TokenExpiredError as exc:
        print(f"!!! token 耗尽: {exc} — 已增量到此,新增 {new_files} 个文件", flush=True)
        raise SystemExit(1)

    latest = days[-1] if days else "-"
    core_failures = [item for item in failures if item[0] in core_set]
    optional_failures = [item for item in failures if item[0] not in core_set]

    if attempted > 0 and successes == 0:
        print(
            f"refresh 全部端点失败: 成功 0/{attempted} — {_format_failures(failures)}",
            flush=True,
        )
        raise SystemExit(1)
    if core_failures:
        print(
            f"refresh 核心失败: {_format_failures(core_failures)} — 拒绝当作刷新成功",
            flush=True,
        )
        raise SystemExit(1)
    if optional_failures:
        print(
            f"refresh 非核心失败: {_format_failures(optional_failures)}",
            flush=True,
        )
        print(
            f"refresh 核心完成: 新增 {new_files} 个缓存文件(最新交易日 {latest})",
            flush=True,
        )
        return
    print(f"refresh done: 新增 {new_files} 个缓存文件(最新交易日 {latest})", flush=True)


if __name__ == "__main__":
    lb = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    cdir = next((a for a in sys.argv[1:] if "/" in a), "data/cache")
    main(lb, cdir)
