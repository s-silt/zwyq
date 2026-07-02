"""Backfill the per-day full-market parquet cache for a date range.

Usage:
    python scripts/backfill.py <start YYYYMMDD> <end YYYYMMDD> [cache_dir]

Idempotent: already-cached (endpoint, trade_date) pulls are skipped, so it is
safe to re-run / resume after an interruption.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ashare_gauntlet.data.cache import read_or_fetch
from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import (
    TokenExpiredError,
    call_with_retry,
    fetch_market_day,
    trading_days_from_cal,
)
from ashare_gauntlet.data.tushare_source import make_pro_api

# Co-pull all three per date together so price and 北向 are time-aligned even if
# the pull is aborted mid-way (1-hour token / credit cap): newest-first means the
# most-recent contiguous block — exactly what the verdict needs — fills first,
# and each date gets daily+adj+hk_hold atomically before moving on.
V1_ENDPOINTS: tuple[str, ...] = ("daily", "adj_factor", "hk_hold")
MAX_WORKERS = 8


def days_to_pull(cal: pd.DataFrame | None, start: str, end: str) -> list[str]:
    """纯函数:trade_cal 结果 + 区间 → 应拉的日期列表(升序 YYYYMMDD)。

    日历可用(``cal`` 非 None)→ 只取 is_open==1 且落在 [start, end] 内的交易日,
    节假日/周末不进拉取队列(不再靠 EmptyMarketDayError 试错,省调用、去 failed 噪声)。
    日历不可用(``cal`` 为 None,调用方已打 warning)→ 回退旧行为:区间内全部自然日
    逐日试,休市日由 fetch_market_day 的 EmptyMarketDayError 响亮暴露——宁可浪费
    调用也不因日历故障静默漏拉行情。
    """
    if cal is None:
        return [d.strftime("%Y%m%d") for d in pd.date_range(start, end)]
    return [d for d in trading_days_from_cal(cal) if start <= d <= end]


def fetch_trade_cal(pro: object, start: str, end: str, cache_dir: str | Path) -> pd.DataFrame | None:
    """拉 [start, end] 的交易日历,按区间缓存到 ``<cache_dir>/trade_cal/<start>_<end>.parquet``。

    fail-loud 但不阻塞:拉不到(接口/网络故障)或返回空/缺列时打 warning 并返回 None,
    由调用方回退全自然日试错——日历只是省调用的优化,日历故障不能反过来卡住行情拉取。
    空日历不落盘(任何非空区间在 trade_cal 里都有行:开市/休市日都在表里,0 行只可能是
    拉取失败;落了盘 read_or_fetch 会永远回放这个假值)。TokenExpiredError 例外向上抛:
    额度耗尽是全局致命错,回退试错只会烧掉更多失败调用。
    """
    path = Path(cache_dir) / "trade_cal" / f"{start}_{end}.parquet"

    def _pull() -> pd.DataFrame:
        df = call_with_retry(lambda: pro.trade_cal(exchange="SSE", start_date=start, end_date=end))
        if df.empty:
            raise RuntimeError(f"trade_cal returned 0 rows for {start}..{end}")
        return df

    try:
        cal = read_or_fetch(path, _pull)
    except TokenExpiredError:
        raise
    except Exception as exc:  # noqa: BLE001 — 响亮降级:warning + 回退,不静默也不阻塞
        print(
            f"warning: trade_cal {start}..{end} 拉取失败({type(exc).__name__}: {str(exc)[:80]})"
            f"—— 回退区间内全自然日逐日试错(休市日会以 EmptyMarketDayError 暴露)",
            flush=True,
        )
        return None
    if not {"cal_date", "is_open"} <= set(cal.columns):
        # 缓存里躺着坏 schema(旧版写入/上游变更):surface 出来并降级,不拿坏日历漏拉行情
        print(
            f"warning: trade_cal 缓存/返回缺列(有 {list(cal.columns)},在 {path})"
            f"—— 回退区间内全自然日逐日试错",
            flush=True,
        )
        return None
    return cal


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
    load_env_local()  # .env.local 权威覆盖 —— 调度任务可能没继承到新源的环境变量
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    cal = fetch_trade_cal(pro, start, end, cache_dir)
    days = days_to_pull(cal, start, end)
    label = "trading days (per trade_cal)" if cal is not None else "calendar days (trade_cal 不可用,逐日试错)"
    print(
        f"{len(days)} {label} {start}..{end}; endpoints={V1_ENDPOINTS}; "
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
