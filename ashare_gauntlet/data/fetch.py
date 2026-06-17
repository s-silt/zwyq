"""Fetch layer: full-market per-trade-date pulls, cached to parquet.

Pull strategy (chosen after measuring the mirror): one call per trade date per
endpoint returns the whole market (~5000 rows in 3-5s under a 120s timeout). With
the mirror's 1500/min regular + 500/min heavy limits and no daily cap, a
multi-year backfill is a few minutes per endpoint.
"""

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import pandas as pd
import requests

from .cache import read_or_fetch

_T = TypeVar("_T")

# Substrings tushare puts in its (bare Exception) API error messages.
_FATAL_MARKERS = ("过期", "欠费", "余额不足")  # token exhausted/expired -> abort
_RATE_MARKERS = ("每分钟", "频率", "频繁", "too many", "rate limit")  # throttle -> retry


class TokenExpiredError(RuntimeError):
    """The data token is expired/out of credits — fatal; abort the backfill."""

# Endpoints pulled per trade date. "daily_basic" / "stk_limit" carry the
# turnover and price-limit fields used by the tradability filters.
MARKET_ENDPOINTS: tuple[str, ...] = ("daily", "adj_factor", "daily_basic", "stk_limit")


def trading_days_from_cal(trade_cal: pd.DataFrame) -> list[str]:
    """Open trading days (``cal_date`` strings) from a ``trade_cal`` pull, sorted."""
    open_days = trade_cal.loc[trade_cal["is_open"] == 1, "cal_date"]
    return sorted(str(d) for d in open_days)


def call_with_retry(
    fn: Callable[[], _T],
    attempts: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Call ``fn``, retrying on transient network errors with linear backoff.

    Only ``requests`` network-layer errors (connection drops, chunked-read
    breaks, timeouts) are retried — a genuine API error is not a transient and
    is left to propagate. The mirror occasionally drops a large full-market
    response mid-stream; without this a single hiccup aborts a whole backfill.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as error:
            message = str(error)
            if any(marker in message for marker in _FATAL_MARKERS):
                raise TokenExpiredError(message) from error
            transient = isinstance(error, requests.exceptions.RequestException) or any(
                marker in message.lower() for marker in _RATE_MARKERS
            )
            if not transient:
                raise
            last_error = error
            if attempt < attempts - 1:
                sleep(base_delay * (attempt + 1))
    assert last_error is not None
    raise last_error


def fetch_market_day(
    pro: object,
    endpoint: str,
    trade_date: str,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """Cached full-market pull of ``endpoint`` for one ``trade_date``."""
    path = Path(cache_dir) / endpoint / f"{trade_date}.parquet"
    fields = (
        "ts_code,trade_date,turnover_rate,circ_mv,total_mv"
        if endpoint == "daily_basic"
        else ""
    )

    def _pull() -> pd.DataFrame:
        method = getattr(pro, endpoint)
        if fields:
            return method(trade_date=trade_date, fields=fields)
        return method(trade_date=trade_date)

    return read_or_fetch(path, lambda: call_with_retry(_pull))


def fetch_range(
    pro: object,
    trade_dates: Sequence[str],
    cache_dir: str | Path,
    endpoints: Sequence[str] = MARKET_ENDPOINTS,
) -> None:
    """Backfill the cache for every (endpoint, trade_date); idempotent."""
    for trade_date in trade_dates:
        for endpoint in endpoints:
            fetch_market_day(pro, endpoint, trade_date, cache_dir)


def fetch_symbol_history(
    pro: object,
    endpoint: str,
    ts_code: str,
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """Cached per-symbol history pull over a date range.

    Per-symbol responses are small (a few hundred rows), so unlike the
    full-market per-day pull they do not get dropped mid-stream by the mirror —
    this is the reliable backfill path. Cached one parquet per (endpoint, code).
    """
    # The window is part of the cache key — otherwise re-pulling the same symbol
    # with a different date range silently serves the stale cached range.
    path = Path(cache_dir) / endpoint / f"{ts_code}_{start_date}_{end_date}.parquet"

    def _pull() -> pd.DataFrame:
        method = getattr(pro, endpoint)
        return method(ts_code=ts_code, start_date=start_date, end_date=end_date)

    return read_or_fetch(path, lambda: call_with_retry(_pull))


def fetch_symbol_table(
    pro: object,
    endpoint: str,
    ts_code: str,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """Cached full-history per-symbol pull (no date window) — for the structured
    fundamentals/risk tables (``income``, ``fina_indicator``, ``share_float``,
    ``pledge_stat``), which return all reported periods for one ``ts_code``.

    One parquet per (endpoint, ts_code); idempotent. The interface-first source
    for the analysis mode's step 3, replacing the web-scrape of Q1 业绩/风险旗标.
    """
    path = Path(cache_dir) / endpoint / f"{ts_code}.parquet"

    def _pull() -> pd.DataFrame:
        return getattr(pro, endpoint)(ts_code=ts_code)

    return read_or_fetch(path, lambda: call_with_retry(_pull))
