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
# Throttle markers must be *specific* to the rate-limit message, not bare
# "频率"/"频繁" — those bare substrings also appear in genuine parameter-error
# wording ("...更频繁", "调用频率参数错误"), which a substring match would
# misclassify as a transient throttle and waste `attempts` retries before
# raising (#11). Keep only phrasings that uniquely mark a real rate limit.
_RATE_MARKERS = (
    "每分钟",          # "抱歉，您每分钟最多访问该接口N次"
    "访问频率",        # "访问频率过快/超限"
    "调用频繁",        # "调用频繁，请稍后再试"
    "too many",
    "rate limit",
)  # throttle -> retry

# Source-side transient unavailability: the (self-hosted) proxy's upstream
# briefly drops and returns "上游数据源暂时不可用" on an otherwise-valid call —
# distinct from rate-limiting but equally retryable (the same call usually
# succeeds on retry). A genuinely persistent case (e.g. a trade date whose data
# isn't published yet) simply exhausts attempts and then propagates, which the
# caller handles (e.g. survivors falls back to the previous trading day).
_TRANSIENT_MARKERS = ("暂时不可用",)  # source temporarily down -> retry


class TokenExpiredError(RuntimeError):
    """The data token is expired/out of credits — fatal; abort the backfill."""


class EmptyCoreTableError(RuntimeError):
    """A core financial-statement table came back empty (0 rows).

    For ``income``/``fina_indicator``/``balancesheet``/``cashflow`` an empty
    pull is never a real "this company has no statements" — it means the pull
    failed or returned nothing, and treating it as a true value (caching it /
    returning it) would poison every downstream computation. Raise loudly so the
    real error surfaces instead of silently producing fabricated facts.
    """


class EmptyMarketDayError(RuntimeError):
    """A full-market per-trade-date pull came back empty (0 rows).

    For ``daily``/``adj_factor``/``daily_basic``/``stk_limit`` the whole market is
    ~5000 rows on any open trading day — 0 rows is never "no trading happened",
    it means the source hasn't published this day's EOD yet (or the pull failed).
    Caching it writes an empty parquet that the *idempotent* backfill then skips
    forever, so the real data can never land until the stale file is deleted by
    hand. Raise loudly and refuse to cache, so the next run simply retries.
    """

# Endpoints pulled per trade date. "daily_basic" / "stk_limit" carry the
# turnover and price-limit fields used by the tradability filters.
MARKET_ENDPOINTS: tuple[str, ...] = ("daily", "adj_factor", "daily_basic", "stk_limit")

# Full-market endpoints whose *emptiness* on an open trading day is never real
# (the whole market is ~5000 rows) -> raise instead of caching the empty pull.
# "hk_hold" (北向持股) is intentionally excluded: it can legitimately be empty.
NONEMPTY_MARKET_ENDPOINTS: frozenset[str] = frozenset(
    {"daily", "adj_factor", "daily_basic", "stk_limit"}
)

# Per-symbol full-history tables whose *emptiness* is meaningful vs. fatal:
#   - core financial statements: 0 rows is never a real value -> raise loudly.
#   - everything else (event tables: share_float/pledge_stat/stk_holdertrade/
#     forecast/express, plus namechange) may legitimately be empty.
# Classifying here (the layer that knows the endpoint) keeps the generic
# read_or_fetch free of any core/event policy, so other callers aren't affected.
CORE_SYMBOL_TABLES: frozenset[str] = frozenset(
    {"income", "fina_indicator", "balancesheet", "cashflow"}
)


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
                marker in message.lower() for marker in (*_RATE_MARKERS, *_TRANSIENT_MARKERS)
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
    """Cached full-market pull of ``endpoint`` for one ``trade_date``.

    For the full-market price/factor endpoints (``NONEMPTY_MARKET_ENDPOINTS``) an
    empty pull means the source hasn't published this day's EOD yet — never a real
    value. Refuse to cache it (and refuse to serve a legacy empty cache), so the
    next run retries instead of an empty parquet blocking the date forever. See
    ``EmptyMarketDayError``.
    """
    path = Path(cache_dir) / endpoint / f"{trade_date}.parquet"
    must_be_nonempty = endpoint in NONEMPTY_MARKET_ENDPOINTS
    # daily_basic 不再窄化 fields:缓存键只有日期不含 fields,窄列表落盘后其它下游(pe_ttm/pb/
    # dv_ttm)会静默读到缺列的旧 schema;全字段落盘 = 一份缓存服务所有下游(历史 daily_basic
    # 不可变,是完美缓存对象)。
    fields = ""

    def _pull() -> pd.DataFrame:
        method = getattr(pro, endpoint)
        df = method(trade_date=trade_date, fields=fields) if fields else method(trade_date=trade_date)
        # Guard *before* read_or_fetch caches it: an empty full-market pull must
        # never be written, or the idempotent backfill skips the date forever.
        if must_be_nonempty and df.empty:
            raise EmptyMarketDayError(
                f"market endpoint {endpoint!r} returned 0 rows for trade_date={trade_date!r} "
                f"— the source likely hasn't published this day's EOD yet; refusing to cache "
                f"an empty full-market pull as a real value"
            )
        return df

    df = read_or_fetch(path, lambda: call_with_retry(_pull))
    # Also guard the cache-hit path: a legacy empty parquet (written before this
    # guard, or by an earlier pre-publish pull) is just as poisonous — surface it
    # so the date refetches once the stale file is gone, instead of serving 0 rows.
    if must_be_nonempty and df.empty:
        raise EmptyMarketDayError(
            f"cached market endpoint {endpoint!r} is empty (0 rows) for trade_date={trade_date!r} "
            f"at {path} — refusing to serve an empty full-market pull as a real value"
        )
    return df


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
    force: bool = False,
) -> pd.DataFrame:
    """Cached full-history per-symbol pull (no date window) — for the structured
    fundamentals/risk tables (``income``, ``fina_indicator``, ``share_float``,
    ``pledge_stat``), which return all reported periods for one ``ts_code``.

    One parquet per (endpoint, ts_code); idempotent. The interface-first source
    for the analysis mode's step 3, replacing the web-scrape of Q1 业绩/风险旗标.
    ``force=True`` 整只重拉覆盖(接口本就返回全历史,覆盖写天然一致)——财报季刷新用,
    否则缓存优先会把财务永远冻结在旧报告期。
    """
    path = Path(cache_dir) / endpoint / f"{ts_code}.parquet"
    is_core = endpoint in CORE_SYMBOL_TABLES

    def _pull() -> pd.DataFrame:
        df = call_with_retry(lambda: getattr(pro, endpoint)(ts_code=ts_code))
        # Guard *before* read_or_fetch caches it: an empty core table must never
        # be written to disk, or the next run serves the fabricated empty value
        # from cache and the error is permanently masked.
        if is_core and df.empty:
            raise EmptyCoreTableError(
                f"core table {endpoint!r} returned 0 rows for ts_code={ts_code!r} "
                f"— refusing to cache an empty financial statement as a real value"
            )
        return df

    df = read_or_fetch(path, _pull, force=force)
    # Also guard the cache-hit path: a legacy/partial empty parquet on disk for a
    # core table is just as poisonous as a fresh empty pull — surface it too.
    if is_core and df.empty:
        raise EmptyCoreTableError(
            f"cached core table {endpoint!r} is empty (0 rows) for ts_code={ts_code!r} "
            f"at {path} — refusing to serve an empty financial statement as a real value"
        )
    return df
