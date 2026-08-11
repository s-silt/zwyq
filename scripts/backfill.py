"""Backfill the per-day full-market parquet cache for a date range.

Usage:
    python -m scripts.backfill <start YYYYMMDD> <end YYYYMMDD> [cache_dir]

The legacy default remains best-effort ``daily/adj_factor/hk_hold``.  MCP and
other completeness-sensitive callers use ``--strict-market`` to require every
open day from ``trade_cal`` and all four core market endpoints.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR, tushare_pro
from ashare_gauntlet.data.cache import read_or_fetch
from ashare_gauntlet.data.fetch import (
    MARKET_ENDPOINTS,
    TokenExpiredError,
    call_with_retry,
    fetch_market_day,
    trading_days_from_cal,
)

LEGACY_ENDPOINTS: tuple[str, ...] = ("daily", "adj_factor", "hk_hold")
MAX_WORKERS = 8
REPORT_PREFIX = "BACKFILL_RESULT_JSON="


class TradeCalendarUnavailableError(RuntimeError):
    """Raised when strict mode cannot establish the complete open-day set."""


def _validate_date_range(start: str, end: str) -> None:
    for name, value in (("start", start), ("end", end)):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a real YYYYMMDD date")
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"{name} must be a real YYYYMMDD date: {value!r}") from exc
    if start > end:
        raise ValueError("start must be <= end")


def _validate_trade_cal(cal: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    required = {"cal_date", "is_open"}
    if not required <= set(cal.columns):
        raise TradeCalendarUnavailableError(
            f"trade_cal 缺列(有 {list(cal.columns)})")
    normalized = cal.copy()
    dates = normalized["cal_date"].astype(str)
    invalid_dates: list[str] = []
    for value in dates:
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError:
            invalid_dates.append(value)
    if invalid_dates:
        raise TradeCalendarUnavailableError(
            f"trade_cal 含非法 cal_date: {invalid_dates[:5]}")
    if dates.duplicated().any():
        duplicates = sorted(set(dates[dates.duplicated(keep=False)]))
        raise TradeCalendarUnavailableError(
            f"trade_cal 含重复 cal_date: {duplicates[:5]}")
    out_of_range = sorted(set(dates[(dates < start) | (dates > end)]))
    if out_of_range:
        raise TradeCalendarUnavailableError(
            f"trade_cal 日期超出请求区间 {start}..{end}: {out_of_range[:5]}")
    # trade_cal 语义是区间内每个自然日一行(开市/休市都在表里);子集返回意味着
    # 缺失日的开闭市状态未知,漏掉的开市日不会进拉取队列,却仍会得出
    # completed_pairs == expected_pairs 的假阳性 ok(codex review P1-1)。
    expected_days = {d.strftime("%Y%m%d") for d in pd.date_range(start, end)}
    missing_days = sorted(expected_days - set(dates))
    if missing_days:
        raise TradeCalendarUnavailableError(
            f"trade_cal 未覆盖区间内全部自然日(缺 {missing_days[:5]} 共 "
            f"{len(missing_days)} 天)——缺失日开市状态未知,不能当作休市")
    is_open = pd.to_numeric(normalized["is_open"], errors="coerce")
    if is_open.isna().any() or not is_open.isin([0, 1]).all():
        raise TradeCalendarUnavailableError("trade_cal is_open 必须逐行是 0 或 1")
    normalized["cal_date"] = dates
    normalized["is_open"] = is_open.astype(int)
    return normalized


def days_to_pull(cal: pd.DataFrame | None, start: str, end: str) -> list[str]:
    """Return ascending dates to pull, falling back to calendar days if needed."""
    if cal is None:
        return [d.strftime("%Y%m%d") for d in pd.date_range(start, end)]
    return [d for d in trading_days_from_cal(cal) if start <= d <= end]


def fetch_trade_cal(
    pro: object,
    start: str,
    end: str,
    cache_dir: str | Path,
    *,
    strict: bool = False,
) -> pd.DataFrame | None:
    """Fetch and cache SSE calendar; strict callers fail instead of degrading."""
    path = Path(cache_dir) / "trade_cal" / f"{start}_{end}.parquet"

    def _pull() -> pd.DataFrame:
        df = call_with_retry(
            lambda: pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
        )
        if df.empty:
            raise RuntimeError(f"trade_cal returned 0 rows for {start}..{end}")
        return df

    try:
        cal = read_or_fetch(path, _pull)
    except TokenExpiredError:
        raise
    except Exception as exc:  # noqa: BLE001 - strict mode converts this to a contract failure.
        if strict:
            raise TradeCalendarUnavailableError(
                f"trade_cal {start}..{end} unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        print(
            f"warning: trade_cal {start}..{end} 拉取失败({type(exc).__name__}: {str(exc)[:80]})"
            "—— 回退区间内全自然日逐日试错(休市日会以 EmptyMarketDayError 暴露)",
            flush=True,
        )
        return None

    try:
        return _validate_trade_cal(cal, start, end)
    except TradeCalendarUnavailableError as exc:
        message = f"{exc}(在 {path})"
        if strict:
            raise TradeCalendarUnavailableError(message) from exc
        print(f"warning: {message}—— 回退区间内全自然日逐日试错", flush=True)
        return None


def _pull_day(
    pro: object,
    day: str,
    cache_dir: str | Path,
    endpoints: Sequence[str],
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    completed = 0
    for endpoint in endpoints:
        try:
            fetch_market_day(pro, endpoint, day, cache_dir)
            completed += 1
        except TokenExpiredError:
            raise
        except Exception as exc:  # noqa: BLE001 - collect every missing endpoint.
            failures.append({
                "trade_date": day,
                "endpoint": endpoint,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
    return {"completed": completed, "failures": failures}


def run_backfill(
    pro: object,
    start: str,
    end: str,
    cache_dir: str | Path = CACHE_DIR,
    *,
    strict_market: bool = False,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    """Run a resumable backfill and return a machine-verifiable coverage report."""
    endpoints = tuple(MARKET_ENDPOINTS if strict_market else LEGACY_ENDPOINTS)
    report: dict[str, Any] = {
        "ok": False,
        "strict_market": strict_market,
        "start_date": start,
        "end_date": end,
        "calendar_status": "unknown",
        "open_days": [],
        "required_endpoints": list(endpoints),
        "expected_pairs": 0,
        "completed_pairs": 0,
        "failed_pairs": [],
        "fatal_error": None,
    }

    try:
        _validate_date_range(start, end)
        if (not isinstance(max_workers, int) or isinstance(max_workers, bool)
                or max_workers <= 0):
            raise ValueError("max_workers must be a positive integer")
    except ValueError as exc:
        report["calendar_status"] = "failed"
        report["fatal_error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
        return report

    try:
        cal = fetch_trade_cal(pro, start, end, cache_dir, strict=strict_market)
    except Exception as exc:  # Includes token expiry and strict calendar failures.
        report["calendar_status"] = "failed"
        report["fatal_error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
        return report

    days = days_to_pull(cal, start, end)
    report["calendar_status"] = "complete" if cal is not None else "fallback"
    report["open_days"] = days
    report["expected_pairs"] = len(days) * len(endpoints)
    label = (
        "trading days (per trade_cal)"
        if cal is not None
        else "calendar days (trade_cal 不可用,逐日试错)"
    )
    print(
        f"{len(days)} {label} {start}..{end}; endpoints={endpoints}; "
        f"~{report['expected_pairs']} credit-calls (before retries); workers={max_workers}",
        flush=True,
    )

    futures = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_pull_day, pro, day, cache_dir, endpoints)
                for day in reversed(days)
            ]
            done = 0
            for future in as_completed(futures):
                result = future.result()
                report["completed_pairs"] += result["completed"]
                report["failed_pairs"].extend(result["failures"])
                done += 1
                if done % 20 == 0 or done == len(days):
                    print(
                        f"  [{done}/{len(days)}] failed_calls={len(report['failed_pairs'])}",
                        flush=True,
                    )
    except Exception as exc:  # Future/worker/pool failures must stay in the report contract.
        for future in futures:
            future.cancel()
        report["fatal_error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }

    report["ok"] = (
        report["fatal_error"] is None
        and not report["failed_pairs"]
        and report["completed_pairs"] == report["expected_pairs"]
        and (not strict_market or report["calendar_status"] == "complete")
    )
    return report


def main(
    start: str,
    end: str,
    cache_dir: str = CACHE_DIR,
    *,
    strict_market: bool = False,
    strict_env: bool = False,
    report_json: bool = False,
) -> dict[str, Any]:
    pro = tushare_pro(strict_env=strict_env)
    report = run_backfill(
        pro,
        start,
        end,
        cache_dir,
        strict_market=strict_market,
    )
    print(f"backfill done; failed_calls={len(report['failed_pairs'])}", flush=True)
    if report["failed_pairs"]:
        print("failed sample:", report["failed_pairs"][:10], flush=True)
    if report["fatal_error"]:
        print(f"!!! FATAL: {report['fatal_error']}", flush=True)
    if report_json:
        print(REPORT_PREFIX + json.dumps(report, ensure_ascii=False), flush=True)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("cache_dir", nargs="?", default=CACHE_DIR)
    parser.add_argument(
        "--strict-market",
        action="store_true",
        help="require trade_cal and all daily/adj_factor/daily_basic/stk_limit pairs",
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="allow only TUSHARE_TOKEN/TUSHARE_HTTP_URL in .env.local",
    )
    parser.add_argument(
        "--report-json",
        action="store_true",
        help="print a final BACKFILL_RESULT_JSON line",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outcome = main(
        args.start,
        args.end,
        args.cache_dir,
        strict_market=args.strict_market,
        strict_env=args.strict_env,
        report_json=args.report_json,
    )
    if args.strict_market and not outcome["ok"]:
        raise SystemExit(1)
