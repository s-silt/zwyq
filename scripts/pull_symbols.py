"""Per-symbol backfill of daily + adj_factor for the survivorship-free universe.

Reliable path (small per-symbol responses) for a date window. Idempotent: cached
symbols are skipped, so it resumes after interruption. v1 pulls only daily+adj
(liquidity comes from daily.amount; 一字板/turnover refinements added later).

Usage: python scripts/pull_symbols.py <start YYYYMMDD> <end YYYYMMDD> [cache_dir]
"""

import datetime as dt
import os
import sys

import pandas as pd

from ashare_gauntlet.data.fetch import fetch_symbol_history
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.universe import build_universe


def _d(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y%m%d").date()


def main(start: str, end: str, cache_dir: str = "data/bystock") -> None:
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    fields = "ts_code,name,list_date,delist_date,list_status,market"
    parts = [pro.stock_basic(exchange="", list_status=st, fields=fields) for st in ("L", "D")]
    uni = build_universe(pd.concat(parts, ignore_index=True))

    start_d, end_d = _d(start), _d(end)
    codes: list[str] = []
    for ts_code, ld, dd in zip(uni["ts_code"], uni["list_date"], uni["delist_date"]):
        if ld is None or ld > end_d:
            continue  # not yet listed in window
        if dd is not None and dd < start_d:
            continue  # delisted before window
        codes.append(ts_code)
    print(f"universe={len(uni)} -> in-window codes={len(codes)} ({start}..{end})", flush=True)

    failed: list[tuple[str, str]] = []
    for i, code in enumerate(codes, 1):
        for endpoint in ("daily", "adj_factor"):
            try:
                fetch_symbol_history(pro, endpoint, code, start, end, cache_dir)
            except Exception as exc:  # noqa: BLE001 — record + continue for max progress
                failed.append((code, f"{endpoint}:{type(exc).__name__}"))
        if i % 200 == 0 or i == len(codes):
            print(f"  [{i}/{len(codes)}] failed={len(failed)}", flush=True)
    print(f"PULL DONE codes={len(codes)} failed={len(failed)}", flush=True)
    if failed:
        print("failed sample:", failed[:10], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "data/bystock")
