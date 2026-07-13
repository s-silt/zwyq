"""持仓盯盘估值 —— IO 薄壳(读 holdings.json + 本地缓存 → 纯函数 → JSON 到 stdout)。

只算不判:输出确定性数字给盯盘 Claude,信号判定留给它(见 ashare_gauntlet.holdings 说明)。
纯计算全在 ashare_gauntlet.holdings(可单测);本层只做 IO 与拼装。

Usage:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.holdings_watch [今日YYYYMMDD]
  传今日 → 校验缓存最新日==今日(backfill 未刷新则 fail-loud);不传 → 用缓存最新日。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

from ashare_gauntlet.config import HOLDINGS_PATH as HOLDINGS
from ashare_gauntlet.holdings import (
    build_position_record,
    is_date_partition,
    qfq_series,
    verify_as_of,
)
from scripts.pick_track import CACHE

WINDOW = 20
HISTORY_DAYS = WINDOW + 1  # 前复权 MA20 需 20 日;多取 1 日与代码库 n+1 约定一致


def _recent(kind: str, n: int) -> tuple[list[str], list[str]]:
    # 只认日期分区文件 YYYYMMDD.parquet —— 缓存目录混有 per-symbol 区间文件,
    # 不滤掉会让 sorted(glob)[-1] 取到 ts_code 命名文件、污染 as_of(见 is_date_partition)
    files = sorted(f for f in glob.glob(f"{CACHE}/{kind}/*.parquet")
                   if is_date_partition(os.path.basename(f)))[-n:]
    return files, [os.path.basename(f)[:8] for f in files]


def main() -> None:
    expected = sys.argv[1] if len(sys.argv) > 1 else None

    with open(HOLDINGS, encoding="utf-8") as fh:
        hold = json.load(fh)

    daily_files, days = _recent("daily", HISTORY_DAYS)
    if not days:
        raise SystemExit("daily 缓存为空 —— 先跑 scripts.backfill 补行情")
    as_of = days[-1]
    if expected is not None:
        verify_as_of(as_of, expected)  # 缓存不是今日 → StaleCacheError 硬失败

    # 当日不复权 close/pct_chg + 每日横截面索引(算前复权序列)
    close_by_day = {d: pd.read_parquet(f, columns=["ts_code", "close", "low", "pct_chg"]).set_index("ts_code")
                    for f, d in zip(daily_files, days)}
    adj_files, adj_days = _recent("adj_factor", HISTORY_DAYS)
    adj_by_day = {}
    for f, d in zip(adj_files, adj_days):
        adf = pd.read_parquet(f)
        if "adj_factor" in adf.columns:
            adj_by_day[d] = adf.set_index("ts_code")["adj_factor"]

    today_px = close_by_day[as_of]
    latest_adj_row = adj_by_day.get(as_of)

    records = []
    for pos in hold["positions"]:
        code = pos["ts_code"]
        if code not in today_px.index:
            records.append(build_position_record(
                pos, close=None, pct_chg=None, qfq_closes=[], qfq_lows=[],
                as_of=as_of, trade_days=days, window=WINDOW))
            continue
        row = today_px.loc[code]
        close = float(row["close"])
        pct_chg = float(row["pct_chg"])

        # 前复权序列:逐日不复权 close/low + 复权因子,归一到当日因子
        raw_c, raw_l, adjs = [], [], []
        latest_adj = (float(latest_adj_row.loc[code])
                      if latest_adj_row is not None and code in latest_adj_row.index else None)
        if latest_adj:
            for d in days:
                cb = close_by_day[d]
                ab = adj_by_day.get(d)
                if code in cb.index and ab is not None and code in ab.index:
                    raw_c.append(float(cb.loc[code]["close"]))
                    raw_l.append(float(cb.loc[code]["low"]))
                    adjs.append(float(ab.loc[code]))
        qfq_c = qfq_series(raw_c, adjs, latest_adj) if latest_adj and raw_c else []
        qfq_l = qfq_series(raw_l, adjs, latest_adj) if latest_adj and raw_l else []

        records.append(build_position_record(
            pos, close=close, pct_chg=round(pct_chg, 2), qfq_closes=qfq_c, qfq_lows=qfq_l,
            as_of=as_of, trade_days=days, window=WINDOW))

    out = {"as_of": as_of, "cash": hold.get("cash"), "positions": records}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
