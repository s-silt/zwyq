"""持仓盯盘估值 —— IO 薄壳(读 holdings.json + 本地缓存 → 纯函数 → JSON 到 stdout)。

只算不判:输出确定性数字给盯盘 Claude,信号判定留给它(见 ashare_gauntlet.holdings 说明)。
纯计算全在 ashare_gauntlet.holdings(可单测);本层只做 IO 与拼装。

Usage:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.holdings_watch [YYYYMMDD] [--stdout-only]
  传今日 → 校验缓存最新日==今日(backfill 未刷新则 fail-loud);不传 → 用缓存最新日。
  --stdout-only: 仅输出 stdout JSON,不写 data/account_state/YYYYMMDD_account_state.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from ashare_gauntlet.account_state import (
    EOD_SCHEMA,
    AccountSchemaError,
    build_eod_account_valuation,
    normalize_account_state,
    require_account_as_of,
)
from ashare_gauntlet.config import (
    ACCOUNT_STATE_DIR as OUT_DIR,
    HOLDINGS_PATH as HOLDINGS,
)
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
    files = sorted(
        f for f in glob.glob(f"{CACHE}/{kind}/*.parquet")
        if is_date_partition(os.path.basename(f))
    )[-n:]
    return files, [os.path.basename(f)[:8] for f in files]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("expected", nargs="?", default=None,
                    help="期望缓存日期 YYYYMMDD")
    ap.add_argument("--stdout-only", action="store_true",
                    help="仅 stdout 输出,不写文件")
    a = ap.parse_args(argv)

    expected = a.expected
    stdout_only = a.stdout_only

    with open(HOLDINGS, encoding="utf-8") as fh:
        hold = json.load(fh)

    daily_files, days = _recent("daily", HISTORY_DAYS)
    if not days:
        raise SystemExit("daily 缓存为空 —— 先跑 scripts.backfill 补行情")
    as_of = days[-1]
    if expected is not None:
        verify_as_of(as_of, expected)  # 缓存不是今日 → StaleCacheError 硬失败

    # 归一化账户(只读,不修改) + 严格门禁；失败发生在任何行情读取/估值/写入前。
    account = normalize_account_state(hold, expected_as_of=as_of)
    require_account_as_of(account, as_of)
    if account["data_status"] != "complete":
        raise AccountSchemaError(
            "ACCOUNT_SCHEMA_ERROR: 账户状态不完整 "
            f"missing={account['missing_fields']} invalid={account['invalid_fields']}"
        )

    # 当日不复权 close/pct_chg + 每日横截面索引(算前复权序列)
    close_by_day = {
        d: pd.read_parquet(f, columns=["ts_code", "close", "low", "pct_chg"]).set_index("ts_code")
        for f, d in zip(daily_files, days)
    }
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

        raw_c, raw_l, adjs = [], [], []
        latest_adj = (
            float(latest_adj_row.loc[code])
            if latest_adj_row is not None and code in latest_adj_row.index
            else None
        )
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
            pos, close=close, pct_chg=round(pct_chg, 2),
            qfq_closes=qfq_c, qfq_lows=qfq_l,
            as_of=as_of, trade_days=days, window=WINDOW))

    # EOD 估值: shares × 当日 EOD close(不使用人工 mv)
    valuation = build_eod_account_valuation(account, as_of, records)

    # 行业权重(EOD 口径)
    eod_industries: dict[str, float] = {}
    for rec in records:
        code = rec["ts_code"]
        pos_data = next(
            (p for p in hold["positions"] if p["ts_code"] == code), None
        )
        if pos_data is None:
            continue
        close = rec.get("close") if rec.get("error") is None else None
        shares = pos_data.get("shares")
        if close is not None and shares and float(shares) == int(shares) and shares > 0:
            ind = str(pos_data.get("industry") or "其他")
            eod_industries[ind] = eod_industries.get(ind, 0.0) + float(int(shares)) * float(close)

    eod_industry_weights = None
    if valuation["total_assets"] is not None and valuation["total_assets"] > 0:
        eod_industry_weights = {
            key: {
                "market_value": round(value, 2),
                "weight": round(value / valuation["total_assets"], 6),
            }
            for key, value in sorted(
                eod_industries.items(), key=lambda item: -item[1]
            )
        }

    # 条件单摘要(不暴露 raw)
    co = account["conditional_orders"]
    co_summary = {
        "status": co["status"],
        "format": co.get("format"),
        "verified_count": co.get("verified_count"),
    }

    out = {
        "schema_version": EOD_SCHEMA,
        "as_of": as_of,
        "account_as_of": account["as_of"],
        "source_schema": account["source_schema"],
        "account_data_status": account["data_status"],
        "account_freshness": account["freshness"],
        "data_status": valuation["status"],
        "valuation": valuation,
        "industry_weights": eod_industry_weights,
        "short_slot": account["short_slot"],
        "conditional_orders": co_summary,
        "positions": records,
    }

    payload = json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)
    if stdout_only:
        print(payload)
        return

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{as_of}_account_state.json"
    # 同目录临时文件 + flush/fsync + 原子替换；失败不覆盖旧快照、不提前打印成功 JSON。
    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".json", prefix=".tmp_account_state_", dir=str(out_dir)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tf:
            tf.write(payload)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_name, str(out_path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    print(payload)


if __name__ == "__main__":
    main()
