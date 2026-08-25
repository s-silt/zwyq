"""执行记录闭环(spec §13.2 影子核对 + RQAlpha 借鉴落点)——建议→次日现实的结构化落库。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.execution_record

对**上一份**四态决策快照(data/decisions/)做次日核对并追加到
data/execution_records.json:
- BUY 的可成交性:一字涨停锁定 / 停牌 / 可买(参考价=当日开盘);
- 纪律偏差:BUY 未买 / EXIT 未清 / 未建议却新增持仓(OFF_LIST_TRADE);
  对照口径=决策快照当时的 holdings(快照内嵌 held 列表)vs 当前 holdings。
影子期(2026-07-20 起 ≥60 交易日)结束后据此统计建议→成交滑点与纪律偏差率。
不回写 holdings.json;按 decision_as_of 幂等——同一快照重跑(含延迟多日)只覆盖不追加。
"""
from __future__ import annotations

import json
import math
import os
import re

from ashare_gauntlet.config import CACHE_DIR as CACHE, HOLDINGS_PATH, tushare_pro
from ashare_gauntlet.data.fetch import fetch_market_day
from ashare_gauntlet.data.partition import date_partition_files
from ashare_gauntlet.decision_snapshot import require_decision_snapshot_ready
from scripts.factor_backtest import one_word_limit_up
from scripts.buy_list import DECISION_DIR, latest_trade_date

RECORDS_PATH = "data/execution_records.json"


def verify_buy(open_px: "float | None", locked: bool, suspended: bool) -> str:
    """BUY 建议在执行日(快照次一交易日)的可成交性判定。

    真值表(Codex review 补齐):矛盾态与数据缺口一律 fail-loud——
    停牌却有价/既锁又停 → ValueError;非停牌但价缺失或非有限(None/NaN)→ ValueError
    (锁板与否都要求有效价,一字涨停也有开盘价);其余:停牌→SUSPENDED,
    锁板→LIMIT_UP_LOCKED,可买→BUYABLE。
    """
    if suspended:
        if open_px is not None or locked:
            raise ValueError(f"矛盾状态:suspended 但 open={open_px!r} locked={locked}")
        return "SUSPENDED"
    if open_px is None or not math.isfinite(float(open_px)):
        raise ValueError(f"非停牌却无有效开盘价(open={open_px!r})——上游数据缺口")
    return "LIMIT_UP_LOCKED" if locked else "BUYABLE"


_DATE8 = re.compile(r"^\d{8}$")
_DECISION_FILE_RE = re.compile(r"^(\d{8})_buy_decisions\.json$")


def next_trade_date(dates: list[str], as_of: str) -> str:
    """交易日列表中严格晚于 as_of 的第一个交易日(执行日语义 NEXT_TRADING_DAY)。

    不依赖调用方清洗:自行排序;非 YYYYMMDD 混入 fail-loud。已知限制:只能看见
    列表里存在的日期——缓存跳日会把更晚日期当次一交易日,分区连续性由每日盘后
    拉取纪律(holdings-daily-monitor 步骤1)维护。
    """
    bad = [d for d in dates if not _DATE8.match(d)]
    if bad:
        raise ValueError(f"非法日期分区名混入:{bad!r}")
    for d in sorted(dates):
        if d > as_of:
            return d
    raise ValueError(f"{as_of} 之后无交易日数据——无法核对执行日")


def upsert_record(records: list[dict], record: dict) -> "tuple[list[dict], str]":
    """按 decision_as_of 幂等落库,历史纪律事实冻结(Codex P1)。

    同一快照同日重跑 → 覆盖(replaced);隔日再跑已核对过的快照 → 不覆盖
    (frozen,divergences 是执行日次日的事实,不得被之后的 holdings 改写);
    新快照 → 追加(inserted)。
    """
    prior = [r for r in records if r["decision_as_of"] == record["decision_as_of"]]
    if prior and prior[0]["verify_date"] != record["verify_date"]:
        return records, "frozen"
    kept = [r for r in records if r["decision_as_of"] != record["decision_as_of"]]
    return kept + [record], "replaced" if prior else "inserted"


def divergences(decisions: list[dict], held_now: "set[str]",
                prev_held: "set[str]") -> list[dict]:
    """建议 vs 实际的纪律偏差清单(HOLD 且仍持有=一致,不记)。"""
    out: list[dict] = []
    for d in decisions:
        ts, st = d["ts_code"], d["state"]
        if st == "BUY":
            out.append({"ts_code": ts, "state": st,
                        "outcome": "FOLLOWED" if ts in held_now else "NOT_FOLLOWED"})
        elif st == "EXIT":
            out.append({"ts_code": ts, "state": st,
                        "outcome": "FOLLOWED" if ts not in held_now else "NOT_FOLLOWED"})
    buy_codes = {d["ts_code"] for d in decisions if d["state"] == "BUY"}
    for ts in sorted(held_now - prev_held):
        if ts not in buy_codes:
            out.append({"ts_code": ts, "state": "NONE", "outcome": "OFF_LIST_TRADE"})
    return out


def main() -> None:
    today = latest_trade_date()
    files = sorted(f for f in os.listdir(DECISION_DIR) if f.endswith("_buy_decisions.json")) \
        if os.path.isdir(DECISION_DIR) else []
    prev_files = [f for f in files if f[:8] < today]
    if not prev_files:
        raise SystemExit("无早于今日的决策快照——影子闭环从第二个交易日开始")
    snap_path = f"{DECISION_DIR}/{prev_files[-1]}"
    snap = json.load(open(snap_path, encoding="utf-8"))
    require_decision_snapshot_ready(snap, source=f"decision snapshot: {snap_path}")
    match = _DECISION_FILE_RE.fullmatch(prev_files[-1])
    if match is None or snap.get("as_of") != match.group(1):
        raise ValueError(
            f"decision snapshot filename date does not match payload as_of: {snap_path}"
        )
    decisions = snap["decisions"]

    hold = json.load(open(HOLDINGS_PATH, encoding="utf-8"))
    held_now = {p["ts_code"] for p in hold["positions"]}
    prev_held = {d["ts_code"] for d in decisions if d["state"] in ("HOLD", "EXIT")}

    # 可成交性按**执行日**核对=快照 as_of 的次一交易日(NEXT_TRADING_DAY 语义,
    # Codex review P1:延迟运行也不得用运行日行情冒充执行日)
    all_dates = [os.path.basename(f)[:8] for f in date_partition_files(CACHE, "daily")]
    execution_date = next_trade_date(all_dates, snap["as_of"])
    da = fetch_market_day(pro := tushare_pro(), "daily", execution_date, CACHE)
    sl = fetch_market_day(pro, "stk_limit", execution_date, CACHE)
    buys = [d for d in decisions if d["state"] == "BUY"]
    codes = [d["ts_code"] for d in buys]
    locked = one_word_limit_up(da, sl, codes) if codes else set()
    open_px = da.set_index("ts_code")["open"] if len(da) else {}
    checks = []
    for d in buys:
        ts = d["ts_code"]
        suspended = ts not in getattr(open_px, "index", [])
        px = None if suspended else float(open_px[ts])
        checks.append({"ts_code": ts, "executability": verify_buy(px, ts in locked, suspended),
                       "ref_open": px})

    record = {"decision_as_of": snap["as_of"], "execution_date": execution_date,
              "verify_date": today,
              "gap_days_note": (None if execution_date == today
                                else f"延迟核对:执行日 {execution_date},运行日 {today}"),
              "buy_checks": checks,
              "divergences": divergences(decisions, held_now, prev_held)}
    try:
        book = json.load(open(RECORDS_PATH, encoding="utf-8"))
    except FileNotFoundError:
        book = {"note": "影子模式执行记录(建议→次日现实);60交易日后统计滑点与纪律偏差率",
                "records": []}
    book["records"], action = upsert_record(book["records"], record)
    if action == "frozen":
        print(f"⚠ 决策 {snap['as_of']} 已有历史核对记录——纪律事实冻结,本次不落库")
    else:
        json.dump(book, open(RECORDS_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    n_dev = sum(1 for x in record["divergences"] if x["outcome"] != "FOLLOWED")
    print(f"执行记录:决策 {snap['as_of']} → 执行日 {execution_date}(运行日 {today});"
          f"BUY 可成交性 {[c['executability'] for c in checks] or '无BUY'};偏差 {n_dev} 项")
    for x in record["divergences"]:
        if x["outcome"] != "FOLLOWED":
            print(f"  ⚠ {x['ts_code']} {x['state']} → {x['outcome']}")
    print(f"→ {RECORDS_PATH}")


if __name__ == "__main__":
    main()
