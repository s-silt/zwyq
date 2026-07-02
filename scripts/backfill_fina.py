"""全市场逐只财务 backfill —— 让"质地排序"建在"全市场财务都在本地"之上(而非 daily_basic 价值预筛近似)。

这源的财务接口只能逐只拉(按报告期一次拉全市场 = 返回空),所以全市场质地排序的前提是
把每只的财务都拉到本地缓存。本脚本逐只过、缓存优先(已拉的自动跳过 = 断点续传)。

模式:
  lean —— 只拉 fina_indicator(含三增 netprofit_yoy/dt_netprofit_yoy/tr_yoy + roe + 经营现金流 ocfps),
          够算 compute_holdscore 质地分;沪深主板 ~3千只,100/min 下约 30-45 分钟。
  full —— 拉 income/fina_indicator/balancesheet/cashflow 四张核心表 + 预警表(forecast/express/
          share_float/pledge_stat/stk_holdertrade/namechange),够算完整 build_record/tier;数小时。

纯净:核心表空值不落盘(fetch_symbol_table 抛 EmptyCoreTableError)→ 本脚本 catch + 响亮上报 + 跳过
(新上市/无财报的票正常会空,不让它中断整轮)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.backfill_fina
       [--mode lean|full] [--board main|all] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import EmptyCoreTableError, call_with_retry, fetch_symbol_table
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.screen import board_of

CACHE = "data/cache"
LEAN_TABLES: tuple[str, ...] = ("fina_indicator",)
CORE_TABLES: tuple[str, ...] = ("income", "fina_indicator", "balancesheet", "cashflow")
FULL_TABLES: tuple[str, ...] = (
    *CORE_TABLES,
    "forecast", "express", "share_float", "pledge_stat", "stk_holdertrade", "namechange",
)
MAIN_BOARDS = ("沪主板", "深主板")


def expected_min_end_date(today: str) -> str:
    """截至 ``today``(YYYYMMDD),法定必须已披露的最新报告期(监管常数,非 magic number)。

    A股披露截止:年报+一季报 ≤4/30、半年报 ≤8/31、三季报 ≤10/31。
    用途:① `--refresh` 判断哪些票的缓存已过期需重拉;② 读端(factor_rank 等)fail-loud
    断言缓存新鲜度——否则财报季后整条管线会静默消费旧财报。
    """
    y, md = today[:4], today[4:]
    if md >= "1101":
        return f"{y}0930"
    if md >= "0901":
        return f"{y}0630"
    if md >= "0501":
        return f"{y}0331"
    return f"{int(y) - 1}0930"


def tables_for_mode(mode: str) -> tuple[str, ...]:
    """模式 → 要拉的表集(纯函数)。lean 只拉 fina_indicator;core 拉 4 核心财报表;full 拉核心+预警表。"""
    if mode == "lean":
        return LEAN_TABLES
    if mode == "core":
        return CORE_TABLES
    if mode == "full":
        return FULL_TABLES
    raise ValueError(f"未知 mode={mode!r}(应为 lean / core / full)")


def _is_fresh(code: str, target: str) -> bool:
    """该股 fina_indicator 缓存是否已含 ``target`` 报告期(refresh 的跳过判据)。"""
    p = f"{CACHE}/fina_indicator/{code}.parquet"
    if not os.path.exists(p):
        return False
    try:
        ed = pd.read_parquet(p, columns=["end_date"])["end_date"].astype(str)
        return bool(len(ed)) and str(ed.max()) >= target
    except Exception:
        return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lean", choices=("lean", "core", "full"))
    ap.add_argument("--board", default="main", choices=("main", "all"))
    ap.add_argument("--limit", type=int, default=0, help="只拉前 N 只(0=全部),用于试跑")
    ap.add_argument("--refresh", action="store_true",
                    help="财报季刷新:按法定披露期判断,缓存缺最新报告期的票整只 force 重拉(否则缓存优先=永不更新)")
    a = ap.parse_args(argv)

    load_env_local()
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    tables = tables_for_mode(a.mode)

    sb = call_with_retry(lambda: pro.stock_basic(list_status="L", fields="ts_code,name"))
    codes = [str(c) for c in sb["ts_code"]]
    if a.board == "main":
        codes = [c for c in codes if board_of(c) in MAIN_BOARDS]
    codes.sort()
    if a.limit:
        codes = codes[: a.limit]

    target = expected_min_end_date(time.strftime("%Y%m%d")) if a.refresh else ""
    if a.refresh:
        print(f"refresh 模式:法定应披露最新报告期 = {target},已含该期的票跳过、过期票整只重拉", flush=True)
    print(f"mode={a.mode} board={a.board} | 待拉 {len(codes)} 只 × {len(tables)} 表 = {tables} "
          f"| {'刷新过期' if a.refresh else '缓存优先(已拉自动跳过)'}", flush=True)
    t0 = time.time()
    done = empty = err = skipped = 0
    for code in codes:
        force = False
        if a.refresh:
            if _is_fresh(code, target):
                skipped += 1
                done += 1
                continue
            force = True  # 过期:该股所有表整只重拉(接口返全历史,覆盖写一致)
        for table in tables:
            try:
                fetch_symbol_table(pro, table, code, CACHE, force=force)
            except EmptyCoreTableError:
                empty += 1  # 新上市/无财报:正常,跳过(已响亮:不落盘)
            except Exception as e:  # 单只单表失败响亮上报、继续
                err += 1
                print(f"  {code}/{table} 失败: {type(e).__name__}: {str(e)[:50]}", file=sys.stderr, flush=True)
        done += 1
        if done % 100 == 0:
            rate = done / max(time.time() - t0, 1) * 60
            print(f"  …{done}/{len(codes)} ({rate:.0f}只/min, 空{empty} 错{err} 新鲜跳过{skipped})", flush=True)

    dt = time.time() - t0
    print(f"\n完成:{done} 只,空核心表 {empty} 次,失败 {err} 次,耗时 {dt/60:.1f} 分钟。", flush=True)
    print(f"→ 现在可跑 scripts.holdscore(质地排序已建在全市场财务本地之上)", flush=True)


if __name__ == "__main__":
    main()
