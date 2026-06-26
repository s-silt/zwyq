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

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import EmptyCoreTableError, call_with_retry, fetch_symbol_table
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.screen import board_of

CACHE = "data/cache"
LEAN_TABLES: tuple[str, ...] = ("fina_indicator",)
FULL_TABLES: tuple[str, ...] = (
    "income", "fina_indicator", "balancesheet", "cashflow",
    "forecast", "express", "share_float", "pledge_stat", "stk_holdertrade", "namechange",
)
MAIN_BOARDS = ("沪主板", "深主板")


def tables_for_mode(mode: str) -> tuple[str, ...]:
    """模式 → 要拉的表集(纯函数)。lean 只拉 fina_indicator;full 拉核心+预警表。"""
    if mode == "lean":
        return LEAN_TABLES
    if mode == "full":
        return FULL_TABLES
    raise ValueError(f"未知 mode={mode!r}(应为 lean / full)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lean", choices=("lean", "full"))
    ap.add_argument("--board", default="main", choices=("main", "all"))
    ap.add_argument("--limit", type=int, default=0, help="只拉前 N 只(0=全部),用于试跑")
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

    print(f"mode={a.mode} board={a.board} | 待拉 {len(codes)} 只 × {len(tables)} 表 = {tables} "
          f"| 缓存优先(已拉自动跳过)", flush=True)
    t0 = time.time()
    done = empty = err = 0
    for code in codes:
        for table in tables:
            try:
                fetch_symbol_table(pro, table, code, CACHE)
            except EmptyCoreTableError:
                empty += 1  # 新上市/无财报:正常,跳过(已响亮:不落盘)
            except Exception as e:  # 单只单表失败响亮上报、继续
                err += 1
                print(f"  {code}/{table} 失败: {type(e).__name__}: {str(e)[:50]}", file=sys.stderr, flush=True)
        done += 1
        if done % 100 == 0:
            rate = done / max(time.time() - t0, 1) * 60
            print(f"  …{done}/{len(codes)} ({rate:.0f}只/min, 空{empty} 错{err})", flush=True)

    dt = time.time() - t0
    print(f"\n完成:{done} 只,空核心表 {empty} 次,失败 {err} 次,耗时 {dt/60:.1f} 分钟。", flush=True)
    print(f"→ 现在可跑 scripts.holdscore(质地排序已建在全市场财务本地之上)", flush=True)


if __name__ == "__main__":
    main()
