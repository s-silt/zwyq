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
from collections.abc import Mapping

import pandas as pd

from ashare_gauntlet.data.fetch import (
    EmptyCoreTableError,
    TokenExpiredError,
    call_with_retry,
    fetch_symbol_table,
)
from ashare_gauntlet.screen import board_of
from ashare_gauntlet.config import CACHE_DIR as CACHE, tushare_pro

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


def latest_period_end(today: str) -> str:
    """截至 ``today``(YYYYMMDD)最近一个**已结束**的报告期(季度末 0331/0630/0930/1231,日历常数)。

    用途:``--refresh`` 披露表优先路径的查询键。法定截止(``expected_min_end_date``)
    只给出"最迟必须已披露"的下限;而披露表(disclosure_date)能看到谁**提前**披露了
    最新一期(如 7 月即出的半年报,法定截止 8/31)——对这些票立即刷新,不等截止日。
    """
    y, md = today[:4], today[4:]
    if md >= "1231":
        return f"{y}1231"
    if md >= "0930":
        return f"{y}0930"
    if md >= "0630":
        return f"{y}0630"
    if md >= "0331":
        return f"{y}0331"
    return f"{int(y) - 1}1231"


def disclosed_stale_codes(
    disclosure: pd.DataFrame, local_max_end: Mapping[str, str], period: str
) -> list[str]:
    """纯函数:全市场披露表 + 本地新鲜度 → ``period`` 期需要重拉的代码列表(升序去重)。

    入选须同时满足:
      ① ``end_date == period``(防御:接口按期查询,仍过滤防混期);
      ② ``actual_date`` 非空(**已实际披露**;只有拟披露日 pre_date 的票拉了也只有旧数据);
      ③ 本地缓存 max(end_date) < period(缺缓存按 "" = 最旧,同样要拉)。
    """
    actual = disclosure["actual_date"]
    disclosed = disclosure.loc[
        (disclosure["end_date"].astype(str) == period)
        & actual.notna()
        & (actual.astype(str).str.strip() != ""),
        "ts_code",
    ]
    return sorted({str(c) for c in disclosed if local_max_end.get(str(c), "") < period})


def _fetch_disclosure(pro: object, period: str) -> pd.DataFrame | None:
    """全市场披露表(``disclosure_date`` 不带 ts_code,一次一期)。

    **不落缓存**:actual_date 每天都在增长(新披露不断填进来),按期缓存会把披露进度
    冻结在首拉那天。拉不到/为空/缺列 → 打 warning 返回 None,调用方回退
    ``expected_min_end_date`` 启发式(fail-loud 但不阻塞刷新)。TokenExpiredError
    例外向上抛:额度耗尽是全局致命错,回退只会烧掉几千次注定失败的逐只调用。
    """
    try:
        df = call_with_retry(lambda: pro.disclosure_date(end_date=period))
    except TokenExpiredError:
        raise
    except Exception as exc:  # noqa: BLE001 — 响亮降级:warning + 回退启发式,不静默
        print(
            f"警告:disclosure_date(end_date={period}) 拉取失败"
            f"({type(exc).__name__}: {str(exc)[:80]})—— 回退 expected_min_end_date 启发式",
            file=sys.stderr, flush=True,
        )
        return None
    if df.empty or not {"ts_code", "end_date", "actual_date"} <= set(df.columns):
        print(
            f"警告:disclosure_date(end_date={period}) 返回空/缺列"
            f"(rows={len(df)}, cols={list(df.columns)})—— 回退 expected_min_end_date 启发式",
            file=sys.stderr, flush=True,
        )
        return None
    return df


def tables_for_mode(mode: str) -> tuple[str, ...]:
    """模式 → 要拉的表集(纯函数)。lean 只拉 fina_indicator;core 拉 4 核心财报表;full 拉核心+预警表。"""
    if mode == "lean":
        return LEAN_TABLES
    if mode == "core":
        return CORE_TABLES
    if mode == "full":
        return FULL_TABLES
    raise ValueError(f"未知 mode={mode!r}(应为 lean / core / full)")


def _local_max_end(code: str) -> str:
    """该股本地 fina_indicator 缓存的最新报告期(YYYYMMDD)。

    无缓存/读不了/空表 → 返回 ""(排序上=最旧):读失败按"最旧"处理会触发整只重拉,
    坏缓存被覆盖修复而非被静默消费。
    """
    p = f"{CACHE}/fina_indicator/{code}.parquet"
    if not os.path.exists(p):
        return ""
    try:
        ed = pd.read_parquet(p, columns=["end_date"])["end_date"].astype(str)
        return str(ed.max()) if len(ed) else ""
    except Exception:
        return ""


def _is_fresh(code: str, target: str) -> bool:
    """该股 fina_indicator 缓存是否已含 ``target`` 报告期(refresh 的跳过判据)。"""
    return _local_max_end(code) >= target


def universe_status(universe: str, refresh: bool) -> str:
    """--universe → stock_basic list_status。P0③ 审计修复:财务缓存此前只按 L 名单回填,
    退市股 0 覆盖=财务侧幸存者偏差(2013 横截面 8.8% 缺席)。delisted=按 D 名单补拉,
    退市股财报永久冻结,一次回填终身有效;--refresh 对其无意义,组合 fail-loud 防误用。"""
    if universe == "delisted":
        if refresh:
            raise ValueError("--universe delisted 不可与 --refresh 组合:退市股财报永久冻结,无新鲜度可刷")
        return "D"
    return "L"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lean", choices=("lean", "core", "full"))
    ap.add_argument("--board", default="main", choices=("main", "all"))
    ap.add_argument("--limit", type=int, default=0, help="只拉前 N 只(0=全部),用于试跑")
    ap.add_argument("--universe", default="listed", choices=("listed", "delisted"),
                    help="delisted=退市股名单回填(P0③:修复财务侧幸存者偏差)")
    ap.add_argument("--refresh", action="store_true",
                    help="财报季刷新:法定披露期判过期 + 披露表(disclosure_date)抓提前披露最新期的票,"
                         "两者并集整只 force 重拉(否则缓存优先=永不更新)")
    a = ap.parse_args(argv)

    pro = tushare_pro()
    tables = tables_for_mode(a.mode)

    status = universe_status(a.universe, a.refresh)
    sb = call_with_retry(lambda: pro.stock_basic(list_status=status, fields="ts_code,name"))
    if sb.empty:
        raise SystemExit(f"stock_basic list_status={status} 拉取为空——源侧异常,拒绝当作'无退市股'")
    codes = [str(c) for c in sb["ts_code"]]
    if a.board == "main":
        codes = [c for c in codes if board_of(c) in MAIN_BOARDS]
    codes.sort()
    if a.limit:
        codes = codes[: a.limit]

    today = time.strftime("%Y%m%d")
    target = expected_min_end_date(today) if a.refresh else ""
    early: set[str] = set()  # 披露表优先路径:已提前披露最新期、且本地缺该期的票
    if a.refresh:
        print(f"refresh 模式:法定应披露最新报告期 = {target},已含该期的票跳过、过期票整只重拉", flush=True)
        # 优先路径:全市场拉一次披露表,把提前披露最新期(latest_period_end)的票也刷进来。
        # 它是法定下限之上的**增量**(并集):法定下限保证"最迟必须已披露"的硬新鲜度,
        # 披露表补上"谁提前出了下一期"——只用披露表会在无人披露时(如 7 月初查 0630)
        # 让 --refresh 空转,连法定早该有的旧期都不补。拉不到披露表则仅剩启发式(有注明)。
        period = latest_period_end(today)
        disc = _fetch_disclosure(pro, period)
        if disc is not None:
            local_max = {c: _local_max_end(c) for c in codes}
            early = set(disclosed_stale_codes(disc, local_max, period)) & set(codes)
            print(f"披露表优先:end_date={period} 全市场 {len(disc)} 行,已实际披露且本地缺该期 "
                  f"→ 额外重拉 {len(early)} 只(叠加在法定下限 {target} 的过期判断之上)", flush=True)
    print(f"mode={a.mode} board={a.board} | 待拉 {len(codes)} 只 × {len(tables)} 表 = {tables} "
          f"| {'刷新过期' if a.refresh else '缓存优先(已拉自动跳过)'}", flush=True)
    t0 = time.time()
    done = empty = err = skipped = 0
    for code in codes:
        force = False
        if a.refresh:
            # 跳过判据:法定下限已满足 且 不在披露表的"提前披露待刷"名单里
            if _is_fresh(code, target) and code not in early:
                skipped += 1
                done += 1
                continue
            force = True  # 过期/已披露新期:该股所有表整只重拉(接口返全历史,覆盖写一致)
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
