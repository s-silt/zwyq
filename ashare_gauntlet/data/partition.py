"""日期分区缓存的统一文件枚举 + adj_factor 完整性断言。

背景(真实事故,非假设):data/cache/daily/ 曾混入 ``<code>_<start>_<end>.parquet``
形态的整段拉取文件(已人工隔离)。直接 ``glob */*.parquet`` 的脚本会把它们当日分区
读进面板 → 交易日历错乱、横截面被重复行污染。此处统一只认
``^\\d{8}\\.parquet$``(fetch_market_day 的日分区落盘约定),混入文件打印 warning
点名列出(surface 不静默)后忽略——不 raise:污染文件不该让全管线瘫痪,隔离动作
留给人工;但也绝不允许静默读入。
"""
from __future__ import annotations

import glob
import os
import re
import sys

import pandas as pd

# 日分区文件名约定:8 位日期 + .parquet(ashare_gauntlet.data.fetch.fetch_market_day 落盘口径)
_DATE_PARQUET = re.compile(r"^\d{8}\.parquet$")


def date_partition_files(cache_dir: str, endpoint: str) -> list[str]:
    """``<cache_dir>/<endpoint>/`` 下的日分区文件完整路径,按日期(=文件名)升序。

    只返回文件名匹配 ``^\\d{8}\\.parquet$`` 的;混入的非日期 .parquet(如整段拉取的
    ``<code>_<start>_<end>.parquet``)不返回,并向 stderr 打印 warning 点名列出。
    目录不存在/为空 → 返回 []( 空缓存要不要 fail-loud 由各调用方按语境决定)。
    """
    files = glob.glob(f"{cache_dir}/{endpoint}/*.parquet")
    good = sorted(f for f in files if _DATE_PARQUET.match(os.path.basename(f)))
    bad = sorted(os.path.basename(f) for f in files
                 if not _DATE_PARQUET.match(os.path.basename(f)))
    if bad:
        print(f"⚠ {endpoint}/ 混入 {len(bad)} 个非日分区 parquet(已忽略,请隔离出缓存目录):"
              f" {' '.join(bad)}", file=sys.stderr)
    return good


def assert_adj_complete(px: pd.DataFrame) -> None:
    """daily×adj_factor 左合并结果的 fail-loud 断言:任一行缺 ``adj_factor`` 即 raise。

    缺 adj_factor 会让该 (票, 日) 的复权价变 NaN,被下游 dropna/notna 静默吞掉
    (MOM/MA20/信号/前向收益悄悄错位)。与 scripts.factor_rank 原有内联断言同一口径,
    抽为纯函数供 factor_rank / pick_track / panel 共用。
    """
    _miss = px.loc[px["adj_factor"].isna(), "trade_date"].unique()
    if len(_miss):
        raise SystemExit(f"adj_factor 缺 {len(_miss)} 个交易日(如 {sorted(_miss)[:4]})——先 backfill 补齐,"
                         "拒绝静默 NaN 传染")
