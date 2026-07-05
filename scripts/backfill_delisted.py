"""退市股财务回填(P0③ 修复)—— VIP 按报告期接口补齐财务侧幸存者偏差。

审计(scripts.fina_coverage_audit)坐实:财务缓存按 L 名单逐票回填,退市股 0 覆盖,
2013 横截面 8.8% 股票缺席 ACC/ROE/GP 样本。个券接口对退市股返回空(镜像实测),
但 *_vip 按期全市场接口含退市股(601558 华锐风电实测在内)。

流程:每 (vip表, 报告期) 分页拉全市场 → 原始按期落盘(data/cache/<ep>_vip/<period>.parquet,
幂等零 API 重跑)→ 过滤主板退市股 → 拆写 per-symbol 布局(data/cache/<table>/<code>.parquet)
—— _load/latest_rows glob 同目录,零改动透明受益。**只写缺文件的票**(在市股既有缓存
绝不覆盖:两套接口的行内容一致性未审计,混写会引入不可追溯的口径差)。
报告期自 20121231(回测样本 2014 起的 PIT 尾部)至法定最新期。
页大小 5000 是请求分页操作参数(镜像单页上限内),非评分常数。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.backfill_delisted
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from ashare_gauntlet.data.cache import read_or_fetch
from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.screen import board_of
from scripts.backfill_fina import MAIN_BOARDS, latest_period_end

CACHE = "data/cache"
PAGE = 5000
VIP = {"income": "income_vip", "balancesheet": "balancesheet_vip",
       "cashflow": "cashflow_vip", "fina_indicator": "fina_indicator_vip"}
FIRST_PERIOD = "20121231"   # 回测首个换仓日(2014-01)的 PIT 尾部所需最早报告期


def fetch_all_pages(page: Callable[[int, int], pd.DataFrame], limit: int = PAGE) -> pd.DataFrame:
    """limit/offset 分页拉全:末页行数 < limit 即停;首页即空返回空表。"""
    out: list[pd.DataFrame] = []
    offset = 0
    while True:
        df = page(limit, offset)
        if df.empty:
            break
        out.append(df)
        if len(df) < limit:
            break
        offset += limit
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def write_missing_symbol_tables(rows: pd.DataFrame, cache_dir: str | Path, table: str) -> list[str]:
    """把按期数据拆写进 per-symbol 布局;**只写缺文件的票**,既有缓存绝不覆盖。

    返回实际写入的代码列表(排序)。在市股的 per-symbol 缓存来自个券接口,与 vip
    接口行内容的一致性未审计——只补缺不混写,口径差异不进存量。
    """
    written: list[str] = []
    for code, g in rows.groupby("ts_code"):
        path = Path(cache_dir) / table / f"{code}.parquet"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        g.reset_index(drop=True).to_parquet(path, index=False)
        written.append(str(code))
    return sorted(written)


def quarterly_periods(first: str, last: str) -> list[str]:
    """[first, last] 内的季度报告期序列(0331/0630/0930/1231,定义性)。"""
    out = []
    for y in range(int(first[:4]), int(last[:4]) + 1):
        for q in ("0331", "0630", "0930", "1231"):
            p = f"{y}{q}"
            if first <= p <= last:
                out.append(p)
    return out


def main() -> None:
    load_env_local()
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    sb = call_with_retry(lambda: pro.stock_basic(list_status="D", fields="ts_code,name,delist_date"))
    if sb.empty:
        raise SystemExit("stock_basic list_status=D 为空——源侧异常,拒绝当作'无退市股'")
    gone = {str(c) for c in sb["ts_code"] if board_of(str(c)) in MAIN_BOARDS}
    periods = quarterly_periods(FIRST_PERIOD, latest_period_end(time.strftime("%Y%m%d")))
    print(f"主板退市股 {len(gone)} 只 | 报告期 {periods[0]}..{periods[-1]}({len(periods)}期)× {len(VIP)} 表", flush=True)

    total_written: dict[str, int] = {}
    for table, ep in VIP.items():
        chunks = []
        for p in periods:
            raw = read_or_fetch(
                Path(CACHE) / ep / f"{p}.parquet",
                lambda e=ep, pp=p: fetch_all_pages(
                    lambda limit, offset: call_with_retry(
                        lambda: getattr(pro, e)(period=pp, limit=limit, offset=offset))))
            if not raw.empty:
                chunks.append(raw[raw["ts_code"].astype(str).isin(gone)])
        rows = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if rows.empty:
            print(f"  {table}: vip 全期无退市股行(异常,人工核查)", flush=True)
            continue
        written = write_missing_symbol_tables(rows, CACHE, table)
        total_written[table] = len(written)
        print(f"  {table}: 退市股行 {len(rows)},新写 {len(written)} 只(既有跳过 "
              f"{rows['ts_code'].nunique() - len(written)})", flush=True)
    print(f"完成:{total_written};重跑 scripts.fina_coverage_audit 验证 gap 收敛", flush=True)


if __name__ == "__main__":
    main()
