"""财务因子退市覆盖率审计(吸纳终榜 P0③)—— 量化财务表侧的幸存者偏差。

价格侧 survivorship 已修(退出 ffill+顺延),但财务表(fina_indicator/income/cashflow/
balancesheet)若不含退市股的历史财报,ACC/EP 等因子的历史横截面就只剩"活下来的公司"
——崩盘前的差公司恰好缺席,因子表现被系统性高估。此前从未量化过这层偏差(对抗轮
双方独立点名为 EP/BP/ACC 可信度的下一个系统偏差源)。

口径:对每个审计时点 t(半年采样,--every 可调):
- 宇宙 = t 日有收盘价的主板股(与 factor_backtest 同源同过滤);
- 按"今天仍在上市名单(stock_basic list_status=L)"切 既存组/已消失组;
- 分别报 ACC 三件套(income.n_income_attr_p + cashflow.n_cashflow_act +
  balancesheet.total_assets)的 PIT 可得率;gap = 既存 − 消失。
判读:gap 持续显著 >0 → 财务侧幸存者偏差实存,ACC/EP 读数需打折标注;
gap≈0 → 数据商回填完整,该风险排除。只报数不改口径(审计非修复)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.fina_coverage_audit [--every 6]
"""
from __future__ import annotations

import argparse
import math
import os

import pandas as pd

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.data.partition import date_partition_files
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.screen import board_of
from scripts.factor_backtest import CACHE, MAIN, _load, _pit


def coverage_split(universe: list[str], have: set[str], listed_now: set[str]) -> dict:
    """宇宙按'今天仍上市'切两组的 PIT 财务可得率。

    cov_listed/cov_gone = 组内可得率;gap = cov_listed − cov_gone(幸存者偏差读数);
    组为空 → NaN 不伪造。全部定义性,无阈值。
    """
    listed = [c for c in universe if c in listed_now]
    gone = [c for c in universe if c not in listed_now]
    cov_l = sum(c in have for c in listed) / len(listed) if listed else math.nan
    cov_g = sum(c in have for c in gone) / len(gone) if gone else math.nan
    return {"n": len(universe), "n_gone": len(gone),
            "cov_listed": cov_l, "cov_gone": cov_g,
            "gap": cov_l - cov_g if not (math.isnan(cov_l) or math.isnan(cov_g)) else math.nan}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=6,
                    help="每第 N 个月末取一个审计点(默认6=半年;覆盖率变化缓慢,无需月度)")
    a = ap.parse_args(argv)
    load_env_local()
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    listed_now = set(call_with_retry(
        lambda: pro.stock_basic(list_status="L", fields="ts_code"))["ts_code"].astype(str))

    inc = _load("income", ["n_income_attr_p"])
    cf = _load("cashflow", ["n_cashflow_act"])
    bs = _load("balancesheet", ["total_assets"])

    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
                    for f in date_partition_files(CACHE, "daily")], ignore_index=True)
    month_last: dict[str, str] = {}
    for d in sorted(da["trade_date"].astype(str).unique()):
        month_last[d[:6]] = d
    points = list(month_last.values())[::max(a.every, 1)]
    by_date = da.groupby("trade_date")["ts_code"]

    print(f"=== 财务因子退市覆盖率审计(ACC三件套 PIT 可得率;{len(points)}个审计点,主板)===")
    print(f"{'时点':>10}{'宇宙':>6}{'已消失':>6}{'既存覆盖':>9}{'消失覆盖':>9}{'gap':>7}")
    rows = []
    for t in points:
        codes = [str(c) for c in by_date.get_group(t) if board_of(str(c)) in MAIN]
        have = (set(_pit(inc, t).index) & set(_pit(cf, t).index) & set(_pit(bs, t).index))
        r = coverage_split(codes, have, listed_now)
        rows.append({"date": t, **r})
        fmt = lambda x: f"{x:>8.1%}" if x == x else "     n/a"
        gap_s = f"{r['gap']:>+6.1%}" if r["gap"] == r["gap"] else "   n/a"
        print(f"{t:>10}{r['n']:>6}{r['n_gone']:>6}{fmt(r['cov_listed'])}{fmt(r['cov_gone'])}{gap_s}")
    df = pd.DataFrame(rows)
    g = df["gap"].dropna()
    print(f"\ngap 均值 {g.mean():+.1%} | 最大 {g.max():+.1%} | 消失组均覆盖 "
          f"{df['cov_gone'].dropna().mean():.1%} vs 既存组 {df['cov_listed'].dropna().mean():.1%}")
    print("判读:gap 持续>0 = 财务侧幸存者偏差实存(退市股缺财报,差公司缺席历史横截面,"
          "ACC/EP 读数偏乐观);gap≈0 = 数据商回填完整,风险排除。")
    os.makedirs("data/holdscore", exist_ok=True)
    df.to_json("data/holdscore/fina_coverage_audit.json", orient="records", force_ascii=False, indent=2)
    print("→ 明细 data/holdscore/fina_coverage_audit.json")


if __name__ == "__main__":
    main()
