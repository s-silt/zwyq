"""market_temp —— 每日盯盘头部一行"市场温度":指导仓位松紧,不选股、不预测。

双仓制下的用途:短线仓(右侧入场、硬止损)对市场情绪极敏感,炸板率抬头/缩量阴跌时
收紧新开仓;长线仓看 regime 分清 α/β。四个读数只并列 surface,不加权综合
(权重会是 magic number,冷热判断留给人)。

as_of 锚定 = 本地 daily 缓存最新日:全部读数对齐同一交易日,缓存未刷新时读数就是
旧日的(日期印在行首,陈旧自然可见),不混用两天的口径。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.market_temp
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.factsheet import north_flow_disclosure
from ashare_gauntlet.market_temp import (
    BASELINE_WINDOW,
    amount_ratio,
    limit_counts,
    north_turnover_recent,
    summary_line,
)
from scripts.pick_track import CACHE, INDEX_CODE, REGIME_WINDOW, load_index_daily, regime_return

# tushare daily.amount 单位=千元;1 亿元 = 1e8 元 = 1e5 千元(单位换算,非阈值)
_QIANYUAN_PER_YI = 1e5


def main() -> None:
    day_files = sorted(glob.glob(f"{CACHE}/daily/*.parquet"))
    if len(day_files) < BASELINE_WINDOW + 1:
        raise SystemExit(f"daily 缓存仅 {len(day_files)} 日,成交额基准需 {BASELINE_WINDOW + 1} 日"
                         "(先跑 scripts.refresh 回填)")
    trade_days = [os.path.basename(f)[:8] for f in day_files]
    as_of = trade_days[-1]

    # ② 全市场(沪深京)成交额:今日 vs 前 20 日均值 —— 本地缓存,零 API
    amounts_yi = [float(pd.read_parquet(f, columns=["amount"])["amount"].sum()) / _QIANYUAN_PER_YI
                  for f in day_files[-(BASELINE_WINDOW + 1):]]
    amt = amount_ratio(amounts_yi)

    from ashare_gauntlet.config import tushare_pro
    pro = tushare_pro()

    # ① 涨停/炸板/跌停:limit_list_d 不在 MARKET_ENDPOINTS,直接单日调用不落缓存
    #    (fetch_market_day 会把未发布日的空拉永久缓存;这里空表由 limit_counts fail-loud,
    #    下次运行自然重试)
    lim_df = call_with_retry(lambda: pro.limit_list_d(trade_date=as_of))  # type: ignore[attr-defined]
    lim = limit_counts(lim_df)

    # ③ 北向成交额:净流入 2024-08-19 制度性停披露(factsheet LANDMINE),只 surface 成交额。
    #    拉取窗口复用 20 交易日约定 —— HK 最长连休后仍留 ≥5 个有数日;1 行/日,直接拉最新
    north_start = trade_days[-(BASELINE_WINDOW + 1)]
    mf = call_with_retry(lambda: pro.moneyflow_hsgt(  # type: ignore[attr-defined]
        start_date=north_start, end_date=as_of))
    north = north_turnover_recent(mf)

    # ④ 沪深300 regime:复用 pick_track 的缓存加载 + 20 日涨跌读数
    idx_days = trade_days[-(REGIME_WINDOW + 1):]
    idx = load_index_daily(pro, INDEX_CODE, start_date=idx_days[0], end_date=as_of,
                           expected_days=idx_days)
    rg = regime_return(idx, REGIME_WINDOW)

    # —— 头部一行 + 各读数明细(明细供追问,单行供盯盘)——
    print(summary_line(as_of, lim, amt, north, regime_pct=rg, regime_window=REGIME_WINDOW))
    print(f"\n  涨跌停: 涨停收盘 {lim['up']} 只 / 炸板 {lim['broken']} 只 / 跌停 {lim['down']} 只;"
          f"炸板率=炸板/(涨停+炸板),抬头=情绪退潮")
    print(f"  成交额: 今日全市场(沪深京) {amt['today']:.0f} 亿,前{amt['window']}日均 "
          f"{amt['baseline']:.0f} 亿,比值 ×{amt['ratio']:.2f}(>1 放量 / <1 缩量,无阈值)")
    print(f"  北向:   最近有数日 {north['latest_date']} 成交 {north['latest_yi']:.0f} 亿,"
          f"近{north['days']}个有数日累计 {north['sum_yi']:.0f} 亿")
    print(f"          ({north_flow_disclosure()})")
    print(f"  regime: 沪深300 最近{REGIME_WINDOW}交易日 {rg * 100:+.1f}%"
          f"(个股与 300 同跌=β 问题,300 涨而个股跌=α 问题)")


if __name__ == "__main__":
    main()
