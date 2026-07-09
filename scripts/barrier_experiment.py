"""止盈止损路径实验 —— 回答"信号买入+到点止盈止损"是否改写门禁结论(研究件,非生产)。

用户质疑:门禁测的是固定持有期,实盘是到点止盈止损,两者不是一回事。本实验做
**逐日路径模拟**:TREND 顶档(追涨信号)T+1 开盘买入,逐日查止盈/止损/到期,
对照 TREND 底档(超跌)与全宇宙(市场基线,分离 β)。

口径:入场剔一字涨停(买不进);跳空按开盘成交(止损跳空更亏,真实);同日双触
止损优先(保守);停牌跳过;退市按最后有效收盘;成本=round_trip(佣金/滑点+
印花税按**实际卖出日** PIT)。参数网格公开全报(不挑参数):tp/sl 取双仓制及常见
倍数档,H=63(季度上限,超过即非"短线止盈止损"语境)。
统计:期均净收益的 NW t、胜率、平均持有天、年化(净/均持有天×252)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.barrier_experiment [--start YYYYMMDD]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ashare_gauntlet.backtest import newey_west_tstat
from ashare_gauntlet.costs import round_trip_cost_rate
from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import fetch_market_day
from ashare_gauntlet.data.partition import assert_adj_complete, date_partition_files
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factor_model import trend_ma_distance
from ashare_gauntlet.screen import board_of
from scripts.factor_backtest import one_word_limit_up

CACHE = "data/cache"
MAIN = ("沪主板", "深主板")
GRID = ((0.08, 0.05), (0.10, 0.07), (0.20, 0.10))   # (止盈,止损):双仓制参数及常见档,全报不挑
H = 63                                               # 持有上限(季度;更长即非"短线"语境)


def barrier_paths(entry: np.ndarray, o: np.ndarray, h: np.ndarray, low: np.ndarray,
                  c: np.ndarray, tp: float, sl: float) -> tuple:
    """向量化逐日路径:行=交易日(0=入场日)、列=股票。返回 (exit_px, days, reason)。

    每日顺序:停牌(NaN)跳过 → 跳空(开盘已越界按开盘)→ 盘中触碰(按触发价,
    同日双触止损优先=保守)→ 到期按期末收盘 → 数据尽头按最后有效收盘("end")。
    """
    n = entry.shape[0]
    tp_px, sl_px = entry * (1.0 + tp), entry * (1.0 - sl)
    exit_px = np.full(n, np.nan)
    days = np.full(n, -1, dtype=int)
    reason = np.array(["end"] * n, dtype=object)
    live = np.ones(n, dtype=bool)
    last_close = np.full(n, np.nan)
    for d in range(len(o)):
        od, hd, ld, cd = o[d], h[d], low[d], c[d]
        has = ~np.isnan(od) & live
        last_close = np.where(~np.isnan(cd) & live, cd, last_close)
        gap_dn = has & (od <= sl_px)
        gap_up = has & ~gap_dn & (od >= tp_px)
        hit_sl = has & ~gap_dn & ~gap_up & (ld <= sl_px)
        hit_tp = has & ~gap_dn & ~gap_up & ~hit_sl & (hd >= tp_px)
        for mask, px, why in ((gap_dn, od, "sl"), (gap_up, od, "tp"),
                              (hit_sl, sl_px, "sl"), (hit_tp, tp_px, "tp")):
            exit_px = np.where(mask, px, exit_px)
            days = np.where(mask, d, days)
            for i in np.flatnonzero(mask):
                reason[i] = why
            live &= ~mask
        if not live.any():
            break
    # 到期未触:期末最后有效收盘;窗内完全无价(入场后即退市)保持 NaN 由调用方丢弃
    tail = live & ~np.isnan(last_close)
    exit_px = np.where(tail, last_close, exit_px)
    days = np.where(tail, len(o) - 1, days)
    for i in np.flatnonzero(tail):
        reason[i] = "time" if not np.isnan(c[-1][i]) else "end"
    return exit_px, days, reason


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    a = ap.parse_args(argv)
    load_env_local()
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])

    cols = ["ts_code", "trade_date", "open", "high", "low", "close"]
    da = pd.concat([pd.read_parquet(f, columns=cols)
                    for f in date_partition_files(CACHE, "daily")], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in date_partition_files(CACHE, "adj_factor")], ignore_index=True)
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    assert_adj_complete(px)
    for k in ("open", "high", "low", "close"):
        px["a" + k] = px[k].astype(float) * px["adj_factor"].astype(float)
    panels = {k: px.pivot_table(index="trade_date", columns="ts_code", values="a" + k)
              for k in ("open", "high", "low", "close")}
    dates = sorted(px["trade_date"].astype(str).unique())
    di = {d: i for i, d in enumerate(dates)}
    month_last: dict[str, str] = {}
    for d in dates:
        month_last[d[:6]] = d
    rebal = [d for d in month_last.values() if di[d] >= 200 and di[d] + 1 + H < len(dates)]
    if a.start:
        rebal = [d for d in rebal if d >= a.start]

    legs = {"TREND顶档(追涨)": [], "TREND底档(超跌)": [], "全宇宙(基线)": []}
    daysbook = {k: [] for k in legs}
    reasons: dict[str, dict] = {k: {"tp": 0, "sl": 0, "time": 0, "end": 0} for k in legs}
    results_by_grid = {g: {k: [] for k in legs} for g in GRID}
    days_by_grid = {g: {k: [] for k in legs} for g in GRID}

    print(f"逐日路径实验:{len(rebal)} 个入场期 × 网格 {GRID} × H={H}", flush=True)
    for k, t in enumerate(rebal):
        it = di[t]
        cp = panels["close"]
        codes = [str(cc) for cc in cp.columns[cp.loc[t].notna()] if board_of(str(cc)) in MAIN]
        entry_date = dates[it + 1]
        locked = one_word_limit_up(fetch_market_day(pro, "daily", entry_date, CACHE),
                                   fetch_market_day(pro, "stk_limit", entry_date, CACHE), codes)
        codes = [cc for cc in codes if cc not in locked]
        trend = trend_ma_distance(cp.iloc[: it + 1][codes]).dropna()
        if len(trend) < 100:
            continue
        dec = pd.qcut(trend.rank(method="first"), 10, labels=False)
        pools = {"TREND顶档(追涨)": list(trend.index[dec == 9]),
                 "TREND底档(超跌)": list(trend.index[dec == 0]),
                 "全宇宙(基线)": list(trend.index)}
        win = slice(it + 1, it + 2 + H)
        for leg, pool in pools.items():
            o = panels["open"].iloc[win][pool].to_numpy()
            hh = panels["high"].iloc[win][pool].to_numpy()
            ll = panels["low"].iloc[win][pool].to_numpy()
            cc_ = panels["close"].iloc[win][pool].to_numpy()
            entry = o[0]
            ok = ~np.isnan(entry)
            for g in GRID:
                ep, dd, why = barrier_paths(entry[ok], o[1:, ok], hh[1:, ok], ll[1:, ok],
                                            cc_[1:, ok], tp=g[0], sl=g[1])
                valid = ~np.isnan(ep)
                if not valid.any():
                    continue
                exit_dates = [dates[min(it + 2 + int(x), len(dates) - 1)] for x in dd[valid]]
                gross = ep[valid] / entry[ok][valid] - 1.0
                cost = np.array([round_trip_cost_rate(entry_date, 0.00025, 0.0015, sell_date=xd)
                                 for xd in exit_dates])
                results_by_grid[g][leg].append(float(np.mean(gross - cost)))
                days_by_grid[g][leg].append(float(np.mean(dd[valid])) + 1)
                if g == GRID[0]:
                    for w in why[valid]:
                        reasons[leg][w] += 1
        if (k + 1) % 30 == 0:
            print(f"  {k + 1}/{len(rebal)}", flush=True)

    print(f"\n=== 止盈止损路径实验(N={len(rebal)}期,成本含实际卖出日印花税;全网格公开)===")
    print(f"{'网格(止盈/止损)':>14}{'腿':>14}{'期均净收益':>10}{'NW t':>7}{'胜率':>6}{'均持有':>7}{'年化净':>8}")
    for g in GRID:
        for leg in legs:
            r = pd.Series(results_by_grid[g][leg])
            if r.empty:
                continue
            _, tstat, _ = newey_west_tstat(r)
            d_mean = float(pd.Series(days_by_grid[g][leg]).mean())
            ann = r.mean() / d_mean * 252
            print(f"{g[0]:.0%}/{g[1]:.0%}".rjust(14) + f"{leg:>14}"
                  f"{r.mean() * 100:>+9.2f}%{tstat:>+7.2f}{(r > 0).mean() * 100:>5.0f}%"
                  f"{d_mean:>6.1f}日{ann * 100:>+7.1f}%")
    print(f"\n离场原因分布(网格 {GRID[0]},全期合计):")
    for leg, rc in reasons.items():
        tot = sum(rc.values()) or 1
        print(f"  {leg}: 止盈{rc['tp']/tot:.0%} 止损{rc['sl']/tot:.0%} 到期{rc['time']/tot:.0%} 退市{rc['end']/tot:.0%}")
    print("(读法:追涨腿若年化净≤基线,则'止盈止损'没有拯救追涨信号;三腿同规则,差异全来自入场信号)")


if __name__ == "__main__":
    main()
