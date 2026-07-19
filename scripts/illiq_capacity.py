"""X-04 ILLIQ 容量/冲击敏感性 —— "成本模型陷阱"的定量解剖(docs/experiments.md 预注册)。

背景(methodology §6 判读3):ILLIQ(Amihud 2002)N=149 机械过五门(t+8.06,
多头腿+0.41%),但收益疑似集中在微流动性票——线性 15bp 滑点对这些票的冲击成本
过于乐观,过门不豁免结构性批评。本实验按预注册口径"分市值桶成本敏感性"回答:

1. 多头腿(高 ILLIQ 五分位,行业+市值双中性后)内部按**市值三分位**(小/中/大)
   拆桶:期均超额毛收益、NW t、换手、15bp 口径净、退出受限率(一字跌停/停牌顺延占比);
2. **break-even 单边滑点**(零新常数,纯代数逆运算):净=超额−τ×(2c+2s+印花税)=0
   解 s*——桶内超额能承受多大的单边冲击成本;
3. 桶内 21 日日均成交额(ADV)分布——判断层拿它对照下单规模(资金/ADV 占比),
   脚本不硬编码任何账户参数。

判定(预注册):若超额集中于最小市值桶且该桶 ADV 显著低(break-even 滑点被真实
冲击成本轻易吞没的量级),则成本陷阱成立、ILLIQ 维持不入分;若大市值桶独立存活
(t 稳、ADV 充足),陷阱降级。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.illiq_capacity
       [--fwd 21] [--commission 0.00025] [--slippage 0.0015]
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from ashare_gauntlet.backtest import newey_west_tstat
from ashare_gauntlet.config import CACHE_DIR as CACHE, HOLDSCORE_DIR, tushare_pro
from ashare_gauntlet.costs import round_trip_cost_rate
from ashare_gauntlet.data.fetch import fetch_market_day
from ashare_gauntlet.data.partition import assert_adj_complete, date_partition_files
from ashare_gauntlet.factor_model import daily_returns, neutralize_industry_size
from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.screen import board_of
from scripts.factor_backtest import (
    MAIN,
    amihud_illiq,
    defer_note,
    first_sellable_open,
    leg_turnover,
    one_word_limit_down,
    one_word_limit_up,
    quantile_legs,
)

MOM_LB = 250     # 与 factor_backtest 同锚:换仓日集合一致(N 可比)
BUCKETS = ("小", "中", "大")


def break_even_slippage(excess: float, turnover: float,
                        commission: float, stamp: float) -> float:
    """净=超额 − τ×(2c+2s+印花税)=0 的单边滑点解:s* = (超额/τ − 2c − 印花税)/2。

    现有成本模型(round_trip=2×佣金+2×滑点+卖出印花税)的纯代数逆运算,零新常数。
    τ<=0(零换手,分母语义消失)或输入 NaN → NaN 不伪造。
    """
    if not (turnover > 0) or math.isnan(excess):
        return float("nan")
    return (excess / turnover - 2.0 * commission - stamp) / 2.0


def mv_terciles(mv: pd.Series) -> pd.Series:
    """市值三分位标签(小/中/大)——小/中/大是最小的规模划分,描述性拆桶非交易规则。

    非空样本 <3 无三分位语义 → 全 NA(与 to_decile 的桶数纪律同精神)。
    """
    if int(mv.notna().sum()) < 3:
        return pd.Series(pd.NA, index=mv.index, dtype=object)
    ranks = mv.rank(method="first")
    return pd.qcut(ranks, 3, labels=list(BUCKETS)).astype(object)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, default=21)
    ap.add_argument("--commission", type=float, default=0.00025,
                    help="单边佣金率(万2.5,与 factor_backtest 同出处)")
    ap.add_argument("--slippage", type=float, default=0.0015,
                    help="单边滑点率(LWZ 2022 JFE 15bp 下沿)——净收益列的基准口径")
    a = ap.parse_args(argv)

    pro = tushare_pro()
    sb = call_with_retry(lambda: pro.stock_basic(
        list_status="L", fields="ts_code,industry")).set_index("ts_code")
    ind_all = sb["industry"].fillna("其他")

    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close", "amount"])
                    for f in date_partition_files(CACHE, "daily")], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in date_partition_files(CACHE, "adj_factor")], ignore_index=True)
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    assert_adj_complete(px)
    px["aclose"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    px["aopen"] = px["open"].astype(float) * px["adj_factor"].astype(float)
    dates = sorted(px["trade_date"].unique())
    close_p = px.pivot_table(index="trade_date", columns="ts_code", values="aclose")
    open_p = px.pivot_table(index="trade_date", columns="ts_code", values="aopen")
    amt_p = px.pivot_table(index="trade_date", columns="ts_code", values="amount")
    ret_p = daily_returns(close_p)

    di = {d: i for i, d in enumerate(dates)}
    month_last: dict[str, str] = {}
    for d in dates:
        month_last[d[:6]] = d
    rebal = [d for d in month_last.values() if di[d] >= MOM_LB and di[d] + 1 + a.fwd < len(dates)]
    print(f"加载完成:{len(dates)}交易日 → {len(rebal)}个月度换仓日;ILLIQ 多头腿"
          f"(高 Amihud 五分位,双中性)× 市值三分桶", flush=True)

    rows: list[dict] = []
    prev_leg: "set[str] | None" = None
    prev_bucket: dict[str, "set[str] | None"] = {b: None for b in BUCKETS}
    for k, t in enumerate(rebal):
        it = di[t]
        codes = [str(c) for c in close_p.columns[close_p.loc[t].notna()]
                 if board_of(str(c)) in MAIN]
        if len(codes) < 50:
            continue
        entry_date = dates[it + 1]
        locked = one_word_limit_up(fetch_market_day(pro, "daily", entry_date, CACHE),
                                   fetch_market_day(pro, "stk_limit", entry_date, CACHE), codes)
        codes = [c for c in codes if c not in locked]
        idx = pd.Index(codes)
        # 前向收益:T+1 开盘 → 窗口末;退出一字跌停/停牌顺延(与引擎同——ILLIQ 腿
        # 恰是微流动性票,跳过退出约束会系统性高估该腿)
        win = open_p.iloc[it + 1: it + 2 + a.fwd][codes]
        entry = win.iloc[0]
        exit_ = win.ffill().iloc[-1].copy()
        exit_pos = it + 1 + a.fwd
        _ld_cache: dict[int, set[str]] = {}

        def _locked(pos: int, c: str, _pool: list[str] = codes) -> bool:
            if pos not in _ld_cache:
                d_ = dates[pos]
                _ld_cache[pos] = one_word_limit_down(
                    fetch_market_day(pro, "daily", d_, CACHE),
                    fetch_market_day(pro, "stk_limit", d_, CACHE), _pool)
            return c in _ld_cache[pos]

        locked_exit = one_word_limit_down(fetch_market_day(pro, "daily", dates[exit_pos], CACHE),
                                          fetch_market_day(pro, "stk_limit", dates[exit_pos], CACHE),
                                          codes)
        suspended = {c for c in codes
                     if pd.isna(open_p.iloc[exit_pos].get(c)) and pd.notna(entry.get(c))}
        constrained = locked_exit | suspended        # 退出受限票(顺延或未解)
        n_def = n_unres = def_days = 0
        for c in constrained:
            r = first_sellable_open(open_p[c], exit_pos + 1, lambda j, _c=c: _locked(j, _c))
            if r is None:
                n_unres += 1
            else:
                exit_[c] = r[0]
                n_def += 1
                def_days += r[1] + 1
        fwd = exit_ / entry - 1.0

        db = fetch_market_day(pro, "daily_basic", t, CACHE).set_index("ts_code")
        mv = pd.to_numeric(db["total_mv"], errors="coerce").reindex(idx)
        logmv = np.log(mv.where(mv > 0))
        ind = ind_all.reindex(idx).fillna("其他")
        raw = amihud_illiq(ret_p.iloc[: it + 1], amt_p.iloc[: it + 1], 21).reindex(idx)
        neu = neutralize_industry_size(raw, ind, logmv)
        _, high = quantile_legs(neu[fwd.notna()], 5)   # 多头腿=高 ILLIQ 五分位(正向因子)
        if not high:
            continue
        leg = pd.Index(sorted(high))
        adv21 = amt_p.iloc[max(0, it - 20): it + 1][list(leg)].mean()   # 千元/日(tushare amount 单位)
        buckets = mv_terciles(mv[leg])
        row: dict = {"date": t, "mkt_fwd": float(fwd[idx].mean()),
                     "cost_rt": round_trip_cost_rate(entry_date, a.commission, a.slippage,
                                                     sell_date=dates[exit_pos]),
                     "stamp": round_trip_cost_rate(entry_date, 0.0, 0.0,
                                                   sell_date=dates[exit_pos]),
                     "n_leg": int(len(leg)),
                     "TO_leg": leg_turnover(prev_leg, set(leg))}
        prev_leg = set(leg)
        for b in BUCKETS:
            m = pd.Index([c for c in leg if buckets.get(c) == b])
            if not len(m):
                row[f"ret_{b}"] = float("nan")
                continue
            row[f"ret_{b}"] = float(fwd[m].mean())
            row[f"n_{b}"] = int(len(m))
            row[f"adv_{b}"] = float(adv21[m].median())          # 桶内中位 ADV(千元/日)
            row[f"TO_{b}"] = leg_turnover(prev_bucket[b], set(m))
            prev_bucket[b] = set(m)
            row[f"constr_{b}"] = float(len(set(m) & constrained) / len(m))
        row["ret_leg"] = float(fwd[leg].mean())
        print(f"  {k + 1}/{len(rebal)} {t} 腿{len(leg)}只 剔涨停{len(locked)}"
              f"{defer_note(n_def, def_days, n_unres)}", flush=True)
        rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("无有效换仓期(检查缓存覆盖)")
    os.makedirs(HOLDSCORE_DIR, exist_ok=True)
    res.to_json(f"{HOLDSCORE_DIR}/illiq_capacity.json", orient="records",
                force_ascii=False, indent=2)   # 先落盘再报告(报告 bug 不毁计算)

    print(f"\n=== X-04 ILLIQ 容量/冲击敏感性(N={len(res)},{res['date'].min()}→"
          f"{res['date'].max()};多头腿=高 Amihud 五分位·双中性;桶=腿内市值三分位)===")
    print(f"{'桶':>3}{'N期':>5}{'均只数':>7}{'超额毛%':>8}{'NW t':>7}{'换手':>6}"
          f"{'净%(15bp)':>10}{'break-even滑点bp':>17}{'ADV中位(万元/日)':>17}{'退出受限率':>10}")
    comm = a.commission
    for b in ("小", "中", "大"):
        ex = (res[f"ret_{b}"] - res["mkt_fwd"]).dropna()
        _, tnw, _ = newey_west_tstat(ex)
        to = res[f"TO_{b}"].mean()
        net = (res[f"ret_{b}"] - res["mkt_fwd"] - res[f"TO_{b}"] * res["cost_rt"]).dropna()
        s_star = break_even_slippage(float(ex.mean()), float(to), comm, float(res["stamp"].mean()))
        adv_wan = res[f"adv_{b}"].median() / 10.0      # 千元 → 万元
        print(f"{b:>3}{len(ex):>5}{res[f'n_{b}'].mean():>7.1f}{ex.mean() * 100:>+7.2f}%"
              f"{tnw:>+7.2f}{to:>6.0%}{net.mean() * 100:>+9.2f}%"
              f"{s_star * 1e4:>16.0f}b{adv_wan:>16.0f}{res[f'constr_{b}'].mean():>10.1%}")
    ex_leg = (res["ret_leg"] - res["mkt_fwd"]).dropna()
    _, t_leg, _ = newey_west_tstat(ex_leg)
    print(f"{'腿':>3}{len(ex_leg):>5}{res['n_leg'].mean():>7.1f}{ex_leg.mean() * 100:>+7.2f}%"
          f"{t_leg:>+7.2f}{res['TO_leg'].mean():>6.0%}")
    print("判定口径(预注册):超额集中最小市值桶且其 ADV 量级不足 → 成本陷阱成立,"
          "ILLIQ 维持不入分;大市值桶独立存活(t 稳+ADV 充足)→ 陷阱降级。"
          "ADV 供判断层对照下单规模(资金/ADV 占比),脚本不设账户参数。")
    print("→ 明细 data/holdscore/illiq_capacity.json(已在报告前落盘)")


if __name__ == "__main__":
    main()
