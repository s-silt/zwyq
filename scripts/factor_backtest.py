"""因子 IC 回测(修正版)—— 把因子模型从"文献支撑"升级到"A股本地实证"。

对每个月度换仓日 t,**point-in-time**(只用 ann_date<=t 已披露财报,防未来函数)构造因子横截面,
配 **T+1 开盘买入的未来收益**,算横截面 IC(秩相关)+ 分组多空 spread(可交易性)。

对抗式审计(2026-07-01)后的修正:
1. 【survivorship】退市/停牌股 exit 用持有窗口内**最后成交价(ffill)**,退市前暴跌收益不再被丢 NaN。
2. 【size 中性】拉 daily_basic.total_mv,因子在 **行业+市值** 双中性后再算 IC(排除"BP=小盘代理")。
3. 【t 虚高】用 AR(1) 自相关修正的 N_eff 算 t(adjusted_tstat),不再 iid 高估。
4. 【MOM horizon】用标准 **12-1 动量**(近 250 日、跳过最近 21 日)而非 120 日不跳月。
5. _pit 先按 end_date 再按 ann_date 取最新期(防乱序重述选错期)。

诚实:样本仍仅 2022-26 一个 regime、单一市场,IC 是实证读数非跨周期保证。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.factor_backtest [--board main] [--fwd 21]
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

from ashare_gauntlet.backtest import adjusted_tstat, information_coefficient, quantile_spread
from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry, fetch_market_day
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factor_model import industry_neutralize
from ashare_gauntlet.screen import board_of

CACHE = "data/cache"
MAIN = ("沪主板", "深主板")


def _load(ep: str, cols: list[str]) -> pd.DataFrame:
    # update_flag 参与排序:同 (end_date,ann_date) 的快照 vs 更正重述行,'1'=更正后(tushare 官方
    # 语义)排最后,_pit 的 tail(1) 才确定性取到更正值(fina_indicator 缓存暂无此列,自动降级两键)。
    need = ["ts_code", "end_date", "ann_date", "update_flag"] + cols
    out = []
    for f in glob.glob(f"{CACHE}/{ep}/*.parquet"):
        try:
            out.append(pd.read_parquet(f, columns=need))
        except Exception:
            df = pd.read_parquet(f)
            out.append(df[[c for c in need if c in df.columns]])
    df = pd.concat(out, ignore_index=True)
    df["ann_date"] = df["ann_date"].astype(str)
    df["end_date"] = df["end_date"].astype(str)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    sort_keys = ["ts_code"] + [k for k in ("end_date", "ann_date", "update_flag") if k in df.columns]
    return df.sort_values(sort_keys, kind="mergesort")


def _pit(df: pd.DataFrame, asof: str) -> pd.DataFrame:
    """截至 asof 已公告(ann_date<=asof)的最新一期(按 end_date/ann_date 已排序)。"""
    d = df[df["ann_date"] <= asof]
    return d.groupby("ts_code", sort=False).tail(1).set_index("ts_code")


def _neutralize(fac: pd.Series, ind: pd.Series, logmv: pd.Series) -> pd.Series:
    """行业中位数去均值 + 市值十分位去均值(双中性:去掉行业与规模的水平效应)。"""
    f1 = industry_neutralize(fac, ind)
    sb = pd.qcut(logmv.reindex(fac.index).rank(method="first"), 10, labels=False, duplicates="drop")
    return f1 - f1.groupby(sb).transform("median")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main")
    ap.add_argument("--fwd", type=int, default=21)
    a = ap.parse_args(argv)
    load_env_local()
    MOM_LB, MOM_SKIP = 250, 21   # 12-1 动量:近250日、跳最近21日

    fina = _load("fina_indicator", ["eps", "bps", "roe"])
    inc = _load("income", ["revenue", "oper_cost", "n_income_attr_p"])
    cf = _load("cashflow", ["n_cashflow_act"])
    bs = _load("balancesheet", ["total_assets"])

    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    sb = call_with_retry(lambda: pro.stock_basic(list_status="L", fields="ts_code,industry")).set_index("ts_code")
    ind_all = sb["industry"].fillna("其他")

    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"])
                    for f in sorted(glob.glob(f"{CACHE}/daily/*.parquet"))], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in sorted(glob.glob(f"{CACHE}/adj_factor/*.parquet"))], ignore_index=True)
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    px["aclose"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    px["aopen"] = px["open"].astype(float) * px["adj_factor"].astype(float)
    dates = sorted(px["trade_date"].unique())
    close_p = px.pivot_table(index="trade_date", columns="ts_code", values="aclose")
    open_p = px.pivot_table(index="trade_date", columns="ts_code", values="aopen")

    di = {d: i for i, d in enumerate(dates)}
    month_last: dict[str, str] = {}
    for d in dates:
        month_last[d[:6]] = d
    rebal = [d for d in month_last.values() if di[d] >= MOM_LB and di[d] + 1 + a.fwd < len(dates)]

    FACTORS = ["EP", "BP", "ROE", "GP", "ACC", "MOM"]
    print(f"加载完成:{len(dates)}交易日 → {len(rebal)}个月度换仓日,逐期算 IC(行业+市值双中性)…", flush=True)
    ic_rows: list[dict] = []
    for k, t in enumerate(rebal):
        if k % 10 == 0:
            print(f"  …{k}/{len(rebal)}", flush=True)
        it = di[t]
        codes = [str(c) for c in close_p.columns[close_p.loc[t].notna()]
                 if a.board != "main" or board_of(str(c)) in MAIN]
        if len(codes) < 50:
            continue
        idx = pd.Index(codes)
        price_t = close_p.loc[t, codes]
        # 未来收益:T+1 开盘买入 → 持有窗口内最后成交价(ffill,退市/停牌不丢崩盘收益)
        win = open_p.iloc[it + 1: it + 2 + a.fwd][codes]
        entry = win.iloc[0]
        exit_ = win.ffill().iloc[-1]
        fwd = exit_ / entry - 1.0
        # 市值(size 中性)
        # 缓存版:历史 daily_basic 不可变,落盘后重跑回测零 API 调用、结果可复现
        db = fetch_market_day(pro, "daily_basic", t, CACHE)
        mv = pd.to_numeric(db.set_index("ts_code")["total_mv"], errors="coerce").reindex(idx)
        logmv = np.log(mv.where(mv > 0))
        # 因子原值
        pf, pi, pc, pb = _pit(fina, t), _pit(inc, t), _pit(cf, t), _pit(bs, t)
        rev = pi["revenue"].reindex(idx); cogs = pi["oper_cost"].reindex(idx); ni = pi["n_income_attr_p"].reindex(idx)
        ocf = pc["n_cashflow_act"].reindex(idx); ta = pb["total_assets"].reindex(idx)
        raw = pd.DataFrame(index=idx)
        raw["EP"] = pf["eps"].reindex(idx) / price_t
        raw["BP"] = pf["bps"].reindex(idx) / price_t
        raw["ROE"] = pf["roe"].reindex(idx)
        raw["GP"] = (rev - cogs) / ta
        raw["ACC"] = -((ni - ocf) / ta)
        raw["MOM"] = close_p.iloc[it - MOM_SKIP][codes] / close_p.iloc[it - MOM_LB][codes] - 1.0
        ind = ind_all.reindex(idx).fillna("其他")
        row: dict = {"date": t, "n": len(codes)}
        for fac in FACTORS:
            neu = _neutralize(raw[fac], ind, logmv)
            row["IC_" + fac] = information_coefficient(neu, fwd)
            row["SPR_" + fac] = quantile_spread(neu, fwd, 5)
        ic_rows.append(row)

    res = pd.DataFrame(ic_rows)
    print(f"\n=== 因子 IC 回测·修正版(主板·月度·未来{a.fwd}日·PIT·T+1·行业+市值双中性·survivorship修) ===")
    print(f"N={len(res)} | {res['date'].min()}→{res['date'].max()} | 均{res['n'].mean():.0f}只/期")
    print(f"{'因子':>5}{'IC均值':>8}{'ICIR':>7}{'t值(NW)':>8}{'Neff':>6}{'Q5-Q1月%':>9}{'胜率':>6}  判定")
    for fac in FACTORS:
        ic = res["IC_" + fac].dropna()
        icir, tnw, neff = adjusted_tstat(ic)
        spr = res["SPR_" + fac].dropna().mean() * 100
        hit = (ic > 0).mean() * 100
        m = ic.mean()
        v = "✓有效" if abs(tnw) > 2 and abs(m) > 0.02 else ("~弱" if abs(tnw) > 1.5 else "✗噪声")
        sign = "(反转)" if m < -0.02 else ""
        print(f"{fac:>5}{m:>+8.3f}{icir:>+7.2f}{tnw:>+8.2f}{neff:>6.0f}{spr:>+8.2f}%{hit:>5.0f}%  {v}{sign}")
    os.makedirs("data/holdscore", exist_ok=True)
    res.to_json("data/holdscore/factor_ic_backtest.json", orient="records", force_ascii=False, indent=2)
    print("→ 明细 data/holdscore/factor_ic_backtest.json")


if __name__ == "__main__":
    main()
