"""因子 IC 回测(修正版)—— 把因子模型从"文献支撑"升级到"A股本地实证"。

对每个月度换仓日 t,**point-in-time**(只用 ann_date<=t 已披露财报,防未来函数)构造因子横截面,
配 **T+1 开盘买入的未来收益**,算横截面 IC(秩相关)+ 分组多空 spread(可交易性)。

对抗式审计(2026-07-01)后的修正:
1. 【survivorship】退市/停牌股 exit 用持有窗口内**最后成交价(ffill)**,退市前暴跌收益不再被丢 NaN。
2. 【size 中性】拉 daily_basic.total_mv,因子在 **行业+市值** 双中性后再算 IC(排除"BP=小盘代理")。
3. 【t 虚高】用 AR(1) 自相关修正的 N_eff 算 t(adjusted_tstat),不再 iid 高估。
4. 【MOM horizon】用标准 **12-1 动量**(近 250 日、跳过最近 21 日)而非 120 日不跳月。
5. _pit 先按 end_date 再按 ann_date 取最新期(防乱序重述选错期)。

外部 review(2026-07)后的修正:
6. 【EP/BP 口径】改用 daily_basic 的 1/pe_ttm、1/pb(与生产 factor_rank 同口径)。旧版
   eps/bps ÷ **前复权价**(close×adj_factor)是错口径:复权乘数每只股票不同,横截面被
   系统性扭曲,且与生产用的未复权估值口径不一致——回测验证的不是生产在用的因子。
7. 【中性化统一】删本地 _neutralize,改调 factor_model.neutralize_industry_size
   (生产/回测同一实现;缺市值行输出 NaN 不塞哨兵桶)。
8. 【IC 配对】information_coefficient/quantile_spread 按索引 inner join,不再位置配对。

成本模型 + 可成交性过滤(2026-07,五路文献研读后):
9. 【一字涨停剔除】入场日 t+1 开=高=低=收且触及涨停价(stk_limit 规则价)的票买不进,
   剔出当期横截面——"涨停可买"是文献一致点名的前向收益虚高源;剔除数每期打印(surface)。
   一字**跌**停(卖不出,影响退出价)本版未处理,docstring 记录为已知未处理项。
10.【成本后月差】新列 = Q5-Q1 月差 − round_trip_cost_rate(t)(2×佣金+2×滑点+当期印花税,
   PIT 分段见 ashare_gauntlet.costs)。**上界口径**:假设月度全换手(每期整仓一买一卖);
   实际相邻期持仓有重叠、换手更低,真实成本 ≤ 该列扣减——保守而非精确。
   --commission 默认 0.00025(券商常见万2.5,用户合同参数可覆盖);
   --slippage 默认 0.0015(LWZ 2022 JFE 中国市场实测 15bp 取下沿,可覆盖)。
11.【--start】YYYYMMDD,只跑该日及之后的换仓日(默认不截;回填完 2013 数据后跑长样本用)。

诚实:样本仍仅 2022-26 一个 regime、单一市场,IC 是实证读数非跨周期保证。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.factor_backtest \
       [--board main] [--fwd 21] [--commission 0.00025] [--slippage 0.0015] [--start YYYYMMDD]
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

from ashare_gauntlet.backtest import adjusted_tstat, information_coefficient, quantile_spread
from ashare_gauntlet.costs import round_trip_cost_rate
from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry, fetch_market_day
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factor_model import neutralize_industry_size
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


def one_word_limit_up(daily_1d: pd.DataFrame, stk_limit_1d: pd.DataFrame,
                      codes: list[str] | None = None) -> set[str]:
    """入场日"一字涨停"票集合:open==high==low==close 且 high ≥ up_limit − 1e-6。

    可成交性过滤的定义性锚:一字板全天封死、开盘即无卖单,T+1 开盘的"入场价"对买方
    不可得——留在样本里等于假设能按看得见吃不到的价格成交,系统性抬高多头收益
    (五路文献研读一致点名的"涨停可买"虚高;Qlib/RQAlpha 的可交易过滤同款处理)。
    盘中触板但非一字(low<high)当日有成交机会,不剔除;涨停价用 stk_limit 交易所规则价
    (监管常数,非 magic number)。低≤开/收≤高恒成立,同一行情源内四价同值即位级相等,
    故 OHLC 用精确相等;对 up_limit(跨源配对)用 1e-6 浮点容差——远小于最小报价单位
    0.01 元,纯数值容差非可调阈值(与 factor_model.touched_limit_up 同约定)。
    codes 非 None 时只判定该子集(停牌股 daily 无行 → 不在子集匹配内,entry 价天然 NaN
    由收益口径处理,不报错);子集内 daily 有行但 stk_limit 缺涨停价 → fail-loud
    (静默不剔除会把"买不进"的票重新混回样本)。
    一字**跌**停(卖不出,影响退出侧)不在本过滤范围,已知未处理项。
    """
    if daily_1d.empty or stk_limit_1d.empty:
        raise ValueError("one_word_limit_up: daily/stk_limit 输入为空——上游缓存缺日,"
                         "拒绝静默当作全部可成交")
    need_daily = {"ts_code", "open", "high", "low", "close"}
    need_limit = {"ts_code", "up_limit"}
    missing = (need_daily - set(daily_1d.columns)) | (need_limit - set(stk_limit_1d.columns))
    if missing:
        raise ValueError(f"one_word_limit_up: 输入缺列 {sorted(missing)}——拒绝静默当作全部可成交")
    d = daily_1d[["ts_code", "open", "high", "low", "close"]].copy()
    d["ts_code"] = d["ts_code"].astype(str)
    if codes is not None:
        d = d[d["ts_code"].isin(set(codes))]
    if d.empty:
        return set()
    lim = stk_limit_1d[["ts_code", "up_limit"]].copy()
    lim["ts_code"] = lim["ts_code"].astype(str)
    m = d.merge(lim, on="ts_code", how="left")
    miss = m["up_limit"].isna()
    if bool(miss.any()):
        sample = m.loc[miss, "ts_code"].head(5).tolist()
        raise ValueError(
            f"one_word_limit_up: daily 有行但 stk_limit 缺涨停价 {int(miss.sum())} 只(如 {sample})"
            f"——拒绝静默当作可成交,先补齐 stk_limit 缓存")
    o, h, low, c = (m[k].astype(float) for k in ("open", "high", "low", "close"))
    flat = (o == h) & (h == low) & (low == c)
    at_limit = h >= m["up_limit"].astype(float) - 1e-6
    return set(m.loc[flat & at_limit, "ts_code"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main")
    ap.add_argument("--fwd", type=int, default=21)
    # 成本参数是用户合同/实测值而非库常数(ashare_gauntlet.costs 不写默认),默认值出处:
    ap.add_argument("--commission", type=float, default=0.00025,
                    help="单边佣金率;默认 0.00025=券商常见万2.5,按用户合同可覆盖")
    ap.add_argument("--slippage", type=float, default=0.0015,
                    help="单边滑点率;默认 0.0015=LWZ(2022 JFE)中国市场实测 15bp 取下沿,可覆盖")
    ap.add_argument("--start", default=None,
                    help="YYYYMMDD,只跑该日及之后的换仓日(默认不截;回填完 2013 数据后跑长样本用)")
    a = ap.parse_args(argv)
    load_env_local()
    MOM_LB, MOM_SKIP = 250, 21   # 12-1 动量:近250日、跳最近21日

    fina = _load("fina_indicator", ["roe"])
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
    if a.start:
        rebal = [d for d in rebal if d >= a.start]

    FACTORS = ["EP", "BP", "ROE", "GP", "ACC", "MOM"]
    print(f"加载完成:{len(dates)}交易日 → {len(rebal)}个月度换仓日,逐期算 IC(行业+市值双中性)…", flush=True)
    ic_rows: list[dict] = []
    for k, t in enumerate(rebal):
        it = di[t]
        codes = [str(c) for c in close_p.columns[close_p.loc[t].notna()]
                 if a.board != "main" or board_of(str(c)) in MAIN]
        # 可成交性过滤:入场日 t+1 一字涨停买不进,剔出当期横截面(停牌股 entry 无行
        # 天然 NaN,无需另处理);剔除数每期打印 surface,不悄悄改样本
        entry_date = dates[it + 1]
        locked = one_word_limit_up(fetch_market_day(pro, "daily", entry_date, CACHE),
                                   fetch_market_day(pro, "stk_limit", entry_date, CACHE), codes)
        codes = [c for c in codes if c not in locked]
        print(f"  {k + 1}/{len(rebal)} {t} 入场{entry_date} 一字涨停剔除{len(locked)}只", flush=True)
        if len(codes) < 50:
            continue
        idx = pd.Index(codes)
        # 未来收益:T+1 开盘买入 → 持有窗口内最后成交价(ffill,退市/停牌不丢崩盘收益)
        win = open_p.iloc[it + 1: it + 2 + a.fwd][codes]
        entry = win.iloc[0]
        exit_ = win.ffill().iloc[-1]
        fwd = exit_ / entry - 1.0
        # 市值(size 中性)+ 估值(EP/BP 与生产 factor_rank 同口径:1/pe_ttm、1/pb,
        # 仅盈利/正净资产下有定义;旧版 eps/bps÷前复权价的复权乘数每只不同→横截面扭曲)
        # 缓存版:历史 daily_basic 不可变,落盘后重跑回测零 API 调用、结果可复现
        db = fetch_market_day(pro, "daily_basic", t, CACHE).set_index("ts_code")
        mv = pd.to_numeric(db["total_mv"], errors="coerce").reindex(idx)
        logmv = np.log(mv.where(mv > 0))
        pe_t = pd.to_numeric(db["pe_ttm"], errors="coerce").reindex(idx)
        pb_t = pd.to_numeric(db["pb"], errors="coerce").reindex(idx)
        # 因子原值
        pf, pi, pc, pb = _pit(fina, t), _pit(inc, t), _pit(cf, t), _pit(bs, t)
        rev = pi["revenue"].reindex(idx); cogs = pi["oper_cost"].reindex(idx); ni = pi["n_income_attr_p"].reindex(idx)
        ocf = pc["n_cashflow_act"].reindex(idx); ta = pb["total_assets"].reindex(idx)
        raw = pd.DataFrame(index=idx)
        raw["EP"] = (1.0 / pe_t).where(pe_t > 0)
        raw["BP"] = (1.0 / pb_t).where(pb_t > 0)
        raw["ROE"] = pf["roe"].reindex(idx)
        raw["GP"] = (rev - cogs) / ta
        raw["ACC"] = -((ni - ocf) / ta)
        raw["MOM"] = close_p.iloc[it - MOM_SKIP][codes] / close_p.iloc[it - MOM_LB][codes] - 1.0
        ind = ind_all.reindex(idx).fillna("其他")
        row: dict = {"date": t, "n": len(codes), "excl_limit_up": len(locked),
                     # round_trip 成本率(2×佣金+2×滑点+当期印花税,PIT 分段)——
                     # "成本后月差"的上界口径:假设月度全换手(每期整仓一买一卖)
                     "cost_rt": round_trip_cost_rate(t, a.commission, a.slippage)}
        for fac in FACTORS:
            neu = neutralize_industry_size(raw[fac], ind, logmv)
            row["IC_" + fac] = information_coefficient(neu, fwd)
            row["SPR_" + fac] = quantile_spread(neu, fwd, 5)
        ic_rows.append(row)

    res = pd.DataFrame(ic_rows)
    if res.empty:
        raise SystemExit("无有效换仓期(检查 --start 是否截掉全部样本 / 缓存是否覆盖区间)")
    print(f"\n=== 因子 IC 回测·修正版(主板·月度·未来{a.fwd}日·PIT·T+1·行业+市值双中性·survivorship修·一字板剔除) ===")
    print(f"N={len(res)} | {res['date'].min()}→{res['date'].max()} | 均{res['n'].mean():.0f}只/期"
          f" | 一字涨停均剔除{res['excl_limit_up'].mean():.1f}只/期")
    print(f"成本口径:佣金万{a.commission * 10000:g}+滑点{a.slippage * 10000:g}bp(单边)+卖出印花税(PIT分段);"
          f"成本后月差=Q5-Q1−round_trip(上界:假设月度全换手,实际换手更低则成本更低)")
    print(f"{'因子':>5}{'IC均值':>8}{'ICIR':>7}{'t值(NW)':>8}{'Neff':>6}{'Q5-Q1月%':>9}{'成本后月%':>9}{'胜率':>6}  判定")
    for fac in FACTORS:
        ic = res["IC_" + fac].dropna()
        icir, tnw, neff = adjusted_tstat(ic)
        spr = res["SPR_" + fac].dropna().mean() * 100
        # 成本后月差:逐期 SPR−当期 round_trip(印花税分段随期变),再取均值——
        # 比"均值−均成本"更诚实(样本跨税改日时两者不等)
        net = (res["SPR_" + fac] - res["cost_rt"]).dropna().mean() * 100
        hit = (ic > 0).mean() * 100
        m = ic.mean()
        v = "✓有效" if abs(tnw) > 2 and abs(m) > 0.02 else ("~弱" if abs(tnw) > 1.5 else "✗噪声")
        sign = "(反转)" if m < -0.02 else ""
        print(f"{fac:>5}{m:>+8.3f}{icir:>+7.2f}{tnw:>+8.2f}{neff:>6.0f}{spr:>+8.2f}%{net:>+8.2f}%{hit:>5.0f}%  {v}{sign}")
    os.makedirs("data/holdscore", exist_ok=True)
    res.to_json("data/holdscore/factor_ic_backtest.json", orient="records", force_ascii=False, indent=2)
    print("→ 明细 data/holdscore/factor_ic_backtest.json")


if __name__ == "__main__":
    main()
