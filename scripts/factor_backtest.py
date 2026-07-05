"""因子 IC 回测(修正版)—— 把因子模型从"文献支撑"升级到"A股本地实证"。

对每个月度换仓日 t,**point-in-time**(只用 ann_date<=t 已披露财报,防未来函数)构造因子横截面,
配 **T+1 开盘买入的未来收益**,算横截面 IC(秩相关)+ 分组多空 spread(可交易性)。

对抗式审计(2026-07-01)后的修正:
1. 【survivorship】退市/停牌股 exit 用持有窗口内**最后成交价(ffill)**,退市前暴跌收益不再被丢 NaN。
2. 【size 中性】拉 daily_basic.total_mv,因子在 **行业+市值** 双中性后再算 IC(排除"BP=小盘代理")。
3. 【t 虚高】真 Newey-West HAC t(Bartlett 核+NW1994 自动带宽,newey_west_tstat);
   初版用 AR(1) N_eff 近似且错标 "NW",外部 review(2026-07-05)点名后换真 HAC。
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

from ashare_gauntlet.backtest import information_coefficient, newey_west_tstat, quantile_spread
from ashare_gauntlet.costs import round_trip_cost_rate
from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry, fetch_market_day
from ashare_gauntlet.data.partition import assert_adj_complete, date_partition_files
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
    # 个券级涨停价缺行(数据商缺口,实测 001914.SZ 重组上市初期):**保守剔除**——
    # 无法验证可成交的股不进多头样本(definitional:宁可少算不可伪造"可买"),
    # 并入返回集合由调用方 surface 数量;整跑 raise 会让 1 只缺行掐死 149 期回测。
    o, h, low, c = (m[k].astype(float) for k in ("open", "high", "low", "close"))
    flat = (o == h) & (h == low) & (low == c)
    at_limit = h >= m["up_limit"].astype(float) - 1e-6
    return set(m.loc[(flat & at_limit) | miss, "ts_code"])


def one_word_limit_down(daily_1d: pd.DataFrame, stk_limit_1d: pd.DataFrame,
                        codes: list[str] | None = None) -> set[str]:
    """退出日"一字跌停"票集合(卖不出):open==high==low==close 且 low ≤ down_limit + 1e-6。

    退出侧可成交性(吸纳终榜 P0①,对称于入场侧 one_word_limit_up):一字跌停全天封死、
    无买单承接,"按 ffill 价卖出"等于假设崩盘途中能出货——系统性**高估**多头收益,方向
    与入场侧过滤相反,两侧都修才对称。只认一字封死;盘中打开(low<high)有成交机会不算。
    与买侧的一处**方向性不对称**:个券缺跌停价 → 视为**可卖**(daily 行里有真实成交价,
    无法证明封死;买侧伪造"可买"会虚增收益故保守剔除,卖侧伪造"卖不掉"是无据地推迟
    真实成交价)。空输入 fail-loud 同买侧。
    """
    if daily_1d.empty or stk_limit_1d.empty:
        raise ValueError("one_word_limit_down: daily/stk_limit 输入为空——上游缓存缺日,"
                         "拒绝静默当作全部可卖")
    need_daily = {"ts_code", "open", "high", "low", "close"}
    need_limit = {"ts_code", "down_limit"}
    missing = (need_daily - set(daily_1d.columns)) | (need_limit - set(stk_limit_1d.columns))
    if missing:
        raise ValueError(f"one_word_limit_down: 输入缺列 {sorted(missing)}——拒绝静默判定")
    d = daily_1d[["ts_code", "open", "high", "low", "close"]].copy()
    d["ts_code"] = d["ts_code"].astype(str)
    if codes is not None:
        d = d[d["ts_code"].isin(set(codes))]
    if d.empty:
        return set()
    lim = stk_limit_1d[["ts_code", "down_limit"]].copy()
    lim["ts_code"] = lim["ts_code"].astype(str)
    m = d.merge(lim, on="ts_code", how="left")
    o, h, low, c = (m[k].astype(float) for k in ("open", "high", "low", "close"))
    flat = (o == h) & (h == low) & (low == c)
    at_limit = low <= m["down_limit"].astype(float) + 1e-6   # 缺跌停价 → NaN 比较=False=可卖
    return set(m.loc[flat & at_limit, "ts_code"])


def first_sellable_open(opens: pd.Series, start: int, locked) -> "tuple[float, int] | None":
    """从位置 ``start`` 起找第一个可卖日:open 非 NaN(未停牌)且 ``locked(pos)`` 为假
    (非一字跌停封死)。返回 (该日开盘价, 顺延交易日数);数据尽头仍不可卖(退市终局)
    → None,调用方保持 ffill 最后成交价并 surface 计数(不伪造未来价格)。

    ``locked`` 注入依赖便于测试与懒取数:只对 open 非 NaN 的日子调用(停牌日无需查涨跌停)。
    """
    valid = np.flatnonzero(opens.iloc[start:].notna().to_numpy())
    for off in valid:
        pos = start + int(off)
        if not locked(pos):
            return float(opens.iloc[pos]), pos - start
    return None


def leg_turnover(prev: "set[str] | None", cur: set[str]) -> float:
    """单腿组合换手率 τ = 1 − |前后期交集| / |当期|(定义性,无参数)。

    真实成本折扣(吸纳终榜 P0②):现有"成本后月差"按全换手扣一次完整 round_trip 是
    上界;实际只有被替换的 τ 部分付成本(卖旧+买新各半个往返≈τ×round_trip)。
    首期无前期组合 → NaN(建仓是一次性成本,不属月度维持换手);当期空腿 → NaN。
    """
    if prev is None or not prev or not cur:
        return math.nan     # 无前期/前期空腿(重建)/当期空腿:维持换手无定义
    return 1.0 - len(prev & cur) / len(cur)


def quantile_legs(factor: pd.Series, q: int = 5) -> "tuple[set[str], set[str]]":
    """组合形成时点的 (低腿, 高腿) 成员集合——按因子分 q 组的底/顶组。

    与 quantile_spread 同约定(rank method="first" 破并列 + qcut + len<q*5 门槛),
    差别:这里只按**因子**分组不配对收益——组合是在 t 日按当时可得因子形成的,
    配对收益分组是评价口径、不是形成口径,算换手必须用形成口径。
    样本不足 → 空腿(调用方 leg_turnover 得 NaN,不造伪组合)。
    """
    f = factor.dropna()
    if len(f) < q * 5:
        return set(), set()
    bucket = pd.qcut(f.rank(method="first"), q, labels=False)
    return (set(f.index[bucket == 0].astype(str)),
            set(f.index[bucket == q - 1].astype(str)))


# ---------- P1 交易行为族候选因子(先过门禁再谈入分;窗口 月=21/年=252 与 MOM 同惯例) ----------

def max_daily_ret(ret: pd.DataFrame, window: int) -> pd.Series:
    """MAX(Bali-Cakici-Whitelaw 2011 彩票需求):近 window 日最大单日收益。

    行=交易日升序(末行=信号日)、列=股票。面板历史不足 window → 全 NaN 不伪造;
    窗内个别停牌日(NaN)按可得日取最大(经济含义:出现过的最大单日暴涨),全缺自然 NaN。
    """
    if len(ret) < window:
        return pd.Series(math.nan, index=ret.columns, dtype=float)
    return ret.tail(window).max()


def ivol_capm(ret: pd.DataFrame, mkt: pd.Series, window: int) -> pd.Series:
    """IVOL(Ang-Hodrick-Xing-Zhang 2006):市场模型残差的日波动,近 window 日。

    **口径显式标注:CAPM(单因子市场模型)残差,非 FF3/CH-3 残差**(对抗轮 B3:
    不许拿总波动冒充;CAPM 残差是文献承认的最简因子模型口径)。mkt=宇宙等权日收益。
    β 用窗内 OLS;面板历史不足 → 全 NaN。
    """
    if len(ret) < window or len(mkt) < window:
        return pd.Series(math.nan, index=ret.columns, dtype=float)
    w = ret.tail(window).reset_index(drop=True)
    m = mkt.tail(window).reset_index(drop=True).astype(float)
    mc = m - m.mean()
    denom = float((mc ** 2).sum())
    if denom <= 0:
        return pd.Series(math.nan, index=ret.columns, dtype=float)
    beta = w.sub(w.mean()).mul(mc, axis=0).sum() / denom
    resid = w.sub(w.mean()) - pd.DataFrame(np.outer(mc, beta), columns=w.columns)
    return resid.std()


def amihud_illiq(ret: pd.DataFrame, amount: pd.DataFrame, window: int) -> pd.Series:
    """ILLIQ(Amihud 2002):mean(|日收益| / 成交额) 近 window 日。

    月窗是 GKX/LWZ 特征工程惯例(Amihud 原式年窗,月窗响应更快、与本族其它因子同窗)。
    amount≤0/NaN(停牌/无成交)不作分母(inf 污染),skipna 均值;历史不足 → 全 NaN。
    """
    if len(ret) < window:
        return pd.Series(math.nan, index=ret.columns, dtype=float)
    r = ret.tail(window).abs()
    a = amount.tail(window).where(amount.tail(window) > 0)
    return (r / a).mean()


def turn_abnormal(turnover: pd.DataFrame, month: int, year: int) -> pd.Series:
    """TURN 异常换手(LWZ 2022 中国最强特征族):ln(近月均换手 / 之前一年均换手)。

    比率剥掉个股常态换手水平(小盘天生高换手),只留"最近异常活跃"的情绪信号;
    ln 使倍增/减半对称。历史不足 year+month → 全 NaN。
    """
    if len(turnover) < year + month:
        return pd.Series(math.nan, index=turnover.columns, dtype=float)
    recent = turnover.tail(month).mean()
    base = turnover.tail(year + month).head(year).mean()
    return np.log(recent / base.where(base > 0))


def defer_note(n_deferred: int, defer_days: int, n_unresolved: int) -> str:
    """退出顺延的每期 surface 文案。均值只在有顺延时才有定义(n_deferred=0 而
    n_unresolved>0 的期——如全是退市终局——曾除零崩掉 fwd=10 整跑)。"""
    if not n_deferred and not n_unresolved:
        return ""
    avg = f"均{defer_days / n_deferred:.1f}日," if n_deferred else ""
    return f" 退出顺延{n_deferred}只({avg}未解{n_unresolved})"


def exclude_shell(mv: pd.Series) -> list[str]:
    """LSY(Liu-Stambaugh-Yuan 2019 JFE, CH-3)剔壳:剔除当期市值最小 30% 的股票。

    A股壳价值污染:微盘股价格含"被借壳"期权价值,系统性抬高价值因子(高BP)多头腿的
    表观收益;30% 为文献常数(LSY 报告对 25-40% cutoff 稳健)。市值 NaN 无法排除
    壳嫌疑 → 保守剔除。用于 --ex-shell30 稳健性开关:BP 若剔壳后塌掉=壳噪声,存活=真价值。
    """
    v = mv.dropna()
    cut = v.quantile(0.30)
    return [str(i) for i in v[v > cut].index]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main")
    ap.add_argument("--fwd", type=int, default=21)
    ap.add_argument("--ex-shell30", action="store_true", dest="ex_shell30",
                    help="剔除市值最小30%(LSY 2019 壳价值稳健性开关)后重算 IC")
    # 成本参数是用户合同/实测值而非库常数(ashare_gauntlet.costs 不写默认),默认值出处:
    ap.add_argument("--commission", type=float, default=0.00025,
                    help="单边佣金率;默认 0.00025=券商常见万2.5,按用户合同可覆盖")
    ap.add_argument("--slippage", type=float, default=0.0015,
                    help="单边滑点率;默认 0.0015=LWZ(2022 JFE)中国市场实测 15bp 取下沿,可覆盖")
    ap.add_argument("--start", default=None,
                    help="YYYYMMDD,只跑该日及之后的换仓日(默认不截;回填完 2013 数据后跑长样本用)")
    ap.add_argument("--every", type=int, default=1,
                    help="每第 N 个月末取一个换仓日(默认1=月度)。fwd=63 配 --every 3 得"
                         "非重叠季度采样——月度采样 63 日收益强重叠,均值 t 会虚高(对抗轮点名)")
    ap.add_argument("--candidates", action="store_true",
                    help="加评 P1 交易行为族候选:MAX/IVOL/ILLIQ/TURN/NLIMIT(需 daily_basic/"
                         "stk_limit 全史缓存;加载多三张面板,启动慢数分钟)。只评不入分:"
                         "入 composite 须过准入纪律(NW t>3+成本后>0+方向稳定+年度切片)")
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

    # date_partition_files:只认 ^\d{8}\.parquet$(daily/ 实际混入过整段拉取文件,直接
    # glob 会把重复 (trade_date, ts_code) 行读进面板,pivot_table 静默聚合改写价格);
    # assert_adj_complete:adj_factor 缺日 fail-loud,不让 NaN 复权价被 dropna 静默吞
    # (与 factor_rank/pick_track 同口径,外部 review 点名回测侧遗漏)
    da_cols = ["ts_code", "trade_date", "open", "close"] + (
        ["high", "amount"] if a.candidates else [])
    da = pd.concat([pd.read_parquet(f, columns=da_cols)
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

    # P1 候选因子的面板(仅 --candidates;窗口 月=21/年=252 与 MOM 同惯例)
    ret_p = amt_p = to_p = touched_p = mkt = None
    if a.candidates:
        ret_p = close_p.pct_change()                       # 前复权日收益(aclose 面板)
        amt_p = px.pivot_table(index="trade_date", columns="ts_code", values="amount")
        mkt = ret_p.mean(axis=1)                           # 宇宙等权市场收益(CAPM 口径锚)
        print("加载 turnover 面板(daily_basic 全史)…", flush=True)
        to_files = date_partition_files(CACHE, "daily_basic")
        to_p = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "turnover_rate"])
                          for f in to_files], ignore_index=True).pivot_table(
            index="trade_date", columns="ts_code", values="turnover_rate")
        print("构建触涨停面板(stk_limit 全史)…", flush=True)
        high_p = px.pivot_table(index="trade_date", columns="ts_code", values="high")
        touched_rows = {}
        for f in date_partition_files(CACHE, "stk_limit"):
            d_ = os.path.basename(f)[:8]
            if d_ not in high_p.index:
                continue
            ul = pd.read_parquet(f, columns=["ts_code", "up_limit"]).set_index("ts_code")["up_limit"]
            hi = high_p.loc[d_]
            touched_rows[d_] = (hi >= ul.reindex(hi.index).astype(float) - 1e-6)
        touched_p = pd.DataFrame(touched_rows).T.sort_index()  # 行=日(bool;缺涨停价=False 由 NaN 比较)

    di = {d: i for i, d in enumerate(dates)}
    month_last: dict[str, str] = {}
    for d in dates:
        month_last[d[:6]] = d
    rebal = [d for d in month_last.values() if di[d] >= MOM_LB and di[d] + 1 + a.fwd < len(dates)]
    if a.start:
        rebal = [d for d in rebal if d >= a.start]
    rebal = rebal[::max(a.every, 1)]

    FACTORS = ["EP", "BP", "ROE", "GP", "ACC", "MOM"]
    if a.candidates:
        FACTORS += ["MAX", "IVOL", "ILLIQ", "TURN", "NLIMIT"]
    print(f"加载完成:{len(dates)}交易日 → {len(rebal)}个月度换仓日,逐期算 IC(行业+市值双中性)"
          f"{'·剔壳30%' if a.ex_shell30 else ''}…", flush=True)
    ic_rows: list[dict] = []
    corr_sum = None   # 因子横截面 Spearman 相关矩阵逐期累计(冗余审查)
    corr_n = 0
    prev_legs: dict[str, "tuple[set[str], set[str]] | None"] = {}   # P0② 换手追踪
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
        if len(codes) < 50:
            continue
        idx = pd.Index(codes)
        # 未来收益:T+1 开盘买入 → 持有窗口内最后成交价(ffill,退市/停牌不丢崩盘收益)
        win = open_p.iloc[it + 1: it + 2 + a.fwd][codes]
        entry = win.iloc[0]
        exit_ = win.ffill().iloc[-1].copy()
        # 退出侧卖出约束(吸纳终榜 P0①):计划退出日一字跌停封死/停牌 → 顺延到第一个
        # 可卖日开盘价(数据尽头仍不可卖=退市终局 → 保持 ffill 并 surface);
        # 入场"买不进"已剔而退出"卖不出"不修=方向性高估多头,两侧对称才诚实
        exit_pos = it + 1 + a.fwd
        exit_date = dates[exit_pos]
        _ld_cache: dict[int, set[str]] = {}

        def _locked(pos: int, c: str, _pool: list[str] = codes) -> bool:
            if pos not in _ld_cache:
                d_ = dates[pos]
                _ld_cache[pos] = one_word_limit_down(
                    fetch_market_day(pro, "daily", d_, CACHE),
                    fetch_market_day(pro, "stk_limit", d_, CACHE), _pool)
            return c in _ld_cache[pos]

        locked_exit = one_word_limit_down(fetch_market_day(pro, "daily", exit_date, CACHE),
                                          fetch_market_day(pro, "stk_limit", exit_date, CACHE), codes)
        suspended_exit = {c for c in codes if pd.isna(open_p.iloc[exit_pos].get(c))
                          and pd.notna(entry.get(c))}   # 入场成功但退出日无行=停牌中
        n_deferred = n_unresolved = defer_days = 0
        for c in locked_exit | suspended_exit:
            r = first_sellable_open(open_p[c], exit_pos + 1, lambda j, _c=c: _locked(j, _c))
            if r is None:
                n_unresolved += 1            # 退市终局:保持窗口内最后成交价(已含崩盘)
            else:
                exit_[c] = r[0]
                n_deferred += 1
                defer_days += r[1] + 1       # +1:从退出日顺延到可卖日至少隔 1 个交易日
        fwd = exit_ / entry - 1.0
        print(f"  {k + 1}/{len(rebal)} {t} 入场{entry_date} 一字涨停剔除{len(locked)}只"
              f"{defer_note(n_deferred, defer_days, n_unresolved)}", flush=True)
        # 市值(size 中性)+ 估值(EP/BP 与生产 factor_rank 同口径:1/pe_ttm、1/pb,
        # 仅盈利/正净资产下有定义;旧版 eps/bps÷前复权价的复权乘数每只不同→横截面扭曲)
        # 缓存版:历史 daily_basic 不可变,落盘后重跑回测零 API 调用、结果可复现
        db = fetch_market_day(pro, "daily_basic", t, CACHE).set_index("ts_code")
        mv = pd.to_numeric(db["total_mv"], errors="coerce").reindex(idx)
        if a.ex_shell30:
            kept = set(exclude_shell(mv))
            idx = pd.Index([c for c in codes if c in kept])
            fwd = fwd.reindex(idx)
            mv = mv.reindex(idx)
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
        raw["MOM"] = (close_p.iloc[it - MOM_SKIP][list(idx)] / close_p.iloc[it - MOM_LB][list(idx)] - 1.0)
        if a.candidates:
            hist = slice(None, it + 1)
            raw["MAX"] = max_daily_ret(ret_p.iloc[hist], 21).reindex(idx)
            raw["IVOL"] = ivol_capm(ret_p.iloc[hist], mkt.iloc[hist], 21).reindex(idx)
            raw["ILLIQ"] = amihud_illiq(ret_p.iloc[hist], amt_p.iloc[hist], 21).reindex(idx)
            raw["TURN"] = turn_abnormal(to_p.reindex(dates[:it + 1]), 21, 252).reindex(idx)
            # NLIMIT=近月(21日)触涨停次数;窗内任一日缺 stk_limit 面板行 → 该期整列 NaN
            # (静默少数会低估计数、把彩票票伪装成安静票)
            win_days = dates[it - 20: it + 1]
            if all(d_ in touched_p.index for d_ in win_days):
                raw["NLIMIT"] = touched_p.loc[win_days].sum().astype(float).reindex(idx)
            else:
                raw["NLIMIT"] = pd.Series(float("nan"), index=idx)
        ind = ind_all.reindex(idx).fillna("其他")
        row: dict = {"date": t, "n": len(idx), "excl_limit_up": len(locked),
                     "exit_deferred": n_deferred, "exit_unresolved": n_unresolved,
                     "mkt_fwd": float(fwd.mean()),   # 宇宙等权前向收益(tearsheet 市场状态切片用)
                     # round_trip 成本率(2×佣金+2×滑点+印花税,PIT 分段)——上界口径:
                     # 假设月度全换手。印花税按**持有窗口末=卖出日**取段(税是卖出时缴的,
                     # 窗口跨税改日时用信号日会错扣——外部 review 点名)
                     "cost_rt": round_trip_cost_rate(entry_date, a.commission, a.slippage,
                                                     sell_date=dates[it + 1 + a.fwd])}
        # 中性化后的因子值攒成一张表:算 IC 之外顺带累计因子横截面相关矩阵
        # (冗余审查:相关>0.65 触发合并/剔除评估——华泰FFScore 口径)
        neu_df = pd.DataFrame({fac: neutralize_industry_size(raw[fac], ind, logmv) for fac in FACTORS})
        for fac in FACTORS:
            row["IC_" + fac] = information_coefficient(neu_df[fac], fwd)
            row["SPR_" + fac] = quantile_spread(neu_df[fac], fwd, 5)
            # 真实换手折扣(P0②):τ=两腿平均换手,真实成本≈τ×round_trip(全换手上界的替代)
            low, high = quantile_legs(neu_df[fac], 5)
            pl = prev_legs.get(fac)
            row["TO_" + fac] = float(pd.Series([leg_turnover(pl[0] if pl else None, low),
                                                leg_turnover(pl[1] if pl else None, high)]).mean())
            prev_legs[fac] = (low, high)
        c = neu_df.corr(method="spearman")
        # 逐单元格 skipna 累计:某期某因子全缺 → 该期该行/列 NaN,直接相加会把 NaN
        # 传染到全样本平均;按"有值单元格计数"分母平均
        if corr_sum is None:
            corr_sum, corr_cnt = c.fillna(0.0), c.notna().astype(int)
        else:
            corr_sum, corr_cnt = corr_sum + c.fillna(0.0), corr_cnt + c.notna().astype(int)
        corr_n += 1
        ic_rows.append(row)

    res = pd.DataFrame(ic_rows)
    if res.empty:
        raise SystemExit("无有效换仓期(检查 --start 是否截掉全部样本 / 缓存是否覆盖区间)")
    print(f"\n=== 因子 IC 回测·修正版(主板·月度·未来{a.fwd}日·PIT·T+1·行业+市值双中性·survivorship修·一字板剔除) ===")
    print(f"N={len(res)} | {res['date'].min()}→{res['date'].max()} | 均{res['n'].mean():.0f}只/期"
          f" | 一字涨停均剔除{res['excl_limit_up'].mean():.1f}只/期")
    print(f"成本口径:佣金万{a.commission * 10000:g}+滑点{a.slippage * 10000:g}bp(单边)+卖出印花税(PIT分段);"
          f"成本后月差=Q5-Q1−round_trip(上界:假设月度全换手,实际换手更低则成本更低)")
    print(f"{'因子':>5}{'IC均值':>8}{'ICIR':>7}{'t值(NW)':>8}{'lag':>5}{'Q5-Q1月%':>9}{'上界净%':>8}{'换手':>6}{'真实净%':>8}{'胜率':>6}  判定")
    for fac in FACTORS:
        ic = res["IC_" + fac].dropna()
        # 真 Newey-West HAC(Bartlett核,NW1994 自动带宽):旧版 adjusted_tstat 是 AR(1)
        # N_eff 近似却错标 "NW"(外部 review 点名)——多阶自相关下 AR(1) 低估标准误
        icir, tnw, nwlag = newey_west_tstat(ic)
        spr = res["SPR_" + fac].dropna().mean() * 100
        # 成本后月差两口径,逐期扣当期 round_trip(印花税分段随期变)再取均值:
        # 上界净=全换手(每期整仓一买一卖);真实净=τ×rt(P0②,只有被替换部分付成本)
        net = (res["SPR_" + fac] - res["cost_rt"]).dropna().mean() * 100
        to = res["TO_" + fac]
        real_net = (res["SPR_" + fac] - to * res["cost_rt"]).dropna().mean() * 100
        hit = (ic > 0).mean() * 100
        m = ic.mean()
        v = "✓有效" if abs(tnw) > 2 and abs(m) > 0.02 else ("~弱" if abs(tnw) > 1.5 else "✗噪声")
        sign = "(反转)" if m < -0.02 else ""
        print(f"{fac:>5}{m:>+8.3f}{icir:>+7.2f}{tnw:>+8.2f}{nwlag:>5d}{spr:>+8.2f}%{net:>+7.2f}%"
              f"{to.mean():>6.0%}{real_net:>+7.2f}%{hit:>5.0f}%  {v}{sign}")
    os.makedirs("data/holdscore", exist_ok=True)
    if corr_sum is not None and corr_n:
        avg_corr = corr_sum / corr_cnt.replace(0, pd.NA)
        print(f"\n=== 因子横截面 Spearman 相关(逐期平均,N={corr_n};冗余审查:>0.65 触发合并/剔除评估)===")
        print(avg_corr.astype(float).round(2).to_string())
    res.to_json("data/holdscore/factor_ic_backtest.json", orient="records", force_ascii=False, indent=2)
    print("→ 明细 data/holdscore/factor_ic_backtest.json")


if __name__ == "__main__":
    main()
