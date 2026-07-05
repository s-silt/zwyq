"""横截面因子模型(纯函数)—— 零 magic number 的"质量×价值"合成,替掉 compute_holdscore 的手定常数加权和。

为什么这套比 compute_holdscore 更 grounded(见审计 scoring-needs-theory):
- **行业内中位数去均值**(industry_neutralize):去掉行业绝对水平差、只留行业内相对——解决
  "PE 全市场绝对阈值跨行业不可比(银行 PE5 vs 科技 PE40 同尺)"。中位数对肥尾稳健、**无阈值常数**。
- **横截面百分位排名**(percentile_rank):把任意分布映成 [0,1] 的序,对肥尾稳健、无正态假设、
  **无 winsorize 的 1%/99% 任意阈值**,也不存在"PE14.9 得+10 / 15.1 得+5"的硬断点。
- **等权合成**(composite):在有滚动回测 IC/IR 之前,等权是诚实的无信息先验;手填权重才是伪精度。
- **十分位分桶**(to_decile):只输出 D1–D10,输出粒度=信息粒度,**不报 1 分粒度伪精度**。

因子由 scripts 层装配,每个映射公认 anomaly:EP=1/PE(Basu 1977/FF 价值)、BP=1/PB(FF 1992)、
ROE(盈利能力)、毛利/总资产(Novy-Marx 2013)、应计 (净利−经营现金流)/总资产(Sloan 1996,越低越好)。
全函数无任何可调常数(中位数/百分位/等权/十分位都是参数-free 或定义性)。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def percentile_rank(s: pd.Series) -> pd.Series:
    """横截面百分位排名 → [0,1](最大=1)。NaN 不参与排名、保持 NaN。对肥尾稳健、无阈值。"""
    return s.rank(pct=True)


def industry_neutralize(s: pd.Series, industry: pd.Series) -> pd.Series:
    """行业内中位数去均值:减去同行业中位数,去掉行业绝对水平差、只留行业内相对强弱。"""
    med = s.groupby(industry).transform("median")
    return s - med


def neutralize_industry_size(s: pd.Series, industry: pd.Series, logmv: pd.Series) -> pd.Series:
    """行业+市值双中性:行业内中位数去均值 → 市值十分位组内去中位。返回中性化后的**原值**。

    生产(factor_percentile)与回测(scripts/factor_backtest)共享的**唯一实现**——
    此前两处各写一套曾出现口径漂移(rank first vs average、NaN 市值一边塞 -1 桶一边天然 NaN),
    "回测验证的形态"与"生产使用的形态"必须是同一个数学对象。
    - tie 用 rank(method="average"):市值并列取同秩→落同一桶(method="first" 会把并列
      硬拆进单元素组、组内中位=自身、因子被抹零);
    - **logmv 为 NaN 的行输出 NaN**:缺市值=无法做 size 中性化,塞哨兵桶继续评分是
      无依据的伪分——保持 NaN 交给下游(percentile_rank/IC/composite 均已保 NaN 语义);
    - qcut 完全退化(全体市值并列→边界去重后无区间→全 NaN 桶)时归单一组,保住组内因子序;
    - 十分位复用 to_decile 既有粒度约定,非新常数。
    """
    neu = industry_neutralize(s, industry)
    mv = logmv.reindex(neu.index)
    out = pd.Series(float("nan"), index=neu.index, dtype=float)
    valid = mv.notna()
    if not bool(valid.any()):
        return out                       # 全体缺市值:无一行可中性化
    sb = pd.qcut(mv[valid].rank(method="average"), 10, labels=False, duplicates="drop")
    sb = pd.Series(sb, index=mv.index[valid])
    if sb.isna().any():
        sb = sb.fillna(0)                # 完全退化(全并列):唯一组标签,定义性非阈值
    sub = neu[valid]
    out[valid] = sub - sub.groupby(sb).transform("median")
    return out


def factor_percentile(s: pd.Series, industry: pd.Series, higher_is_better: bool = True,
                      logmv: "pd.Series | None" = None) -> pd.Series:
    """单因子 → 行业中性(可选:+市值中性)→ 百分位。higher_is_better=False(如应计)取负。

    logmv 给定时调 neutralize_industry_size 做行业+市值**双中性**(与 factor_backtest
    同一实现):回测(修正版)证明 BP 未去 size 就是小盘/低价代理,生产因子必须与被验证
    的形态一致。缺市值的行输出 NaN(见 neutralize_industry_size),不再 fillna(-1) 混桶。
    """
    raw = s if higher_is_better else -s
    if logmv is None:
        neu = industry_neutralize(raw, industry)
    else:
        neu = neutralize_industry_size(raw, industry, logmv)
    return percentile_rank(neu)


def composite(factor_ranks: pd.DataFrame) -> pd.Series:
    """等权合成各因子百分位。缺某因子用可得因子均值(skipna),**不当 0 填**(0 填=无依据地惩罚)。"""
    return factor_ranks.mean(axis=1, skipna=True)


def momentum_return(adj_close: pd.Series, lookback: int = 120) -> float | None:
    """动量(Carhart MOM):前复权收益率 = 最新 / lookback 个交易日前 − 1(默认 ~6 个月)。

    用长周期(~6mo)而非短期:持续上行的趋势(如洁美 +49%)得高分,一日涨停脉冲对 6 月收益
    贡献甚微——天然区分"趋势 vs 脉冲"。需前复权价(close×adj_factor)消除分红/送转。
    历史不足返回 None(新上市)。是 Jegadeesh-Titman 1993 / Carhart 1997 的实证最稳健因子之一,
    之前被错误排除(洁美 +49% 是收据),现补回。
    """
    s = adj_close.dropna()
    if len(s) < lookback + 1:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - lookback] - 1.0)


def max_daily_ret(ret: pd.DataFrame, window: int) -> pd.Series:
    """MAX(Bali-Cakici-Whitelaw 2011 彩票需求):近 window 日最大单日收益。

    行=交易日升序(末行=信号日)、列=股票。面板历史不足 window → 全 NaN 不伪造;
    窗内个别停牌日(NaN)按可得日取最大,全缺自然 NaN。生产(🎰标签)与回测
    (--candidates 门禁)共用同一实现——被验证的形态=生产使用的形态。
    """
    if len(ret) < window:
        return pd.Series(math.nan, index=ret.columns, dtype=float)
    return ret.tail(window).max()


def ivol_capm(ret: pd.DataFrame, mkt: pd.Series, window: int) -> pd.Series:
    """IVOL(Ang-Hodrick-Xing-Zhang 2006):市场模型残差的日波动,近 window 日。

    **口径显式标注:CAPM(单因子市场模型)残差,非 FF3/CH-3 残差**。mkt=宇宙等权
    日收益。β 用窗内 OLS;面板历史不足 → 全 NaN。P1 门禁全过(N=149:NW t-14.7、
    13折 LOYO 无变号、涨跌市同号、多头腿成本后+0.34%/期)后入 composite(负向),
    生产/回测共用同一实现。
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


def to_decile(s: pd.Series) -> pd.Series:
    """合成分 → 十分位 D1..D10(D10=最好)。只输出分桶,避免 1 分粒度伪精度。NaN 保持 NaN。

    非空样本 < 10(=十分位的定义性桶数,非新常数)时全返回 NA:不足 10 个样本没有
    "十分位"语义,qcut 硬分只会给 5 只票也标出 D1/D10 的伪精度档位。
    """
    q = 10
    if int(s.notna().sum()) < q:
        return pd.Series(pd.NA, index=s.index, dtype="Int64")
    ranks = s.rank(method="first")  # 先打破并列,保证 qcut 边界唯一
    return pd.qcut(ranks, q, labels=range(1, q + 1)).astype("Int64")


def touched_limit_up(daily_5d: pd.DataFrame, stk_limit_5d: pd.DataFrame) -> set[str]:
    """窗口内触及过涨停的票集合:同日 daily.high >= stk_limit.up_limit 即触及。

    ⚡脉冲的定义性锚(替代 "ret5>15%" magic number):涨停价是交易所规则给出的监管常数
    (主板 ±10%、ST ±5% 等,tushare stk_limit 已按规则算好),"触及涨停"直接服务
    "新强名先问是不是涨停顶上来的"铁律(见 memory momentum-screen-limitup)。
    按 (ts_code, trade_date) 同日配对,不跨日错配;窗口内任一日触及即入集合(之后回落也算)。
    浮点比较容差 1e-6 远小于报价最小变动单位 0.01 元——纯数值容差,非可调阈值。
    输入为空/缺列 fail-loud:静默返回空集会让 ⚡ 标注全体消失、看似"这几天没人涨停"。
    """
    if daily_5d.empty or stk_limit_5d.empty:
        raise ValueError("touched_limit_up: daily/stk_limit 输入为空——上游缓存缺日,拒绝静默返回空集")
    need_daily = {"ts_code", "trade_date", "high"}
    need_limit = {"ts_code", "trade_date", "up_limit"}
    missing = (need_daily - set(daily_5d.columns)) | (need_limit - set(stk_limit_5d.columns))
    if missing:
        raise ValueError(f"touched_limit_up: 输入缺列 {sorted(missing)}——拒绝静默当作无人触及")
    # left merge + 缺配对 fail-loud:inner join 会把 stk_limit 缺行的 (ts_code, trade_date)
    # 静默丢掉——该票该日"看似没涨停",⚡ 标注静默消失。样例截断仅为展示,非评分常数。
    m = daily_5d[["ts_code", "trade_date", "high"]].merge(
        stk_limit_5d[["ts_code", "trade_date", "up_limit"]],
        on=["ts_code", "trade_date"], how="left")
    miss = m["up_limit"].isna()
    if bool(miss.any()):
        sample = [f"{r.ts_code}@{r.trade_date}"
                  for r in m.loc[miss, ["ts_code", "trade_date"]].head(5).itertuples()]
        raise ValueError(
            f"touched_limit_up: daily 有行但 stk_limit 缺涨停价 {int(miss.sum())} 行"
            f"(如 {sample})——拒绝静默丢行当作无人触及,先补齐 stk_limit 缓存")
    hit = m["high"].astype(float) >= m["up_limit"].astype(float) - 1e-6
    return set(m.loc[hit, "ts_code"].astype(str))
