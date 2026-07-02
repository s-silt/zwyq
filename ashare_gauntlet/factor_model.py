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

import pandas as pd


def percentile_rank(s: pd.Series) -> pd.Series:
    """横截面百分位排名 → [0,1](最大=1)。NaN 不参与排名、保持 NaN。对肥尾稳健、无阈值。"""
    return s.rank(pct=True)


def industry_neutralize(s: pd.Series, industry: pd.Series) -> pd.Series:
    """行业内中位数去均值:减去同行业中位数,去掉行业绝对水平差、只留行业内相对强弱。"""
    med = s.groupby(industry).transform("median")
    return s - med


def factor_percentile(s: pd.Series, industry: pd.Series, higher_is_better: bool = True,
                      logmv: "pd.Series | None" = None) -> pd.Series:
    """单因子 → 行业中性(可选:+市值中性)→ 百分位。higher_is_better=False(如应计)取负。

    logmv 给定时做行业+市值**双中性**(市值十分位组内去中位)——与 factor_backtest 的
    _neutralize 同一数学对象:回测(修正版)证明 BP 未去 size 就是小盘/低价代理,
    生产因子必须与被验证的形态一致。十分位复用 to_decile 既有粒度约定,非新常数。
    """
    raw = s if higher_is_better else -s
    neu = industry_neutralize(raw, industry)
    if logmv is not None:
        # rank(average):市值并列取同秩→落同一桶(method="first" 会把并列硬拆进单元素组、抹掉因子);
        # qcut 退化(边界全并列→NaN 桶)时归单一组,保住组内因子序
        sb = pd.qcut(logmv.reindex(neu.index).rank(method="average"), 10, labels=False, duplicates="drop")
        sb = pd.Series(sb, index=neu.index).fillna(-1)
        neu = neu - neu.groupby(sb).transform("median")
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


def to_decile(s: pd.Series) -> pd.Series:
    """合成分 → 十分位 D1..D10(D10=最好)。只输出分桶,避免 1 分粒度伪精度。NaN 保持 NaN。"""
    ranks = s.rank(method="first")  # 先打破并列,保证 qcut 边界唯一
    return pd.qcut(ranks, 10, labels=range(1, 11)).astype("Int64")
