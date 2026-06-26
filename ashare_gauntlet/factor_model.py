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


def factor_percentile(s: pd.Series, industry: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """单因子 → 行业中性 → 百分位。higher_is_better=False(如应计利润)取负,使"好"恒映到高分位。"""
    raw = s if higher_is_better else -s
    return percentile_rank(industry_neutralize(raw, industry))


def composite(factor_ranks: pd.DataFrame) -> pd.Series:
    """等权合成各因子百分位。缺某因子用可得因子均值(skipna),**不当 0 填**(0 填=无依据地惩罚)。"""
    return factor_ranks.mean(axis=1, skipna=True)


def to_decile(s: pd.Series) -> pd.Series:
    """合成分 → 十分位 D1..D10(D10=最好)。只输出分桶,避免 1 分粒度伪精度。NaN 保持 NaN。"""
    ranks = s.rank(method="first")  # 先打破并列,保证 qcut 边界唯一
    return pd.qcut(ranks, 10, labels=range(1, 11)).astype("Int64")
