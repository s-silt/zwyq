"""Backtest entry/exit timing.

The functions here encode the fill rule so the rest of the harness cannot
accidentally peek into the future: a signal known at the close of day t can only
be acted on at the next bar's open, and a forward return that would require bars
beyond the end of available data is unrealized (NaN), never fabricated.
"""

import math
from typing import cast

import pandas as pd

from .portfolio import bucket_mean_return, long_short_spread
from .signals import assign_quantile_buckets


def information_coefficient(factor: pd.Series, fwd_return: pd.Series) -> float:
    """横截面 IC = 因子值与未来收益的 Spearman 秩相关(一个换仓日、一个因子)。

    用秩相关(非 Pearson)对肥尾稳健、不受单位影响;成对丢弃 NaN;有效对 <3 返回 NaN
    (样本太小的相关无意义)。IC 均值衡量因子方向有效性,IC 均值/标准差(ICIR)衡量稳定性。

    **按索引 inner join 配对**(非 reset_index 位置配对):因子与收益都以 ts_code 为索引,
    位置配对在两边顺序/成分稍有不同时会把 A 票的因子配到 B 票的收益上——静默错配出假 IC。
    索引不交集的行(只有因子没收益、或反之)丢弃。
    """
    paired = pd.concat({"f": factor, "r": fwd_return}, axis=1, join="inner").dropna()
    if len(paired) < 3:
        return math.nan
    # Spearman = 秩的 Pearson 相关(手算避免 scipy 依赖)
    return float(paired["f"].rank().corr(paired["r"].rank()))


def adjusted_tstat(ic_series: pd.Series) -> "tuple[float, float, float]":
    """IC 序列 → (ICIR, 自相关修正 t值, 有效样本 N_eff)。

    朴素 t=ICIR·√N 假设各期 IC 独立;但 point-in-time 财务季度间被相邻月度换仓复用,
    IC 序列有正自相关 → 会虚高显著性。用 AR(1) 估有效样本 N_eff=N·(1-ρ)/(1+ρ),
    t=ICIR·√N_eff(Newey-West 的简化)。ρ 夹到 [-0.95,0.95] 防退化。
    """
    s = ic_series.dropna()
    n = len(s)
    if n < 4 or s.std() == 0:
        return math.nan, math.nan, float(n)
    icir = float(s.mean() / s.std())
    rho = s.autocorr(1)
    rho = 0.0 if (rho is None or math.isnan(rho)) else max(min(float(rho), 0.95), -0.95)
    neff = max(n * (1.0 - rho) / (1.0 + rho), 1.0)
    return icir, icir * math.sqrt(neff), neff


def newey_west_tstat(ic_series: pd.Series, lag: "int | None" = None) -> "tuple[float, float, int]":
    """IC 序列 → (ICIR, 真 Newey-West HAC t 值, 所用 lag)。

    对 IC 均值做 HAC 方差:S = γ0 + 2·Σ_{k=1..q}(1 − k/(q+1))·γk(Bartlett 核,
    保证 S 半正定),t = mean / √(S/N)。γk 用 1/N 分母(NW 原式)。
    lag 缺省用 NW(1994) 自动带宽 q = ⌊4·(N/100)^{2/9}⌋——文献公式非手拍常数。

    与 :func:`adjusted_tstat`(AR(1) N_eff 近似)的差别:AR(1) 只吸收一阶自相关,
    IC 存在 lag2/lag3 相关(重叠持仓/财报季复用)时会低估标准误、虚高 t;
    此前回测输出列名标 "NW" 实为 AR(1) 近似(外部 review 点名),本函数是名实相符版。
    """
    s = ic_series.dropna().astype(float)
    n = len(s)
    if n < 4 or s.std() == 0:
        return math.nan, math.nan, 0
    icir = float(s.mean() / s.std())
    q = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lag is None else int(lag)
    q = max(0, min(q, n - 2))
    x = (s - s.mean()).to_numpy()
    S = float((x * x).sum() / n)                       # γ0
    for k in range(1, q + 1):
        gk = float((x[k:] * x[:-k]).sum() / n)         # γk,1/N 分母(NW 原式)
        S += 2.0 * (1.0 - k / (q + 1.0)) * gk
    if S <= 0:                                          # Bartlett 核下仅数值零可能触发
        return icir, math.nan, q
    return icir, float(s.mean() / math.sqrt(S / n)), q


def quantile_spread(factor: pd.Series, fwd_return: pd.Series, q: int = 5) -> float:
    """可交易性检验:顶组减底组的未来收益(top-bottom spread),验因子是否单调可交易。

    只看 IC(秩相关)不够——IC 显著但分组不单调/多空 spread 被成本吃光=不可交易。
    按因子分 q 组,返回 最高组均值收益 − 最低组均值收益。样本不足返回 NaN。
    配对口径与 information_coefficient 相同:按索引 inner join,不按位置(防静默错配)。
    """
    df = pd.concat({"f": factor, "r": fwd_return}, axis=1, join="inner").dropna()
    if len(df) < q * 5:
        return math.nan
    bucket = pd.qcut(df["f"].rank(method="first"), q, labels=False)
    return float(df.loc[bucket == q - 1, "r"].mean() - df.loc[bucket == 0, "r"].mean())


def restated_visibility(df: pd.DataFrame) -> pd.DataFrame:
    """重述行可见日修正:ann_date ← f_ann_date(实际发布日)优先。

    tushare income/cashflow/balancesheet 的更正重述行(update_flag='1')保留**原
    公告日** ann_date,真实发布日在 f_ann_date(2026-07 四镜头验证实测:同 ann_date
    不同取值的组 income 444/24万、balancesheet 631/23.7万,重述滞后中位 ~360 天)。
    不修正则 PIT 过滤(ann_date<=asof)会让更正值在原公告日即"可见"=前视。
    f_ann_date 缺失或非 YYYYMMDD 的行保留原 ann_date;无该列的表(fina_indicator
    缓存)原样返回——其重述泄漏(profit_dedt 1,793 组)为已知残余,记 methodology。
    """
    if "f_ann_date" not in df.columns:
        return df
    # 数值规范先行(Codex P2):parquet 可能把日期存 float(20230429.0),直接
    # astype(str) 会被八位正则静默拒绝、修复失效;经 Int64 归一再转字符串
    fa = pd.to_numeric(df["f_ann_date"], errors="coerce").astype("Int64").astype(str)
    ok = fa.str.fullmatch(r"\d{8}", na=False)
    out = df.copy()
    out.loc[ok, "ann_date"] = fa[ok]
    return out.drop(columns=["f_ann_date"])


def point_in_time(hist: pd.DataFrame, asof: str, ann_col: str = "ann_date") -> "pd.Series | None":
    """防未来函数选期:返回截至 ``asof`` **已公告**(ann_date<=asof)的最新一期财务行。

    回测的命门——在换仓日 t 只能用当时已披露的财报(否则偷看未来利润=虚假 IC)。
    截至 asof 无任何已公告财报则返回 None(该股当日不入选)。
    """
    d = hist[hist[ann_col].astype(str) <= str(asof)]
    if d.empty:
        return None
    return d.sort_values(ann_col).iloc[-1]


def forward_return_from_next_open(
    opens: pd.Series,
    decision_idx: int,
    holding_days: int,
) -> float:
    """Forward return for a decision made at the close of ``decision_idx``.

    Entry is the NEXT bar's open (``decision_idx + 1``) — real T+1, and never the
    decision day's own price — and exit is ``holding_days`` opens later. If the
    entry or exit bar lies beyond the available data the return is unrealized and
    returned as NaN.
    """
    entry_idx = decision_idx + 1
    exit_idx = entry_idx + holding_days
    if entry_idx >= len(opens) or exit_idx >= len(opens):
        return math.nan
    entry = opens.iloc[entry_idx]
    exit_price = opens.iloc[exit_idx]
    return float(exit_price / entry - 1.0)


def daily_long_short(
    panel: pd.DataFrame,
    n_buckets: int,
    low: int,
    high: int,
) -> pd.Series:
    """Per-decision-date reversal long-short spread.

    ``panel`` is tidy: columns ``trade_date``, ``ts_code``, ``signal`` (sort key,
    low = bigger loser = buy leg), ``fwd_ret`` (realized T+1 forward return).
    Returns a Series indexed by ``trade_date``.
    """

    def _spread(group: pd.DataFrame) -> float:
        codes = group["ts_code"].to_numpy()
        sig = pd.Series(group["signal"].to_numpy(), index=codes)
        fwd = pd.Series(group["fwd_ret"].to_numpy(), index=codes)
        buckets = assign_quantile_buckets(sig, n_buckets)
        return long_short_spread(fwd, buckets, low, high)

    return cast("pd.Series", panel.groupby("trade_date", sort=True).apply(_spread))


def daily_long_only_excess(
    panel: pd.DataFrame,
    n_buckets: int,
    low: int,
) -> pd.Series:
    """Per-decision-date long-only buy-leg return in excess of the equal-weight
    universe (the cross-sectional demean / apple-to-apple baseline): does buying
    the loser bucket beat just holding the whole tradable universe that day?
    """

    def _excess(group: pd.DataFrame) -> float:
        codes = group["ts_code"].to_numpy()
        sig = pd.Series(group["signal"].to_numpy(), index=codes)
        fwd = pd.Series(group["fwd_ret"].to_numpy(), index=codes)
        buckets = assign_quantile_buckets(sig, n_buckets)
        buy_leg = bucket_mean_return(fwd, buckets, low)
        universe = float(fwd.mean())
        return buy_leg - universe

    return cast("pd.Series", panel.groupby("trade_date", sort=True).apply(_excess))
