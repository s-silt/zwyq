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
    """
    paired = pd.DataFrame({"f": factor.reset_index(drop=True), "r": fwd_return.reset_index(drop=True)}).dropna()
    if len(paired) < 3:
        return math.nan
    # Spearman = 秩的 Pearson 相关(手算避免 scipy 依赖)
    return float(paired["f"].rank().corr(paired["r"].rank()))


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
