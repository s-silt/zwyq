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
