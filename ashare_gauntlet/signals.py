"""Cross-sectional signal operations for the gauntlet.

These are deliberately small, pure functions: the falsification logic should be
expressible as a composition of operations that are each trivially testable.
"""

import pandas as pd


def cross_sectional_demean(section: pd.Series) -> pd.Series:
    """Subtract the cross-sectional mean from a one-day cross-section.

    The mean skips NaN (suspended / missing names) so data holes do not bias the
    surviving names. NaN positions are preserved in the output.
    """
    return section - section.mean()


def assign_quantile_buckets(section: pd.Series, n_buckets: int) -> pd.Series:
    """Split a one-day cross-section into ``n_buckets`` quantile buckets.

    Bucket 0 holds the lowest values (for reversal, the biggest losers = the
    buy/long leg). Ties are broken by rank order before bucketing so that runs
    of identical returns (e.g. many names pinned at the ±10% daily limit) do not
    collapse the quantile edges. NaN inputs stay unbucketed (NaN out): a
    suspended/missing name is not tradable and must not silently land in a leg.
    """
    if int(section.notna().sum()) < n_buckets:
        # Too few tradable names to form n_buckets quantiles -> drop this date.
        return pd.Series(float("nan"), index=section.index, name=section.name)
    ranks = section.rank(method="first")
    buckets = pd.qcut(ranks, n_buckets, labels=False)
    return pd.Series(buckets, index=section.index, name=section.name)
