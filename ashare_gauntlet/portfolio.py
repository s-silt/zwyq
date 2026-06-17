"""Portfolio leg construction from quantile buckets and realized returns.

Kept separate from the bucketing (signals) and the timing (backtest) so each
piece stays independently testable: this module only knows how to turn
"bucket assignments + forward returns" into leg/spread returns.
"""

import pandas as pd


def bucket_mean_return(
    forward_returns: pd.Series,
    buckets: pd.Series,
    bucket_id: int,
) -> float:
    """Equal-weighted forward return of the names in ``bucket_id``.

    NaN forward returns (unrealized / untradable names) are skipped, so a data
    hole drops the name from the leg instead of biasing the average.
    """
    members = forward_returns[buckets == bucket_id]
    return float(members.mean())


def long_short_spread(
    forward_returns: pd.Series,
    buckets: pd.Series,
    low: int,
    high: int,
) -> float:
    """Reversal long-short return: long the ``low`` bucket (losers), short the
    ``high`` bucket (winners). Equals ``mean(low) - mean(high)``.
    """
    return (
        bucket_mean_return(forward_returns, buckets, low)
        - bucket_mean_return(forward_returns, buckets, high)
    )
