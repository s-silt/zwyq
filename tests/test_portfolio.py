"""Tests for portfolio leg construction.

Turns a day's quantile buckets + realized forward returns into leg returns:
the equal-weighted return of a bucket, and the reversal long-short spread
(long the losers = bucket 0, short the winners = top bucket). Names with no
realized forward return (NaN) drop out of the leg rather than poisoning it.
"""

import pandas as pd
import pytest

from ashare_gauntlet.portfolio import bucket_mean_return, long_short_spread


def test_bucket_mean_return_equal_weights_names_in_bucket():
    fwd = pd.Series({"A": 0.05, "B": -0.03, "C": 0.01, "D": 0.09})
    buckets = pd.Series({"A": 0, "B": 0, "C": 1, "D": 1})

    # Equal-weight mean within each bucket.
    assert bucket_mean_return(fwd, buckets, 0) == pytest.approx(0.01)
    assert bucket_mean_return(fwd, buckets, 1) == pytest.approx(0.05)


def test_bucket_mean_return_skips_nan_forward_returns():
    # A name whose forward window ran past the end of data (NaN) is not tradable
    # and must drop out, not drag the leg average.
    fwd = pd.Series({"A": 0.04, "B": float("nan"), "C": 0.02})
    buckets = pd.Series({"A": 0, "B": 0, "C": 0})

    assert bucket_mean_return(fwd, buckets, 0) == pytest.approx(0.03)


def test_long_short_spread_is_low_minus_high_bucket():
    # Reversal: long the losers (bucket 0), short the winners (bucket 1).
    fwd = pd.Series({"A": 0.05, "B": -0.03, "C": 0.01, "D": 0.09})
    buckets = pd.Series({"A": 0, "B": 0, "C": 1, "D": 1})

    # long bucket0 (0.01) - short bucket1 (0.05) = -0.04
    assert long_short_spread(fwd, buckets, low=0, high=1) == pytest.approx(-0.04)
