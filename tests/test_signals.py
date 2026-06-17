"""Tests for cross-sectional signal operations.

`cross_sectional_demean` is the "beta 照妖镜" carried over from the crypto bot:
subtracting the cross-sectional mean strips the market-wide (beta) component, so
what remains is the part of a name's return that is genuinely *relative* to its
peers. A signal whose edge vanishes after demeaning was only ever riding the
market, not selecting stocks.
"""

import pandas as pd
import pytest

from ashare_gauntlet.signals import assign_quantile_buckets, cross_sectional_demean


def test_demean_subtracts_cross_sectional_mean_and_sums_to_zero():
    # Cross-section of one day's returns; mean = 0.01.
    section = pd.Series({"600000.SH": 0.03, "000001.SZ": 0.01, "300750.SZ": -0.01})

    out = cross_sectional_demean(section)

    assert out["600000.SH"] == pytest.approx(0.02)
    assert out["000001.SZ"] == pytest.approx(0.0)
    assert out["300750.SZ"] == pytest.approx(-0.02)
    # A demeaned cross-section sums to zero by construction (pure market-neutral).
    assert out.sum() == pytest.approx(0.0)


def test_demean_ignores_nans_in_the_mean():
    # Suspended / missing names (NaN) must not poison the cross-sectional mean,
    # otherwise a few data holes would bias every other name's demeaned value.
    section = pd.Series({"A": 0.04, "B": 0.02, "C": float("nan")})  # mean of {0.04,0.02}=0.03

    out = cross_sectional_demean(section)

    assert out["A"] == pytest.approx(0.01)
    assert out["B"] == pytest.approx(-0.01)
    assert pd.isna(out["C"])


def test_quantile_buckets_put_lowest_returns_in_bucket_zero():
    # Past-k returns; reversal expects bucket 0 (biggest losers) to be the
    # long/buy leg. Bucket index must increase with the value so callers can
    # rely on "0 = lowest".
    section = pd.Series({"A": -0.05, "B": -0.01, "C": 0.02, "D": 0.06})

    buckets = assign_quantile_buckets(section, n_buckets=2)

    assert buckets["A"] == 0
    assert buckets["B"] == 0
    assert buckets["C"] == 1
    assert buckets["D"] == 1


def test_quantile_buckets_leave_nan_unbucketed():
    # Suspended / missing names must not be assigned a bucket (they are not
    # tradable that day and must not silently land in the long or short leg).
    section = pd.Series({"A": -0.05, "B": 0.01, "C": float("nan"), "D": 0.06})

    buckets = assign_quantile_buckets(section, n_buckets=2)

    assert pd.isna(buckets["C"])
    assert buckets["A"] == 0
