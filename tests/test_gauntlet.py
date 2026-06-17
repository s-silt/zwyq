"""Tests for the falsification gauntlet metrics and orchestration."""

import math

import numpy as np
import pandas as pd
import pytest

from ashare_gauntlet.gauntlet import (
    annualized_sharpe,
    oos_split,
    per_symbol_contribution,
)


def _two_day_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["d1"] * 4 + ["d2"] * 4,
            "ts_code": ["A", "B", "C", "D"] * 2,
            "signal": [-0.05, -0.01, 0.02, 0.06, 0.03, 0.01, -0.02, -0.04],
            "fwd_ret": [0.05, -0.03, 0.01, 0.09, 0.02, 0.00, 0.04, -0.01],
        }
    )


def test_annualized_sharpe_scales_mean_over_std_by_sqrt_periods():
    r = pd.Series([0.01, -0.01, 0.02, 0.00, 0.03, -0.02])
    ppy = 50
    expected = r.mean() / r.std() * math.sqrt(ppy)  # r.std() is sample (ddof=1)

    assert annualized_sharpe(r, periods_per_year=ppy) == pytest.approx(expected)


def test_annualized_sharpe_skips_nan_periods():
    r = pd.Series([0.01, np.nan, -0.01, 0.02, np.nan, 0.00, 0.03, -0.02])
    clean = pd.Series([0.01, -0.01, 0.02, 0.00, 0.03, -0.02])
    ppy = 50

    assert annualized_sharpe(r, periods_per_year=ppy) == pytest.approx(
        clean.mean() / clean.std() * math.sqrt(ppy)
    )


def test_annualized_sharpe_is_nan_without_dispersion_or_data():
    assert pd.isna(annualized_sharpe(pd.Series([0.01, 0.01, 0.01]), periods_per_year=50))
    assert pd.isna(annualized_sharpe(pd.Series([0.01]), periods_per_year=50))
    assert pd.isna(annualized_sharpe(pd.Series([], dtype=float), periods_per_year=50))


def test_oos_split_reports_in_and_oos_sharpe_at_each_cut():
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.04, -0.01, 0.02, 0.00])

    rep = oos_split(r, periods_per_year=50, cut_fractions=(0.3, 0.5)).set_index("cut")

    # cut 0.5 -> first 5 in-sample, last 5 out-of-sample.
    assert rep.loc[0.5, "n_in"] == 5
    assert rep.loc[0.5, "n_oos"] == 5
    assert rep.loc[0.5, "in_sharpe"] == pytest.approx(annualized_sharpe(r.iloc[:5], 50))
    assert rep.loc[0.5, "oos_sharpe"] == pytest.approx(annualized_sharpe(r.iloc[5:], 50))
    # cut 0.3 -> first 3 in-sample, last 7 out-of-sample.
    assert rep.loc[0.3, "n_in"] == 3
    assert rep.loc[0.3, "n_oos"] == 7


def test_per_symbol_contribution_is_additive_to_total_spread():
    contrib = per_symbol_contribution(_two_day_panel(), n_buckets=2, low=0, high=1)

    # Additive attribution: each name in the buy bucket adds +fwd/n_low, each in
    # the sell bucket adds -fwd/n_high, summed across days.
    # A: +0.05/2 (d1 buy) - 0.02/2 (d2 sell) = 0.025 - 0.010 = 0.015
    # B: -0.03/2 (d1 buy) - 0.00/2 (d2 sell) = -0.015 + 0.000 = -0.015
    # C: -0.01/2 (d1 sell) + 0.04/2 (d2 buy) = -0.005 + 0.020 = 0.015
    # D: -0.09/2 (d1 sell) - 0.01/2 (d2 buy) = -0.045 - 0.005 = -0.050
    assert contrib["A"] == pytest.approx(0.015)
    assert contrib["B"] == pytest.approx(-0.015)
    assert contrib["C"] == pytest.approx(0.015)
    assert contrib["D"] == pytest.approx(-0.050)
    # The attribution must sum to the total cumulative long-short spread.
    assert contrib.sum() == pytest.approx(-0.035)
