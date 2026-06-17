"""Tests for the falsification gauntlet metrics and orchestration."""

import math

import numpy as np
import pandas as pd
import pytest

from ashare_gauntlet.gauntlet import (
    annualized_sharpe,
    oos_split,
    per_symbol_contribution,
    run_gauntlet,
)


def _planted_reversal_panel(n_days: int = 40, n_names: int = 10) -> pd.DataFrame:
    # fwd = -beta_d * signal: the lowest-signal names (losers) get the highest
    # forward return, so buying bucket 0 beats the universe. beta_d varies by day
    # so the demeaned excess has dispersion (a finite, large, positive Sharpe).
    rows = []
    for d in range(n_days):
        beta = 1.0 + (d % 4) * 0.25
        for i in range(n_names):
            sig = (i - (n_names - 1) / 2) * 0.01  # symmetric around 0
            rows.append(
                {"trade_date": f"d{d:03d}", "ts_code": f"S{i}", "signal": sig, "fwd_ret": -beta * sig}
            )
    return pd.DataFrame(rows)


def _noise_panel(n_days: int = 40, n_names: int = 10) -> pd.DataFrame:
    # Forward return is a common per-day move with no relation to the signal, so
    # the demeaned selection excess is exactly zero — no edge.
    rows = []
    for d in range(n_days):
        common = 0.001 * (d % 3)
        for i in range(n_names):
            sig = (i - (n_names - 1) / 2) * 0.01
            rows.append(
                {"trade_date": f"d{d:03d}", "ts_code": f"S{i}", "signal": sig, "fwd_ret": common}
            )
    return pd.DataFrame(rows)


def test_run_gauntlet_passes_a_planted_reversal_edge():
    rep = run_gauntlet(_planted_reversal_panel(), n_buckets=2, periods_per_year=50)

    assert rep.long_only_excess_sharpe > 0
    assert rep.verdict == "GO"
    assert rep.reasons == []


def test_run_gauntlet_rejects_a_no_edge_noise_panel():
    rep = run_gauntlet(_noise_panel(), n_buckets=2, periods_per_year=50)

    assert rep.verdict == "NO_GO"
    assert rep.reasons  # non-empty: it must say why


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
