"""Tests for backtest entry/exit timing.

The single rule these pin is the no-look-ahead / real-T+1 fill: a decision made
at the close of day t is realized by entering at the NEXT bar's open, never the
decision day's own price. Getting this wrong is the timing leak that the crypto
bot's oracle controls exist to catch.
"""

import pandas as pd
import pytest

from ashare_gauntlet.backtest import (
    daily_long_only_excess,
    daily_long_short,
    forward_return_from_next_open,
)


def _two_day_panel() -> pd.DataFrame:
    # Decision dates d1, d2; 4 names each. `signal` = past-k return (low = bigger
    # loser = reversal buy leg); `fwd_ret` = realized t+1-open forward return.
    return pd.DataFrame(
        {
            "trade_date": ["d1"] * 4 + ["d2"] * 4,
            "ts_code": ["A", "B", "C", "D"] * 2,
            "signal": [-0.05, -0.01, 0.02, 0.06, 0.03, 0.01, -0.02, -0.04],
            "fwd_ret": [0.05, -0.03, 0.01, 0.09, 0.02, 0.00, 0.04, -0.01],
        }
    )


def test_daily_long_short_spread_per_decision_date():
    out = daily_long_short(_two_day_panel(), n_buckets=2, low=0, high=1)

    # d1: signal ranks A<B<C<D -> bucket0={A,B}, bucket1={C,D};
    #     spread = mean(0.05,-0.03) - mean(0.01,0.09) = 0.01 - 0.05 = -0.04
    assert out.loc["d1"] == pytest.approx(-0.04)
    # d2: signal ranks D<C<B<A -> bucket0={C,D}, bucket1={A,B};
    #     spread = mean(0.04,-0.01) - mean(0.02,0.00) = 0.015 - 0.01 = 0.005
    assert out.loc["d2"] == pytest.approx(0.005)


def test_daily_long_only_excess_is_buy_leg_minus_universe_mean():
    # Long-only realizable leg, demeaned against the equal-weight universe: this
    # is the "apple-to-apple" question — does selecting the losers beat just
    # holding the whole market that day, stripped of market beta?
    out = daily_long_only_excess(_two_day_panel(), n_buckets=2, low=0)

    # d1: buy {A,B} mean 0.01; universe mean 0.03; excess = -0.02
    assert out.loc["d1"] == pytest.approx(-0.02)
    # d2: buy {C,D} mean 0.015; universe mean 0.0125; excess = 0.0025
    assert out.loc["d2"] == pytest.approx(0.0025)


def test_entry_is_next_open_and_exit_is_holding_days_later():
    # Opens for trade days 0..4.
    opens = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])

    # Decide at close of day 1 -> enter day 2 open (12.0), hold 2 days,
    # exit day 4 open (14.0).
    r = forward_return_from_next_open(opens, decision_idx=1, holding_days=2)

    assert r == pytest.approx(14.0 / 12.0 - 1.0)


def test_no_realized_return_when_window_runs_past_end_of_data():
    # Near the end of history the forward window does not exist yet; it must come
    # back NaN rather than peeking at (nonexistent) future bars.
    opens = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])

    r = forward_return_from_next_open(opens, decision_idx=3, holding_days=2)

    assert pd.isna(r)
