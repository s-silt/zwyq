"""Tests for backtest entry/exit timing.

The single rule these pin is the no-look-ahead / real-T+1 fill: a decision made
at the close of day t is realized by entering at the NEXT bar's open, never the
decision day's own price. Getting this wrong is the timing leak that the crypto
bot's oracle controls exist to catch.
"""

import pandas as pd
import pytest

from ashare_gauntlet.backtest import forward_return_from_next_open


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
