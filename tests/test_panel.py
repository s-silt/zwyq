"""Tests for panel assembly: back-adjustment and signal / forward-return.

These pin the two transforms that turn raw per-day pulls into the tidy panel the
gauntlet consumes, with the T+1 no-look-ahead fill baked into the forward return.
"""

import datetime as dt

import pandas as pd
import pytest

from ashare_gauntlet.panel import (
    add_adjusted_prices,
    add_signal_and_forward,
    build_gauntlet_panel,
    mark_entry_locked,
)


def test_add_adjusted_prices_back_adjusts_open_and_close():
    daily = pd.DataFrame(
        {
            "ts_code": ["A", "A"],
            "trade_date": ["20240102", "20240103"],
            "open": [10.0, 11.0],
            "close": [10.5, 11.5],
            "amount": [1000.0, 2000.0],
        }
    )
    adj = pd.DataFrame(
        {
            "ts_code": ["A", "A"],
            "trade_date": ["20240102", "20240103"],
            "adj_factor": [2.0, 2.0],
        }
    )

    out = add_adjusted_prices(daily, adj).set_index("trade_date")

    assert out.loc["20240102", "hfq_close"] == pytest.approx(21.0)
    assert out.loc["20240102", "hfq_open"] == pytest.approx(20.0)
    assert out.loc["20240103", "hfq_close"] == pytest.approx(23.0)


def test_add_signal_and_forward_uses_past_close_and_next_open_window():
    n = 8
    df = pd.DataFrame(
        {
            "ts_code": ["A"] * n,
            "trade_date": [f"2024010{i}" for i in range(1, n + 1)],
            "hfq_close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "hfq_open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        }
    )

    out = add_signal_and_forward(df, k=2, h=2).set_index("trade_date")

    # signal(t) = hfq_close(t)/hfq_close(t-2) - 1; at index 2 (20240103): 12/10 - 1
    assert out.loc["20240103", "signal"] == pytest.approx(12 / 10 - 1)
    # fwd_ret(t) = open(t+1+h)/open(t+1) - 1 with h=2; at index 2: open[5]/open[3] - 1
    assert out.loc["20240103", "fwd_ret"] == pytest.approx(15 / 13 - 1)
    # No look-ahead at the edges: first rows have no past window, last rows no fwd.
    assert pd.isna(out.loc["20240101", "signal"])
    assert pd.isna(out.loc["20240108", "fwd_ret"])


def test_mark_entry_locked_flags_next_day_open_at_limit():
    df = pd.DataFrame(
        {
            "ts_code": ["A"] * 3,
            "trade_date": ["20240101", "20240102", "20240103"],
            "open": [10.0, 11.0, 9.0],
            "up_limit": [11.0, 11.0, 9.9],
            "down_limit": [9.0, 9.0, 8.1],
        }
    )

    out = mark_entry_locked(df).set_index("trade_date")

    # Decision 20240101: next open (11.0) == next up_limit (11.0) -> locked, can't buy.
    assert bool(out.loc["20240101", "entry_locked"]) is True
    # Decision 20240102: next open (9.0) sits inside [8.1, 9.9] -> tradable.
    assert bool(out.loc["20240102", "entry_locked"]) is False
    # Last row has no next day -> not flagged (NaN comparison is False).
    assert bool(out.loc["20240103", "entry_locked"]) is False


def test_build_gauntlet_panel_keeps_only_tradable_decision_rows():
    panel = pd.DataFrame(
        {
            "ts_code": ["GOOD", "NANSIG", "LOCKED", "ILLIQ", "NEW", "OFFDATE"],
            "trade_date": ["20240110", "20240110", "20240110", "20240110", "20240110", "20240111"],
            "signal": [0.05, float("nan"), 0.05, 0.05, 0.05, 0.05],
            "fwd_ret": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "entry_locked": [False, False, True, False, False, False],
            "amount": [1e5, 1e5, 1e5, 1.0, 1e5, 1e5],
        }
    )
    universe = pd.DataFrame(
        {
            "ts_code": ["GOOD", "NANSIG", "LOCKED", "ILLIQ", "NEW", "OFFDATE"],
            "list_date": [dt.date(2000, 1, 1)] * 6,
            "delist_date": [None] * 6,
            "market": ["主板"] * 6,
        }
    )
    universe.loc[universe["ts_code"] == "NEW", "list_date"] = dt.date(2023, 12, 1)  # ~40d -> 次新

    out = build_gauntlet_panel(
        panel,
        universe,
        decision_dates={"20240110"},
        min_amount=1000.0,
        min_list_days=90,
    )

    # Only GOOD survives: NANSIG (no signal), LOCKED (一字板), ILLIQ (amount),
    # NEW (次新 <90d), OFFDATE (not a decision date) all filtered.
    assert set(out["ts_code"]) == {"GOOD"}
    assert list(out.columns) == ["trade_date", "ts_code", "signal", "fwd_ret"]
