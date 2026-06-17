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
    assemble_panel,
    build_gauntlet_panel,
    mark_entry_locked,
    assemble_flow_panel,
    north_flow_signal,
    universe_from_daily,
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


def test_assemble_panel_composes_pipeline_and_spaces_decision_dates():
    days = [f"2024010{i}" for i in range(1, 10)] + ["20240110"]  # 10 ordered days
    codes = ["A", "B"]
    daily = pd.DataFrame(
        [
            {
                "ts_code": c,
                "trade_date": d,
                "open": 10.0 + j,
                "close": 10.0 + j,
                "amount": 1e6,
            }
            for c in codes
            for j, d in enumerate(days)
        ]
    )
    adj = pd.DataFrame(
        [{"ts_code": c, "trade_date": d, "adj_factor": 1.0} for c in codes for d in days]
    )
    universe = pd.DataFrame(
        {
            "ts_code": codes,
            "list_date": [dt.date(2000, 1, 1)] * 2,
            "delist_date": [None] * 2,
            "market": ["主板"] * 2,
        }
    )

    out = assemble_panel(
        daily, adj, universe, k=2, h=2, rebalance=3, min_amount=0.0, min_list_days=0
    )

    # Decision dates = every 3rd day (idx 0,3,6,9). idx0 has no past-k signal and
    # idx9 has no forward window, so only 20240104 and 20240107 yield rows.
    assert set(out["trade_date"]) == {"20240104", "20240107"}
    assert set(out["ts_code"]) == {"A", "B"}
    assert list(out.columns) == ["trade_date", "ts_code", "signal", "fwd_ret"]


def test_universe_from_daily_derives_codes_without_stock_basic():
    # When stock_basic is unavailable (token exhausted), the cached daily data is
    # itself the survivorship-free PIT universe: each code appears only on the
    # days it actually traded. We can't recover list_date, so 次新 filtering is
    # disabled (list_date set far in the past) — a documented limitation.
    daily = pd.DataFrame(
        {
            "ts_code": ["A", "B", "A", "C"],
            "trade_date": ["20250102", "20250102", "20250103", "20250103"],
        }
    )

    uni = universe_from_daily(daily)

    assert sorted(uni["ts_code"]) == ["A", "B", "C"]
    assert all(d == dt.date(1990, 1, 1) for d in uni["list_date"])
    assert all(d is None for d in uni["delist_date"])


def test_north_flow_signal_is_negative_change_in_holding_ratio():
    # hk_hold.ratio = 北向持股占流通股比例. 增持 (ratio up) is the smart-money-buy
    # hypothesis, and the gauntlet's buy leg is bucket 0 (lowest signal), so the
    # signal must be the NEGATED change: 增持 -> low signal -> buy leg.
    hk = pd.DataFrame(
        {
            "ts_code": ["A", "A", "A", "B", "B", "B"],
            "trade_date": ["20250102", "20250103", "20250104"] * 2,
            "ratio": [1.0, 1.5, 2.0, 3.0, 2.0, 1.0],  # A 增持, B 减持
        }
    )

    out = north_flow_signal(hk, k=1).set_index(["ts_code", "trade_date"])

    # A 0103: +0.5 增持 -> signal -0.5 (buy leg); B 0103: -1.0 减持 -> signal +1.0
    assert out.loc[("A", "20250103"), "signal"] == pytest.approx(-0.5)
    assert out.loc[("B", "20250103"), "signal"] == pytest.approx(1.0)
    # No prior day -> no signal.
    assert pd.isna(out.loc[("A", "20250102"), "signal"])


def test_assemble_flow_panel_uses_flow_signal_with_price_forward_return():
    days = [f"2024010{i}" for i in range(1, 10)] + ["20240110"]
    codes = ["A", "B"]
    daily = pd.DataFrame(
        [
            {"ts_code": c, "trade_date": d, "open": 10.0 + j, "close": 10.0 + j, "amount": 1e6}
            for c in codes
            for j, d in enumerate(days)
        ]
    )
    adj = pd.DataFrame(
        [{"ts_code": c, "trade_date": d, "adj_factor": 1.0} for c in codes for d in days]
    )
    # A 北向增持 (ratio j+1 rising), B 减持 (falling).
    hk = pd.DataFrame(
        [{"ts_code": "A", "trade_date": d, "ratio": float(j + 1)} for j, d in enumerate(days)]
        + [{"ts_code": "B", "trade_date": d, "ratio": float(10 - j)} for j, d in enumerate(days)]
    )
    universe = universe_from_daily(daily)

    out = assemble_flow_panel(
        daily, adj, hk, universe, k=1, h=2, rebalance=3, min_amount=0.0, min_list_days=0
    ).set_index(["ts_code", "trade_date"])

    assert list(out.columns) == ["signal", "fwd_ret"]
    # Decision date 20240104 (idx3): A 增持 -> signal = -(4-3) = -1 (flow, not reversal).
    assert out.loc[("A", "20240104"), "signal"] == pytest.approx(-1.0)
    assert out.loc[("B", "20240104"), "signal"] == pytest.approx(1.0)
