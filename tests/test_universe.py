"""Tests for point-in-time (PIT) universe membership.

These pin the survivorship-bias fix: a stock is tradable on day `t` only if it
was already listed and not yet delisted as of `t`. Using the full (current)
listing as the universe for historical days is exactly the leak the soilbot
TSMOM long-scale post-mortem identified as the killer for stock-like universes.
"""

import datetime as dt

from ashare_gauntlet.universe import is_listed_on


def test_delisted_stock_drops_out_on_and_after_delist_date():
    list_date = dt.date(2010, 1, 4)
    delist_date = dt.date(2020, 6, 30)

    # Day before delisting: still in the tradable universe.
    assert is_listed_on(list_date, delist_date, dt.date(2020, 6, 29)) is True

    # Delisting day and after: must drop out (survivorship fix). Trading a name
    # on/after its delisting date would require future knowledge and an
    # untradable fill.
    assert is_listed_on(list_date, delist_date, dt.date(2020, 6, 30)) is False
    assert is_listed_on(list_date, delist_date, dt.date(2021, 1, 4)) is False


def test_stock_not_in_universe_before_its_list_date():
    list_date = dt.date(2015, 5, 11)
    delist_date = None  # still listed today

    # Before IPO: not a member (no look-ahead onto names that don't trade yet).
    assert is_listed_on(list_date, delist_date, dt.date(2015, 5, 8)) is False
    # First trading day onward: member.
    assert is_listed_on(list_date, delist_date, dt.date(2015, 5, 11)) is True
    assert is_listed_on(list_date, delist_date, dt.date(2026, 1, 1)) is True
