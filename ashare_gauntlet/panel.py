"""Assemble the tidy gauntlet panel from raw per-day market pulls.

Back-adjust prices, then derive the reversal signal (past-k back-adjusted close
return) and the realized forward return (enter next open, exit h opens later).
The forward window is built with group-wise shifts so a name's first/last rows
correctly have no signal / no forward return rather than borrowing another
name's data or peeking past the end of history.
"""

import datetime as dt
from collections.abc import Collection
from typing import cast

import pandas as pd

from ashare_gauntlet.data.partition import assert_adj_complete


def universe_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Derive a universe table from cached daily data when ``stock_basic`` is
    unavailable (e.g. token exhausted).

    The daily data is already the survivorship-free PIT universe — a code only
    has bars on days it traded. ``list_date`` is unknown here, so it is set far
    in the past, which makes the 次新 (min_list_days) filter a no-op: a
    documented limitation of running purely off cached prices.
    """
    codes = sorted(daily["ts_code"].unique())
    return pd.DataFrame(
        {
            "ts_code": codes,
            "list_date": [dt.date(1990, 1, 1)] * len(codes),
            "delist_date": [None] * len(codes),
            "market": [""] * len(codes),
        }
    )


def north_flow_signal(hk_hold: pd.DataFrame, k: int) -> pd.DataFrame:
    """Northbound-flow signal from per-stock 北向持股 ratio.

    ``signal = -(ratio(t) - ratio(t-k))`` per symbol: 增持 (ratio rising) is the
    smart-money-buy hypothesis and the gauntlet's buy leg is bucket 0 (lowest
    signal), so the change is negated to put 增持 names in bucket 0. Returns
    columns ``ts_code``, ``trade_date``, ``signal``.
    """
    # hk_hold mixes 北向 (A-shares on SSE/SZSE, .SH/.SZ) and 南向 (港股通, .HK).
    # The northbound signal is A-shares only; ratio comes back as object/strings.
    out = cast(pd.DataFrame, hk_hold[hk_hold["ts_code"].str.endswith((".SH", ".SZ"))].copy())
    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce")
    out = out.sort_values(["ts_code", "trade_date"])
    delta = out.groupby("ts_code", group_keys=False)["ratio"].transform(
        lambda s: s - s.shift(k)
    )
    out["signal"] = -delta
    return cast(pd.DataFrame, out[["ts_code", "trade_date", "signal"]].reset_index(drop=True))


def add_adjusted_prices(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Merge daily OHLC with adjustment factors and add back-adjusted
    ``hfq_open`` / ``hfq_close`` (raw price x adj_factor).

    Fail-loud when any traded row lacks ``adj_factor`` (与 factor_rank 同一断言,
    见 data.partition.assert_adj_complete):缺行会让 hfq 价 NaN,被下游
    signal/fwd_ret 的 notna 过滤静默吞掉 —— 信号与前向收益悄悄错位而非报错。
    """
    merged = daily.merge(
        adj[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"],
        how="left",
    )
    assert_adj_complete(merged)
    merged["hfq_close"] = merged["close"] * merged["adj_factor"]
    merged["hfq_open"] = merged["open"] * merged["adj_factor"]
    return merged


def add_signal_and_forward(panel: pd.DataFrame, k: int, h: int) -> pd.DataFrame:
    """Add ``signal`` (past-k back-adjusted close return) and ``fwd_ret``
    (enter at the next open, exit h opens later) per symbol.

    ``signal(t)  = hfq_close(t) / hfq_close(t-k) - 1``
    ``fwd_ret(t) = hfq_open(t+1+h) / hfq_open(t+1) - 1``  (real T+1 entry)
    """
    out = panel.sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code", group_keys=False)
    out["signal"] = grouped["hfq_close"].transform(lambda s: s / s.shift(k) - 1.0)
    entry = grouped["hfq_open"].transform(lambda s: s.shift(-1))
    exit_open = grouped["hfq_open"].transform(lambda s: s.shift(-(1 + h)))
    out["fwd_ret"] = exit_open / entry - 1.0
    return out


def mark_entry_locked(panel: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose *next* day opens at a price limit (一字板).

    For a decision at t, entry is the next open; if that open is at the up- or
    down-limit the position cannot actually be established/filled, so the
    realized forward return is fictional and the row must be dropped.
    """
    out = panel.sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code", group_keys=False)
    next_open = grouped["open"].transform(lambda s: s.shift(-1))
    next_up = grouped["up_limit"].transform(lambda s: s.shift(-1))
    next_down = grouped["down_limit"].transform(lambda s: s.shift(-1))
    out["entry_locked"] = (next_open >= next_up) | (next_open <= next_down)
    return out


def _days_since_list(trade_date: str, list_date: dt.date) -> int:
    decided = dt.datetime.strptime(str(trade_date), "%Y%m%d").date()
    return (decided - list_date).days


def build_gauntlet_panel(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    decision_dates: Collection[str],
    min_amount: float,
    min_list_days: int,
) -> pd.DataFrame:
    """Reduce the enriched price panel to the tidy, tradable decision rows.

    Keeps a row only if: its date is a (non-overlapping) decision date; the
    signal and forward return both exist; the next-day entry is not limit-locked;
    turnover ``amount`` clears the liquidity floor; and the name is past its
    新股/次新 window (``min_list_days`` calendar days since listing).
    """
    list_map = dict(zip(universe["ts_code"], universe["list_date"]))

    mask = (
        panel["trade_date"].isin(list(decision_dates))
        & panel["signal"].notna()
        & panel["fwd_ret"].notna()
        & (~panel["entry_locked"].astype(bool))
        & (panel["amount"] >= min_amount)
    )
    rows = panel[mask].copy()

    seasoned = [
        (list_map.get(tc) is not None and _days_since_list(td, list_map[tc]) >= min_list_days)
        for tc, td in zip(rows["ts_code"], rows["trade_date"])
    ]
    rows = cast(pd.DataFrame, rows[pd.Series(seasoned, index=rows.index)])
    tidy = rows[["trade_date", "ts_code", "signal", "fwd_ret"]].reset_index(drop=True)
    return cast(pd.DataFrame, tidy)


def assemble_panel(
    daily: pd.DataFrame,
    adj: pd.DataFrame,
    universe: pd.DataFrame,
    k: int,
    h: int,
    rebalance: int,
    min_amount: float,
    min_list_days: int,
) -> pd.DataFrame:
    """Full pipeline: raw daily + adj -> tidy gauntlet panel.

    Back-adjusts, derives signal/forward returns, spaces decision dates every
    ``rebalance``-th trading day (non-overlapping holds), and applies the
    tradability filters. v1 has no ``stk_limit`` feed, so the 一字板 entry filter
    is a no-op (``entry_locked = False``) — a documented refinement, not silent.
    """
    priced = add_adjusted_prices(daily, adj)
    enriched = add_signal_and_forward(priced, k, h)
    enriched["entry_locked"] = False  # v1: no stk_limit; 一字板 filter pending

    all_dates = sorted(enriched["trade_date"].unique())
    decision_dates = set(all_dates[::rebalance])
    return build_gauntlet_panel(enriched, universe, decision_dates, min_amount, min_list_days)


def assemble_flow_panel(
    daily: pd.DataFrame,
    adj: pd.DataFrame,
    hk_hold: pd.DataFrame,
    universe: pd.DataFrame,
    k: int,
    h: int,
    rebalance: int,
    min_amount: float,
    min_list_days: int,
) -> pd.DataFrame:
    """Northbound-flow panel: ``signal`` is the 北向 flow signal, ``fwd_ret`` is
    the same T+1 price forward return used by the reversal gauntlet.

    The inner join on (ts_code, trade_date) naturally restricts the universe to
    北向-eligible names (only those have hk_hold rows) — the correct universe for
    a northbound-flow signal. Runs through the same ``run_gauntlet``.
    """
    priced = add_adjusted_prices(daily, adj)
    priced = add_signal_and_forward(priced, k=1, h=h)  # only fwd_ret is used here
    priced = priced.drop(columns=["signal"])
    flow = north_flow_signal(hk_hold, k)

    merged = priced.merge(flow, on=["ts_code", "trade_date"], how="inner")
    merged["entry_locked"] = False
    all_dates = sorted(merged["trade_date"].unique())
    decision_dates = set(all_dates[::rebalance])
    return build_gauntlet_panel(merged, universe, decision_dates, min_amount, min_list_days)
