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


def add_adjusted_prices(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Merge daily OHLC with adjustment factors and add back-adjusted
    ``hfq_open`` / ``hfq_close`` (raw price x adj_factor)."""
    merged = daily.merge(
        adj[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"],
        how="left",
    )
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
