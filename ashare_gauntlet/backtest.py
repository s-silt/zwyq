"""Backtest entry/exit timing.

The functions here encode the fill rule so the rest of the harness cannot
accidentally peek into the future: a signal known at the close of day t can only
be acted on at the next bar's open, and a forward return that would require bars
beyond the end of available data is unrealized (NaN), never fabricated.
"""

import math

import pandas as pd


def forward_return_from_next_open(
    opens: pd.Series,
    decision_idx: int,
    holding_days: int,
) -> float:
    """Forward return for a decision made at the close of ``decision_idx``.

    Entry is the NEXT bar's open (``decision_idx + 1``) — real T+1, and never the
    decision day's own price — and exit is ``holding_days`` opens later. If the
    entry or exit bar lies beyond the available data the return is unrealized and
    returned as NaN.
    """
    entry_idx = decision_idx + 1
    exit_idx = entry_idx + holding_days
    if entry_idx >= len(opens) or exit_idx >= len(opens):
        return math.nan
    entry = opens.iloc[entry_idx]
    exit_price = opens.iloc[exit_idx]
    return float(exit_price / entry - 1.0)
