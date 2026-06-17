"""Falsification gauntlet: metrics and (later) the orchestrated GO/NO_GO run.

The gauntlet consumes a daily strategy-return series (and, for some steps, the
tidy panel) and asks the soilbot questions: does an edge survive out-of-sample
splits, cross-regime sign checks, per-symbol concentration, demeaning, and
realistic costs — or is it an artifact?
"""

import math
from collections.abc import Sequence

import pandas as pd

from .signals import assign_quantile_buckets


def annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    """Annualized Sharpe of a return series (NaN periods skipped).

    Returns NaN when there is no dispersion or fewer than two observations, so a
    degenerate series never masquerades as an infinite-Sharpe edge.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return math.nan
    sd = float(clean.std())  # sample std, ddof=1
    if sd == 0.0:
        return math.nan
    return float(clean.mean() / sd * math.sqrt(periods_per_year))


def oos_split(
    returns: pd.Series,
    periods_per_year: float,
    cut_fractions: Sequence[float] = (0.3, 0.5, 0.7),
) -> pd.DataFrame:
    """In-sample vs out-of-sample Sharpe at each time cut.

    The series is ordered by decision date; for each cut fraction the earlier
    part is in-sample and the later part is OOS. An edge that is real should keep
    its sign across cuts; sign flips or OOS collapse are the soilbot tell that the
    in-sample result was overfit or look-back.
    """
    clean = returns.dropna()
    n = len(clean)
    rows = []
    for cut in cut_fractions:
        k = int(n * cut)
        in_sample = clean.iloc[:k]
        out_sample = clean.iloc[k:]
        rows.append(
            {
                "cut": cut,
                "n_in": len(in_sample),
                "n_oos": len(out_sample),
                "in_sharpe": annualized_sharpe(in_sample, periods_per_year),
                "oos_sharpe": annualized_sharpe(out_sample, periods_per_year),
            }
        )
    return pd.DataFrame(rows)


def per_symbol_contribution(
    panel: pd.DataFrame,
    n_buckets: int,
    low: int,
    high: int,
) -> pd.Series:
    """Each symbol's additive contribution to the cumulative long-short spread.

    A name in the buy bucket on a day adds ``+fwd_ret / n_low``; a name in the
    sell bucket adds ``-fwd_ret / n_high``. Summed over days, the contributions
    add up exactly to the total cumulative spread, so a single name (or a few)
    secretly carrying the whole edge shows up as a dominant share — the
    single-event / single-symbol disguise the soilbot post-mortems kept finding.
    """
    contrib: dict[object, float] = {}
    for _date, group in panel.groupby("trade_date"):
        codes = group["ts_code"].to_numpy()
        sig = pd.Series(group["signal"].to_numpy(), index=codes)
        bucket_of = dict(zip(codes, assign_quantile_buckets(sig, n_buckets).to_numpy()))
        fwd_of = dict(zip(codes, group["fwd_ret"].to_numpy()))
        low_codes = [c for c in codes if bucket_of[c] == low]
        high_codes = [c for c in codes if bucket_of[c] == high]
        n_low = len(low_codes)
        n_high = len(high_codes)
        for code in low_codes:
            contrib[code] = contrib.get(code, 0.0) + float(fwd_of[code]) / n_low
        for code in high_codes:
            contrib[code] = contrib.get(code, 0.0) - float(fwd_of[code]) / n_high
    return pd.Series(contrib)
