"""Falsification gauntlet: metrics and (later) the orchestrated GO/NO_GO run.

⚠️ LEGACY(2026-07 标注):本模块是项目早期的通用证伪器,阈值(min_excess_sharpe
等)为未标定 placeholder,仅剩 scripts/first_verdict.py 与 scripts/flow_verdict.py
两个早期一次性脚本引用,**不在现役选股链路中**。现役准入门禁 = scripts/
factor_tearsheet.py 的五门纪律(NW t>3 出处 Harvey-Liu-Zhu 2016 + 真实净>0 +
LOYO 逐年同号 + 涨跌市同号 + 多头腿成本后>0),标定依据见 docs/methodology.md。
保留本模块因其测试仍钉住底层指标函数(annualized_sharpe/oos_split 等)的语义。

The gauntlet consumes a daily strategy-return series (and, for some steps, the
tidy panel) and asks the soilbot questions: does an edge survive out-of-sample
splits, cross-regime sign checks, per-symbol concentration, demeaning, and
realistic costs — or is it an artifact?
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from .backtest import daily_long_only_excess, daily_long_short
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


@dataclass
class GauntletReport:
    """The gauntlet's verdict and the numbers behind it."""

    n_decisions: int
    long_short_sharpe: float
    long_only_excess_sharpe: float
    oos: pd.DataFrame
    top_symbol: object
    top_symbol_share: float
    verdict: str
    reasons: list[str] = field(default_factory=list)


def run_gauntlet(
    panel: pd.DataFrame,
    n_buckets: int = 10,
    low: int = 0,
    high: int | None = None,
    periods_per_year: float = 50.0,
    cut_fractions: Sequence[float] = (0.3, 0.5, 0.7),
    min_excess_sharpe: float = 0.3,
    max_top_share: float = 0.5,
) -> GauntletReport:
    """Run the falsification steps on a tidy panel and return a GO/NO_GO verdict.

    The verdict is driven by the demeaned long-only *excess* return (buying the
    loser bucket vs. just holding the universe) — the apple-to-apple "does
    selection add alpha" question. GO requires all of: positive demeaned-excess
    Sharpe above ``min_excess_sharpe``; that excess staying positive across every
    OOS cut; and no single symbol carrying more than ``max_top_share`` of the
    gross long-short attribution (the single-symbol-disguise guard).

    Thresholds are deliberate placeholders to be recalibrated against real data.
    """
    if high is None:
        high = n_buckets - 1

    long_short = daily_long_short(panel, n_buckets, low, high)
    excess = daily_long_only_excess(panel, n_buckets, low)
    ls_sharpe = annualized_sharpe(long_short, periods_per_year)
    excess_sharpe = annualized_sharpe(excess, periods_per_year)
    oos = oos_split(excess, periods_per_year, cut_fractions)

    contrib = per_symbol_contribution(panel, n_buckets, low, high)
    gross = float(contrib.abs().sum())
    top_symbol = contrib.abs().idxmax() if len(contrib) else None
    top_share = float(contrib.abs().max() / gross) if gross > 0 else math.nan

    reasons: list[str] = []
    if not (excess_sharpe > min_excess_sharpe):
        reasons.append(
            f"demeaned-excess Sharpe {excess_sharpe:.2f} is not > {min_excess_sharpe}"
        )
    oos_sharpes = [float(s) for s in oos["oos_sharpe"]]
    if not all(s > 0 for s in oos_sharpes):
        reasons.append("demeaned-excess OOS Sharpe is not positive across all cuts")
    if not (top_share < max_top_share):
        reasons.append(
            f"top symbol carries {top_share:.0%} of gross attribution (>= {max_top_share:.0%})"
        )

    return GauntletReport(
        n_decisions=int(panel["trade_date"].nunique()),
        long_short_sharpe=ls_sharpe,
        long_only_excess_sharpe=excess_sharpe,
        oos=oos,
        top_symbol=top_symbol,
        top_symbol_share=top_share,
        verdict="GO" if not reasons else "NO_GO",
        reasons=reasons,
    )
