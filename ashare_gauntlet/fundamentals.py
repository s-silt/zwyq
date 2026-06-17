"""Interface-first fundamentals — turn cached tushare tables into fact-layer
numbers, deterministically (no web-scrape, no hallucination).

The fixed analysis mode's step 3 is now interface-first: pull the structured
tables (income / fina_indicator / share_float / pledge_stat / index_daily) with
``scripts/fundamentals.py`` and read the numbers off here; the web layer only
supplies the narrative the interface lacks (重组进展、公司澄清、业绩说明会口径).

Everything is descriptive — real reported numbers, never invented or predicted.
"""

import datetime as dt
from typing import Any, cast

import pandas as pd


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def latest_quarter(income: pd.DataFrame, fina: pd.DataFrame | None) -> dict[str, Any]:
    """Latest reported quarter's headline facts from cached ``income`` +
    ``fina_indicator``.

    Revenue / 归母净利 come from ``income`` (元 → 亿元); the year-over-year % and
    毛利率 come from ``fina_indicator`` for the same ``end_date``. ``report_type``
    is filtered to '1' (合并报表) so 母公司/调整口径 rows don't shadow the real one.
    Missing fina fields stay ``None`` rather than being invented.
    """
    if income is None or income.empty:
        return {}
    inc = income.copy()
    inc["end_date"] = inc["end_date"].astype(str)
    if "report_type" in inc.columns:
        inc = inc[inc["report_type"].astype(str) == "1"]
    if inc.empty:
        return {}
    inc = cast(pd.DataFrame, inc).sort_values("end_date").drop_duplicates("end_date", keep="last")
    last = inc.iloc[-1]
    end = str(last["end_date"])
    rev = _num(last.get("total_revenue"))
    npr = _num(last.get("n_income_attr_p"))
    out: dict[str, Any] = {
        "end_date": end,
        "revenue_yi": rev / 1e8 if rev is not None else None,
        "net_profit_yi": npr / 1e8 if npr is not None else None,
        "profitable": (npr > 0) if npr is not None else None,
        "revenue_yoy_pct": None,
        "net_profit_yoy_pct": None,
        "gross_margin_pct": None,
    }
    if fina is not None and not fina.empty:
        fr = fina.copy()
        fr["end_date"] = fr["end_date"].astype(str)
        fr = cast(pd.DataFrame, fr[fr["end_date"] == end]).sort_values("end_date")
        if not fr.empty:
            f = fr.iloc[-1]
            out["revenue_yoy_pct"] = _num(f.get("or_yoy"))
            out["net_profit_yoy_pct"] = _num(f.get("netprofit_yoy"))
            out["gross_margin_pct"] = _num(f.get("grossprofit_margin"))
            if out["gross_margin_pct"] is None:
                out["gross_margin_pct"] = _num(f.get("gross_margin"))
    return out


def pledge_ratio(pledge_stat: pd.DataFrame, as_of: str | None = None) -> float | None:
    """Latest 控股股东体系质押比例 (%) at/before ``as_of`` from ``pledge_stat``."""
    if pledge_stat is None or pledge_stat.empty:
        return None
    df = pledge_stat.copy()
    df["end_date"] = df["end_date"].astype(str)
    if as_of is not None:
        df = df[df["end_date"] <= str(as_of)]
    if df.empty:
        return None
    return _num(cast(pd.DataFrame, df).sort_values("end_date").iloc[-1].get("pledge_ratio"))


def upcoming_unlocks(share_float: pd.DataFrame, as_of: str, within_days: int = 180) -> list[dict[str, Any]]:
    """限售解禁 records whose ``float_date`` falls in (as_of, as_of+within_days]."""
    if share_float is None or share_float.empty:
        return []
    start = dt.datetime.strptime(str(as_of), "%Y%m%d").date()
    end = start + dt.timedelta(days=within_days)
    out: list[dict[str, Any]] = []
    for _, r in share_float.iterrows():
        raw = str(r.get("float_date"))
        try:
            day = dt.datetime.strptime(raw, "%Y%m%d").date()
        except ValueError:
            continue
        if start < day <= end:
            out.append({
                "float_date": raw,
                "float_share": _num(r.get("float_share")),
                "float_ratio": _num(r.get("float_ratio")),
                "holder_name": str(r.get("holder_name") or ""),
            })
    return sorted(out, key=lambda x: x["float_date"])


def index_changes(index_daily: pd.DataFrame, as_of: str) -> dict[str, dict[str, float | None]]:
    """``{ts_code: {close, pct_chg}}`` for ``as_of`` from a cached ``index_daily`` frame."""
    if index_daily is None or index_daily.empty:
        return {}
    df = index_daily.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df[df["trade_date"] == str(as_of)]
    return {
        str(r["ts_code"]): {"close": _num(r.get("close")), "pct_chg": _num(r.get("pct_chg"))}
        for _, r in df.iterrows()
    }
