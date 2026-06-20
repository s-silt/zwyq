"""Interface-first fundamentals — turn cached tushare tables into fact-layer
numbers, deterministically (no web-scrape, no hallucination).

The fixed analysis mode's step 3 is now interface-first: pull the structured
tables (income / fina_indicator / share_float / pledge_stat / index_daily) with
``scripts/fundamentals.py`` and read the numbers off here; the web layer only
supplies the narrative the interface lacks (重组进展、公司澄清、业绩说明会口径).

Everything is descriptive — real reported numbers, never invented or predicted.
"""

import datetime as dt
import sys
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


def _yi(value: Any) -> float | None:
    """Convert a 元-denominated value to 亿元."""
    v = _num(value)
    return v / 1e8 if v is not None else None


def _period_row(df: pd.DataFrame | None, end_date: str | None = None) -> "pd.Series | None":
    """The report row for ``end_date`` (or the latest), 合并报表口径(report_type=1
    if the column exists), with one row per period (latest ann wins)."""
    if df is None or df.empty:
        return None
    d = df.copy()
    d["end_date"] = d["end_date"].astype(str)
    if "report_type" in d.columns:
        d = d[d["report_type"].astype(str) == "1"]
    if d.empty:
        return None
    d = cast(pd.DataFrame, d).sort_values("end_date").drop_duplicates("end_date", keep="last")
    if end_date is not None:
        sel = d[d["end_date"] == str(end_date)]
        return sel.iloc[-1] if not sel.empty else None
    return d.iloc[-1]


def latest_quarter(income: pd.DataFrame, fina: pd.DataFrame | None) -> dict[str, Any]:
    """Latest reported quarter's headline facts from cached ``income`` +
    ``fina_indicator``.

    Revenue / 归母净利 come from ``income`` (元 → 亿元); the year-over-year % and
    毛利率 come from ``fina_indicator`` for the same ``end_date``. 合并报表口径;
    missing fina fields stay ``None`` rather than being invented.
    """
    r = _period_row(income)
    if r is None:
        return {}
    end = str(r["end_date"])
    rev = _num(r.get("total_revenue"))
    npr = _num(r.get("n_income_attr_p"))
    out: dict[str, Any] = {
        "end_date": end,
        "revenue_yi": rev / 1e8 if rev is not None else None,
        "net_profit_yi": npr / 1e8 if npr is not None else None,
        "profitable": (npr > 0) if npr is not None else None,
        "revenue_yoy_pct": None,
        "net_profit_yoy_pct": None,
        "gross_margin_pct": None,
    }
    fr = _period_row(fina, end)
    if fr is not None:
        out["revenue_yoy_pct"] = _num(fr.get("or_yoy"))
        out["net_profit_yoy_pct"] = _num(fr.get("netprofit_yoy"))
        # 只取真·百分率 grossprofit_margin;缺失就保持 None。
        # 不回退 gross_margin 列——那是绝对毛利额(元),拿元当 % 是伪造。
        out["gross_margin_pct"] = _num(fr.get("grossprofit_margin"))
    return out


def _latest_annual_net(income: pd.DataFrame) -> float | None:
    """Latest full-year (end_date …1231) 归母净利 (亿元) from ``income``."""
    if income is None or income.empty:
        return None
    d = income.copy()
    d["end_date"] = d["end_date"].astype(str)
    d = cast(pd.DataFrame, d[d["end_date"].str.endswith("1231")])
    r = _period_row(d)
    return _yi(r.get("n_income_attr_p")) if r is not None else None


def receivables_ratio(balancesheet: pd.DataFrame, income: pd.DataFrame) -> float | None:
    """应收账款 / 最近年报归母净利 (%) — 回款风险(>~300% 偏高)。"""
    ar = balance_facts(balancesheet).get("accounts_receiv_yi")
    annual = _latest_annual_net(income)
    if ar is None or annual is None or annual <= 0:
        return None
    return ar / annual * 100


def peg(pe_ttm: Any, net_profit_yoy_pct: Any) -> float | None:
    """PEG = PE_TTM / 净利增速。增速≤0 或 PE 缺失(亏损)时返回 None。"""
    pe = _num(pe_ttm)
    g = _num(net_profit_yoy_pct)
    if pe is None or g is None or g <= 0:
        return None
    return pe / g


def latest_forecast(forecast: pd.DataFrame) -> dict[str, Any]:
    """Newest-period 业绩预告 {end_date, type, p_change_min/max} from ``forecast``."""
    if forecast is None or forecast.empty:
        return {}
    d = forecast.copy()
    d["end_date"] = d["end_date"].astype(str)
    d = cast(pd.DataFrame, d).sort_values("end_date").drop_duplicates("end_date", keep="last")
    r = d.iloc[-1]
    return {
        "end_date": str(r["end_date"]),
        "type": str(r.get("type") or ""),
        "p_change_min": _num(r.get("p_change_min")),
        "p_change_max": _num(r.get("p_change_max")),
    }


def latest_express(express: pd.DataFrame) -> dict[str, Any]:
    """Newest-period 业绩快报 {end_date, revenue_yi, net_profit_yi, yoy_net_profit_pct}."""
    if express is None or express.empty:
        return {}
    d = express.copy()
    d["end_date"] = d["end_date"].astype(str)
    d = cast(pd.DataFrame, d).sort_values("end_date").drop_duplicates("end_date", keep="last")
    r = d.iloc[-1]
    return {
        "end_date": str(r["end_date"]),
        "revenue_yi": _yi(r.get("revenue")),
        "net_profit_yi": _yi(r.get("n_income")),
        "yoy_net_profit_pct": _num(r.get("yoy_net_profit")),
    }


def balance_facts(balancesheet: pd.DataFrame, end_date: str | None = None) -> dict[str, Any]:
    """应收账款 / 商誉 / 货币资金 (亿元) for ``end_date`` (or latest) from ``balancesheet``."""
    r = _period_row(balancesheet, end_date)
    if r is None:
        return {}
    return {
        "end_date": str(r["end_date"]),
        "accounts_receiv_yi": _yi(r.get("accounts_receiv")),
        "goodwill_yi": _yi(r.get("goodwill")),
        "money_cap_yi": _yi(r.get("money_cap")),
    }


def cashflow_facts(cashflow: pd.DataFrame, end_date: str | None = None) -> dict[str, Any]:
    """经营活动现金流量净额 (亿元) for ``end_date`` (or latest) from ``cashflow``."""
    r = _period_row(cashflow, end_date)
    if r is None:
        return {}
    return {"end_date": str(r["end_date"]), "op_cashflow_yi": _yi(r.get("n_cashflow_act"))}


def recent_holder_trades(stk_holdertrade: pd.DataFrame, as_of: str, within_days: int = 365) -> list[dict[str, Any]]:
    """股东增减持 announced in the last ``within_days``, newest first. ``in_de``:
    DE=减持 / IN=增持 — the deterministic 减持 risk flag."""
    if stk_holdertrade is None or stk_holdertrade.empty:
        return []
    end = dt.datetime.strptime(str(as_of), "%Y%m%d").date()
    floor = end - dt.timedelta(days=within_days)
    out: list[dict[str, Any]] = []
    for _, r in stk_holdertrade.iterrows():
        raw = str(r.get("ann_date"))
        try:
            day = dt.datetime.strptime(raw, "%Y%m%d").date()
        except ValueError:
            # 坏日期跳过但留痕(不静默吞):上报 ts_code + 原始值到 stderr。
            print(
                f"recent_holder_trades: skip row ts_code={r.get('ts_code')!r} "
                f"bad ann_date={raw!r}",
                file=sys.stderr,
            )
            continue
        if floor <= day <= end:
            out.append({
                "ann_date": raw,
                "holder_name": str(r.get("holder_name") or ""),
                "direction": "减持" if str(r.get("in_de")) == "DE" else "增持",
                "change_vol": _num(r.get("change_vol")),
                "change_ratio": _num(r.get("change_ratio")),
                "avg_price": _num(r.get("avg_price")),
            })
    return sorted(out, key=lambda x: x["ann_date"], reverse=True)


def st_status(namechange: pd.DataFrame) -> dict[str, Any]:
    """Current name + ST history from ``namechange`` — flags 当前ST 或 曾ST/刚摘帽
    (e.g. 华微电子 2026-05-20 撤销退市风险警示)."""
    if namechange is None or namechange.empty:
        return {}
    df = cast(pd.DataFrame, namechange.copy())
    df["start_date"] = df["start_date"].astype(str)
    df = df.sort_values("start_date")
    names = [str(n) for n in df["name"]]
    cur = df.iloc[-1]
    current_name = str(cur["name"])
    return {
        "current_name": current_name,
        "is_st": "ST" in current_name,
        "ever_st": any("ST" in n for n in names),
        "last_change_date": str(cur["start_date"]),
        "last_change_reason": str(cur.get("change_reason") or ""),
    }


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
            # 坏日期跳过但留痕(不静默吞):上报 ts_code + 原始值到 stderr。
            print(
                f"upcoming_unlocks: skip row ts_code={r.get('ts_code')!r} "
                f"bad float_date={raw!r}",
                file=sys.stderr,
            )
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
