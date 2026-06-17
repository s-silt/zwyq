"""Reusable stock screen — the '维度优' lens as pure, testable filtering.

``board_of`` encodes the user's account constraint (Shanghai main board only —
see the trading-constraints memory). ``screen_candidates`` applies the screen to
a tidy per-stock facts table (technical + valuation) and returns the filtered,
sorted result. All I/O (cache load, interface pulls) lives in
``scripts/screen.py``; this module is pure so it can be unit-tested.
"""

from collections.abc import Sequence
from typing import cast

import pandas as pd

_SH_MAIN = ("600", "601", "603", "605")
_STAR = ("688", "689")
_CHINEXT = ("300", "301")
_SZ_MAIN = ("000", "001", "002", "003")


def board_of(ts_code: str) -> str:
    """Map a ts_code to its board — the level at which trading permission gates.

    沪主板 (basic Shanghai A account) vs 科创板/创业板/深主板/北交所 (each needs a
    separately-opened permission). Used to honour the 'only 沪市A股' constraint.
    """
    code, _, ex = ts_code.partition(".")
    p3 = code[:3]
    if ex == "SH":
        if p3 in _STAR:
            return "科创板"
        return "沪主板" if p3 in _SH_MAIN else "沪其他"
    if ex == "SZ":
        if p3 in _CHINEXT:
            return "创业板"
        return "深主板" if p3 in _SZ_MAIN else "深其他"
    if ex == "BJ":
        return "北交所"
    return "未知"


def screen_candidates(
    df: pd.DataFrame,
    *,
    boards: Sequence[str] | None = None,
    max_price: float | None = None,
    min_pct20: float | None = None,
    max_dist60: float | None = None,
    trends: Sequence[str] | None = None,
    require_profitable: bool = False,
    max_pe: float | None = None,
    max_pb: float | None = None,
    industries: Sequence[str] | None = None,
    sort_by: str | None = "pct20",
    ascending: bool = False,
    top: int | None = None,
) -> pd.DataFrame:
    """Filter a per-stock facts table down to the screen and sort it.

    Every filter is optional (``None`` = skip). Expected columns: ``ts_code`` and
    whichever the active filters reference (``close``/``pct20``/``dist60``/
    ``trend``/``pe_ttm``/``pb``/``industry``). ``industries`` matches by substring
    (tushare's ``industry`` is a short label like '通信设备'). ``require_profitable``
    keeps only rows with a positive ``pe_ttm`` (亏损股 have NaN PE).
    """
    out = df.copy()
    if boards is not None:
        out = out[cast(pd.Series, out["ts_code"].map(board_of)).isin(list(boards))]
    if max_price is not None:
        out = out[out["close"] <= max_price]
    if min_pct20 is not None:
        out = out[out["pct20"] >= min_pct20]
    if max_dist60 is not None:
        out = out[out["dist60"] <= max_dist60]
    if trends is not None:
        out = out[out["trend"].isin(list(trends))]
    if require_profitable:
        pe = cast(pd.Series, pd.to_numeric(out["pe_ttm"], errors="coerce"))
        out = out[pe.notna() & (pe > 0)]
    if max_pe is not None:
        pe = cast(pd.Series, pd.to_numeric(out["pe_ttm"], errors="coerce"))
        out = out[pe.notna() & (pe <= max_pe)]
    if max_pb is not None:
        pb = cast(pd.Series, pd.to_numeric(out["pb"], errors="coerce"))
        out = out[pb.notna() & (pb <= max_pb)]
    if industries is not None:
        ind = cast(pd.DataFrame, out)["industry"].fillna("").astype(str)
        keys = tuple(industries)
        out = out[ind.apply(lambda i: any(k in i for k in keys))]
    result = cast(pd.DataFrame, out)
    if sort_by and sort_by in result.columns:
        result = result.sort_values(by=sort_by, ascending=ascending)
    if top is not None:
        result = result.head(top)
    return result.reset_index(drop=True)
