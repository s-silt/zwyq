"""Point-in-time (PIT) universe construction for A-share daily selection.

The membership rule here is the single most leak-critical unit in the whole
gauntlet: using the *current* listing as the universe for *historical* days
silently injects survivorship bias (delisted names never appear; surviving
names appear from the start of history). Every membership decision must be
answerable using only information available as of the decision day.
"""

import datetime as dt

import pandas as pd


def _parse_tushare_date(value: object) -> dt.date | None:
    """Parse a Tushare 'YYYYMMDD' date; empty/None means not set (still listed)."""
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    return dt.datetime.strptime(str(value), "%Y%m%d").date()


def build_universe(stock_basic_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw ``stock_basic`` pull (list_status L + D) into a
    survivorship-free universe table with parsed dates.

    Columns out: ``ts_code``, ``list_date`` (date), ``delist_date`` (date | None,
    None = still listed), ``market``.
    """
    return pd.DataFrame(
        {
            "ts_code": stock_basic_raw["ts_code"].to_numpy(),
            "list_date": [_parse_tushare_date(v) for v in stock_basic_raw["list_date"]],
            "delist_date": [
                _parse_tushare_date(v) for v in stock_basic_raw["delist_date"]
            ],
            "market": stock_basic_raw["market"].to_numpy(),
        }
    )


def is_listed_on(
    list_date: dt.date,
    delist_date: dt.date | None,
    day: dt.date,
) -> bool:
    """Whether a stock is in the tradable universe on ``day``.

    A name is a member iff it was already listed and not yet delisted as of
    ``day``. Membership ends strictly before ``delist_date``: we never trade a
    stock on or after its delisting date.
    """
    if day < list_date:
        return False
    if delist_date is not None and day >= delist_date:
        return False
    return True
