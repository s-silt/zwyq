"""Point-in-time (PIT) universe construction for A-share daily selection.

The membership rule here is the single most leak-critical unit in the whole
gauntlet: using the *current* listing as the universe for *historical* days
silently injects survivorship bias (delisted names never appear; surviving
names appear from the start of history). Every membership decision must be
answerable using only information available as of the decision day.
"""

import datetime as dt


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
