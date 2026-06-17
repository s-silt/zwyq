"""Period-correct A-share transaction cost model.

Costs must be charged at the rate in force on the trade date, not a single
fixed rate, because the stamp-tax regime changed mid-sample.
"""

import datetime as dt

# 印花税减半生效日: 2023-08-28, 由 0.1% 降到 0.05% (卖方单边).
STAMP_TAX_CUT_DATE = dt.date(2023, 8, 28)
STAMP_TAX_RATE_BEFORE = 0.001
STAMP_TAX_RATE_AFTER = 0.0005


def stamp_tax_rate(trade_date: dt.date) -> float:
    """Sell-side stamp tax rate in force on ``trade_date``."""
    if trade_date >= STAMP_TAX_CUT_DATE:
        return STAMP_TAX_RATE_AFTER
    return STAMP_TAX_RATE_BEFORE
