"""Tests for the period-correct A-share transaction cost model.

Stamp tax (印花税) is the cost component that changed regime mid-sample: it was
0.1% sell-side until 2023-08-28, when it was halved to 0.05%. A high-turnover
reversal strategy's net edge is sensitive to this, so a fixed rate would be
self-deception — exactly the "backtest fidelity" discipline carried over from
the crypto bot.
"""

import datetime as dt

from ashare_gauntlet.costs import stamp_tax_rate


def test_stamp_tax_is_halved_on_2023_08_28():
    # 0.1% before the cut.
    assert stamp_tax_rate(dt.date(2023, 8, 27)) == 0.001
    # 0.05% on the cut date and after.
    assert stamp_tax_rate(dt.date(2023, 8, 28)) == 0.0005
    assert stamp_tax_rate(dt.date(2025, 1, 1)) == 0.0005
