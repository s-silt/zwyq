"""Tests for the interface-first fundamentals extractors (pure, no I/O).

These turn cached tushare tables (income / fina_indicator / share_float /
pledge_stat / index_daily) into fact-layer numbers deterministically — replacing
the slow web-scrape of Q1 业绩 and risk flags.
"""

import pandas as pd
import pytest

from ashare_gauntlet.fundamentals import (
    index_changes,
    latest_quarter,
    pledge_ratio,
    upcoming_unlocks,
)


def test_latest_quarter_merges_income_and_fina_into_yi_units():
    income = pd.DataFrame(
        [
            {"end_date": "20251231", "report_type": "1", "total_revenue": 9.0e11, "n_income_attr_p": 3.5e10},
            {"end_date": "20260331", "report_type": "1", "total_revenue": 2.51e11, "n_income_attr_p": 1.06e10},
            {"end_date": "20260331", "report_type": "2", "total_revenue": 1.0, "n_income_attr_p": 1.0},  # 母公司,应过滤
        ]
    )
    fina = pd.DataFrame(
        [
            {"end_date": "20260331", "or_yoy": 56.52, "netprofit_yoy": 102.55, "grossprofit_margin": 7.35},
            {"end_date": "20251231", "or_yoy": 48.0, "netprofit_yoy": 52.0, "grossprofit_margin": 6.98},
        ]
    )
    q = latest_quarter(income, fina)
    assert q["end_date"] == "20260331"
    assert q["revenue_yi"] == pytest.approx(2510.0, abs=1)  # 2.51e11 / 1e8
    assert q["net_profit_yi"] == pytest.approx(106.0, abs=1)
    assert q["revenue_yoy_pct"] == pytest.approx(56.52)
    assert q["net_profit_yoy_pct"] == pytest.approx(102.55)
    assert q["gross_margin_pct"] == pytest.approx(7.35)
    assert q["profitable"] is True


def test_latest_quarter_flags_a_loss():
    income = pd.DataFrame([{"end_date": "20260331", "report_type": "1", "total_revenue": 9.33e9, "n_income_attr_p": -1.17e7}])
    q = latest_quarter(income, pd.DataFrame())
    assert q["profitable"] is False
    assert q["net_profit_yi"] < 0
    assert q["revenue_yoy_pct"] is None  # 无 fina 时同比缺失,不编


def test_pledge_ratio_takes_latest_period():
    p = pd.DataFrame([{"end_date": "20260605", "pledge_ratio": 0.29}, {"end_date": "20260612", "pledge_ratio": 0.31}])
    assert pledge_ratio(p) == pytest.approx(0.31)


def test_upcoming_unlocks_keeps_only_future_within_window():
    sf = pd.DataFrame(
        [
            {"float_date": "20220608", "float_share": 1e7, "float_ratio": 0.5, "holder_name": "老解禁"},
            {"float_date": "20260630", "float_share": 2.6e7, "float_ratio": 1.5, "holder_name": "定增对象"},
            {"float_date": "20270101", "float_share": 5e6, "float_ratio": 0.3, "holder_name": "太远"},
        ]
    )
    out = upcoming_unlocks(sf, as_of="20260617", within_days=180)
    assert [u["float_date"] for u in out] == ["20260630"]  # 过去/太远的都排除


def test_index_changes_picks_the_as_of_row():
    idx = pd.DataFrame(
        [
            {"ts_code": "000001.SH", "trade_date": "20260617", "close": 4108.08, "pct_chg": 0.40},
            {"ts_code": "399006.SZ", "trade_date": "20260617", "close": 4167.05, "pct_chg": 1.56},
            {"ts_code": "000001.SH", "trade_date": "20260616", "close": 4091.89, "pct_chg": -0.11},
        ]
    )
    ch = index_changes(idx, "20260617")
    assert ch["000001.SH"]["close"] == pytest.approx(4108.08)
    assert ch["399006.SZ"]["pct_chg"] == pytest.approx(1.56)
    assert "20260616" not in [str(v) for v in ch]  # 只取 as_of 当日
