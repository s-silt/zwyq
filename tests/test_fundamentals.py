"""Tests for the interface-first fundamentals extractors (pure, no I/O).

These turn cached tushare tables (income / fina_indicator / share_float /
pledge_stat / index_daily) into fact-layer numbers deterministically — replacing
the slow web-scrape of Q1 业绩 and risk flags.
"""

import pandas as pd
import pytest

from ashare_gauntlet.fundamentals import (
    balance_facts,
    cashflow_facts,
    index_changes,
    latest_quarter,
    pledge_ratio,
    recent_holder_trades,
    st_status,
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


def test_balance_facts_to_yi_filters_report_type():
    bs = pd.DataFrame(
        [
            {"end_date": "20260331", "report_type": "1", "accounts_receiv": 1.025e11, "goodwill": 3.28e8, "money_cap": 1.02e11},
            {"end_date": "20260331", "report_type": "2", "accounts_receiv": 1.0, "goodwill": 1.0, "money_cap": 1.0},  # 母公司,过滤
        ]
    )
    b = balance_facts(bs)
    assert b["accounts_receiv_yi"] == pytest.approx(1025.0, abs=1)
    assert b["goodwill_yi"] == pytest.approx(3.28, abs=0.01)
    assert b["money_cap_yi"] == pytest.approx(1020.0, abs=2)


def test_cashflow_facts_to_yi():
    cf = pd.DataFrame([{"end_date": "20260331", "report_type": "1", "n_cashflow_act": 2.5e10}])
    assert cashflow_facts(cf)["op_cashflow_yi"] == pytest.approx(250.0, abs=1)


def test_recent_holder_trades_flags_reductions_in_window():
    ht = pd.DataFrame(
        [
            {"ann_date": "20260430", "holder_name": "薛革文", "in_de": "DE", "change_vol": 8.66e6, "change_ratio": 1.94, "avg_price": 17.4},
            {"ann_date": "20240101", "holder_name": "老增持", "in_de": "IN", "change_vol": 1e6, "change_ratio": 0.2, "avg_price": 10.0},  # 太久,排除
        ]
    )
    out = recent_holder_trades(ht, as_of="20260617", within_days=365)
    assert len(out) == 1
    assert out[0]["holder_name"] == "薛革文"
    assert out[0]["direction"] == "减持"


def test_st_status_detects_current_and_history():
    nc = pd.DataFrame(
        [
            {"name": "*ST华微", "start_date": "20240501", "change_reason": "实施退市风险警示"},
            {"name": "华微电子", "start_date": "20260520", "change_reason": "撤销退市风险警示"},
        ]
    )
    s = st_status(nc)
    assert s["current_name"] == "华微电子"
    assert s["is_st"] is False
    assert s["ever_st"] is True  # 曾是 *ST
    assert s["last_change_date"] == "20260520"


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
