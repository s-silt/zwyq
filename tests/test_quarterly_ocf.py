"""Tests for 单季经营现金流(铁律:现金看单季不看年报/累计)—— quarterly_ocf + tier_of 接线。"""
import pandas as pd

from ashare_gauntlet.fundamentals import quarterly_ocf
from ashare_gauntlet.record import tier_of


def _cf(rows):
    return pd.DataFrame(rows)


def test_q1_single_quarter_equals_cumulative():
    cf = _cf([{"end_date": "20260331", "ann_date": "20260428", "n_cashflow_act": 5e8}])
    out = quarterly_ocf(cf)
    assert out["end_date"] == "20260331"
    assert abs(out["ocf_q_yi"] - 5.0) < 1e-9


def test_h1_diffs_against_q1():
    cf = _cf([
        {"end_date": "20260331", "ann_date": "20260428", "n_cashflow_act": 5e8},
        {"end_date": "20260630", "ann_date": "20260830", "n_cashflow_act": 3e8},  # 累计3亿
    ])
    out = quarterly_ocf(cf)
    # Q2 单季 = 3 − 5 = −2亿:累计为正掩盖单季转负 —— 正是铁律要抓的
    assert out["end_date"] == "20260630"
    assert abs(out["ocf_q_yi"] - (-2.0)) < 1e-9


def test_h1_without_q1_returns_none_not_fabricated():
    cf = _cf([{"end_date": "20260630", "ann_date": "20260830", "n_cashflow_act": 3e8}])
    out = quarterly_ocf(cf)
    assert out["ocf_q_yi"] is None  # 上一季缺失:如实 None,不拿累计冒充单季


def test_empty_returns_empty():
    assert quarterly_ocf(pd.DataFrame()) == {}


def _rec(ocf_annual, ocf_q):
    return {
        "fundamental": {"profitable": True, "np_yoy": 20.0, "dedt_yoy": 22.0, "rev_yoy": 15.0,
                        "np_yi": 5.0, "dedt_yi": 4.8},
        "quality": {"op_cashflow_yi": ocf_annual, "op_cashflow_q_yi": ocf_q},
        "balance": {}, "status": {}, "flags": [],
    }


def test_tier_prefers_quarterly_ocf_over_annual():
    # 年报累计为正但最新单季转负 → 不给 🟢(单季口径拦下)
    t = tier_of(_rec(ocf_annual=10.0, ocf_q=-1.0))
    assert t["grade"] == "🟡"
    assert any("单季" in r for r in t["reasons"])


def test_tier_quarterly_positive_passes_green():
    t = tier_of(_rec(ocf_annual=10.0, ocf_q=2.0))
    assert t["grade"] == "🟢"


def test_tier_falls_back_to_annual_when_quarterly_missing():
    # 单季不可算(上一季缺失)→ 退回年报口径,行为与历史一致(不因缺失误杀)
    t = tier_of(_rec(ocf_annual=10.0, ocf_q=None))
    assert t["grade"] == "🟢"
