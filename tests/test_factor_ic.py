"""Tests for factor IC backtest —— IC(秩相关)+ point-in-time 防未来函数选期。"""
import math

import pandas as pd

from ashare_gauntlet.backtest import information_coefficient, point_in_time


def test_ic_perfect_positive():
    assert abs(information_coefficient(pd.Series([1.0, 2, 3, 4, 5]), pd.Series([1.0, 2, 3, 4, 5])) - 1.0) < 1e-9


def test_ic_perfect_negative():
    assert abs(information_coefficient(pd.Series([1.0, 2, 3, 4, 5]), pd.Series([5.0, 4, 3, 2, 1])) + 1.0) < 1e-9


def test_ic_rank_based_robust_to_outlier():
    # 秩相关对肥尾稳健:末位一个极端 return 不翻符号(仍 +1 秩序)
    assert information_coefficient(pd.Series([1.0, 2, 3, 4, 5]), pd.Series([1.0, 2, 3, 4, 999])) > 0.9


def test_ic_drops_nan_pairs():
    assert information_coefficient(pd.Series([1.0, 2, None, 4, 5]), pd.Series([1.0, 2, 3, None, 5])) > 0


def test_ic_insufficient_pairs_returns_nan():
    assert math.isnan(information_coefficient(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])))


def test_point_in_time_uses_latest_announced_before_asof():
    hist = pd.DataFrame({
        "end_date": ["20250331", "20250630", "20250930"],
        "ann_date": ["20250428", "20250815", "20251028"],
        "roe": [5.0, 6.0, 7.0],
    })
    row = point_in_time(hist, "20250901")   # Q3 未披露,只能用 Q2
    assert row is not None and row["roe"] == 6.0


def test_point_in_time_excludes_future_announcement():
    hist = pd.DataFrame({
        "end_date": ["20250331", "20250630"],
        "ann_date": ["20250428", "20250815"],
        "roe": [5.0, 6.0],
    })
    row = point_in_time(hist, "20250801")   # Q2(0815)还没公告,只能用 Q1
    assert row is not None and row["roe"] == 5.0


def test_point_in_time_none_when_nothing_announced_yet():
    hist = pd.DataFrame({"end_date": ["20250331"], "ann_date": ["20250428"], "roe": [5.0]})
    assert point_in_time(hist, "20250101") is None
