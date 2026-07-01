"""Tests for factor IC backtest —— IC(秩相关)+ point-in-time 防未来函数选期。"""
import math

import pandas as pd

from ashare_gauntlet.backtest import (
    adjusted_tstat,
    information_coefficient,
    point_in_time,
    quantile_spread,
)


def test_adjusted_tstat_penalizes_autocorrelation():
    # 强正自相关(月月复用的因子IC)→ 有效样本 N_eff < N → |t| 小于 iid 的 ICIR·√N
    s = pd.Series([0.03] * 5 + [0.01] * 5)  # 持续型、均值正
    icir, t, neff = adjusted_tstat(s)
    assert neff < len(s)
    iid_t = (s.mean() / s.std()) * math.sqrt(len(s))
    assert 0 < t < iid_t


def test_adjusted_tstat_returns_finite_for_normal_series():
    s = pd.Series([0.05, 0.01, 0.03, 0.02, 0.04, 0.00, 0.03, 0.01])  # 正均值、无极端结构
    icir, t, neff = adjusted_tstat(s)
    assert math.isfinite(icir) and math.isfinite(t) and neff > 0


def test_quantile_spread_positive_for_monotone():
    f = pd.Series([float(i) for i in range(50)])
    r = pd.Series([float(i) for i in range(50)])  # 因子高→收益高
    assert quantile_spread(f, r, 5) > 0


def test_quantile_spread_negative_for_inverse():
    f = pd.Series([float(i) for i in range(50)])
    r = pd.Series([float(i) for i in range(49, -1, -1)])
    assert quantile_spread(f, r, 5) < 0


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
