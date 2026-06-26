"""Tests for factor_model —— 横截面因子模型纯函数(零 magic number:中位数去均值+百分位+等权)。"""
import pandas as pd

from ashare_gauntlet.factor_model import (
    composite,
    factor_percentile,
    industry_neutralize,
    percentile_rank,
    to_decile,
)


def test_percentile_rank_orders_and_bounds():
    r = percentile_rank(pd.Series([10.0, 20.0, 30.0, 40.0]))
    assert r.iloc[0] < r.iloc[-1]
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert r.iloc[-1] == 1.0  # 最大 → 顶


def test_percentile_rank_preserves_nan():
    r = percentile_rank(pd.Series([10.0, None, 30.0]))
    assert pd.isna(r.iloc[1])           # 缺失不参与排名、保持 NaN
    assert r.iloc[2] > r.iloc[0]


def test_industry_neutralize_demeans_within_industry_median():
    # A 中位 15 → [-5,5];B 中位 105 → [-5,5](去掉行业绝对水平差,只留行业内相对)
    s = pd.Series([10.0, 20.0, 100.0, 110.0])
    ind = pd.Series(["银行", "银行", "半导体", "半导体"])
    n = industry_neutralize(s, ind)
    assert n.iloc[0] == -5.0 and n.iloc[1] == 5.0
    assert n.iloc[2] == -5.0 and n.iloc[3] == 5.0


def test_factor_percentile_higher_is_better():
    s = pd.Series([1.0, 2.0, 3.0])
    ind = pd.Series(["A", "A", "A"])
    r = factor_percentile(s, ind, higher_is_better=True)
    assert r.iloc[2] > r.iloc[0]


def test_factor_percentile_lower_is_better_inverts():
    # 应计利润:越低越好 → 最小 raw 拿最高百分位
    s = pd.Series([1.0, 2.0, 3.0])
    ind = pd.Series(["A", "A", "A"])
    r = factor_percentile(s, ind, higher_is_better=False)
    assert r.iloc[0] > r.iloc[2]


def test_composite_equal_weight_average():
    c = composite(pd.DataFrame({"f1": [0.8, 0.2], "f2": [0.6, 0.4]}))
    assert abs(c.iloc[0] - 0.7) < 1e-9
    assert abs(c.iloc[1] - 0.3) < 1e-9


def test_composite_skips_missing_factor_not_zero_fill():
    # 缺某因子时用可得因子均值,不当 0 填(0 填会无依据地惩罚)
    c = composite(pd.DataFrame({"f1": [0.8, 0.2], "f2": [None, 0.4]}))
    assert abs(c.iloc[0] - 0.8) < 1e-9   # 只有 f1
    assert abs(c.iloc[1] - 0.3) < 1e-9   # mean(0.2,0.4)


def test_to_decile_top_and_bottom():
    d = to_decile(pd.Series([float(i) for i in range(100)]))
    assert d.iloc[-1] == 10
    assert d.iloc[0] == 1
