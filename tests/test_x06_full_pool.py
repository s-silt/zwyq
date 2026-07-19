"""X-06:分位边界在全因子池上划,T+1 涨停只在评估侧缺席(composite P0-1 同修)。"""
from __future__ import annotations

import math

import pandas as pd
import pytest


def test_bucket_edges_from_full_pool_not_tradable_subset():
    from scripts.factor_backtest import full_pool_quantile_stats

    # 25 只票因子 1..25(≥ q×5 门槛),最高分那只 T+1 一字涨停(fwd NaN,买不进)
    neu = pd.Series({f"s{i}": float(i) for i in range(1, 26)})
    fwd = pd.Series({f"s{i}": i * 0.01 for i in range(1, 26)})
    fwd["s25"] = float("nan")
    spread, qlo, qhi, low_exec, high_exec = full_pool_quantile_stats(neu, fwd, q=5)
    # 边界在全池上划:顶组={s21..s25};s20 不得因 s25 不可成交而被顶进顶组
    assert high_exec == {"s21", "s22", "s23", "s24"}   # 可成交成员=顶组∩有收益
    assert qhi == pytest.approx((0.21 + 0.22 + 0.23 + 0.24) / 4)  # s25 缺席不伪造
    assert low_exec == {"s1", "s2", "s3", "s4", "s5"}
    assert qlo == pytest.approx(0.03)
    assert spread == pytest.approx(qhi - qlo)


def test_full_pool_quantile_stats_small_sample_nan():
    from scripts.factor_backtest import full_pool_quantile_stats

    neu = pd.Series({"a": 1.0, "b": 2.0})
    fwd = pd.Series({"a": 0.01, "b": 0.02})
    spread, qlo, qhi, low_exec, high_exec = full_pool_quantile_stats(neu, fwd, q=5)
    assert math.isnan(spread) and math.isnan(qlo) and math.isnan(qhi)
    assert low_exec == set() and high_exec == set()
