"""X-08 市值三分位子组合:size_tilt_members 纯函数(D10 内按市值切小/中/大桶)。

动机:X-04 证明高 ILLIQ 腿超额集中于小桶但机构不可投资;X-08 检验同一规模效应
是否存在于 PROD D10 内部——对万元级账户小桶冲击近零,若成立即为期望提升路径。
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_small_and_large_terciles_by_mv():
    from scripts.composite_backtest import size_tilt_members

    mv = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0, "F": 60.0})
    d10 = {"A", "B", "C", "D", "E", "F"}
    assert size_tilt_members(d10, mv, "小") == {"A", "B"}
    assert size_tilt_members(d10, mv, "中") == {"C", "D"}
    assert size_tilt_members(d10, mv, "大") == {"E", "F"}


def test_split_ranks_within_d10_only():
    """三分位必须在 D10 成员内部排名——mv 里存在的非成员不得影响切分。"""
    from scripts.composite_backtest import size_tilt_members

    mv = pd.Series({c: float(i) for i, c in enumerate(
        ["x1", "x2", "x3", "x4", "A", "B", "C", "D", "E", "F"], start=1)})
    d10 = {"A", "B", "C", "D", "E", "F"}     # 全市场里偏大,但桶内照常三分
    assert size_tilt_members(d10, mv, "小") == {"A", "B"}
    assert size_tilt_members(d10, mv, "大") == {"E", "F"}


def test_mv_nan_member_in_no_bucket():
    from scripts.composite_backtest import size_tilt_members

    mv = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0,
                    "F": float("nan")})
    d10 = {"A", "B", "C", "D", "E", "F"}
    got = (size_tilt_members(d10, mv, "小") | size_tilt_members(d10, mv, "中")
           | size_tilt_members(d10, mv, "大"))
    assert "F" not in got                    # 无市值=无桶,不得静默塞进任何桶


def test_fewer_than_three_members_empty():
    from scripts.composite_backtest import size_tilt_members

    mv = pd.Series({"A": 10.0, "B": 20.0})
    assert size_tilt_members({"A", "B"}, mv, "小") == set()   # 无三分位语义(同 X-04)


def test_unknown_bucket_raises():
    from scripts.composite_backtest import size_tilt_members

    mv = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0})
    with pytest.raises(ValueError):
        size_tilt_members({"A", "B", "C"}, mv, "micro")
