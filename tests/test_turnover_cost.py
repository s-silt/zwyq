"""P0② 真实换手成本折扣(吸纳终榜 2026-07-05 P0 第2条)。

现有"成本后月差"按每期全换手扣一次完整 round_trip=上界;实际相邻期 Q5/Q1 持仓有重叠,
只有被替换的那部分付成本。τ_t = 1 − |前后期交集|/|当期|(定义性),真实成本 ≈ τ_t × rt_t。
对抗轮量化:上界 40-45bp/月、重叠 40-60% 时差 16-27bp——足以改变 ACC 类边际因子的判定。
"""
import math

import pandas as pd

from scripts.factor_backtest import leg_turnover, quantile_legs


def test_leg_turnover_identical_zero():
    assert leg_turnover({"a", "b", "c"}, {"a", "b", "c"}) == 0.0


def test_leg_turnover_disjoint_one():
    assert leg_turnover({"a", "b"}, {"c", "d"}) == 1.0


def test_leg_turnover_half():
    assert leg_turnover({"a", "b"}, {"a", "c"}) == 0.5


def test_leg_turnover_no_prev_is_nan():
    # 首期无前期组合 → NaN(建仓成本是一次性的,不属于"月度维持换手")
    assert math.isnan(leg_turnover(None, {"a"}))
    assert math.isnan(leg_turnover({"a"}, set()))   # 当期空腿同样无定义


def test_quantile_legs_bottom_and_top():
    # 50 只按因子值排 → 低腿=最小10只,高腿=最大10只(与 quantile_spread 同 5 分位约定)
    f = pd.Series({f"c{i:02d}": float(i) for i in range(50)})
    low, high = quantile_legs(f, 5)
    assert low == {f"c{i:02d}" for i in range(10)}
    assert high == {f"c{i:02d}" for i in range(40, 50)}


def test_quantile_legs_nan_excluded_and_small_sample_empty():
    f = pd.Series({f"c{i}": float(i) for i in range(30)})
    f.iloc[0] = float("nan")
    low, high = quantile_legs(f, 5)
    assert "c0" not in low and len(low) > 0
    # 样本 < q*5(quantile_spread 同门槛)→ 空腿,不造伪组合
    small = pd.Series({f"c{i}": float(i) for i in range(20)})
    assert quantile_legs(small, 5) == (set(), set())
