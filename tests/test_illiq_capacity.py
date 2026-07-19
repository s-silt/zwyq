"""X-04 ILLIQ 容量/冲击敏感性:分市值桶 + break-even 单边滑点(零新常数)。"""
from __future__ import annotations

import math

import pandas as pd
import pytest


def test_break_even_slippage_inverts_cost_model():
    from scripts.illiq_capacity import break_even_slippage

    # 净=超额 − τ×(2c+2s+印花税)=0 ⇒ s* = (超额/τ − 2c − 印花税)/2(纯代数逆运算)
    s = break_even_slippage(excess=0.006, turnover=1.0, commission=0.00025, stamp=0.0005)
    assert s == pytest.approx((0.006 / 1.0 - 2 * 0.00025 - 0.0005) / 2)
    # τ=0(组合零换手)→ 分母语义消失,NaN 不伪造
    assert math.isnan(break_even_slippage(0.006, 0.0, 0.00025, 0.0005))
    assert math.isnan(break_even_slippage(float("nan"), 1.0, 0.00025, 0.0005))


def test_mv_terciles_small_mid_large():
    from scripts.illiq_capacity import mv_terciles

    mv = pd.Series({c: v for c, v in zip("ABCDEFGHI", [1, 2, 3, 10, 20, 30, 100, 200, 300])})
    t = mv_terciles(mv)
    assert list(t[["A", "B", "C"]]) == ["小", "小", "小"]
    assert list(t[["D", "E", "F"]]) == ["中", "中", "中"]
    assert list(t[["G", "H", "I"]]) == ["大", "大", "大"]
    # 样本 <3 无三分位语义:全 NA(与 to_decile 同精神)
    assert mv_terciles(pd.Series({"A": 1.0, "B": 2.0})).isna().all()
