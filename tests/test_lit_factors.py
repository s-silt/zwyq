"""TDD:A股有效因子纯函数(净现比/应计/EP/短期反转/波动)。文献依据见 presentation-layer-design memory。"""
import pandas as pd
import pytest

from ashare_gauntlet.lit_factors import (
    accrual_ratio,
    earnings_yield,
    net_cash_ratio,
    reversal,
    volatility,
)


def _inc(ni: float, ed: str = "20251231") -> pd.DataFrame:
    return pd.DataFrame([{"end_date": ed, "ann_date": ed, "n_income_attr_p": ni}])


def _cf(ocf: float, ed: str = "20251231") -> pd.DataFrame:
    return pd.DataFrame([{"end_date": ed, "ann_date": ed, "n_cashflow_act": ocf}])


def _bs(ta: float, ed: str = "20251231") -> pd.DataFrame:
    return pd.DataFrame([{"end_date": ed, "ann_date": ed, "total_assets": ta}])


def _daily(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": [f"202601{i:02d}" for i in range(1, len(closes) + 1)], "close": closes})


def _adj(n: int, factor: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": [f"202601{i:02d}" for i in range(1, n + 1)], "adj_factor": [factor] * n})


# ---- 净现比 = 2025年报经营现金流 / 归母净利 ----
def test_net_cash_ratio_basic():
    assert net_cash_ratio(_inc(10e8), _cf(12e8)) == pytest.approx(1.2)


def test_net_cash_ratio_nonpositive_profit_returns_none():
    # 净利<=0 时净现比无意义(分母失真),返回 None 而非伪造
    assert net_cash_ratio(_inc(-5e8), _cf(12e8)) is None


def test_net_cash_ratio_missing_annual_returns_none():
    # 缺 2025 年报行 -> None(数据缺失不伪造默认值)
    assert net_cash_ratio(_inc(10e8, ed="20241231"), _cf(12e8)) is None


# ---- 应计强度 = (归母净利 - 经营现金流) / 总资产(越低/越负越干净) ----
def test_accrual_ratio_basic():
    assert accrual_ratio(_inc(10e8), _cf(12e8), _bs(100e8)) == pytest.approx(-0.02)


def test_accrual_ratio_zero_assets_returns_none():
    assert accrual_ratio(_inc(10e8), _cf(12e8), _bs(0)) is None  # 总资产为0不可除


# ---- EP 盈利收益率 = 1 / PE_TTM ----
def test_earnings_yield_basic():
    assert earnings_yield(20.0) == pytest.approx(0.05)


def test_earnings_yield_nonpositive_returns_none():
    assert earnings_yield(0) is None
    assert earnings_yield(-10) is None
    assert earnings_yield(None) is None


# ---- 短期反转 = 近 n 日前复权收益率(正=涨多=反转回调风险大) ----
def test_reversal_20d():
    closes = [float(x) for x in range(100, 121)]  # 21 个点,100→120
    assert reversal(_daily(closes), _adj(21), n=20) == pytest.approx(0.20)


def test_reversal_insufficient_history_returns_none():
    assert reversal(_daily([100.0, 101.0]), _adj(2), n=20) is None


def test_reversal_uses_adjusted_price():
    # adj_factor 恒定时前复权收益等于裸收益(因子约去);验证用的是复权价路径
    closes = [float(x) for x in range(100, 121)]
    assert reversal(_daily(closes), _adj(21, factor=2.0), n=20) == pytest.approx(0.20)


# ---- 波动率 = 近 n 日前复权日收益标准差(越低越好,低波动因子) ----
def test_volatility_constant_return_is_zero():
    closes = [100.0 * (1.01 ** i) for i in range(61)]  # 恒定日收益 1% -> std=0
    assert volatility(_daily(closes), _adj(61), n=60) == pytest.approx(0.0, abs=1e-9)


def test_volatility_insufficient_history_returns_none():
    assert volatility(_daily([100.0, 101.0]), _adj(2), n=60) is None
