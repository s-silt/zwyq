"""TDD:A股有效因子纯函数(净现比/应计/EP/短期反转/波动)。文献依据见 presentation-layer-design memory。"""
import pandas as pd
import pytest

from ashare_gauntlet.lit_factors import (
    accrual_ratio,
    earnings_yield,
    latest_annual_end,
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


# ---- 契约C3:最近年报期从数据动态取 max(end_date endswith '1231'),不硬编码 ----
def test_latest_annual_end_picks_max_1231():
    # 多年年报 + 季报混在一起 → 取最大的 …1231(最近年报),忽略季报 …0331/0630
    inc = pd.DataFrame([
        {"end_date": "20231231", "n_income_attr_p": 1e8},
        {"end_date": "20241231", "n_income_attr_p": 2e8},
        {"end_date": "20250331", "n_income_attr_p": 3e8},  # 季报,非年报
    ])
    assert latest_annual_end(inc) == "20241231"


def test_latest_annual_end_uses_max_across_frames():
    # 跨表(income/cashflow)取并集最大年报期 —— cashflow 比 income 多一年年报
    inc = pd.DataFrame([{"end_date": "20241231", "n_income_attr_p": 2e8}])
    cf = pd.DataFrame([
        {"end_date": "20241231", "n_cashflow_act": 1e8},
        {"end_date": "20251231", "n_cashflow_act": 2e8},
    ])
    assert latest_annual_end(inc, cf) == "20251231"


def test_latest_annual_end_returns_none_when_no_annual():
    # 只有季报、无任何 …1231 → 取不到年报期返回 None(不套固定日期伪造)
    inc = pd.DataFrame([{"end_date": "20250331", "n_income_attr_p": 3e8}])
    assert latest_annual_end(inc) is None


def test_latest_annual_end_handles_empty_and_none():
    assert latest_annual_end(pd.DataFrame()) is None
    assert latest_annual_end(None) is None


def test_net_cash_ratio_honours_dynamic_end():
    # 把动态取到的年报期传进去仍可算(不再依赖硬编码 20251231)
    inc = pd.DataFrame([{"end_date": "20241231", "ann_date": "20250401", "n_income_attr_p": 10e8}])
    cf = pd.DataFrame([{"end_date": "20241231", "ann_date": "20250401", "n_cashflow_act": 12e8}])
    end = latest_annual_end(inc, cf)
    assert end == "20241231"
    assert net_cash_ratio(inc, cf, end=end) == pytest.approx(1.2)


# ---- 波动率 = 近 n 日前复权日收益标准差(越低越好,低波动因子) ----
def test_volatility_constant_return_is_zero():
    closes = [100.0 * (1.01 ** i) for i in range(61)]  # 恒定日收益 1% -> std=0
    assert volatility(_daily(closes), _adj(61), n=60) == pytest.approx(0.0, abs=1e-9)


def test_volatility_insufficient_history_returns_none():
    assert volatility(_daily([100.0, 101.0]), _adj(2), n=60) is None
