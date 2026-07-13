"""composite 端到端组合回测的纯函数测试(外部评审二轮:D10 组合级证据)。"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------- max_drawdown:期收益序列 → 最大回撤(负数,复利口径) ----------

def test_max_drawdown_known_path():
    from scripts.composite_backtest import max_drawdown

    # 净值 1 → 1.1 → 0.99 → 1.0395:峰值 1.1,谷 0.99 → 回撤 -10%
    r = pd.Series([0.10, -0.10, 0.05])
    assert max_drawdown(r) == pytest.approx(-0.10)


def test_max_drawdown_monotonic_up_is_zero():
    from scripts.composite_backtest import max_drawdown

    assert max_drawdown(pd.Series([0.01, 0.02, 0.0])) == pytest.approx(0.0)


def test_max_drawdown_skips_nan_and_empty():
    from scripts.composite_backtest import max_drawdown

    r = pd.Series([0.10, float("nan"), -0.20])
    assert max_drawdown(r) == pytest.approx(-0.20)
    assert pd.isna(max_drawdown(pd.Series(dtype=float)))


# ---------- industry_hhi:等权组合的行业集中度(Herfindahl,1=全一个行业) ----------

def test_industry_hhi_uniform_and_concentrated():
    from scripts.composite_backtest import industry_hhi

    # 4 只票 4 个行业:HHI = 4×(1/4)² = 0.25;全同行业 = 1
    assert industry_hhi(pd.Series(["a", "b", "c", "d"])) == pytest.approx(0.25)
    assert industry_hhi(pd.Series(["a", "a", "a"])) == pytest.approx(1.0)


def test_industry_hhi_empty_is_nan():
    from scripts.composite_backtest import industry_hhi

    assert pd.isna(industry_hhi(pd.Series(dtype=object)))


# ---------- top_contrib_share:单票贡献集中度(伪装成组合的单票行情要现形) ----------

def test_top_contrib_share():
    from scripts.composite_backtest import top_contrib_share

    # 贡献 [9, 1, -1, 1]:|9| / (9+1+1+1) = 0.75
    c = pd.Series([9.0, 1.0, -1.0, 1.0], index=list("abcd"))
    assert top_contrib_share(c) == pytest.approx(0.75)
    assert pd.isna(top_contrib_share(pd.Series(dtype=float)))


# ---------- dedt_ttm_rows:扣非 TTM(YTD 累计口径 → 滚动四季)PIT 逐行构造 ----------

def _fina(rows):
    return pd.DataFrame(rows, columns=["end_date", "ann_date", "profit_dedt"])


def test_dedt_ttm_mid_year_quarter():
    from scripts.composite_backtest import dedt_ttm_rows

    df = _fina([
        ("20240630", "20240820", 40.0),   # 上年同期 YTD
        ("20241231", "20250328", 100.0),  # 上年年报
        ("20250630", "20250815", 55.0),   # 当期 YTD
    ])
    out = dedt_ttm_rows(df)
    # TTM(20250630) = 55 + 100 − 40 = 115;年报行 TTM=自身
    assert out.loc[out["end_date"] == "20250630", "dedt_ttm"].iloc[0] == pytest.approx(115.0)
    assert out.loc[out["end_date"] == "20241231", "dedt_ttm"].iloc[0] == pytest.approx(100.0)


def test_dedt_ttm_missing_pieces_is_nan():
    from scripts.composite_backtest import dedt_ttm_rows

    # 缺上年年报 → 无法构造 TTM,必须 NaN(fail-honest,不得退化成 YTD 冒充 TTM)
    df = _fina([("20240630", "20240820", 40.0), ("20250630", "20250815", 55.0)])
    out = dedt_ttm_rows(df)
    assert pd.isna(out.loc[out["end_date"] == "20250630", "dedt_ttm"].iloc[0])


def test_dedt_ttm_dedupes_corrections_by_latest_ann():
    from scripts.composite_backtest import dedt_ttm_rows

    # 同 end_date 多行(更正公告):取 ann_date 最新一行的值
    df = _fina([
        ("20240630", "20240820", 40.0),
        ("20241231", "20250328", 100.0),
        ("20241231", "20250428", 90.0),   # 年报更正:100 → 90
        ("20250630", "20250815", 55.0),
    ])
    out = dedt_ttm_rows(df)
    assert out.loc[out["end_date"] == "20250630", "dedt_ttm"].iloc[0] == pytest.approx(55 + 90 - 40)


def test_dedt_ttm_preserves_row_order_and_index():
    from scripts.composite_backtest import dedt_ttm_rows

    # 行序/索引契约:_pit 的 groupby.tail(1) 依赖输入行序(end_date/ann_date 已排序);
    # 函数内 reset_index/重排会静默错选行(对抗审查 P2-9 的回归防线)
    df = _fina([
        ("20240630", "20240820", 40.0),
        ("20241231", "20250328", 100.0),
        ("20250630", "20250815", 55.0),
    ])
    df.index = [7, 3, 9]                      # 故意非常规索引
    out = dedt_ttm_rows(df)
    assert list(out.index) == [7, 3, 9]
    assert list(out["end_date"]) == ["20240630", "20241231", "20250630"]
