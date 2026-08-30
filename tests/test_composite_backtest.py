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


# ---------- c2_step:C2 退出规则状态机(M3;跌出首期保留/连续第2期剔/回档清零) ----------

def test_c2_step_state_machine():
    from scripts.composite_backtest import c2_step

    tradable = {"A", "B", "C"}
    # 首期:B 跌出 D10 但保留(streak=1)
    m1, s1 = c2_step({"A", "B"}, {}, {"A"}, tradable)
    assert m1 == {"A", "B"} and s1 == {"B": 1}
    # 次期仍在档外:B 被剔
    m2, s2 = c2_step(m1, s1, {"A"}, tradable)
    assert m2 == {"A"} and s2 == {}
    # 回档清零:B 回到 D10 后再跌出,重新从 streak=1 开始
    m3, s3 = c2_step({"A", "B"}, {"B": 1}, {"A", "B"}, tradable)
    assert m3 == {"A", "B"} and s3 == {}
    m4, s4 = c2_step(m3, s3, {"A"}, tradable)
    assert "B" in m4 and s4 == {"B": 1}
    # 不可交易的跌出票不保留
    m5, s5 = c2_step({"A", "D"}, {}, {"A"}, tradable)
    assert m5 == {"A"} and s5 == {}


# ---------- band_step:B8 排名带缓冲状态机(X-14;带内保留时间无界/跌穿带即剔) ----------

def test_band_step_state_machine():
    from scripts.composite_backtest import band_step

    tradable = {"A", "B", "C", "D"}
    d10 = {"A"}
    # 首期:B 跌出 D10 但仍在 D8 带内 → 保留(时间无界)
    assert band_step({"A", "B"}, d10, {"A", "B"}, tradable) == {"A", "B"}
    # 次期 B 仍在带内 → 继续保留(与 C2 不同:无连续期数上限)
    assert band_step({"A", "B"}, d10, {"A", "B"}, tradable) == {"A", "B"}
    # B 跌穿带(≤D7)→ 立即移出
    assert band_step({"A", "B"}, d10, {"A"}, tradable) == {"A"}
    # 带内但不可交易(停牌,E 不在 tradable)→ 不保留
    assert band_step({"A", "E"}, d10, {"A", "E"}, tradable) == {"A"}
    # 当期 D10 新成员照收(即使上期不在持仓)
    assert band_step(set(), {"A", "C"}, {"A", "C"}, tradable) == {"A", "C"}


# ---------- dedt_ttm_pit:扣非 TTM,构件严格 PIT(评审三轮 R6:不得用未来更正值) ----------

def _fina(rows, code="000001.SZ"):
    df = pd.DataFrame(rows, columns=["end_date", "ann_date", "profit_dedt"])
    df.insert(0, "ts_code", code)
    return df.sort_values(["ts_code", "end_date", "ann_date"], kind="mergesort")


def test_dedt_ttm_pit_mid_year_quarter():
    from scripts.composite_backtest import dedt_ttm_pit

    df = _fina([
        ("20240630", "20240820", 40.0),   # 上年同期 YTD
        ("20241231", "20250328", 100.0),  # 上年年报
        ("20250630", "20250815", 55.0),   # 当期 YTD
    ])
    # asof=20250901:TTM = 55 + 100 − 40 = 115;asof=20250401:最新期=年报,TTM=自身
    assert dedt_ttm_pit(df, "20250901")["000001.SZ"] == pytest.approx(115.0)
    assert dedt_ttm_pit(df, "20250401")["000001.SZ"] == pytest.approx(100.0)


def test_dedt_ttm_pit_missing_pieces_is_nan():
    from scripts.composite_backtest import dedt_ttm_pit

    # 缺上年年报 → NaN(fail-honest,不得退化成 YTD 冒充 TTM)
    df = _fina([("20240630", "20240820", 40.0), ("20250630", "20250815", 55.0)])
    assert pd.isna(dedt_ttm_pit(df, "20250901")["000001.SZ"])


def test_dedt_ttm_pit_future_correction_invisible():
    from scripts.composite_backtest import dedt_ttm_pit

    # 评审三轮 P0 场景:年报更正公告晚于查询日,构件必须用当时已知版本(100),
    # 不得用未来更正值(80);更正公告日之后查询才切换到 80
    df = _fina([
        ("20240930", "20241030", 30.0),
        ("20241231", "20250328", 100.0),
        ("20241231", "20251201", 80.0),    # 年报更正,公告于 2025-12-01
        ("20250930", "20251030", 35.0),
    ])
    assert dedt_ttm_pit(df, "20251115")["000001.SZ"] == pytest.approx(35 + 100 - 30)
    assert dedt_ttm_pit(df, "20251215")["000001.SZ"] == pytest.approx(35 + 80 - 30)


def test_dedt_ttm_pit_nothing_announced_yet():
    from scripts.composite_backtest import dedt_ttm_pit

    # 查询日早于一切公告 → 该股缺席结果(reindex 后自然 NaN),不得伪造
    df = _fina([("20241231", "20250328", 100.0)])
    out = dedt_ttm_pit(df, "20250101")
    assert "000001.SZ" not in out.index
