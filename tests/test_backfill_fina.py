"""Tests for backfill_fina —— 全市场逐只财务 backfill 的模式→表集映射(纯函数)。"""
import pytest

from scripts.backfill_fina import tables_for_mode


def test_lean_is_only_fina_indicator():
    # 精简版只拉 fina_indicator(含三增+ROE+经营现金流,够算质地分)
    assert tables_for_mode("lean") == ("fina_indicator",)


def test_full_includes_four_core_tables():
    t = set(tables_for_mode("full"))
    assert {"income", "fina_indicator", "balancesheet", "cashflow"} <= t


def test_fina_indicator_in_both_modes():
    assert "fina_indicator" in tables_for_mode("lean")
    assert "fina_indicator" in tables_for_mode("full")


def test_full_superset_of_lean():
    assert set(tables_for_mode("lean")) <= set(tables_for_mode("full"))


def test_core_is_four_core_financial_tables():
    # core = 4 核心财报表(income/fina_indicator/balancesheet/cashflow),无预警表
    assert set(tables_for_mode("core")) == {"income", "fina_indicator", "balancesheet", "cashflow"}


def test_lean_subset_of_core_subset_of_full():
    assert set(tables_for_mode("lean")) <= set(tables_for_mode("core")) <= set(tables_for_mode("full"))


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        tables_for_mode("bogus")


# ---- expected_min_end_date:法定披露截止日(监管常数,非 magic number)----
from scripts.backfill_fina import expected_min_end_date


def test_expected_after_q1_deadline():
    assert expected_min_end_date("20260702") == "20260331"   # 4/30 后必有 Q1


def test_expected_after_h1_deadline():
    assert expected_min_end_date("20260901") == "20260630"   # 8/31 后必有 H1


def test_expected_after_q3_deadline():
    assert expected_min_end_date("20261101") == "20260930"   # 10/31 后必有 Q3


def test_expected_before_annual_deadline_falls_back_to_prev_q3():
    assert expected_min_end_date("20260430") == "20250930"   # 年报/Q1 截止日当天还没强制
    assert expected_min_end_date("20260215") == "20250930"


# ---- latest_period_end:最近一个已结束报告期(季度末是日历常数,非 magic number)----
from scripts.backfill_fina import latest_period_end


def test_latest_period_after_h1_end():
    assert latest_period_end("20260702") == "20260630"   # H1 已结束,披露表该查 0630


def test_latest_period_in_q2():
    assert latest_period_end("20260401") == "20260331"


def test_latest_period_on_quarter_end_itself():
    assert latest_period_end("20260331") == "20260331"   # 季度末当天该期已结束


def test_latest_period_just_before_quarter_end():
    assert latest_period_end("20260330") == "20251231"


def test_latest_period_q4_and_year_end():
    assert latest_period_end("20261001") == "20260930"
    assert latest_period_end("20261231") == "20261231"
    assert latest_period_end("20260101") == "20251231"


# ---- disclosed_stale_codes:披露表 + 本地新鲜度 → 待刷新代码列表(纯函数)----
import pandas as pd

from scripts.backfill_fina import disclosed_stale_codes


def _disc(rows):
    """(ts_code, end_date, actual_date) 行 → disclosure_date 形状的 df。"""
    return pd.DataFrame(rows, columns=["ts_code", "end_date", "actual_date"])


def test_disclosed_and_stale_is_refreshed():
    disc = _disc([("600000.SH", "20260630", "20260710")])
    local = {"600000.SH": "20260331"}   # 本地还停在 Q1,该票已披露 H1 → 重拉
    assert disclosed_stale_codes(disc, local, "20260630") == ["600000.SH"]


def test_not_yet_disclosed_is_skipped():
    # actual_date 为空 = 只有拟披露日(pre_date),拉了也只有旧数据 → 不重拉
    disc = _disc([("600000.SH", "20260630", None)])
    assert disclosed_stale_codes(disc, {"600000.SH": "20260331"}, "20260630") == []


def test_empty_string_actual_date_treated_as_not_disclosed():
    disc = _disc([("600000.SH", "20260630", "")])
    assert disclosed_stale_codes(disc, {"600000.SH": "20260331"}, "20260630") == []


def test_locally_fresh_is_skipped():
    # 本地缓存已含该期 → 已刷新过,不重拉
    disc = _disc([("600000.SH", "20260630", "20260710")])
    assert disclosed_stale_codes(disc, {"600000.SH": "20260630"}, "20260630") == []


def test_missing_local_cache_counts_as_stale():
    # 本地无缓存 = 最旧:已披露就该拉
    disc = _disc([("600000.SH", "20260630", "20260710")])
    assert disclosed_stale_codes(disc, {}, "20260630") == ["600000.SH"]


def test_other_period_rows_are_ignored():
    # 防御:混入别的报告期的行不应触发刷新
    disc = _disc([("600000.SH", "20260331", "20260408")])
    assert disclosed_stale_codes(disc, {"600000.SH": "20251231"}, "20260630") == []


def test_output_sorted_and_unique():
    disc = _disc([
        ("600002.SH", "20260630", "20260711"),
        ("600001.SH", "20260630", "20260710"),
        ("600001.SH", "20260630", "20260712"),   # 改期后同票多行:去重
    ])
    assert disclosed_stale_codes(disc, {}, "20260630") == ["600001.SH", "600002.SH"]
