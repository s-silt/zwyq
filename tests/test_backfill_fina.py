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
