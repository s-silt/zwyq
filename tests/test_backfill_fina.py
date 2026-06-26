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


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        tables_for_mode("bogus")
