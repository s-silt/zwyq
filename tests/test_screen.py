"""Tests for the reusable screen logic (pure filtering, no I/O).

board_of encodes the user's account constraint (Shanghai main board only);
screen_candidates is the 维度优 lens applied to a tidy facts table.
"""

import pandas as pd

from ashare_gauntlet.screen import board_of, screen_candidates


def test_board_of_classifies_exchange_and_board():
    assert board_of("601138.SH") == "沪主板"
    assert board_of("603890.SH") == "沪主板"
    assert board_of("688347.SH") == "科创板"
    assert board_of("002463.SZ") == "深主板"
    assert board_of("300319.SZ") == "创业板"
    assert board_of("920971.BJ") == "北交所"


def _facts() -> pd.DataFrame:
    # ts_code, close, trend, dist60, pct20, pe_ttm, pb, industry
    return pd.DataFrame(
        [
            {"ts_code": "601138.SH", "close": 72.6, "trend": "多头", "dist60": -10, "pct20": 87, "pe_ttm": 35.0, "pb": 8.2, "industry": "通信设备"},
            {"ts_code": "603890.SH", "close": 25.1, "trend": "多头", "dist60": -20, "pct20": 95, "pe_ttm": 37.0, "pb": 3.4, "industry": "元器件"},
            {"ts_code": "688347.SH", "close": 260.7, "trend": "多头", "dist60": 0, "pct20": 98, "pe_ttm": 918.0, "pb": 10.0, "industry": "半导体"},
            {"ts_code": "002463.SZ", "close": 146.5, "trend": "多头", "dist60": 0, "pct20": 98, "pe_ttm": 66.0, "pb": 16.8, "industry": "元器件"},
            {"ts_code": "600539.SH", "close": 16.0, "trend": "纠缠", "dist60": -22, "pct20": 92, "pe_ttm": float("nan"), "pb": 12.3, "industry": "互联网"},
        ]
    )


def test_screen_keeps_sh_main_tech_profitable_and_cheap():
    out = screen_candidates(
        _facts(), boards=["沪主板"], require_profitable=True, max_pe=80,
        industries=("通信", "元器件", "半导体"),
    )
    codes = set(out["ts_code"])
    assert {"601138.SH", "603890.SH"} == codes  # 沪主板 + 科技 + 盈利 + PE≤80
    # 688347 科创板且 PE918>80;002463 深市;600539 亏损(pe NaN)且行业非目标 —— 全部剔除


def test_screen_max_dist60_keeps_only_deep_pullbacks():
    out = screen_candidates(_facts(), boards=["沪主板"], require_profitable=True, max_dist60=-15)
    assert list(out["ts_code"]) == ["603890.SH"]  # 仅 -20 过线(601138 -10;600539 亏损)


def test_screen_sorts_and_caps_top():
    out = screen_candidates(_facts(), boards=["沪主板"], require_profitable=True,
                            sort_by="pct20", ascending=False, top=1)
    assert list(out["ts_code"]) == ["603890.SH"]  # pct20 95 > 87
