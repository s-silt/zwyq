"""Tests for pick_track —— 筛选器命中率闭环(D10 进出 diff + 前向收益),纯测量不打分。"""
import pandas as pd

from scripts.pick_track import diff_picks, forward_returns


def test_diff_picks_new_and_dropped():
    prev = ["A", "B", "C"]
    curr = ["B", "C", "D", "E"]
    d = diff_picks(prev, curr)
    assert d["new"] == ["D", "E"]
    assert d["dropped"] == ["A"]
    assert d["stay"] == ["B", "C"]


def test_diff_picks_no_prev_all_new():
    d = diff_picks([], ["A", "B"])
    assert d["new"] == ["A", "B"] and d["dropped"] == [] and d["stay"] == []


def _panel():
    # 两只股,4 个交易日,前复权价
    rows = []
    for i, dte in enumerate(["20260101", "20260102", "20260103", "20260106"]):
        rows.append({"ts_code": "A", "trade_date": dte, "adj_close": 10.0 + i})       # 10,11,12,13
        rows.append({"ts_code": "B", "trade_date": dte, "adj_close": 20.0 - i})       # 20,19,18,17
    return pd.DataFrame(rows)


def test_forward_returns_from_snapshot_date():
    px = _panel()
    r = forward_returns(["A", "B"], "20260101", px)
    # A: 13/10-1=+30%;B: 17/20-1=-15%(截至面板最新日)
    assert abs(r["A"] - 0.30) < 1e-9
    assert abs(r["B"] + 0.15) < 1e-9


def test_forward_returns_snapshot_date_not_traded_uses_next_available():
    px = _panel()
    r = forward_returns(["A"], "20260104", px)   # 周末快照 → 用其后首个交易日 20260106 起算
    assert abs(r["A"] - 0.0) < 1e-9              # 起点即最新日 → 0%


def test_forward_returns_missing_code_is_nan():
    px = _panel()
    r = forward_returns(["Z"], "20260101", px)
    assert r["Z"] != r["Z"]  # NaN
