"""Tests for execution —— 右侧确认(entry_readiness)+ A股整手仓位公式(position_size)。

双仓制执行层(制度出处 memory trading-constraints):判据全定义性比较、零可调常数;
仓位 = 风险预算 / 单股风险,向下取整手。fail-loud:数据不足 / 参数非法都 raise。
"""
import pandas as pd
import pytest

from ashare_gauntlet.execution import (
    LABEL_LEFT,
    LABEL_RIGHT,
    LABEL_STABILIZING,
    entry_readiness,
    position_size,
)


# ---------- entry_readiness:右侧确认(缩量企稳 + 收复5日线) ----------

def _series(vals):
    return pd.Series([float(v) for v in vals])


def test_entry_readiness_both_true_is_right_side():
    # 收复5日线:最新 11 > MA5(10,10,10,10,11)=10.2;缩量:最新 50 < 前5日均量 100
    close = _series([10, 10, 10, 10, 10, 11])
    vol = _series([100, 100, 100, 100, 100, 50])
    r = entry_readiness(close, vol)
    assert r["above_ma5"] is True
    assert r["shrinking_vol"] is True
    assert r["label"] == LABEL_RIGHT


def test_entry_readiness_only_above_ma5_is_stabilizing():
    # 站上5日线但放量(150 > 前5日均量 100)→ 仅一真 → 企稳中
    close = _series([10, 10, 10, 10, 10, 11])
    vol = _series([100, 100, 100, 100, 100, 150])
    r = entry_readiness(close, vol)
    assert r["above_ma5"] is True and r["shrinking_vol"] is False
    assert r["label"] == LABEL_STABILIZING


def test_entry_readiness_only_shrinking_vol_is_stabilizing():
    # 缩量但仍在5日线下(9 < MA5(10,10,10,10,9)=9.8)→ 仅一真 → 企稳中
    close = _series([10, 10, 10, 10, 10, 9])
    vol = _series([100, 100, 100, 100, 100, 50])
    r = entry_readiness(close, vol)
    assert r["above_ma5"] is False and r["shrinking_vol"] is True
    assert r["label"] == LABEL_STABILIZING


def test_entry_readiness_neither_is_left_side():
    # 线下 + 放量 → 左侧⚠
    close = _series([10, 10, 10, 10, 10, 9])
    vol = _series([100, 100, 100, 100, 100, 200])
    r = entry_readiness(close, vol)
    assert r["label"] == LABEL_LEFT


def test_entry_readiness_boundary_equal_is_not_confirmed():
    # 严格不等:恰好等于5日均/等于前5日均量 → 都不算确认(平走≠收复,平量≠缩量)
    close = _series([10, 10, 10, 10, 10, 10])   # MA5 = 10,10 > 10 为假
    vol = _series([100, 100, 100, 100, 100, 100])  # 前5日均量 = 100,100 < 100 为假
    r = entry_readiness(close, vol)
    assert r["above_ma5"] is False and r["shrinking_vol"] is False
    assert r["label"] == LABEL_LEFT


def test_entry_readiness_returns_evidence_numbers():
    # 返回判定所用的数字证据(CLI 直接打印,不重算)
    close = _series([10, 10, 10, 10, 10, 11])
    vol = _series([100, 100, 100, 100, 100, 50])
    r = entry_readiness(close, vol)
    assert abs(r["close"] - 11.0) < 1e-9
    assert abs(r["ma5"] - 10.2) < 1e-9
    assert abs(r["vol"] - 50.0) < 1e-9
    assert abs(r["vol_ma5"] - 100.0) < 1e-9


def test_entry_readiness_insufficient_bars_fails_loud():
    # <6 根(缩量基准=不含最新的前5日,需 5+1 根)→ fail-loud,不静默给判定
    with pytest.raises(ValueError, match="6"):
        entry_readiness(_series([10, 10, 10, 10, 11]), _series([100, 100, 100, 100, 50]))


def test_entry_readiness_nan_bars_dont_count():
    # NaN 不算有效根数:表面 6 根、有效 5 根 → 同样 fail-loud(不拿 NaN 凑数)
    close = pd.Series([float("nan"), 10.0, 10.0, 10.0, 10.0, 11.0])
    vol = _series([100, 100, 100, 100, 100, 50])
    with pytest.raises(ValueError):
        entry_readiness(close, vol)


# ---------- position_size:风险预算 → A股整手仓位 ----------

def test_position_size_basic_formula():
    # 75000×1% = 750 风险预算;单股风险 10−9.3 = 0.7;750/0.7 = 1071.4 股 → 10 手 = 1000 股
    r = position_size(75000, 0.01, 10.0, 9.3)
    assert r["shares"] == 1000
    assert r["lots"] == 10
    assert abs(r["cost"] - 10000.0) < 1e-9
    assert abs(r["max_loss"] - 700.0) < 1e-9


def test_position_size_max_loss_never_exceeds_budget():
    # 向下取整手 ⇒ 触止损的真实亏损 ≤ 风险预算(整手化只会降杠杆不会升)
    r = position_size(75000, 0.01, 10.0, 9.3)
    assert r["max_loss"] <= 75000 * 0.01 + 1e-9


def test_position_size_zero_lots_when_budget_too_small():
    # 风险预算不够一手的风险 → 0 手(如实返回,不硬凑一手突破风险预算)
    r = position_size(10000, 0.01, 50.0, 46.5)   # 预算 100,一手风险 350
    assert r["shares"] == 0 and r["lots"] == 0
    assert r["cost"] == 0.0 and r["max_loss"] == 0.0


def test_position_size_stop_at_or_above_entry_fails_loud():
    # 止损价 ≥ 入场价:单股风险 ≤ 0,公式失义 → fail-loud
    with pytest.raises(ValueError):
        position_size(75000, 0.01, 10.0, 10.0)
    with pytest.raises(ValueError):
        position_size(75000, 0.01, 10.0, 10.5)


@pytest.mark.parametrize("account,risk,entry,stop", [
    (0, 0.01, 10.0, 9.3),        # 账户非正
    (-1, 0.01, 10.0, 9.3),
    (75000, 0, 10.0, 9.3),       # 风险比例非正
    (75000, -0.01, 10.0, 9.3),
    (75000, 0.01, 0, -0.7),      # 入场价非正
    (75000, 0.01, 10.0, 0),      # 止损价非正
    (75000, 0.01, 10.0, -1.0),
])
def test_position_size_nonpositive_params_fail_loud(account, risk, entry, stop):
    with pytest.raises(ValueError):
        position_size(account, risk, entry, stop)
