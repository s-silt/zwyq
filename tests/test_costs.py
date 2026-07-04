"""Tests for costs —— A股交易成本模型:印花税 PIT 分段 / 5元佣金地板 / round-trip 上界口径。

五路文献研读(Qlib/QuantsPlaybook/RQAlpha/LWZ)一致判定:无成本回测的前向收益系统性
虚高,成本模型是所有边际因子取舍的前提。费率必须按**交易日当期**取(印花税 2023-08-28
减半,横跨样本期),单一固定费率是自欺。
"""
import pytest

from ashare_gauntlet.costs import (
    MIN_COMMISSION_CNY,
    STAMP_DUTY_SEGMENTS,
    round_trip_cost_rate,
    stamp_duty,
    stamp_duty_rate,
    trade_cost,
)


# ---------- 印花税:PIT 分段(监管常数) ----------

def test_stamp_duty_rate_regime_change_on_20230828():
    # 财政部 税务总局公告2023年第39号:2023-08-28 起减半征收 0.1% → 0.05%
    assert stamp_duty_rate("20230827") == 0.001
    assert stamp_duty_rate("20230828") == 0.0005
    assert stamp_duty_rate("20260101") == 0.0005


def test_stamp_duty_amount_uses_trade_date_rate():
    # 同额卖出,税改前后税额差一倍(卖出单边)
    assert stamp_duty("20230827", 100_000.0) == pytest.approx(100.0)
    assert stamp_duty("20230828", 100_000.0) == pytest.approx(50.0)


def test_stamp_duty_before_earliest_segment_fails_loud():
    # 分段表最早段 = 2008-09-19(单边征收起点);更早属双边/其它税率 regime,拒绝外推
    with pytest.raises(ValueError, match=STAMP_DUTY_SEGMENTS[0][0]):
        stamp_duty_rate("20080101")


def test_stamp_duty_malformed_date_fails_loud():
    with pytest.raises(ValueError):
        stamp_duty_rate("2023-08-28")   # 全库日期约定是 YYYYMMDD 纯数字串
    with pytest.raises(ValueError):
        stamp_duty_rate("")


def test_stamp_duty_negative_amount_fails_loud():
    with pytest.raises(ValueError):
        stamp_duty("20260101", -1.0)


# ---------- trade_cost:5元佣金地板(行业最低收费惯例) ----------

def test_trade_cost_min_commission_floor_triggers_on_small_order():
    # 1万×万2.5=2.5元 < 5元 → 地板生效;买入无印花税
    c = trade_cost("20260101", "buy", 10_000.0, commission_rate=0.00025, slippage_rate=0.0015)
    assert c["commission"] == pytest.approx(MIN_COMMISSION_CNY)
    assert c["stamp"] == 0.0
    assert c["slippage"] == pytest.approx(15.0)
    assert c["total"] == pytest.approx(5.0 + 0.0 + 15.0)


def test_trade_cost_min_commission_floor_not_triggered_on_large_order():
    # 10万×万2.5=25元 > 5元 → 按比例计;卖出含当期印花税(20260101 处于 0.05% 段)
    c = trade_cost("20260101", "sell", 100_000.0, commission_rate=0.00025, slippage_rate=0.0015)
    assert c["commission"] == pytest.approx(25.0)
    assert c["stamp"] == pytest.approx(50.0)
    assert c["slippage"] == pytest.approx(150.0)
    assert c["total"] == pytest.approx(25.0 + 50.0 + 150.0)


def test_trade_cost_sell_stamp_follows_pit_segment():
    # 税改前一日卖出 → 0.1% 段
    c = trade_cost("20230827", "sell", 100_000.0, commission_rate=0.00025, slippage_rate=0.0015)
    assert c["stamp"] == pytest.approx(100.0)


def test_trade_cost_invalid_side_fails_loud():
    with pytest.raises(ValueError, match="side"):
        trade_cost("20260101", "hold", 10_000.0, commission_rate=0.00025, slippage_rate=0.0015)


def test_trade_cost_nonpositive_amount_fails_loud():
    for bad in (0.0, -100.0, float("nan")):
        with pytest.raises(ValueError):
            trade_cost("20260101", "buy", bad, commission_rate=0.00025, slippage_rate=0.0015)


def test_trade_cost_negative_rates_fail_loud():
    with pytest.raises(ValueError):
        trade_cost("20260101", "buy", 10_000.0, commission_rate=-0.0001, slippage_rate=0.0015)
    with pytest.raises(ValueError):
        trade_cost("20260101", "buy", 10_000.0, commission_rate=0.00025, slippage_rate=-0.001)


# ---------- round_trip_cost_rate:单次完整买卖的费率 ----------

def test_round_trip_cost_rate_combines_all_legs():
    # 2×佣金 + 2×滑点 + 卖出印花税:2×0.00025 + 2×0.0015 + 0.0005 = 0.004
    assert round_trip_cost_rate("20260101", 0.00025, 0.0015) == pytest.approx(0.004)


def test_round_trip_cost_rate_pit_stamp_segment():
    # 税改前后 round_trip 差 = 印花税差 0.05%
    pre = round_trip_cost_rate("20230827", 0.00025, 0.0015)
    post = round_trip_cost_rate("20230828", 0.00025, 0.0015)
    assert pre == pytest.approx(0.0045)
    assert (pre - post) == pytest.approx(0.0005)


def test_round_trip_cost_rate_negative_rates_fail_loud():
    with pytest.raises(ValueError):
        round_trip_cost_rate("20260101", -0.00025, 0.0015)
