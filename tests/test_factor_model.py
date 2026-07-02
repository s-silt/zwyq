"""Tests for factor_model —— 横截面因子模型纯函数(零 magic number:中位数去均值+百分位+等权)。"""
import pandas as pd
import pytest

from ashare_gauntlet.factor_model import (
    composite,
    factor_percentile,
    industry_neutralize,
    momentum_return,
    percentile_rank,
    to_decile,
    touched_limit_up,
)


def test_momentum_rising_series_positive():
    s = pd.Series([float(i) for i in range(1, 201)])  # 1..200 递增
    assert momentum_return(s, lookback=120) > 0


def test_momentum_falling_series_negative():
    s = pd.Series([float(i) for i in range(200, 0, -1)])  # 200..1 递减
    assert momentum_return(s, lookback=120) < 0


def test_momentum_exact_ratio():
    s = pd.Series([float(i) for i in range(1, 201)])
    # 末值200 / 120日前(iloc[-121]=80) - 1 = 1.5
    assert abs(momentum_return(s, lookback=120) - 1.5) < 1e-9


def test_momentum_insufficient_history_none():
    s = pd.Series([1.0, 2.0, 3.0])
    assert momentum_return(s, lookback=120) is None


def test_percentile_rank_orders_and_bounds():
    r = percentile_rank(pd.Series([10.0, 20.0, 30.0, 40.0]))
    assert r.iloc[0] < r.iloc[-1]
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert r.iloc[-1] == 1.0  # 最大 → 顶


def test_percentile_rank_preserves_nan():
    r = percentile_rank(pd.Series([10.0, None, 30.0]))
    assert pd.isna(r.iloc[1])           # 缺失不参与排名、保持 NaN
    assert r.iloc[2] > r.iloc[0]


def test_industry_neutralize_demeans_within_industry_median():
    # A 中位 15 → [-5,5];B 中位 105 → [-5,5](去掉行业绝对水平差,只留行业内相对)
    s = pd.Series([10.0, 20.0, 100.0, 110.0])
    ind = pd.Series(["银行", "银行", "半导体", "半导体"])
    n = industry_neutralize(s, ind)
    assert n.iloc[0] == -5.0 and n.iloc[1] == 5.0
    assert n.iloc[2] == -5.0 and n.iloc[3] == 5.0


def test_factor_percentile_higher_is_better():
    s = pd.Series([1.0, 2.0, 3.0])
    ind = pd.Series(["A", "A", "A"])
    r = factor_percentile(s, ind, higher_is_better=True)
    assert r.iloc[2] > r.iloc[0]


def test_factor_percentile_lower_is_better_inverts():
    # 应计利润:越低越好 → 最小 raw 拿最高百分位
    s = pd.Series([1.0, 2.0, 3.0])
    ind = pd.Series(["A", "A", "A"])
    r = factor_percentile(s, ind, higher_is_better=False)
    assert r.iloc[0] > r.iloc[2]


def test_factor_percentile_size_neutral_removes_size_effect():
    # 因子值与市值完全同向(纯小盘/大盘代理):做 size 中性后,组内应无差异 → 分位≈组内中位
    # 20只:2个行业×流动市值梯度;因子=市值本身(极端代理情形)
    n = 20
    s = pd.Series([float(i) for i in range(n)])            # 因子=“市值”
    ind = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2))
    logmv = pd.Series([float(i) for i in range(n)])        # 市值同序
    r_no = factor_percentile(s, ind, higher_is_better=True)
    r_sz = factor_percentile(s, ind, higher_is_better=True, logmv=logmv)
    # 无 size 中性:最大市值票拿最高分位;有 size 中性:纯市值代理被去掉,顶部分位应下降
    assert r_no.iloc[-1] == r_no.max()
    assert r_sz.iloc[-1] < r_no.iloc[-1]


def test_factor_percentile_size_neutral_keeps_within_size_ranking():
    # 同一市值档内的真实因子差异应保留:市值同一档、因子不同 → 排序不变
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    ind = pd.Series(["A"] * 4)
    logmv = pd.Series([10.0, 10.0, 10.0, 10.0])  # 市值全同 → size 中性不应扭曲因子序
    r = factor_percentile(s, ind, higher_is_better=True, logmv=logmv)
    assert r.iloc[3] > r.iloc[0]


def test_composite_equal_weight_average():
    c = composite(pd.DataFrame({"f1": [0.8, 0.2], "f2": [0.6, 0.4]}))
    assert abs(c.iloc[0] - 0.7) < 1e-9
    assert abs(c.iloc[1] - 0.3) < 1e-9


def test_composite_skips_missing_factor_not_zero_fill():
    # 缺某因子时用可得因子均值,不当 0 填(0 填会无依据地惩罚)
    c = composite(pd.DataFrame({"f1": [0.8, 0.2], "f2": [None, 0.4]}))
    assert abs(c.iloc[0] - 0.8) < 1e-9   # 只有 f1
    assert abs(c.iloc[1] - 0.3) < 1e-9   # mean(0.2,0.4)


def test_to_decile_top_and_bottom():
    d = to_decile(pd.Series([float(i) for i in range(100)]))
    assert d.iloc[-1] == 10
    assert d.iloc[0] == 1


# ---- touched_limit_up:⚡脉冲的定义性锚(近5交易日 high 触及 stk_limit 涨停价) ----

def _daily(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts_code", "trade_date", "high"])


def _limit(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts_code", "trade_date", "up_limit"])


def test_touched_limit_up_hit_vs_miss():
    # high==涨停价 → 触及;差 1 分钱 → 未触及
    daily = _daily([("600000.SH", "20260701", 11.00), ("000001.SZ", "20260701", 10.99)])
    lim = _limit([("600000.SH", "20260701", 11.00), ("000001.SZ", "20260701", 11.00)])
    assert touched_limit_up(daily, lim) == {"600000.SH"}


def test_touched_limit_up_float_tolerance():
    # 浮点表示误差(如 21.45*1.1 存出 23.594999…)不应漏判:容差 1e-6 远小于报价最小变动 0.01
    daily = _daily([("600000.SH", "20260701", 23.595 - 1e-9)])
    lim = _limit([("600000.SH", "20260701", 23.595)])
    assert touched_limit_up(daily, lim) == {"600000.SH"}


def test_touched_limit_up_same_day_pairing():
    # 涨停价按 (ts_code, trade_date) 同日配对,不跨日错配:
    # 0630 high=11 只对 0630 的涨停价 12(未触及),不能拿去比 0701 的涨停价 11
    daily = _daily([("600000.SH", "20260630", 11.00), ("600000.SH", "20260701", 10.00)])
    lim = _limit([("600000.SH", "20260630", 12.00), ("600000.SH", "20260701", 11.00)])
    assert touched_limit_up(daily, lim) == set()


def test_touched_limit_up_any_day_in_window():
    # 窗口内任一日触及即入集合(之后回落也算——"涨停顶上来的"正是要抓的形态)
    daily = _daily([("600000.SH", "20260629", 11.00), ("600000.SH", "20260630", 10.20),
                    ("600000.SH", "20260701", 10.10)])
    lim = _limit([("600000.SH", "20260629", 11.00), ("600000.SH", "20260630", 12.10),
                  ("600000.SH", "20260701", 11.22)])
    assert touched_limit_up(daily, lim) == {"600000.SH"}


def test_touched_limit_up_empty_input_fail_loud():
    # 输入为空 → fail-loud:静默返回空集会让 ⚡ 全体消失、看似"这5天没人涨停"
    daily = _daily([("600000.SH", "20260701", 11.00)])
    lim = _limit([("600000.SH", "20260701", 11.00)])
    with pytest.raises(ValueError):
        touched_limit_up(daily.iloc[0:0], lim)
    with pytest.raises(ValueError):
        touched_limit_up(daily, lim.iloc[0:0])


def test_touched_limit_up_missing_column_fail_loud():
    # 缺关键列(如 up_limit)→ fail-loud,不静默当作没人触及
    daily = _daily([("600000.SH", "20260701", 11.00)])
    bad = pd.DataFrame([("600000.SH", "20260701")], columns=["ts_code", "trade_date"])
    with pytest.raises(ValueError):
        touched_limit_up(daily, bad)
