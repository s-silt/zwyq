"""Tests for market_temp —— 市场温度读数(涨停/炸板、成交额比值、北向成交额、摘要行)。

只 surface 读数不加权综合;全部纯函数,空表/坏数据 fail-loud 不伪造。
"""
import math

import pandas as pd
import pytest

from ashare_gauntlet.market_temp import (
    BASELINE_WINDOW,
    EmptyLimitListError,
    EmptyNorthTurnoverError,
    InsufficientAmountHistoryError,
    InsufficientNorthHistoryError,
    UnknownLimitStatusError,
    amount_ratio,
    limit_counts,
    north_turnover_recent,
    summary_line,
)


# ---------- 涨停/炸板/跌停计数(limit_list_d 的 limit 列:U/Z/D) ----------

def _limit_df(statuses: list[str]) -> pd.DataFrame:
    # 显式给列名:空表也带 schema(真实 tushare 空拉返回的就是带列的空 DataFrame)
    return pd.DataFrame([{"ts_code": f"{i:06d}.SZ", "limit": s} for i, s in enumerate(statuses)],
                        columns=["ts_code", "limit"])


def test_limit_counts_basic():
    lim = limit_counts(_limit_df(["U", "U", "U", "Z", "D"]))
    assert lim["up"] == 3
    assert lim["broken"] == 1
    assert lim["down"] == 1
    # 炸板率 = Z/(U+Z) = 1/4
    assert abs(lim["broken_rate"] - 0.25) < 1e-9


def test_limit_counts_no_up_no_broken_rate_is_nan():
    # 只有跌停、无涨停也无炸板 → 炸板率分母为 0,NaN 不伪造成 0
    lim = limit_counts(_limit_df(["D", "D"]))
    assert lim["up"] == 0 and lim["broken"] == 0 and lim["down"] == 2
    assert lim["broken_rate"] != lim["broken_rate"]  # NaN


def test_limit_counts_empty_fails_loud():
    # A 股全市场 ~5000 只,真实交易日涨跌停名单不可能 0 行 —— 空表=拉取失败,fail-loud
    with pytest.raises(EmptyLimitListError):
        limit_counts(_limit_df([]))


def test_limit_counts_unknown_status_fails_loud():
    # limit 列编码漂移(镜像返回 U/Z/D 之外的值)→ 拒绝静默丢行,fail-loud
    with pytest.raises(UnknownLimitStatusError, match="X"):
        limit_counts(_limit_df(["U", "X"]))


# ---------- 全市场成交额:今日 vs 前 20 日均值的比值(比值无阈值) ----------

def test_amount_ratio_today_vs_prior_window_mean():
    # 默认窗口 = 代码库既有 20 日约定:基准为不含今日的前 20 日均值
    amounts = [100.0] * BASELINE_WINDOW + [150.0]
    r = amount_ratio(amounts)
    assert abs(r["today"] - 150.0) < 1e-9
    assert abs(r["baseline"] - 100.0) < 1e-9
    assert abs(r["ratio"] - 1.5) < 1e-9
    assert r["window"] == BASELINE_WINDOW == 20


def test_amount_ratio_uses_only_last_window_days_as_baseline():
    # 更久远的历史不进基准:窗口外的 1e9 不应影响结果
    amounts = [1e9] * 3 + [200.0] * BASELINE_WINDOW + [100.0]
    r = amount_ratio(amounts)
    assert abs(r["baseline"] - 200.0) < 1e-9
    assert abs(r["ratio"] - 0.5) < 1e-9


def test_amount_ratio_insufficient_history_fails_loud():
    # 需要 window+1 个值(前 window 日基准 + 今日),同 regime_return 的 n+1 约定
    with pytest.raises(InsufficientAmountHistoryError):
        amount_ratio([100.0] * BASELINE_WINDOW)   # 只有 20 个 → 差 1
    with pytest.raises(InsufficientAmountHistoryError):
        amount_ratio([])                          # 空序列同理


def test_amount_ratio_nan_in_window_fails_loud():
    # 窗口内 NaN=某日缓存坏值,静默跳过会错移基准 → fail-loud 不藏不补
    amounts = [100.0] * (BASELINE_WINDOW - 1) + [math.nan] + [100.0, 150.0]
    with pytest.raises(InsufficientAmountHistoryError, match="NaN"):
        amount_ratio(amounts)


# ---------- 北向成交额(净流入 2024-08-19 制度性停披露 → 只 surface 成交额) ----------

def _mf(rows: list[tuple[str, object]]) -> pd.DataFrame:
    # north_money/hgt/sgt 模拟真实 API:字符串、单位百万元
    return pd.DataFrame(
        [{"trade_date": d, "hgt": "0", "sgt": "0", "north_money": v} for d, v in rows],
        columns=["trade_date", "hgt", "sgt", "north_money"])


def test_north_turnover_recent_latest_and_sum():
    mf = _mf([("20260625", "100000.0"), ("20260626", "200000.0"), ("20260629", "300000.0"),
              ("20260630", "400000.0"), ("20260701", "500000.0"), ("20260702", "600000.0")])
    n = north_turnover_recent(mf)
    assert n["latest_date"] == "20260702"
    assert abs(n["latest_yi"] - 6000.0) < 1e-9          # 百万→亿 = /100
    # 近 5 个有数日累计:(2+3+4+5+6)万百万 = 20000 亿(最早的 20260625 不计)
    assert abs(n["sum_yi"] - 20000.0) < 1e-9
    assert n["days"] == 5


def test_north_turnover_recent_pre_cutoff_fails_loud():
    # 2024-08-19 前 north_money 是净流入语义(~50x 量级差)→ 拒绝混入,防灾难性误读
    mf = _mf([("20240812", "5000.0"), ("20240813", "5000.0"), ("20240814", "5000.0"),
              ("20240815", "5000.0"), ("20240816", "5000.0")])
    with pytest.raises(ValueError, match="20240819"):
        north_turnover_recent(mf)


def test_north_turnover_recent_empty_fails_loud():
    with pytest.raises(EmptyNorthTurnoverError):
        north_turnover_recent(_mf([]))


def test_north_turnover_recent_insufficient_days_fails_loud():
    # 拉取窗口(20 交易日)内有数日必然 ≥5;不足=拉取失败,fail-loud 不缩窗伪造
    mf = _mf([("20260701", "500000.0"), ("20260702", "600000.0")])
    with pytest.raises(InsufficientNorthHistoryError):
        north_turnover_recent(mf)


def test_north_turnover_recent_nan_fails_loud():
    # 有行但 north_money 为空值=坏数据(HK 休市日是整行缺失,不是 NaN 行)→ fail-loud
    mf = _mf([("20260625", "100000.0"), ("20260626", "200000.0"), ("20260629", "300000.0"),
              ("20260630", "400000.0"), ("20260701", None), ("20260702", "600000.0")])
    with pytest.raises(EmptyNorthTurnoverError, match="20260701"):
        north_turnover_recent(mf)


# ---------- 摘要行:一行 surface 全部读数,不加权综合 ----------

def test_summary_line_contains_all_readings():
    lim = {"up": 67, "broken": 15, "down": 3, "broken_rate": 15 / 82}
    amt = {"today": 34739.0, "baseline": 30000.0, "ratio": 34739.0 / 30000.0, "window": 20}
    north = {"latest_date": "20260702", "latest_yi": 3851.0, "sum_yi": 19696.0, "days": 5}
    line = summary_line("20260702", lim, amt, north, regime_pct=0.023, regime_window=20)
    for piece in ("20260702", "涨停67", "炸板15", "18%", "跌停3", "34739亿", "×1.16",
                  "3851亿", "19696亿", "+2.3%"):
        assert piece in line, f"摘要行缺 {piece!r}: {line}"
    assert "\n" not in line  # 一行


def test_summary_line_nan_rendered_as_na_not_zero():
    # NaN 读数(炸板率分母 0 / regime 数据不足)显示 n/a,不伪造成 0
    lim = {"up": 0, "broken": 0, "down": 2, "broken_rate": math.nan}
    amt = {"today": 100.0, "baseline": 100.0, "ratio": 1.0, "window": 20}
    north = {"latest_date": "20260702", "latest_yi": 100.0, "sum_yi": 500.0, "days": 5}
    line = summary_line("20260702", lim, amt, north, regime_pct=math.nan, regime_window=20)
    assert line.count("n/a") == 2
